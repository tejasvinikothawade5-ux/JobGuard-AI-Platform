"""
security.py
------------
All security helpers for JobGuard AI in one place (no extra libraries needed):

  * load_dotenv / load_secret_key  -> secrets come from .env / a random file, never a known default
  * CSRF tokens                    -> every state-changing form must carry a per-session token
  * same_origin                    -> rejects cross-site POSTs (defence in depth)
  * RateLimiter                    -> slows brute force on login / reset / signup
  * safe_next_url                  -> stops "open redirect" tricks via ?next=
  * password_problem / valid_email -> input validation
  * apply_security_headers         -> CSP, clickjacking, MIME-sniffing, caching headers

Note: the rate limiter lives in memory, so it resets when the server restarts and
is per-process. That is fine for a single-server project / demo. For a multi-server
deployment, move the counters to Redis.
"""

import hmac
import os
import re
import secrets
import threading
import time
from collections import defaultdict, deque
from urllib.parse import urlparse

from flask import request, session

# ---------------------------------------------------------------------
# Secrets / configuration
# ---------------------------------------------------------------------

def load_dotenv(path):
    """Tiny .env reader (KEY=VALUE per line). Never overrides real environment variables."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


def load_secret_key(base_dir):
    """
    Order of preference:
      1. JOBGUARD_SECRET_KEY environment variable / .env (use this in production)
      2. a random key generated once and saved to .secret_key (git-ignored)
    There is deliberately NO hard-coded fallback: a published default key lets anyone
    forge login cookies.
    """
    env_key = os.environ.get("JOBGUARD_SECRET_KEY", "")
    if len(env_key) >= 16:
        return env_key

    path = os.path.join(base_dir, ".secret_key")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            existing = fh.read().strip()
        if len(existing) >= 32:
            return existing

    new_key = secrets.token_hex(32)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(new_key)
    try:
        os.chmod(path, 0o600)  # owner-only (ignored on Windows)
    except OSError:
        pass
    return new_key


# ---------------------------------------------------------------------
# CSRF protection
# ---------------------------------------------------------------------

_CSRF_KEY = "_csrf"


def csrf_token():
    token = session.get(_CSRF_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[_CSRF_KEY] = token
    return token


def csrf_valid(req):
    expected = session.get(_CSRF_KEY)
    sent = req.form.get("csrf_token") or req.headers.get("X-CSRF-Token") or ""
    return bool(expected) and hmac.compare_digest(str(expected), str(sent))


def same_origin(req):
    """If the browser tells us where the request came from, it must be us."""
    origin = req.headers.get("Origin")
    if origin:
        return urlparse(origin).netloc == req.host
    referer = req.headers.get("Referer")
    if referer:
        return urlparse(referer).netloc == req.host
    return True  # no header at all (older browsers / tools): the CSRF token still protects forms


# ---------------------------------------------------------------------
# Rate limiting (in memory, sliding window)
# ---------------------------------------------------------------------

class RateLimiter:
    def __init__(self):
        self._events = defaultdict(deque)
        self._lock = threading.Lock()
        self._calls = 0

    def _trim(self, dq, window, now):
        while dq and dq[0] <= now - window:
            dq.popleft()

    def _prune(self, now):
        self._calls += 1
        if self._calls % 500:
            return
        for key in list(self._events):
            dq = self._events[key]
            self._trim(dq, 3600, now)
            if not dq:
                del self._events[key]

    def blocked(self, key, limit, window):
        """Read-only: is this key already at its limit? -> (blocked, seconds_to_wait)"""
        now = time.time()
        with self._lock:
            dq = self._events[key]
            self._trim(dq, window, now)
            if len(dq) >= limit:
                return True, max(1, int(dq[0] + window - now))
            return False, 0

    def hit(self, key, window=None):
        now = time.time()
        with self._lock:
            self._events[key].append(now)
            self._prune(now)

    def check_and_hit(self, key, limit, window):
        """Count this attempt and say whether it is allowed -> (allowed, seconds_to_wait)"""
        now = time.time()
        with self._lock:
            dq = self._events[key]
            self._trim(dq, window, now)
            if len(dq) >= limit:
                return False, max(1, int(dq[0] + window - now))
            dq.append(now)
            self._prune(now)
            return True, 0

    def reset(self, key):
        with self._lock:
            self._events.pop(key, None)


limiter = RateLimiter()


def client_ip():
    # Deliberately NOT trusting X-Forwarded-For: it is trivially spoofed unless you sit
    # behind a proxy you control.
    return request.remote_addr or "unknown"


# ---------------------------------------------------------------------
# Redirect safety
# ---------------------------------------------------------------------

def safe_next_url(target, default):
    """Only allow plain same-site paths like /history. Blocks //evil.com, /\\evil.com, http://evil.com, etc."""
    if not target:
        return default
    t = target.strip()
    if not t.startswith("/") or t.startswith("//") or "\\" in t:
        return default
    if any(ord(c) < 32 for c in t):  # tabs/newlines are stripped by browsers: "/\t/evil.com" -> "//evil.com"
        return default
    parsed = urlparse(t)
    if parsed.scheme or parsed.netloc:
        return default
    return t


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890",
    "qwerty123", "qwertyuiop", "iloveyou1", "admin123", "welcome1", "welcome123",
    "abc12345", "letmein123", "passw0rd", "pass1234", "india123", "test1234",
}


def valid_email(email):
    return bool(email) and len(email) <= 254 and bool(_EMAIL_RE.match(email))


def password_problem(password, email=""):
    """Return a human-readable problem with the password, or None if it is acceptable."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if len(password) > 128:
        return "Password is too long (maximum 128 characters)."
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "Use at least one letter and one number in your password."
    if password.lower() in _COMMON_PASSWORDS:
        return "That password is too common. Please choose something less guessable."
    local = (email or "").split("@")[0].lower()
    if len(local) >= 4 and local in password.lower():
        return "Your password shouldn't contain your email name."
    return None


# ---------------------------------------------------------------------
# Response headers
# ---------------------------------------------------------------------

_CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self' https://fonts.googleapis.com",
    "font-src https://fonts.gstatic.com",
    "img-src 'self' data:",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
])


def apply_security_headers(resp, no_store=False):
    resp.headers["Content-Security-Policy"] = _CSP
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if no_store:
        # stops the browser's Back button from showing a logged-out user their old history page
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Pragma"] = "no-cache"
    return resp
