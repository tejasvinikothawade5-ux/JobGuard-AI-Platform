"""
app.py
------
JobGuard AI - Flask backend.

Serves the interactive frontend and exposes:
  POST /predict   -> analyze a job posting (title, description, url, etc.)
  GET  /health     -> simple health check
  GET  /model-info -> metrics of the currently loaded model

Accounts:  /signup  /login  /logout  /forgot-password  /reset-password/<token>  /change-password
Job seeker: /profile  /recommendations  /history

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import functools
import json
import logging
import math
import os
import secrets
import sqlite3
import sys
from datetime import datetime, timedelta
from urllib.parse import urlparse

import joblib
import numpy as np
from scipy.sparse import hstack, csr_matrix
from flask import (Flask, request, jsonify, render_template, session, redirect,
                   url_for, flash, g)
from werkzeug.security import generate_password_hash, check_password_hash

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "model"))

import security as sec

sec.load_dotenv(os.path.join(HERE, ".env"))  # optional: SMTP settings, secret key, etc.

from feature_engineering import (
    extract_all_features, FEATURE_ORDER, build_red_flags,
    risk_score_from_probability, risk_level, apply_recommendation,
)
import database as db
import mailer
from recommender import recommend_jobs, get_listing

ARTIFACT_PATH = os.path.join(HERE, "model", "artifacts.joblib")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=sec.load_secret_key(HERE),
    SESSION_COOKIE_HTTPONLY=True,          # JavaScript cannot read the login cookie
    SESSION_COOKIE_SAMESITE="Lax",         # not sent on cross-site POSTs
    SESSION_COOKIE_SECURE=os.environ.get("JOBGUARD_HTTPS") == "1",  # set JOBGUARD_HTTPS=1 once deployed on https
    PERMANENT_SESSION_LIFETIME=timedelta(days=7),
    MAX_CONTENT_LENGTH=1 * 1024 * 1024,    # reject request bodies over 1 MB
)

db.init_db()

# ---------------------------------------------------------------------
# Load trained model artifacts once at startup
# ---------------------------------------------------------------------
if not os.path.exists(ARTIFACT_PATH):
    raise RuntimeError(
        "Model artifacts not found. Run:\n"
        "    cd model && python dataset_generator.py && python train_model.py\n"
        "before starting the app."
    )

_artifacts = joblib.load(ARTIFACT_PATH)
MODEL = _artifacts["model"]
VECTORIZER = _artifacts["vectorizer"]
SCALER = _artifacts["scaler"]
METRICS = _artifacts["metrics"]


def predict_job(title, description, company_profile, requirements, url):
    full_text = f"{title} {description} {company_profile} {requirements}"
    X_text = VECTORIZER.transform([full_text])

    feats = extract_all_features(title, description, company_profile, requirements, url)
    X_num = SCALER.transform(np.array([[feats.get(k, 0) for k in FEATURE_ORDER]]))

    X = hstack([X_text, csr_matrix(X_num)])

    proba = MODEL.predict_proba(X)[0]
    prob_fraud = float(proba[1])
    prediction = int(MODEL.predict(X)[0])

    score = risk_score_from_probability(prob_fraud)
    level = risk_level(score)
    recommendation = apply_recommendation(score)
    red_flags = build_red_flags(feats, url)

    # Sort red flags by severity (high -> medium -> low) for display
    severity_order = {"high": 0, "medium": 1, "low": 2}
    red_flags_sorted = sorted(red_flags, key=lambda f: severity_order.get(f[0], 3))

    return {
        "is_fraudulent": bool(prediction),
        "label": "Fraudulent" if prediction == 1 else "Genuine",
        "probability_fraud": round(prob_fraud * 100, 2),
        "probability_genuine": round((1 - prob_fraud) * 100, 2),
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation,
        "red_flags": [{"severity": s, "message": m} for s, m in red_flags_sorted],
        "feature_snapshot": feats,
    }


# =====================================================================
#  SECURITY GATE  (runs before every request)
# =====================================================================

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# /predict takes JSON only (a cross-site <form> cannot send application/json, and other sites are
# blocked by CORS) and is also covered by the same-origin check + rate limit below, so it does not
# need a form token. Leaving it exempt keeps the analyzer working exactly as before for everyone.
CSRF_EXEMPT_ENDPOINTS = {"predict"}

# pages whose content must never be cached by the browser
NO_STORE_ENDPOINTS = {"login", "signup", "forgot_password", "reset_password", "predict"}

# a pre-computed hash so that "unknown email" and "wrong password" take the same time
_DUMMY_HASH = generate_password_hash(secrets.token_hex(8))

LOGIN_MAX_PER_ACCOUNT_IP = 5     # failed logins per (ip + email) ...
LOGIN_MAX_PER_IP = 20            # ... and per ip, within
LOGIN_WINDOW = 15 * 60           # 15 minutes


def _wants_json():
    return request.path == "/predict" or request.path.startswith("/api/")


def _error_response(status, message):
    if _wants_json():
        return jsonify({"error": message}), status
    return render_template("error.html", code=status, message=message, active_page=""), status


@app.before_request
def security_gate():
    g.user = None
    if request.endpoint == "static":
        return None

    # 1) Who is logged in? The session must match the account's current token_version, so a
    #    password change/reset instantly logs out every other device.
    uid = session.get("user_id")
    if uid:
        user = db.get_user_by_id(uid)
        if user and session.get("sv") == user["token_version"]:
            g.user = user
        else:
            session.pop("user_id", None)
            session.pop("sv", None)

    # 2) Block cross-site state-changing requests
    if request.method in UNSAFE_METHODS:
        if not sec.same_origin(request):
            return _error_response(403, "Cross-site request blocked.")
        if request.endpoint not in CSRF_EXEMPT_ENDPOINTS and not sec.csrf_valid(request):
            return _error_response(
                400, "Your session expired or the form was invalid. Please go back, refresh the page and try again.")
    return None


@app.after_request
def add_security_headers(resp):
    no_store = bool(g.get("user")) or request.endpoint in NO_STORE_ENDPOINTS
    return sec.apply_security_headers(resp, no_store=no_store and request.endpoint != "static")


@app.context_processor
def inject_template_globals():
    return {
        "current_user": g.get("user"),
        "csrf_token": sec.csrf_token,
        "history_statuses": db.HISTORY_STATUSES,
    }


def login_required(view_fn):
    @functools.wraps(view_fn)
    def wrapped(*args, **kwargs):
        if not g.get("user"):
            if _wants_json():
                return jsonify({"error": "Please log in to continue."}), 401
            flash("Please log in to continue.", "error")
            return redirect(url_for("login", next=request.path))
        return view_fn(*args, **kwargs)
    return wrapped


def _start_session(user):
    """Fresh session on login: drops anything an attacker may have planted (session fixation)."""
    session.clear()
    session.permanent = True
    session["user_id"] = user["id"]
    session["sv"] = user["token_version"]


def _wait_text(seconds):
    minutes = max(1, math.ceil(seconds / 60))
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


# =====================================================================
#  AUTH ROUTES
# =====================================================================

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = (request.form.get("name") or "").strip()[:80]
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""

        allowed, wait = sec.limiter.check_and_hit(f"signup:{sec.client_ip()}", 10, 3600)
        if not allowed:
            flash(f"Too many sign-up attempts. Please try again in {_wait_text(wait)}.", "error")
            return render_template("signup.html", active_page="login"), 429

        if not name or not email or not password:
            flash("Please fill in all fields.", "error")
            return render_template("signup.html", active_page="login")
        if not sec.valid_email(email):
            flash("Please enter a valid email address.", "error")
            return render_template("signup.html", active_page="login")
        problem = sec.password_problem(password, email)
        if problem:
            flash(problem, "error")
            return render_template("signup.html", active_page="login")
        if db.get_user_by_email(email):
            flash("An account with that email already exists. Try logging in.", "error")
            return render_template("signup.html", active_page="login")

        try:
            user_id = db.create_user(name, email, generate_password_hash(password))
        except sqlite3.IntegrityError:  # two sign-ups with the same email at the same instant
            flash("An account with that email already exists. Try logging in.", "error")
            return render_template("signup.html", active_page="login")

        _start_session(db.get_user_by_id(user_id))
        flash("Account created! Let's set up your profile.", "success")
        return redirect(url_for("profile"))

    return render_template("signup.html", active_page="login")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()[:254]
        password = request.form.get("password") or ""

        ip = sec.client_ip()
        acct_key = f"login:{ip}:{email}"
        ip_key = f"login:ip:{ip}"
        blocked_a, wait_a = sec.limiter.blocked(acct_key, LOGIN_MAX_PER_ACCOUNT_IP, LOGIN_WINDOW)
        blocked_i, wait_i = sec.limiter.blocked(ip_key, LOGIN_MAX_PER_IP, LOGIN_WINDOW)
        if blocked_a or blocked_i:
            flash(f"Too many failed attempts. Please try again in {_wait_text(max(wait_a, wait_i))}.", "error")
            return render_template("login.html", active_page="login"), 429

        user = db.get_user_by_email(email) if email else None
        # always run one password check, so a wrong email and a wrong password take the same time
        password_ok = check_password_hash(user["password_hash"] if user else _DUMMY_HASH, password)

        if not user or not password_ok:
            sec.limiter.hit(acct_key)
            sec.limiter.hit(ip_key)
            flash("Incorrect email or password.", "error")
            return render_template("login.html", active_page="login"), 401

        sec.limiter.reset(acct_key)
        _start_session(user)
        flash(f"Welcome back, {user['name']}!", "success")
        return redirect(sec.safe_next_url(request.args.get("next"), url_for("recommendations")))

    if g.user:
        return redirect(url_for("recommendations"))
    return render_template("login.html", active_page="login")


@app.route("/logout", methods=["GET", "POST"])
def logout():
    if request.method == "POST":  # POST + CSRF token, so another site can't log you out
        session.clear()
        flash("You've been logged out.", "success")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------
# Forgot / reset password
# ---------------------------------------------------------------------

def _reset_base_url():
    """
    Where the emailed link points. Using the browser's Host header would let an attacker poison
    the link ("reset poisoning"), so for real emails set JOBGUARD_BASE_URL in .env.
    """
    configured = os.environ.get("JOBGUARD_BASE_URL", "").strip().rstrip("/")
    if configured:
        return configured
    if request.host.split(":")[0] in ("127.0.0.1", "localhost"):
        return request.host_url.rstrip("/")
    return request.host_url.rstrip("/") if not mailer.smtp_configured() else None


def _issue_reset(user):
    base = _reset_base_url()
    if base is None:
        logging.getLogger("jobguard.mail").error(
            "Reset email NOT sent: set JOBGUARD_BASE_URL in .env (e.g. https://yourapp.com) before using real email.")
        return
    token = db.create_reset_token(user["id"])
    link = base + url_for("reset_password", token=token)
    mailer.send_reset_email(user["email"], user["name"], link, db.RESET_TTL_MINUTES)


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    sent = False
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()[:254]
        ip_ok, _ = sec.limiter.check_and_hit(f"forgot:ip:{sec.client_ip()}", 5, 900)
        if not ip_ok:
            flash("Too many reset requests. Please wait a few minutes and try again.", "error")
            return render_template("forgot_password.html", sent=False, active_page="login"), 429

        email_ok = True
        if email:
            email_ok, _ = sec.limiter.check_and_hit(f"forgot:email:{email}", 3, 3600)

        if email_ok and sec.valid_email(email):
            user = db.get_user_by_email(email)
            if user:
                _issue_reset(user)

        # The SAME answer whether or not the email has an account, so this page can't be used
        # to discover who is registered.
        sent = True
    return render_template("forgot_password.html", sent=sent, active_page="login")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    if len(token) > 100:
        return _error_response(404, "That page doesn't exist.")

    row = db.get_reset(token)
    if not row:
        return render_template("reset_password.html", invalid=True, token=token, active_page="login"), 400

    if request.method == "POST":
        allowed, wait = sec.limiter.check_and_hit(f"reset:ip:{sec.client_ip()}", 10, 900)
        if not allowed:
            flash(f"Too many attempts. Please try again in {_wait_text(wait)}.", "error")
            return render_template("reset_password.html", invalid=False, token=token, active_page="login"), 429

        password = request.form.get("password") or ""
        confirm = request.form.get("confirm_password") or ""
        owner = db.get_user_by_id(row["user_id"])

        if password != confirm:
            flash("The two passwords don't match.", "error")
            return render_template("reset_password.html", invalid=False, token=token, active_page="login")
        problem = sec.password_problem(password, owner["email"] if owner else "")
        if problem:
            flash(problem, "error")
            return render_template("reset_password.html", invalid=False, token=token, active_page="login")

        user_id = db.consume_reset(token)  # atomic: the link works exactly once
        if not user_id:
            return render_template("reset_password.html", invalid=True, token=token, active_page="login"), 400

        db.set_password(user_id, generate_password_hash(password))
        session.clear()
        flash("Password updated. Please log in with your new password.", "success")
        return redirect(url_for("login"))

    return render_template("reset_password.html", invalid=False, token=token, active_page="login")


@app.route("/change-password", methods=["POST"])
@login_required
def change_password():
    user = g.user
    current = request.form.get("current_password") or ""
    new = request.form.get("new_password") or ""
    confirm = request.form.get("confirm_password") or ""

    key = f"chgpw:{user['id']}"
    blocked, wait = sec.limiter.blocked(key, 5, 900)
    if blocked:
        flash(f"Too many attempts. Please try again in {_wait_text(wait)}.", "error")
        return redirect(url_for("profile", _anchor="security"))

    if not check_password_hash(user["password_hash"], current):
        sec.limiter.hit(key)
        flash("Your current password is incorrect.", "error")
    elif new != confirm:
        flash("The two new passwords don't match.", "error")
    elif sec.password_problem(new, user["email"]):
        flash(sec.password_problem(new, user["email"]), "error")
    else:
        new_version = db.set_password(user["id"], generate_password_hash(new))
        session["sv"] = new_version  # keep THIS browser logged in; every other device is logged out
        flash("Password changed. Other devices have been logged out.", "success")
    return redirect(url_for("profile", _anchor="security"))


# ---------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user_id = g.user["id"]

    if request.method == "POST":
        skills = (request.form.get("skills") or "").strip()[:500]
        experience_years = request.form.get("experience_years") or 0
        preferred_role = (request.form.get("preferred_role") or "").strip()[:100]
        preferred_location = (request.form.get("preferred_location") or "").strip()[:100]
        bio = (request.form.get("bio") or "").strip()[:1000]

        try:
            experience_years = max(0, min(60, int(experience_years)))
        except ValueError:
            experience_years = 0

        db.upsert_profile(user_id, skills, experience_years, preferred_role, preferred_location, bio)
        flash("Profile saved.", "success")
        return redirect(url_for("profile"))

    profile_data = db.get_profile(user_id)
    return render_template("profile.html", profile=profile_data, active_page="profile")


# ---------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------

@app.route("/recommendations")
@login_required
def recommendations():
    profile_data = db.get_profile(g.user["id"]) or {}
    skills = profile_data.get("skills", "")
    preferred_role = profile_data.get("preferred_role", "")
    preferred_location = profile_data.get("preferred_location", "")

    jobs, screened_out = [], 0
    if skills or preferred_role:
        jobs, screened_out = recommend_jobs(
            skills=skills,
            preferred_role=preferred_role,
            preferred_location=preferred_location,
            predict_fn=predict_job,
            top_n=6,
        )

    status_map = db.history_status_map(g.user["id"])
    status_labels = dict(db.HISTORY_STATUSES)
    for job in jobs:
        key = f"{job['title'].lower()}|{job['company'].lower()}"
        st = status_map.get(key)
        job["status"] = st
        job["status_label"] = status_labels.get(st) if st and st != "analyzed" else None

    return render_template(
        "recommendations.html",
        jobs=jobs,
        screened_out=screened_out,
        has_profile=bool(skills or preferred_role),
        active_page="recommendations",
    )


# ---------------------------------------------------------------------
# History (analyzed / saved / applied jobs)
# ---------------------------------------------------------------------

def _present_history(row):
    try:
        flags = json.loads(row["red_flags"])
    except (ValueError, TypeError):
        flags = []

    def fmt(ts):
        try:
            return datetime.fromisoformat(ts).strftime("%d %b %Y")
        except (ValueError, TypeError):
            return ""

    url = row["url"] or ""
    host = urlparse(url if "://" in url else "http://" + url).netloc if url else ""
    out = dict(row)
    out.update(
        flags=flags,
        host=host,
        checked_on=fmt(row["created_at"]),
        updated_on=fmt(row["updated_at"]),
        applied_on=fmt(row["applied_at"]) if row["applied_at"] else "",
        status_label=dict(db.HISTORY_STATUSES).get(row["status"], row["status"].title()),
        risk_class="high" if row["risk_score"] >= 70 else ("medium" if row["risk_score"] >= 40 else "low"),
    )
    return out


def _high_risk_apply_warning(item):
    return ("Heads up: this listing was flagged high-risk. Never pay a fee or share bank/ID details "
            "with this employer, and consider verifying the company independently.")


@app.route("/history")
@login_required
def history():
    uid = g.user["id"]
    status = request.args.get("status", "")
    query = (request.args.get("q") or "").strip()[:100]
    if status not in db.STATUS_KEYS:
        status = ""

    rows = db.list_history(uid, status or None, query or None)
    return render_template(
        "history.html",
        items=[_present_history(r) for r in rows],
        stats=db.history_stats(uid),
        counts=db.history_status_counts(uid),
        active_status=status,
        query=query,
        active_page="history",
    )


@app.route("/history/<int:history_id>/update", methods=["POST"])
@login_required
def history_update(history_id):
    uid = g.user["id"]
    status = request.form.get("status", "")
    notes = (request.form.get("notes") or "")[:1000]

    item = db.get_history_item(uid, history_id)
    if not item:
        return _error_response(404, "That history entry doesn't exist.")
    if status not in db.STATUS_KEYS:
        flash("Please choose a valid status.", "error")
        return redirect(url_for("history"))

    db.update_history(uid, history_id, status, notes)
    if status in db.APPLIED_STATES and item["risk_score"] >= db.HIGH_RISK_SCORE:
        flash(_high_risk_apply_warning(item), "error")
    else:
        flash("History updated.", "success")
    return redirect(url_for("history", _anchor=f"item-{history_id}"))


@app.route("/history/<int:history_id>/delete", methods=["POST"])
@login_required
def history_delete(history_id):
    if db.delete_history(g.user["id"], history_id):
        flash("Removed from your history.", "success")
        return redirect(url_for("history"))
    return _error_response(404, "That history entry doesn't exist.")


@app.route("/history/from-recommendation", methods=["POST"])
@login_required
def history_from_recommendation():
    """Save / mark-as-applied a recommended job. The risk score is computed here on the server,
    never trusted from the browser."""
    action = request.form.get("action", "")
    title = (request.form.get("title") or "")[:200]
    company = (request.form.get("company") or "")[:200]
    if action not in ("saved", "applied", "skipped"):
        return _error_response(400, "Unknown action.")

    listing = get_listing(title, company)
    if not listing:
        return _error_response(404, "That listing is no longer available.")

    result = predict_job(
        title=listing["title"], description=listing["description"],
        company_profile=listing["company_profile"], requirements=listing["requirements"], url=listing["url"],
    )
    uid = g.user["id"]
    hid, _ = db.upsert_history(
        uid, "recommended", listing["title"], listing["company"], listing["url"], listing["description"],
        result["risk_score"], result["risk_level"], result["recommendation"]["verdict"], result["red_flags"],
    )
    db.update_history(uid, hid, action)
    label = dict(db.HISTORY_STATUSES)[action]
    flash(f"“{listing['title']} · {listing['company']}” marked as {label}. See it in your history.", "success")
    return redirect(url_for("recommendations"))


@app.route("/api/history/<int:history_id>/status", methods=["POST"])
@login_required
def api_history_status(history_id):
    """Used by the buttons on the Analyze page (“I applied” / “Save” / “Skip”)."""
    data = request.get_json(silent=True)
    status = data.get("status") if isinstance(data, dict) else None
    if status not in db.STATUS_KEYS:
        return jsonify({"error": "Invalid status."}), 400

    uid = g.user["id"]
    item = db.get_history_item(uid, history_id)
    if not item:
        return jsonify({"error": "Not found."}), 404

    db.update_history(uid, history_id, status)
    payload = {"ok": True, "status": status, "label": dict(db.HISTORY_STATUSES)[status]}
    if status in db.APPLIED_STATES and item["risk_score"] >= db.HIGH_RISK_SCORE:
        payload["warning"] = _high_risk_apply_warning(item)
    return jsonify(payload)


# =====================================================================
#  ANALYZER  (the detection logic is unchanged)
# =====================================================================

@app.route("/")
def index():
    return render_template("index.html", active_page="analyze")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/model-info")
def model_info():
    return jsonify({
        "model_type": type(MODEL).__name__,
        "metrics": METRICS,
        "feature_count_text": len(VECTORIZER.get_feature_names_out()),
        "feature_count_engineered": len(FEATURE_ORDER),
    })


@app.route("/predict", methods=["POST"])
def predict():
    allowed, _ = sec.limiter.check_and_hit(f"predict:{sec.client_ip()}", 60, 60)
    if not allowed:
        return jsonify({"error": "Too many analyses in a short time. Please wait a moment and try again."}), 429

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = {}

    title = (data.get("title") or "").strip()
    description = (data.get("description") or "").strip()
    company_profile = (data.get("company_profile") or "").strip()
    requirements = (data.get("requirements") or "").strip()
    url = (data.get("url") or "").strip()
    company = (data.get("company") or "").strip()

    if not description and not title:
        return jsonify({"error": "Please provide at least a job title or description."}), 400

    try:
        result = predict_job(title, description, company_profile, requirements, url)
    except Exception:
        logging.getLogger("jobguard").exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Please check your input and try again."}), 500

    # NEW: logged-in users get this analysis saved to their history automatically.
    # A problem here must never break the analysis itself.
    if g.user:
        try:
            hid, status = db.upsert_history(
                g.user["id"], "analyzer", title, company, url, description,
                result["risk_score"], result["risk_level"], result["recommendation"]["verdict"],
                result["red_flags"],
            )
            result["history_id"] = hid
            result["history_status"] = status
        except Exception:
            logging.getLogger("jobguard").exception("Could not save history")

    return jsonify(result)


# =====================================================================
#  ERRORS
# =====================================================================

@app.errorhandler(404)
def not_found(_e):
    return _error_response(404, "That page doesn't exist.")


@app.errorhandler(405)
def method_not_allowed(_e):
    return _error_response(405, "That action isn't allowed here.")


@app.errorhandler(413)
def too_large(_e):
    return _error_response(413, "That request was too large.")


@app.errorhandler(500)
def server_error(_e):
    return _error_response(500, "Something went wrong on our side. Please try again.")


# =====================================================================
#  RUN
# =====================================================================

if __name__ == "__main__":
    host = os.environ.get("JOBGUARD_HOST", "127.0.0.1")
    port = int(os.environ.get("JOBGUARD_PORT", "5000"))
    debug = os.environ.get("JOBGUARD_DEBUG") == "1"

    if debug and host not in ("127.0.0.1", "localhost"):
        # Flask's debugger lets anyone who can reach it run code on your computer.
        print("!! Refusing to run debug mode on a network address. Starting with debug OFF.")
        debug = False

    print(f"JobGuard AI running at http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}")
    if host == "0.0.0.0":
        print("Network mode ON: other devices on your WiFi can open this app (use your PC's local IP).")
    print("Password reset emails: " + ("SMTP configured" if mailer.smtp_configured()
                                       else "console mode (links print in this window)"))
    app.run(debug=debug, host=host, port=port)
