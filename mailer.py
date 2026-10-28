"""
mailer.py
----------
Sends the password-reset email.

Two modes (chosen automatically):

  1. SMTP mode  - if JOBGUARD_SMTP_HOST / _USER / _PASSWORD are set (in .env), a real email
                  is sent. Works with Gmail (use an "App Password"), Outlook, Brevo, etc.
  2. Console mode - if SMTP is not configured, the reset link is PRINTED IN THE TERMINAL
                  running app.py. This is for local demos only: the link never appears in
                  the web page, so a stranger cannot reset someone else's account.

The email is sent from a background thread so the page responds in the same time whether or
not the email address has an account (otherwise response time would reveal which emails exist).
"""

import logging
import os
import smtplib
import ssl
import threading
from email.message import EmailMessage

log = logging.getLogger("jobguard.mail")


def smtp_configured():
    return all(os.environ.get(k) for k in ("JOBGUARD_SMTP_HOST", "JOBGUARD_SMTP_USER", "JOBGUARD_SMTP_PASSWORD"))


def _build_message(to_email, name, link, minutes):
    sender = os.environ.get("JOBGUARD_SMTP_FROM") or os.environ.get("JOBGUARD_SMTP_USER") or "jobguard@localhost"
    msg = EmailMessage()
    msg["Subject"] = "Reset your JobGuard AI password"
    msg["From"] = sender
    msg["To"] = to_email
    msg.set_content(
        f"Hi {name},\n\n"
        f"We received a request to reset your JobGuard AI password.\n\n"
        f"Reset it here (the link works once and expires in {minutes} minutes):\n{link}\n\n"
        f"If you didn't ask for this, you can ignore this email - your password will stay the same.\n\n"
        f"JobGuard AI will never ask you for your password, bank details or any payment by email.\n"
    )
    return msg


def _smtp_send(msg):
    host = os.environ["JOBGUARD_SMTP_HOST"]
    port = int(os.environ.get("JOBGUARD_SMTP_PORT", "587"))
    user = os.environ["JOBGUARD_SMTP_USER"]
    password = os.environ["JOBGUARD_SMTP_PASSWORD"]
    context = ssl.create_default_context()
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=15, context=context) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.starttls(context=context)
                server.login(user, password)
                server.send_message(msg)
        log.info("Password reset email sent.")
    except Exception as exc:  # never crash the request thread, never print the password
        log.error("Could not send reset email (%s): %s", exc.__class__.__name__, exc)


def send_reset_email(to_email, name, link, minutes):
    """Returns 'smtp' or 'console' so callers/tests know which path was used."""
    if smtp_configured():
        msg = _build_message(to_email, name, link, minutes)
        threading.Thread(target=_smtp_send, args=(msg,), daemon=True).start()
        return "smtp"

    bar = "=" * 72
    print(f"\n{bar}\n  PASSWORD RESET (console mode - SMTP is not configured)\n"
          f"  Account : {to_email}\n  Link    : {link}\n"
          f"  Expires : {minutes} minutes, single use\n{bar}\n", flush=True)
    return "console"
