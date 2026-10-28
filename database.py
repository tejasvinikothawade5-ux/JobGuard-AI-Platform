"""
database.py
------------
SQLite persistence for JobGuard AI. Uses Python's built-in sqlite3 module -- no extra
dependencies required. Every query is parameterised (no string-built SQL), so user input
can never change the meaning of a query.

Tables:
  users(id, name, email UNIQUE, password_hash, created_at, token_version, password_changed_at)
  profiles(user_id PK/FK, skills, experience_years, preferred_role, preferred_location, bio, updated_at)
  password_resets(id, user_id FK, token_hash, expires_at, created_at)      <- NEW
  job_history(id, user_id FK, content_hash, source, title, company, url, description,
              risk_score, risk_level, verdict, red_flags, status, notes,
              applied_at, check_count, created_at, updated_at)             <- NEW

Existing databases are upgraded in place by init_db() (new tables / columns are added only
if missing), so current accounts and profiles keep working.
"""

import hashlib
import json
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "jobguard.db")

RESET_TTL_MINUTES = 30
HIGH_RISK_SCORE = 70

HISTORY_STATUSES = [
    ("analyzed", "Analyzed"),
    ("saved", "Saved"),
    ("applied", "Applied"),
    ("interview", "Interview"),
    ("offer", "Offer"),
    ("rejected", "Rejected"),
    ("skipped", "Skipped"),
]
STATUS_KEYS = {key for key, _ in HISTORY_STATUSES}
APPLIED_STATES = ("applied", "interview", "offer", "rejected")  # statuses that mean "I did apply"


def _now():
    """UTC timestamp as an ISO string (seconds precision, so string comparison is safe)."""
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat(timespec="seconds")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_column(conn, table, column, ddl):
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")


def init_db():
    conn = get_connection()
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS profiles (
            user_id INTEGER PRIMARY KEY,
            skills TEXT DEFAULT '',
            experience_years INTEGER DEFAULT 0,
            preferred_role TEXT DEFAULT '',
            preferred_location TEXT DEFAULT '',
            bio TEXT DEFAULT '',
            updated_at TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS job_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'analyzer',
            title TEXT NOT NULL DEFAULT '',
            company TEXT NOT NULL DEFAULT '',
            url TEXT NOT NULL DEFAULT '',
            description TEXT NOT NULL DEFAULT '',
            risk_score INTEGER NOT NULL DEFAULT 0,
            risk_level TEXT NOT NULL DEFAULT '',
            verdict TEXT NOT NULL DEFAULT '',
            red_flags TEXT NOT NULL DEFAULT '[]',
            status TEXT NOT NULL DEFAULT 'analyzed',
            notes TEXT NOT NULL DEFAULT '',
            applied_at TEXT,
            check_count INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (user_id, content_hash),
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_history_user_updated ON job_history (user_id, updated_at DESC);
        """
    )
    # upgrade older databases that were created before these columns existed
    _ensure_column(conn, "users", "token_version", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "users", "password_changed_at", "TEXT")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------

def create_user(name, email, password_hash):
    """Raises sqlite3.IntegrityError if the email is already registered."""
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email.lower().strip(), password_hash, _now()),
        )
        user_id = cur.lastrowid
        # create an empty profile row up front so profile updates are always an UPDATE
        conn.execute("INSERT INTO profiles (user_id, updated_at) VALUES (?, ?)", (user_id, _now()))
        conn.commit()
        return user_id
    finally:
        conn.close()


def get_user_by_email(email):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def set_password(user_id, password_hash):
    """
    Change a password. Bumps token_version, which instantly logs the account out of every
    other browser/device (sessions remember the version they were created with), and deletes
    any outstanding reset links. Returns the new token_version.
    """
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE users SET password_hash = ?, token_version = token_version + 1, "
            "password_changed_at = ? WHERE id = ?",
            (password_hash, _now(), user_id),
        )
        conn.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
        conn.commit()
        row = conn.execute("SELECT token_version FROM users WHERE id = ?", (user_id,)).fetchone()
        return row["token_version"] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Password reset tokens
#   The raw token is only ever in the emailed link. The database stores its SHA-256 hash,
#   so even someone who steals jobguard.db cannot use it to take over an account.
# ---------------------------------------------------------------------

def _hash_token(raw):
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def create_reset_token(user_id, ttl_minutes=RESET_TTL_MINUTES):
    raw = secrets.token_urlsafe(32)  # 256 bits of randomness
    expires = (datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(minutes=ttl_minutes)).isoformat(timespec="seconds")
    conn = get_connection()
    try:
        conn.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))  # only the newest link works
        conn.execute("DELETE FROM password_resets WHERE expires_at < ?", (_now(),))  # tidy up
        conn.execute(
            "INSERT INTO password_resets (user_id, token_hash, expires_at, created_at) VALUES (?, ?, ?, ?)",
            (user_id, _hash_token(raw), expires, _now()),
        )
        conn.commit()
        return raw
    finally:
        conn.close()


def get_reset(raw):
    """Return the reset row if the token exists and has not expired, else None."""
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM password_resets WHERE token_hash = ?", (_hash_token(raw),)).fetchone()
        if not row or row["expires_at"] < _now():
            return None
        return dict(row)
    finally:
        conn.close()


def consume_reset(raw):
    """Atomically use up a token (single use). Returns the user_id, or None if invalid/expired/already used."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM password_resets WHERE token_hash = ?", (_hash_token(raw),)).fetchone()
        if not row or row["expires_at"] < _now():
            conn.rollback()
            return None
        conn.execute("DELETE FROM password_resets WHERE id = ?", (row["id"],))
        conn.commit()
        return row["user_id"]
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------

def get_profile(user_id):
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def upsert_profile(user_id, skills, experience_years, preferred_role, preferred_location, bio):
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO profiles (user_id, skills, experience_years, preferred_role, preferred_location, bio, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                skills=excluded.skills,
                experience_years=excluded.experience_years,
                preferred_role=excluded.preferred_role,
                preferred_location=excluded.preferred_location,
                bio=excluded.bio,
                updated_at=excluded.updated_at
            """,
            (user_id, skills, experience_years, preferred_role, preferred_location, bio, _now()),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------
# Job history (analyzed / saved / applied listings)
#   Every query includes "user_id = ?" so one user can never read or change another
#   user's rows, even by guessing ids.
# ---------------------------------------------------------------------

def _content_hash(title, company, url, description):
    blob = "|".join(s.strip().lower() for s in (title, company, url, description))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def upsert_history(user_id, source, title, company, url, description,
                   risk_score, risk_level, verdict, red_flags):
    """
    Save a checked listing. Checking the exact same listing again does not create a duplicate:
    it refreshes the score and bumps check_count, keeping the user's status and notes.
    Returns (history_id, status).
    """
    title, company, url = title[:200], company[:200], url[:500]
    description = description[:4000]
    chash = _content_hash(title, company, url, description)
    flags_json = json.dumps(red_flags)[:20000]
    now = _now()

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT id, status FROM job_history WHERE user_id = ? AND content_hash = ?", (user_id, chash)
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE job_history SET risk_score = ?, risk_level = ?, verdict = ?, red_flags = ?, "
                "check_count = check_count + 1, updated_at = ? WHERE id = ? AND user_id = ?",
                (risk_score, risk_level, verdict, flags_json, now, row["id"], user_id),
            )
            conn.commit()
            return row["id"], row["status"]

        cur = conn.execute(
            "INSERT INTO job_history (user_id, content_hash, source, title, company, url, description, "
            "risk_score, risk_level, verdict, red_flags, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'analyzed', ?, ?)",
            (user_id, chash, source, title, company, url, description,
             risk_score, risk_level, verdict, flags_json, now, now),
        )
        conn.commit()
        return cur.lastrowid, "analyzed"
    finally:
        conn.close()


def get_history_item(user_id, history_id):
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT * FROM job_history WHERE id = ? AND user_id = ?", (history_id, user_id)
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def update_history(user_id, history_id, status, notes=None):
    """Change status (and optionally notes). Returns True if a row of THIS user was updated."""
    if status not in STATUS_KEYS:
        return False
    now = _now()
    applied_clause = "applied_at = COALESCE(applied_at, ?)" if status in APPLIED_STATES else "applied_at = NULL"
    params = [status]
    sql = f"UPDATE job_history SET status = ?, {applied_clause}, updated_at = ?"
    if status in APPLIED_STATES:
        params.append(now)
    params.append(now)
    if notes is not None:
        sql += ", notes = ?"
        params.append(notes[:1000])
    sql += " WHERE id = ? AND user_id = ?"
    params.extend([history_id, user_id])

    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def delete_history(user_id, history_id):
    conn = get_connection()
    try:
        cur = conn.execute("DELETE FROM job_history WHERE id = ? AND user_id = ?", (history_id, user_id))
        conn.commit()
        return cur.rowcount == 1
    finally:
        conn.close()


def list_history(user_id, status=None, query=None, limit=100):
    sql = "SELECT * FROM job_history WHERE user_id = ?"
    params = [user_id]
    if status in STATUS_KEYS:
        sql += " AND status = ?"
        params.append(status)
    if query:
        like = "%" + query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        sql += " AND (title LIKE ? ESCAPE '\\' OR company LIKE ? ESCAPE '\\' OR notes LIKE ? ESCAPE '\\')"
        params.extend([like, like, like])
    sql += " ORDER BY updated_at DESC, id DESC LIMIT ?"
    params.append(int(limit))

    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def history_stats(user_id):
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT
              COUNT(*)                                                         AS total,
              COALESCE(SUM(CASE WHEN risk_score >= ? THEN 1 ELSE 0 END), 0)    AS flagged,
              COALESCE(SUM(CASE WHEN applied_at IS NOT NULL THEN 1 ELSE 0 END), 0) AS applied,
              COALESCE(SUM(CASE WHEN risk_score >= ? AND applied_at IS NULL THEN 1 ELSE 0 END), 0) AS avoided
            FROM job_history WHERE user_id = ?
            """,
            (HIGH_RISK_SCORE, HIGH_RISK_SCORE, user_id),
        ).fetchone()
        return dict(row)
    finally:
        conn.close()


def history_status_counts(user_id):
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS n FROM job_history WHERE user_id = ? GROUP BY status", (user_id,)
        ).fetchall()
        return {r["status"]: r["n"] for r in rows}
    finally:
        conn.close()


def history_status_map(user_id):
    """{'title|company' (lower-case): status} -- lets the recommendations page show 'Applied' badges."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT title, company, status FROM job_history WHERE user_id = ?", (user_id,)).fetchall()
        return {f"{r['title'].lower()}|{r['company'].lower()}": r["status"] for r in rows}
    finally:
        conn.close()
