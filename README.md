# JobGuard AI — Fake Job Posting Detector & Safe Job Recommender

An NLP + Machine Learning web application with two parts:

1. **Analyze** (no login needed) — paste a job posting's description,
   requirements and URL, and it predicts whether the listing is
   **fraudulent or genuine**, explains **why** with ranked red flags,
   gives a **0–100 risk score**, and recommends **whether you should apply**.
2. **Job seeker accounts** — sign up, build a profile (skills, preferred
   role, location), and get **recommended job listings** that are
   automatically **pre-screened through the same fraud model**, so scam
   postings never make it into your recommendations.

Built for the Aavishkar competition (socially beneficial project track).

---

## 1. How it works (architecture)

```
Browser (HTML/CSS/JS)
        │  fetch POST /predict  { title, description, url, ... }
        ▼
Flask backend (app.py)
        │
        ├─► feature_engineering.py
        │     • Parses the URL (HTTPS?, free-host domain?, IP address?,
        │       shortener?, suspicious TLD?)
        │     • Scans the text for scam-language patterns (urgency,
        │       upfront-payment requests, unrealistic pay, vague
        │       requirements, generic titles, excessive punctuation, etc.)
        │
        ├─► TF-IDF vectorizer (scikit-learn) — turns the job text into
        │     numeric word/phrase features (unigrams + bigrams)
        │
        ├─► RandomForestClassifier — trained on TF-IDF features +
        │     the engineered red-flag features together
        │
        └─► Returns: prediction, probability, risk score, ranked list
              of red flags (with severity), and an apply/don't-apply
              recommendation
        ▼
Browser renders an animated risk gauge, confidence bars, and a
red-flag "scan log"
```

**Why both TF-IDF and engineered features?** TF-IDF alone learns *which
words* correlate with scams, but engineered features encode
*domain knowledge* (e.g. "asks for a registration fee" or "hosted on a
free Wix subdomain") which is what lets the app explain **why** a
posting was flagged in plain English — not just spit out a black-box
probability.

**About the training data:** Real fake-job datasets (like the public
EMSCAD "Fake Job Postings" corpus) require an internet download. This
project instead ships `model/dataset_generator.py`, which programmatically
generates ~1,000 realistic job postings using template pools that encode
well-documented fraud patterns (urgency language, upfront-fee requests,
unrealistic pay claims, free-hosting/IP/shortener URLs, vague
requirements) versus genuine posting patterns (specific requirements,
professional company profiles, proper corporate domains). This keeps the
project fully self-contained and reproducible offline. **For a
production deployment, swap in a real labeled dataset (e.g. EMSCAD from
Kaggle) in the same CSV shape** — the rest of the pipeline (feature
engineering, training, app) works unchanged. This is worth mentioning
to judges as a "next step" — it shows you understand the limits of
synthetic data.

---

## 2. Project structure

```
jobguard_ai/
├── app.py                        # Flask backend: analyze API + auth + profile + recommendations
├── database.py                   # SQLite helpers (users, profiles) — creates jobguard.db
├── recommender.py                # Matches profile -> job listings, screens out risky ones
├── requirements.txt
├── README.md
├── model/
│   ├── dataset_generator.py      # Generates job_dataset.csv (fraud-detection training data)
│   ├── job_listings_generator.py # Generates job_listings.csv (recommendation pool)
│   ├── feature_engineering.py    # Shared feature/red-flag logic
│   ├── train_model.py            # Trains & saves the fraud-detection model
│   ├── job_dataset.csv           # Generated dataset (created by you)
│   ├── job_listings.csv          # Generated listings pool (created by you)
│   └── artifacts.joblib          # Trained model (created by you)
├── templates/
│   ├── _nav.html                 # Shared nav bar included on every page
│   ├── index.html                # Analyze page (unchanged core behavior)
│   ├── signup.html / login.html  # Auth pages
│   ├── profile.html              # Job-seeker profile form
│   └── recommendations.html      # Matched + fraud-screened job listings
└── static/
    ├── style.css                  # UI styling
    └── script.js                  # Analyze-page logic (calls /predict)
```

**How recommendations work:** `recommender.py` builds a TF-IDF similarity
between your profile's skills/preferred role and a pool of ~30 sample job
listings (`model/job_listings.csv`), which intentionally includes a few
scam-style postings. Before any match is shown to you, it's run through
the *exact same* fraud-detection pipeline used on the Analyze page — any
listing scoring 60+ risk is silently excluded, and the page tells you how
many were filtered out.

---

## 3. Step-by-step setup (VS Code)

### Step 1 — Open the project
Unzip the project folder and open it in VS Code: `File → Open Folder…` →
select `jobguard_ai`.

### Step 2 — Create a virtual environment (recommended)
Open a terminal in VS Code (`` Ctrl+` ``) and run:

```bash
python -m venv venv
```

Activate it:
- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Generate the training dataset
```bash
cd model
python dataset_generator.py
```
Output:
```
Saved 1000 rows to .../model/job_dataset.csv
fraudulent
0    550
1    450
```

### Step 5 — Generate the job listings pool (for recommendations)
```bash
python job_listings_generator.py
```
Output:
```
Saved 33 job listings to .../model/job_listings.csv (3 intentional scam traps included)
```

### Step 6 — Train the fraud-detection model
```bash
python train_model.py
```
This prints accuracy/precision/recall/F1 on a held-out test split and
saves `model/artifacts.joblib`. Example output:
```
Accuracy : 0.98+
Precision: 0.97+
Recall   : 0.97+
F1 Score : 0.97+
```
(Exact numbers vary slightly by scikit-learn version / random seed.)

### Step 7 — Run the web app
```bash
cd ..
python app.py
```
You should see:
```
JobGuard AI running at http://127.0.0.1:5000
```
A `jobguard.db` SQLite file is created automatically on first run to
store user accounts and profiles.

### Step 8 — Open it in your browser
Go to **http://127.0.0.1:5000**

**Analyze page (no login needed):**
- Click **"Suspicious listing"** or **"Genuine listing"** to auto-fill a
  demo example, or paste in a real posting.
- Click **Analyze listing**.
- You'll see: a risk-score gauge, verdict (Do NOT Apply / Proceed with
  Caution / Likely Safe to Apply), model confidence bars, a ranked list
  of red flags in plain English, and an expandable technical feature
  readout (for judges who ask "how does it actually decide?").

**Job seeker account flow:**
1. Click **Sign up** (top right), create an account.
2. You'll land on **My profile** — enter your skills (comma-separated),
   preferred role, experience, and preferred location, then **Save profile**.
3. Click **Recommended jobs** in the nav — you'll see matched listings
   ranked by match %, each already screened for fraud risk. If your
   skills happen to match one of the intentional scam traps (try skills
   like `typing, data entry`), you'll see a note that some matches were
   filtered out for fraud risk — a good thing to point out live to judges.

---

## 4. What to demo/explain to the judges

1. **The problem**: Online job scams trick job-seekers into paying fake
   "registration fees" or leaking personal/financial data.
2. **The approach**: Combine NLP (TF-IDF on the job text) with
   domain-expert red-flag rules (URL structure + scam-language
   detection) so the system is both accurate *and* explainable — it
   doesn't just say "fake," it says *why*.
3. **Live demo**: Use the two sample chips to instantly show a correctly
   flagged scam vs. a correctly cleared genuine posting.
4. **Social benefit**: Protects job seekers (especially students, and
   first-time job hunters) from financial loss and data theft, at zero
   cost, in real time, before they apply.
5. **Limitations & future work** (good for Q&A): synthetic training
   data — a production version would retrain on a real labeled corpus
   (e.g. EMSCAD) and could add browser-extension integration for
   scanning postings directly on LinkedIn/Naukri/Indeed.

---

## 5. Troubleshooting

| Problem | Fix |
|---|---|
| `Model artifacts not found` error on `python app.py` | Run Steps 4–6 first (dataset + listings + training). |
| `job_listings.csv not found` error on the Recommendations page | Run Step 5 (`python job_listings_generator.py` inside `model/`). |
| `ModuleNotFoundError` | Make sure your venv is activated and you ran `pip install -r requirements.txt`. |
| Port 5000 already in use | Edit the last line of `app.py`: change `port=5000` to e.g. `port=5050`. |
| Want to retrain with more/different data | Edit the template pools in `model/dataset_generator.py` or `model/job_listings_generator.py`, then rerun the relevant step. |
| Want to reset all user accounts | Delete `jobguard.db` and restart `app.py` — a fresh empty database is created automatically. |


---

# New in this version: history, forgot password, security

## 1. Application history ("My history" in the menu)
* When you are logged in, every listing you analyze is saved automatically (title, company, link, risk score, verdict, red flags).
* After an analysis, buttons appear: **Save / I applied / Skip**. On *Recommended jobs* each card has **Save / I applied**.
* On the **My history** page: filter by status (Analyzed, Saved, Applied, Interview, Offer, Rejected, Skipped), search, add notes, change status, remove.
* Counters show how many listings you checked, applied to, and how many high-risk ones you found/avoided.
* If you mark a **high-risk** listing as applied, you get a safety warning (never pay a fee).
* Re-analyzing the same listing updates the existing entry instead of duplicating it.

## 2. Forgot password
Login page → **Forgot password?** → enter email → a one-time link (valid 30 minutes) lets you set a new password.
* **Demo mode (default):** no email setup needed. The reset link is **printed in the terminal** where `python app.py` runs.
* **Real email mode:** copy `.env.example` to `.env` and fill in `JOBGUARD_SMTP_*` and `JOBGUARD_BASE_URL`.
  For Gmail: turn on 2-Step Verification → create an *App password* → use it as `JOBGUARD_SMTP_PASSWORD` (never your normal password).
* Logged-in users can also change their password under **My profile → Security**.

## 3. Security measures added
| Risk | Protection |
|---|---|
| Stolen database | Passwords stored as salted hashes (scrypt); reset tokens stored only as SHA-256 hashes |
| Guessing passwords | Login rate limit (5 failed tries per account/IP per 15 min), password rules (8+ chars, letter + number, not common) |
| Finding which emails are registered | Same message/timing for unknown email vs wrong password; forgot-password always says "if an account exists..." |
| Fake forms from other sites (CSRF) | Token in every form + same-origin check; logout is now a POST |
| Reset-link reuse | Link works once, expires in 30 min, new request invalidates older ones |
| Stolen sessions | Changing/resetting password logs out every other device; HttpOnly + SameSite cookies; sessions last 7 days |
| Session fixation | Fresh session created at login |
| Open redirect via `?next=` | Only same-site paths are followed |
| Injection / XSS | Parameterised SQL everywhere; templates auto-escape; strict Content-Security-Policy |
| Clickjacking, MIME sniffing | `X-Frame-Options`, `nosniff`, `Referrer-Policy` headers |
| Seeing other users' history | Every history query is tied to the logged-in user id |
| Debug mode / public default key | Debug is OFF; app listens on 127.0.0.1 only; secret key is random (or from `.env`) |
| Oversized / abusive requests | 1 MB limit; `/predict` rate limit; no internal error text shown |

To let your phone open the app on the same WiFi: `set JOBGUARD_HOST=0.0.0.0` (Windows) / `export JOBGUARD_HOST=0.0.0.0`, then open `http://<your-PC-IP>:5000`. Do not do this on public WiFi.

## Before putting it on the internet
1. Use HTTPS (e.g. Render/Railway/PythonAnywhere give it free) and set `JOBGUARD_HTTPS=1`.
2. Run with a production server: `pip install gunicorn` then `gunicorn app:app` (not `python app.py`).
3. Set `JOBGUARD_SECRET_KEY` and `JOBGUARD_BASE_URL` as environment variables on the host; never commit `.env`.
4. The rate limiter is in memory (resets on restart, per process); use Redis-based limiting if you run several servers.
5. SQLite is fine for a demo; move to PostgreSQL for many users, and back up the database.
6. Not yet included (good "future scope" for judges): email verification at signup, two-factor login, account deletion, security audit/logging.

The fake-job detection model (`model/`, `predict_job`) was **not changed**.
