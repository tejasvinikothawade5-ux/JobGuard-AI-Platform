"""
recommender.py
----------------
Content-based job recommendation engine.

Matches a job seeker's profile (skills + preferred role/location) against
the sample job listings pool (model/job_listings.csv) using TF-IDF +
cosine similarity over skills/title/description text, boosted by a
location preference match.

Every candidate listing is also passed through the SAME fraud-detection
pipeline used on the Analyze page (feature_engineering + the trained
model), so recommendations are automatically screened: listings with a
high fraud risk score are excluded from what's shown to the job seeker,
with a note about how many were filtered out.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
LISTINGS_PATH = os.path.join(HERE, "model", "job_listings.csv")

RISK_EXCLUDE_THRESHOLD = 60  # listings scoring >= this on the fraud model are hidden

_listings_df = None
_vectorizer = None
_listing_vectors = None


def _load_listings():
    global _listings_df, _vectorizer, _listing_vectors
    if _listings_df is not None:
        return
    if not os.path.exists(LISTINGS_PATH):
        raise FileNotFoundError(
            "job_listings.csv not found. Run `python model/job_listings_generator.py` first."
        )
    df = pd.read_csv(LISTINGS_PATH).fillna("")
    corpus = df["title"] + " " + df["skills"] + " " + df["description"]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(corpus)

    _listings_df = df
    _vectorizer = vectorizer
    _listing_vectors = vectors


def recommend_jobs(skills: str, preferred_role: str, preferred_location: str,
                    predict_fn, top_n: int = 6):
    """
    skills, preferred_role, preferred_location: from the user's profile
    predict_fn: the app's predict_job(title, description, company_profile,
                requirements, url) function, used to screen each candidate
                for fraud risk before it's shown.
    Returns: (recommendations: list[dict], screened_out_count: int)
    """
    _load_listings()

    query_text = f"{skills} {preferred_role}".strip()
    if not query_text:
        return [], 0

    query_vec = _vectorizer.transform([query_text])
    sims = cosine_similarity(query_vec, _listing_vectors).flatten()

    df = _listings_df.copy()
    df["match_score"] = sims

    if preferred_location and preferred_location.strip().lower() != "remote":
        loc = preferred_location.strip().lower()
        df.loc[df["location"].str.lower() == loc, "match_score"] += 0.15
    if preferred_location and preferred_location.strip().lower() == "remote":
        df.loc[df["location"].str.lower() == "remote", "match_score"] += 0.15

    df = df.sort_values("match_score", ascending=False)

    candidates = df.head(top_n * 3)  # take a wider pool, then screen for fraud

    results = []
    screened_out = 0
    for row in candidates.itertuples():
        if row.match_score <= 0:
            continue
        fraud_result = predict_fn(
            title=row.title,
            description=row.description,
            company_profile=row.company_profile,
            requirements=row.requirements,
            url=row.url,
        )
        if fraud_result["risk_score"] >= RISK_EXCLUDE_THRESHOLD:
            screened_out += 1
            continue

        results.append({
            "title": row.title,
            "company": row.company,
            "location": row.location,
            "skills": row.skills,
            "description": row.description,
            "url": row.url,
            "match_score": round(min(row.match_score, 1.0) * 100, 1),
            "risk_score": fraud_result["risk_score"],
            "risk_level": fraud_result["risk_level"],
        })
        if len(results) >= top_n:
            break

    return results, screened_out


def get_listing(title, company):
    """Look up one listing from the pool (used so the server -- not the browser -- decides a job's risk score)."""
    _load_listings()
    match = _listings_df[(_listings_df["title"] == title) & (_listings_df["company"] == company)]
    return match.iloc[0].to_dict() if len(match) else None
