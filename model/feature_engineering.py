"""
feature_engineering.py
-----------------------
Extracts numeric "red flag" features from a job posting's text and URL.
These features are combined with TF-IDF text features inside the model
pipeline, AND reused at prediction time to generate human-readable
explanations (why a job was flagged, risk factors, etc).

Keeping this logic in one shared module guarantees the features seen
during training exactly match the features used for live predictions.
"""

import re
from urllib.parse import urlparse

FREE_HOSTS = ["wixsite.com", "blogspot.com", "weebly.com", "000webhostapp.com",
              "sites.google.com", "wordpress.com", "yolasite.com", "tumblr.com"]
SUSPICIOUS_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top", "info", "click"]
URL_SHORTENERS = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "is.gd"]
PERSONAL_EMAIL_DOMAINS = ["gmail.com", "yahoo.com", "hotmail.com", "rediffmail.com", "outlook.com"]

URGENCY_WORDS = [
    "urgent", "urgently", "immediate", "immediately", "hurry", "limited seats",
    "act fast", "don't miss", "apply now", "first come first serve",
    "few positions left", "last chance",
]
PAYMENT_WORDS = [
    "registration fee", "processing fee", "security deposit", "pay rs",
    "pay a fee", "refundable fee", "send money", "training fee",
    "activation fee", "one time payment", "bank details", "aadhar",
    "aadhaar", "pan card", "otp", "wire transfer",
]
UNREALISTIC_PAY_WORDS = [
    "earn up to", "guaranteed income", "guaranteed salary", "no boss",
    "no targets", "work 1-2 hours", "work 2 hours", "daily payout",
    "weekly payout", "easy money", "quick money", "100% guaranteed",
]
VAGUE_WORDS = [
    "no experience needed", "no experience required", "anyone can apply",
    "no qualification required", "no interview required", "no interview",
    "copy paste", "whatsapp only", "no calls please",
]
GENERIC_TITLE_HINTS = [
    "data entry", "form filling", "ad posting", "copy paste", "typist",
    "survey executive", "package handling",
]


def _lower(x):
    return (x or "").lower()


def extract_url_features(url: str) -> dict:
    url = url or ""
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.lower()
    feats = {
        "url_no_https": int(parsed.scheme != "https"),
        "url_free_host": int(any(h in host for h in FREE_HOSTS)),
        "url_suspicious_tld": int(any(host.split(".")[-1] == t for t in SUSPICIOUS_TLDS)),
        "url_is_ip": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host))),
        "url_shortener": int(any(s in host for s in URL_SHORTENERS)),
        "url_excessive_hyphens": int(host.count("-") >= 2),
        "url_length": len(url),
    }
    return feats


def extract_text_features(title: str, description: str, company_profile: str, requirements: str) -> dict:
    full_text = _lower(f"{title} {description} {company_profile} {requirements}")

    def count_hits(word_list):
        return sum(1 for w in word_list if w in full_text)

    exclamations = description.count("!") if description else 0
    caps_words = len(re.findall(r"\b[A-Z]{4,}\b", description or ""))
    word_count = len((description or "").split())

    feats = {
        "urgency_score": count_hits(URGENCY_WORDS),
        "payment_score": count_hits(PAYMENT_WORDS),
        "unrealistic_pay_score": count_hits(UNREALISTIC_PAY_WORDS),
        "vague_score": count_hits(VAGUE_WORDS),
        "generic_title": int(any(w in _lower(title) for w in GENERIC_TITLE_HINTS)),
        "exclamation_count": exclamations,
        "caps_word_count": caps_words,
        "description_word_count": word_count,
        "has_company_profile": int(bool((company_profile or "").strip())),
        "has_requirements": int(bool((requirements or "").strip()) and
                                 _lower(requirements) not in ("no requirements", "anyone can apply", "")),
        "mentions_personal_email": int(any(d in full_text for d in PERSONAL_EMAIL_DOMAINS)),
    }
    return feats


def extract_all_features(title, description, company_profile, requirements, url) -> dict:
    feats = {}
    feats.update(extract_url_features(url))
    feats.update(extract_text_features(title, description, company_profile, requirements))
    return feats


FEATURE_ORDER = [
    "url_no_https", "url_free_host", "url_suspicious_tld", "url_is_ip",
    "url_shortener", "url_excessive_hyphens", "url_length",
    "urgency_score", "payment_score", "unrealistic_pay_score", "vague_score",
    "generic_title", "exclamation_count", "caps_word_count",
    "description_word_count", "has_company_profile", "has_requirements",
    "mentions_personal_email",
]


def features_to_vector(feats: dict):
    return [feats.get(k, 0) for k in FEATURE_ORDER]


# ---------------------------------------------------------------------
# Human-readable explanation layer (used by the Flask app)
# ---------------------------------------------------------------------

def build_red_flags(feats: dict, url: str) -> list:
    """Return a list of (severity, message) tuples explaining risk factors."""
    flags = []

    if feats["url_no_https"]:
        flags.append(("high", "The job link does not use a secure HTTPS connection."))
    if feats["url_free_host"]:
        flags.append(("high", "The posting is hosted on a free website builder (e.g. Wix, Blogspot, Weebly) instead of a proper company domain."))
    if feats["url_suspicious_tld"]:
        flags.append(("high", "The URL uses an unusual/free domain extension commonly abused for scams (.tk, .xyz, .top, etc.)."))
    if feats["url_is_ip"]:
        flags.append(("high", "The job link points to a raw IP address instead of a real domain name — a strong scam indicator."))
    if feats["url_shortener"]:
        flags.append(("medium", "The link uses a URL shortener, which can be used to hide the real destination."))
    if feats["url_excessive_hyphens"]:
        flags.append(("low", "The domain name contains multiple hyphens, a pattern often seen in spoofed domains."))

    if feats["payment_score"] > 0:
        flags.append(("high", "The posting asks for money, a 'registration/processing fee', or sensitive financial/ID details. Legitimate employers never ask you to pay to get hired."))
    if feats["unrealistic_pay_score"] > 0:
        flags.append(("high", "The posting promises unrealistically high pay for minimal work/hours, a classic scam lure."))
    if feats["urgency_score"] > 0:
        flags.append(("medium", "The posting uses high-pressure urgency language ('apply now', 'limited seats') to rush your decision."))
    if feats["vague_score"] > 0:
        flags.append(("medium", "The posting explicitly states no experience/qualification/interview is required, which is unusual for genuine roles."))
    if feats["generic_title"]:
        flags.append(("low", "The job title matches patterns common in generic data-entry/ad-posting scam listings."))
    if feats["mentions_personal_email"]:
        flags.append(("medium", "The posting references contact via a personal email domain (Gmail/Yahoo/etc.) rather than a corporate one."))
    if not feats["has_company_profile"]:
        flags.append(("medium", "No real company profile or background information is provided."))
    if not feats["has_requirements"]:
        flags.append(("low", "No meaningful job requirements or qualifications are listed."))
    if feats["exclamation_count"] >= 2:
        flags.append(("low", "Excessive use of exclamation marks, common in promotional/scam copy."))
    if feats["caps_word_count"] >= 2:
        flags.append(("low", "Excessive use of ALL-CAPS words, a pattern typical of spammy postings."))
    if feats["description_word_count"] < 25:
        flags.append(("low", "The job description is unusually short and lacks detail about the role."))

    return flags


def risk_score_from_probability(prob_fraud: float) -> int:
    """Map model probability (0-1) to a 0-100 risk score."""
    return int(round(prob_fraud * 100))


def risk_level(score: int) -> str:
    if score >= 70:
        return "High Risk"
    elif score >= 40:
        return "Medium Risk"
    else:
        return "Low Risk"


def apply_recommendation(score: int) -> dict:
    if score >= 70:
        return {
            "verdict": "Do NOT Apply",
            "advice": "This posting shows strong indicators of being fraudulent. Avoid sharing personal details, paying any fee, or proceeding further. Report the listing if possible."
        }
    elif score >= 40:
        return {
            "verdict": "Proceed with Caution",
            "advice": "This posting has some suspicious characteristics. Verify the company independently (official website, LinkedIn, reviews) before sharing any personal information or applying."
        }
    else:
        return {
            "verdict": "Likely Safe to Apply",
            "advice": "This posting matches patterns typical of genuine job listings. Still, always verify company legitimacy and never pay any money during a hiring process."
        }
