"""
dataset_generator.py
---------------------
Generates a synthetic but pattern-realistic dataset of job postings
(fraudulent vs genuine) for training JobGuard AI.

Why synthetic? A real labeled corpus (e.g. EMSCAD) requires an internet
download which this environment doesn't have. Instead we encode the
well-documented linguistic + structural signals of fake job postings
(urgency, upfront payment, vague roles, free-hosting URLs, personal
email domains, unrealistic pay, poor grammar) into template pools and
sample from them, so the model learns real, transferable signal rather
than memorizing a handful of examples.

Run directly to write model/job_dataset.csv
"""

import os
import random
import pandas as pd

random.seed(42)

# ---------------------------------------------------------------------
# Vocabulary pools
# ---------------------------------------------------------------------

REAL_COMPANIES = [
    "Infosys", "TCS", "Wipro", "Accenture", "Capgemini", "Deloitte",
    "IBM India", "Cognizant", "HCL Technologies", "Tech Mahindra",
    "Persistent Systems", "Zensar Technologies", "Mindtree", "L&T Infotech",
    "Siemens", "Bosch", "Amazon", "Flipkart", "Zomato", "Swiggy",
    "Razorpay", "Freshworks", "Paytm", "PhonePe", "Nykaa"
]

FAKE_COMPANY_NAMES = [
    "Global Home Earnings Ltd", "QuickCash Solutions", "WorkFromHome247",
    "EasyMoney International", "Prime Data Entry Hub", "Bright Future Careers",
    "InstantJob Network", "Rapid Hire Global", "TopEarn Online Pvt",
    "GreenSignal Recruiters", "MegaProfit Jobs", "SwiftPay Careers"
]

REAL_DOMAINS = ["com", "in", "co.in", "io"]
FREE_HOSTS = ["wixsite.com", "blogspot.com", "weebly.com", "000webhostapp.com",
              "sites.google.com", "wordpress.com", "yolasite.com"]
SUSPICIOUS_TLDS = ["tk", "ml", "ga", "cf", "gq", "xyz", "top"]

REAL_TITLES = [
    "Software Engineer", "Data Analyst", "HR Executive", "Backend Developer",
    "Marketing Associate", "Business Analyst", "QA Engineer", "UI/UX Designer",
    "Sales Executive", "Content Writer", "Financial Analyst", "DevOps Engineer",
    "Product Manager", "Customer Support Specialist", "Machine Learning Engineer"
]

FAKE_TITLES = [
    "Data Entry Operator (Work From Home)", "Online Form Filling Executive",
    "Part Time Typist - Earn Daily", "Home Based Data Entry Job",
    "Online Survey Executive - No Experience", "Copy Paste Job - Daily Payout",
    "WhatsApp Marketing Executive", "Online Chat Support - Instant Joining",
    "Package Handling Agent - Work From Home", "Ad Posting Job - Weekly Payment"
]

REQUIREMENTS_REAL = [
    "Bachelor's degree in Computer Science or related field",
    "2+ years of relevant industry experience",
    "Strong problem solving and communication skills",
    "Proficiency in Python, SQL and data structures",
    "Experience with agile development methodology",
    "Excellent written and verbal communication skills",
    "Relevant certification in the domain preferred",
]

RESPONSIBILITIES_REAL = [
    "collaborate with cross-functional teams to deliver product features",
    "design, develop and maintain scalable backend services",
    "analyze data trends and prepare actionable business reports",
    "participate in code reviews and maintain coding standards",
    "coordinate with stakeholders to gather and refine requirements",
    "own end-to-end delivery of assigned modules and features",
]

FAKE_HOOKS = [
    "No experience needed, immediate joining!!!",
    "Earn up to Rs. 50,000 per week working just 2 hours a day from home",
    "Limited seats available, apply now before offer closes",
    "100% guaranteed income, no boss, no targets",
    "Just copy paste ads and get paid daily via UPI",
    "Selected candidates must pay a refundable registration fee of Rs.999",
    "Send Rs.500 processing fee to confirm your slot",
    "No interview required, direct joining letter provided",
    "Work only 1-2 hours daily and earn a full time salary",
    "Share your bank details and Aadhar card to start earning today",
    "This is a golden opportunity, don't miss it, act fast!!!",
    "Contact us on WhatsApp only, no calls please",
]

REAL_CLOSERS = [
    "Interested candidates can apply through our official careers portal.",
    "We are an equal opportunity employer committed to diversity and inclusion.",
    "Shortlisted candidates will be contacted for a formal interview process.",
    "Please attach your updated resume and portfolio while applying.",
    "Compensation will be as per industry standards and experience.",
]

FAKE_CLOSERS = [
    "Hurry up, only a few positions left today!",
    "First come first serve basis, register immediately.",
    "For quick response, WhatsApp your details now.",
    "No qualification required, anyone can apply and start earning.",
    "Payment will be made daily/weekly directly to your account.",
]


def _slugify(name: str) -> str:
    return name.lower().replace(" ", "").replace("&", "and").replace(".", "")


def make_real_url(company: str) -> str:
    slug = _slugify(company)
    tld = random.choice(REAL_DOMAINS)
    path = random.choice(["/careers/apply", "/jobs/opening", "/careers", "/job-openings"])
    return f"https://www.{slug}.{tld}{path}"


def make_fake_url(company: str) -> str:
    slug = _slugify(company)
    choice = random.random()
    if choice < 0.35:
        host = random.choice(FREE_HOSTS)
        return f"http://{slug}.{host}/apply-now"
    elif choice < 0.6:
        tld = random.choice(SUSPICIOUS_TLDS)
        return f"http://{slug}job.{tld}/hiring"
    elif choice < 0.8:
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
        return f"http://{ip}/jobs/apply"
    else:
        return f"http://bit.ly/{slug}{random.randint(100,999)}"


def generate_real_row():
    company = random.choice(REAL_COMPANIES)
    title = random.choice(REAL_TITLES)
    reqs = ", ".join(random.sample(REQUIREMENTS_REAL, k=3))
    resp = "; ".join(random.sample(RESPONSIBILITIES_REAL, k=2))
    closer = random.choice(REAL_CLOSERS)
    salary_mentioned = random.random() < 0.4
    salary = f"Salary: Rs. {random.randint(4,18)},00,000 per annum (based on experience)." if salary_mentioned else ""
    description = (
        f"{company} is looking for a {title} to join our growing team. "
        f"The ideal candidate will {resp}. "
        f"Requirements: {reqs}. {salary} {closer}"
    )
    company_profile = (
        f"{company} is an established organization with a strong presence in the industry, "
        f"known for its professional work culture and employee growth programs."
    )
    return {
        "title": title,
        "company": company,
        "company_profile": company_profile,
        "description": description,
        "requirements": reqs,
        "url": make_real_url(company),
        "has_company_logo": 1,
        "salary_mentioned": int(salary_mentioned),
        "fraudulent": 0,
    }


def generate_fake_row():
    company = random.choice(FAKE_COMPANY_NAMES)
    title = random.choice(FAKE_TITLES)
    hooks = " ".join(random.sample(FAKE_HOOKS, k=random.randint(2, 4)))
    closer = random.choice(FAKE_CLOSERS)
    description = (
        f"{title} needed urgently at {company}. {hooks} {closer}"
    )
    company_profile = random.choice([
        "", "We are a fast growing company offering flexible online jobs.",
        "Genuine work from home opportunity for students and housewives."
    ])
    return {
        "title": title,
        "company": company,
        "company_profile": company_profile,
        "description": description,
        "requirements": random.choice(["No requirements", "Anyone can apply", "18+ age, smartphone needed", ""]),
        "url": make_fake_url(company),
        "has_company_logo": 0,
        "salary_mentioned": 1,
        "fraudulent": 1,
    }


def build_dataset(n_real=550, n_fake=450) -> pd.DataFrame:
    rows = [generate_real_row() for _ in range(n_real)] + \
           [generate_fake_row() for _ in range(n_fake)]
    random.shuffle(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "job_dataset.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df['fraudulent'].value_counts())
