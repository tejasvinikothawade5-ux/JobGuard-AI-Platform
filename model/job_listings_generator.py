"""
job_listings_generator.py
--------------------------
Generates model/job_listings.csv: a pool of sample job postings across
multiple roles that the recommendation engine matches a job seeker's
profile against. A handful of postings are deliberately built from the
"scam" templates (same ones used in dataset_generator.py) so the
recommender can demonstrate screening them out using the same fraud
model that powers the Analyze page.

Run:  python job_listings_generator.py
"""

import os
import random
import pandas as pd

random.seed(7)

HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = [
    # (title, skills, location, company)
    ("Software Engineer", "python, java, data structures, git, rest apis", "Bengaluru", "Infosys"),
    ("Backend Developer", "python, django, sql, docker, apis", "Pune", "Persistent Systems"),
    ("Frontend Developer", "javascript, react, css, html, typescript", "Hyderabad", "Zensar Technologies"),
    ("Data Analyst", "sql, excel, python, tableau, statistics", "Mumbai", "TCS"),
    ("Data Scientist", "python, machine learning, pandas, statistics, sql", "Bengaluru", "Flipkart"),
    ("Machine Learning Engineer", "python, tensorflow, pytorch, ml, deep learning", "Bengaluru", "Amazon"),
    ("DevOps Engineer", "aws, docker, kubernetes, ci/cd, linux", "Pune", "Wipro"),
    ("QA Engineer", "selenium, testing, automation, java, sql", "Chennai", "Cognizant"),
    ("UI/UX Designer", "figma, adobe xd, wireframing, user research", "Bengaluru", "Freshworks"),
    ("Business Analyst", "excel, sql, requirements gathering, communication", "Mumbai", "Deloitte"),
    ("HR Executive", "recruitment, communication, ms office, onboarding", "Delhi", "HCL Technologies"),
    ("Marketing Associate", "seo, content marketing, social media, analytics", "Mumbai", "Nykaa"),
    ("Sales Executive", "communication, negotiation, crm, lead generation", "Delhi", "Paytm"),
    ("Content Writer", "writing, seo, research, editing", "Remote", "Zomato"),
    ("Financial Analyst", "excel, financial modeling, accounting, sql", "Mumbai", "Deloitte"),
    ("Product Manager", "product strategy, agile, communication, analytics", "Bengaluru", "Swiggy"),
    ("Customer Support Specialist", "communication, crm, problem solving", "Hyderabad", "Razorpay"),
    ("Network Engineer", "networking, ccna, firewalls, troubleshooting", "Chennai", "Bosch"),
    ("Mobile App Developer", "android, kotlin, java, apis", "Pune", "PhonePe"),
    ("Cloud Engineer", "aws, azure, terraform, cloud architecture", "Bengaluru", "IBM India"),
    ("Graphic Designer", "photoshop, illustrator, branding, creativity", "Remote", "Nykaa"),
    ("Operations Executive", "ms office, coordination, logistics, communication", "Delhi", "Flipkart"),
    ("Full Stack Developer", "javascript, react, node.js, mongodb, sql", "Hyderabad", "Tech Mahindra"),
    ("System Administrator", "linux, windows server, networking, scripting", "Chennai", "Siemens"),
    ("Digital Marketing Executive", "google ads, seo, sem, analytics", "Mumbai", "Swiggy"),
    ("Research Analyst", "excel, research, statistics, report writing", "Delhi", "Accenture"),
    ("Technical Support Engineer", "troubleshooting, networking, communication", "Pune", "IBM India"),
    ("Junior Software Developer", "java, python, sql, oop", "Bengaluru", "L&T Infotech"),
    ("Data Engineer", "python, sql, spark, etl, cloud", "Bengaluru", "Mindtree"),
    ("Recruitment Specialist", "recruitment, sourcing, interviewing, communication", "Delhi", "Capgemini"),
]

REAL_CLOSER = "Interested candidates can apply through our official careers portal."
REAL_PROFILE = "{company} is an established organization known for its professional work culture and employee growth programs."


def _slugify(name):
    return name.lower().replace(" ", "").replace("&", "and").replace(".", "")


def make_real_url(company):
    slug = _slugify(company)
    return f"https://www.{slug}.com/careers/apply"


def build_real_listing(title, skills, location, company):
    description = (
        f"{company} is hiring a {title} for our {location} office. "
        f"The role involves working with {skills} and collaborating with cross-functional teams "
        f"to deliver high quality outcomes. {REAL_CLOSER}"
    )
    return {
        "title": title,
        "company": company,
        "skills": skills,
        "location": location,
        "description": description,
        "company_profile": REAL_PROFILE.format(company=company),
        "requirements": f"Relevant experience with {skills}. Strong communication and problem-solving skills.",
        "url": make_real_url(company),
        "is_trap": 0,
    }


# A handful of scam-style postings mixed into the pool, reusing the
# same fraud patterns as dataset_generator.py, so the recommender can
# demonstrate filtering them out via the shared fraud model.
TRAP_LISTINGS = [
    {
        "title": "Data Entry Executive - Work From Home",
        "company": "QuickCash Solutions",
        "skills": "typing, data entry, ms office",
        "location": "Remote",
        "description": ("URGENT HIRING!!! No experience needed, immediate joining! Earn up to "
                         "Rs. 40,000 per week working just 2 hours a day. Limited seats available, "
                         "apply now. Pay a refundable registration fee of Rs.999 to confirm your slot. "
                         "Contact us on WhatsApp only, no calls please. 100% guaranteed income!"),
        "company_profile": "We are a fast growing company offering flexible online jobs.",
        "requirements": "No requirements, anyone can apply, 18+ age, smartphone needed.",
        "url": "http://quickcashsolutions.wixsite.com/apply-now",
        "is_trap": 1,
    },
    {
        "title": "Online Form Filling Executive",
        "company": "EasyMoney International",
        "skills": "typing, internet browsing",
        "location": "Remote",
        "description": ("Earn easy money from home! No qualification required, no interview "
                         "required. Just copy paste ads and get paid daily via UPI. Send Rs.500 "
                         "processing fee to confirm your slot. Hurry up, only a few positions left today!"),
        "company_profile": "",
        "requirements": "Anyone can apply",
        "url": "http://easymoney-jobs.tk/hiring",
        "is_trap": 1,
    },
    {
        "title": "Part Time Typist - Daily Payout",
        "company": "TopEarn Online Pvt",
        "skills": "typing",
        "location": "Remote",
        "description": ("Home based typing job, work only 1-2 hours daily and earn a full time "
                         "salary. No boss, no targets. Share your bank details and Aadhar card to "
                         "start earning today. First come first serve basis, register immediately."),
        "company_profile": "Genuine work from home opportunity for students and housewives.",
        "requirements": "",
        "url": "http://198.51.100.24/jobs/apply",
        "is_trap": 1,
    },
]


def build_dataset():
    rows = [build_real_listing(*job) for job in JOBS]
    rows.extend(TRAP_LISTINGS)
    random.shuffle(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_dataset()
    out_path = os.path.join(HERE, "job_listings.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} job listings to {out_path} ({df['is_trap'].sum()} intentional scam traps included)")
