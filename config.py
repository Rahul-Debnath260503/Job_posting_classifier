# config.py

import os

# --- Website Configuration ---
# config.py
# config.py
WEBSITE_URL = "https://www.karkidi.com/Find-Jobs/"
KARKIDI_JOB_LISTING_PATTERN = "https://www.karkidi.com/Find-Jobs/?page={}"
# (The rest of your config.py remains the same)
# --- File Paths ---
# Data storage paths
SCRAPED_JOBS_CSV_PATH = os.path.join("data", "latest_jobs.csv")
TRACKED_JOBS_JSON_PATH = os.path.join("data", "job_tracker.json")

# ML Model paths
CLASSIFIER_MODEL_PATH = os.path.join("ml_model", "trained_model", "job_classifier_model.pkl")
VECTORIZER_PATH = os.path.join("ml_model", "trained_model", "job_vectorizer.pkl")


# --- Job Classification Configuration ---
# Common skill keywords for initial extraction (extend as needed)
COMMON_SKILL_KEYWORDS = [
    "Python", "Java", "C++", "JavaScript", "React", "Angular", "Vue.js", "Node.js",
    "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "Oracle",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "DevOps",
    "Machine Learning", "Deep Learning", "Data Science", "AI", "NLP", "Computer Vision",
    "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "Keras", "PyTorch",
    "HTML", "CSS", "Frontend", "Backend", "Fullstack",
    "REST API", "Microservices", "Agile", "Scrum",
    "Cybersecurity", "Network Security", "Information Security",
    "Data Analysis", "Business Intelligence", "Tableau", "Power BI", "Excel",
    "Project Management", "Agile Methodology", "Scrum Master", "PMP",
    "Human Resources", "Recruitment", "Talent Acquisition", "HRIS",
    "Sales", "Marketing", "Digital Marketing", "SEO", "SEM", "Content Marketing",
    "Customer Service", "Support", "CRM",
    "Cloud Computing", "Linux", "Windows Server", "Virtualization",
    "Big Data", "Spark", "Hadoop",
    "Software Development Life Cycle", "SDLC", "Git", "Version Control",
    "UI/UX Design", "Figma", "Sketch", "Adobe XD",
    "Financial Analysis", "Accounting", "Auditing"
]
# Model parameters for job classification (if you have them)
CLASSIFIER_MODEL_PATH = "data/job_classifier_model.pkl"
VECTORIZER_PATH = "data/job_vectorizer.pkl"

# Define paths for scraped data and job tracking
SCRAPED_JOBS_CSV_PATH = "data/latest_jobs.csv" # Ensure scraper saves here
TRACKED_JOBS_JSON_PATH = "data/job_tracker.json"