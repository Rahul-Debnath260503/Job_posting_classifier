# main.py
import logging
import pandas as pd
import os
import sys
import nltk
import ssl

# --- NLTK SSL Workaround (Keep this) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# --- End NLTK SSL Workaround ---

# APScheduler imports
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger

# Project imports
from job_classifier import JobClassifier
from job_processing.tracker import JobTracker
from scraper.karkidi_scraper import KarkidiScraper
from config import (
    WEBSITE_URL,
    SCRAPED_JOBS_CSV_PATH,
    TRACKED_JOBS_JSON_PATH,
    CLASSIFIER_MODEL_PATH,
    VECTORIZER_PATH
)

# --- Setup Logging ---
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", "job_scraper.log"))
    ]
)
logger = logging.getLogger(__name__)

# --- Ensure data directory exists ---
os.makedirs('data', exist_ok=True)
os.makedirs(os.path.dirname(CLASSIFIER_MODEL_PATH), exist_ok=True) # Ensure trained_model dir exists


# --- Automatic Category Assignment Function ---
def assign_initial_category(job_title, job_description):
    """
    Assigns an initial category based on keywords in job title and description.
    This is a rule-based system to provide initial labels for ML training.
    """
    title_lower = str(job_title).lower()
    desc_lower = str(job_description).lower()

    if "software" in title_lower or "developer" in title_lower or \
       "engineer" in title_lower or "programmer" in title_lower or \
       "java" in desc_lower or "python" in desc_lower or "react" in desc_lower or \
       "angular" in desc_lower or "node" in desc_lower or "fullstack" in desc_lower:
        return "Software Development"
    elif "data scientist" in title_lower or "machine learning" in title_lower or \
         "ai" in title_lower or "data analyst" in title_lower or "big data" in title_lower or \
         "sql" in desc_lower or "r" in desc_lower or "tableau" in desc_lower or \
         "power bi" in desc_lower or "ml" in desc_lower or "deep learning" in desc_lower:
        return "Data Science & Analytics"
    elif "marketing" in title_lower or "seo" in title_lower or "sem" in title_lower or \
         "content creator" in title_lower or "social media" in title_lower:
        return "Marketing"
    elif "hr" in title_lower or "human resources" in title_lower or \
         "recruiter" in title_lower or "talent acquisition" in title_lower:
        return "Human Resources"
    elif "sales" in title_lower or "business development" in title_lower or \
         "account manager" in title_lower:
        return "Sales"
    elif "project manager" in title_lower or "scrum master" in title_lower or \
         "product owner" in title_lower or "program manager" in title_lower:
        return "Project Management"
    elif "support" in title_lower or "helpdesk" in title_lower or \
         "technical support" in title_lower or "it technician" in title_lower:
        return "IT Support"
    elif "qa" in title_lower or "quality assurance" in title_lower or \
         "tester" in title_lower or "software testing" in title_lower:
        return "QA & Testing"
    elif "designer" in title_lower or "ux" in title_lower or "ui" in title_lower or \
         "graphic" in title_lower:
        return "Design & UI/UX"
    elif "finance" in title_lower or "accountant" in title_lower or \
         "auditor" in title_lower or "financial analyst" in title_lower:
        return "Finance & Accounting"
    else:
        return "Other" # Catch-all category

# --- Main Scraper and Processing Function ---
def run_scraper_and_process():
    """
    Function to be called by the scheduler.
    It runs the scraper, processes new jobs, and updates tracked jobs.
    """
    logger.info("--- Starting scheduled job scraping and processing ---")
    try:
        scraper = KarkidiScraper(WEBSITE_URL)
        scraped_jobs_df = pd.DataFrame(scraper.scrape_jobs())

        if scraped_jobs_df.empty:
            logger.warning("No jobs scraped in this run.")
            return

        logger.info(f"Successfully scraped {len(scraped_jobs_df)} jobs.")

        # --- Automated Initial Categorization ---
        # Apply the rule-based categorization to create the 'category' column
        scraped_jobs_df['category'] = scraped_jobs_df.apply(
            lambda row: assign_initial_category(row['job_title'], row['job_description']),
            axis=1
        )
        logger.info("Initial categories assigned based on keywords.")
        # --- End Automated Categorization ---

        # Save the raw scraped data (now with initial categories) to CSV for Streamlit and manual review
        scraped_jobs_df.to_csv(SCRAPED_JOBS_CSV_PATH, index=False, encoding='utf-8')
        logger.info(f"Scraped data (with initial categories) saved to {SCRAPED_JOBS_CSV_PATH}")

        # --- Job Tracking and Classification ---
        job_tracker = JobTracker(TRACKED_JOBS_JSON_PATH)
        new_jobs_df = job_tracker.get_new_jobs(scraped_jobs_df)

        if not new_jobs_df.empty:
            logger.info(f"Found {len(new_jobs_df)} new jobs.")

            classifier = JobClassifier(CLASSIFIER_MODEL_PATH, VECTORIZER_PATH)

            # Attempt to load model. If fails, train it using the (now categorized) scraped_jobs_df.
            if not classifier.load_model():
                logger.info("Classifier model not found or failed to load. Attempting to train using scraped data...")
                
                # Check if there are enough distinct categories for training
                if len(scraped_jobs_df['category'].unique()) < 2:
                    logger.error(f"Cannot train: only one or zero distinct categories found in scraped data ({scraped_jobs_df['category'].unique()}).")
                    logger.error("Please ensure the keyword-based categorization creates at least two different categories.")
                    logger.warning("New jobs will be 'Uncategorized' as model training failed.")
                    new_jobs_df['category'] = 'Uncategorized' # Fallback
                else:
                    classifier.train(scraped_jobs_df) # Train with the *initially categorized* full dataset
                    # After training, if successful, classify the new jobs
                    if classifier.model and classifier.vectorizer:
                        new_jobs_df['category'] = new_jobs_df.apply(
                            lambda row: classifier.predict_category(row['job_description']), axis=1
                        )
                        logger.info("New jobs classified using the newly trained model.")
                    else:
                        logger.warning("Classifier model not available after attempted training. New jobs will be 'Uncategorized'.")
                        new_jobs_df['category'] = 'Uncategorized' # Fallback
            else:
                # Model loaded successfully, now classify new jobs
                if 'job_description' in new_jobs_df.columns:
                    new_jobs_df['category'] = new_jobs_df.apply(
                        lambda row: classifier.predict_category(row['job_description']), axis=1
                    )
                    logger.info("New jobs classified using the loaded model.")
                else:
                    logger.warning("No 'job_description' column in new_jobs_df for classification.")
                    new_jobs_df['category'] = 'Uncategorized'

            # Update tracker with new jobs (including their classification)
            job_tracker.add_jobs(new_jobs_df)
            job_tracker.save_jobs()
            logger.info(f"New jobs added to tracker and saved to {TRACKED_JOBS_JSON_PATH}")

        else:
            logger.info("No new jobs found in this run.")

    except Exception as e:
        logger.error(f"An unexpected error occurred during the scheduled run: {e}", exc_info=True)
    finally:
        logger.info("--- Scheduled job scraping and processing completed ---")

if __name__ == "__main__":
    logger.info("Application starting...")

    # Run scraper once immediately on startup
    run_scraper_and_process()

    # Setup the scheduler
    scheduler = BlockingScheduler()
    scheduler.add_job(run_scraper_and_process, IntervalTrigger(hours=24), id='daily_job_scrape', name='Daily Karkidi Job Scrape', replace_existing=True)

    logger.info("Scheduler started. Scraping will run every 24 hours. Press Ctrl+C to exit.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        logger.info("Scheduler shut down.")