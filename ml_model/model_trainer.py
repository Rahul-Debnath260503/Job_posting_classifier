# ml_model/model_trainer.py
import pandas as pd
import os
import sys
import logging
import nltk # Important: Make sure nltk is imported before NLTK path config

# --- For relative imports to work (ensure this allows importing from job_processing etc.) ---
# This appends the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# --- End of sys.path fix ---

# --- NLTK Data Path Configuration (Crucial for finding downloads) ---
# Define a directory for NLTK data relative to your project root
# This will be 'JOB_POSTING_CLASIF/data/nltk_data'
# Since model_trainer.py is in ml_model/, need to go up one dir and then into data/
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'nltk_data')

if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Ensure the directory exists
os.makedirs(nltk_data_dir, exist_ok=True)

# Add the SSL workaround (important for Windows when downloading NLTK data)
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# --- End NLTK Data Path Configuration ---

from job_classifier import JobClassifier
from config import SCRAPED_JOBS_CSV_PATH, CLASSIFIER_MODEL_PATH, VECTORIZER_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_new_model():
    """
    Loads scraped data, trains the job classifier, and saves the model.
    This function should be run manually when you want to retrain the model.
    """
    logger.info("Starting model training process...")

    if not os.path.exists(SCRAPED_JOBS_CSV_PATH):
        logger.error(f"Scraped jobs CSV not found at {SCRAPED_JOBS_CSV_PATH}. Please run scraper first.")
        return

    try:
        df = pd.read_csv(SCRAPED_JOBS_CSV_PATH, encoding='utf-8')
        logger.info(f"Loaded {len(df)} jobs from {SCRAPED_JOBS_CSV_PATH}")

        # IMPORTANT: For training, your DataFrame MUST have a 'category' column
        # If your scraped CSV doesn't have it, you need to manually add it
        # or have a process to label a subset of your data for training.
        if 'category' not in df.columns or df['category'].isnull().all():
            logger.error("No 'category' column found or all categories are missing in the scraped data for training.")
            logger.error("Please manually label some jobs in data/latest_jobs.csv with at least two distinct categories (e.g., 'Software Development', 'Data Science', 'Marketing').")
            logger.error("Model training cannot proceed without labeled data.")
            return

        classifier = JobClassifier(CLASSIFIER_MODEL_PATH, VECTORIZER_PATH)
        classifier.train(df) # This will train and save the model
        logger.info("Model training completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}", exc_info=True)

if __name__ == "__main__":
    train_new_model()# ml_model/model_trainer.py
import pandas as pd
import os
import sys
import logging
import nltk # Important: Make sure nltk is imported before NLTK path config

# --- For relative imports to work (ensure this allows importing from job_processing etc.) ---
# This appends the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# --- End of sys.path fix ---

# --- NLTK Data Path Configuration (Crucial for finding downloads) ---
# Define a directory for NLTK data relative to your project root
# This will be 'JOB_POSTING_CLASIF/data/nltk_data'
# Since model_trainer.py is in ml_model/, need to go up one dir and then into data/
nltk_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'nltk_data')

if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Ensure the directory exists
os.makedirs(nltk_data_dir, exist_ok=True)

# Add the SSL workaround (important for Windows when downloading NLTK data)
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# --- End NLTK Data Path Configuration ---

from job_classifier import JobClassifier
from config import SCRAPED_JOBS_CSV_PATH, CLASSIFIER_MODEL_PATH, VECTORIZER_PATH

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_new_model():
    """
    Loads scraped data, trains the job classifier, and saves the model.
    This function should be run manually when you want to retrain the model.
    """
    logger.info("Starting model training process...")

    if not os.path.exists(SCRAPED_JOBS_CSV_PATH):
        logger.error(f"Scraped jobs CSV not found at {SCRAPED_JOBS_CSV_PATH}. Please run scraper first.")
        return

    try:
        df = pd.read_csv(SCRAPED_JOBS_CSV_PATH, encoding='utf-8')
        logger.info(f"Loaded {len(df)} jobs from {SCRAPED_JOBS_CSV_PATH}")

        # IMPORTANT: For training, your DataFrame MUST have a 'category' column
        # If your scraped CSV doesn't have it, you need to manually add it
        # or have a process to label a subset of your data for training.
        if 'category' not in df.columns or df['category'].isnull().all():
            logger.error("No 'category' column found or all categories are missing in the scraped data for training.")
            logger.error("Please manually label some jobs in data/latest_jobs.csv with at least two distinct categories (e.g., 'Software Development', 'Data Science', 'Marketing').")
            logger.error("Model training cannot proceed without labeled data.")
            return

        classifier = JobClassifier(CLASSIFIER_MODEL_PATH, VECTORIZER_PATH)
        classifier.train(df) # This will train and save the model
        logger.info("Model training completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred during model training: {e}", exc_info=True)

if __name__ == "__main__":
    train_new_model()