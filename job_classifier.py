# ml_model/job_classifier.py
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression # Or another classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import logging
import ssl # Ensure this is imported for the SSL fix

# --- NLTK Data Path Configuration (Crucial for finding downloads) ---
# Define a directory for NLTK data relative to your project root
# This will be 'JOB_POSTING_CLASIF/data/nltk_data'
# This logic assumes job_classifier.py is in ml_model/
# So, os.path.dirname(os.path.abspath(__file__)) gives ml_model/
# Then '..' gives JOB_POSTING_CLASIF/
# Then os.path.join('data', 'nltk_data') gives data/nltk_data


# Add the SSL workaround (important for Windows when downloading NLTK data)
# ml_model/job_classifier.py (and similar changes for main.py, model_trainer.py)

# Ensure 'ssl' is imported at the top of the file if it's not already
# import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
# --- End NLTK Data Path Configuration --- # (This comment line is fine, just indicates a section)


# Download NLTK data if not already downloaded (run once manually if error)
# Note: The `try-except` blocks here are for robustness, but ideally
# you'd run `python -m nltk.downloader all` once.
# The WARNING messages are expected if the resources aren't found initially.

try:
    nltk.data.find('corpora/stopwords')
except Exception as e: # Catch a broader exception
    logging.warning(f"NLTK stopwords not found, attempting download: {e}")
    nltk.download('stopwords') # REMOVED: download_dir=nltk_data_dir
try:
    nltk.data.find('corpora/wordnet')
except Exception as e: # Catch a broader exception
    logging.warning(f"NLTK wordnet not found, attempting download: {e}")
    nltk.download('wordnet')   # REMOVED: download_dir=nltk_data_dir
try:
    nltk.data.find('tokenizers/punkt')
except Exception as e: # Catch a broader exception
    logging.warning(f"NLTK punkt not found, attempting download: {e}")
    nltk.download('punkt')     # REMOVED: download_dir=nltk_data_dir
class JobClassifier:
    def __init__(self, model_path, vectorizer_path):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.model = None
        self.vectorizer = None
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.logger = logging.getLogger(__name__)

    def _preprocess_text(self, text):
        """Cleans and tokenizes text for vectorization."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        text = text.lower() # Lowercase
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove non-alphabetic characters
        tokens = nltk.word_tokenize(text) # Tokenize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords] # Lemmatize and remove stopwords
        return " ".join(tokens)

    def train(self, df):
        """
        Trains the classifier model and saves it along with the vectorizer.
        Expects DataFrame with 'job_description' and 'category' columns.
        """
        if df.empty or 'job_description' not in df.columns or 'category' not in df.columns:
            self.logger.warning("DataFrame is empty or missing 'job_description'/'category' columns for training.")
            return

        # Ensure 'category' column is clean and fill NaN if any
        df['category'] = df['category'].fillna('Uncategorized').astype(str)

        # Preprocess job descriptions
        df['processed_description'] = df['job_description'].apply(self._preprocess_text)

        X = df['processed_description']
        y = df['category']

        if X.empty or y.empty:
            self.logger.warning("No data after preprocessing for training.")
            return

        # Check for single class in y (target variable)
        if len(y.unique()) < 2:
            self.logger.error(f"Cannot train: only one class found in 'category' column: {y.unique()}")
            self.logger.error("Please ensure your training data (latest_jobs.csv) has at least two distinct categories.")
            return # Exit training if only one class

        # Initialize and fit TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000) # Limit features for efficiency
        X_vectorized = self.vectorizer.fit_transform(X)

        # Train a simple classifier (e.g., Logistic Regression)
        self.model = LogisticRegression(max_iter=1000, random_state=42)

        # Split data for evaluation (optional, but good practice)
        # Using stratify=y is good for imbalanced classes
        X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42, stratify=y)

        self.model.fit(X_train, y_train)
        self.logger.info("Classifier model trained successfully.")

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        self.logger.info("Classification Report on test set:\n" + classification_report(y_test, y_pred, zero_division=0))

        # Save model and vectorizer
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        self.logger.info(f"Model saved to {self.model_path}")
        self.logger.info(f"Vectorizer saved to {self.vectorizer_path}")

    def load_model(self):
        """Loads the trained model and vectorizer."""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
            self.logger.info("Classifier model and vectorizer loaded successfully.")
            return True
        else:
            self.logger.warning("Model or vectorizer files not found. Training needed.")
            return False

    def predict_category(self, text):
        """Predicts the category of a given job description text."""
        if not self.model or not self.vectorizer:
            self.logger.error("Model or vectorizer not loaded. Cannot predict.")
            return "Uncategorized" # Fallback category

        processed_text = self._preprocess_text(text)
        if not processed_text:
            return "Uncategorized" # Return default if text is empty after processing

        try:
            text_vectorized = self.vectorizer.transform([processed_text])
            prediction = self.model.predict(text_vectorized)
            return prediction[0]
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return "Error_Classification" # Indicate a classification error

# Example usage (for testing purposes when running job_classifier.py directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create dummy data for training with at least two categories
    data = {
        'job_title': ['Software Engineer', 'Data Scientist', 'Marketing Manager', 'HR Specialist', 'QA Tester', 'Project Manager'],
        'job_description': [
            'Develops web applications using Python and Django.',
            'Analyzes large datasets using SQL and machine learning models.',
            'Manages marketing campaigns and social media.',
            'Handles recruitment and employee relations.',
            'Tests software applications and writes test cases.',
            'Leads cross-functional teams and manages project timelines.'
        ],
        'category': ['Software Development', 'Data Science', 'Marketing', 'Human Resources', 'QA', 'Management']
    }
    df_dummy = pd.DataFrame(data)

    # Define paths for test model/vectorizer within ml_model/trained_model/
    model_path_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model", "job_classifier_model_test.pkl")
    vectorizer_path_test = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model", "job_vectorizer_test.pkl")
# ...
    os.makedirs(os.path.dirname(model_path_test), exist_ok=True) # Ensure directory exists

    classifier = JobClassifier(model_path_test, vectorizer_path_test)
    classifier.train(df_dummy) # Train with dummy data

    if classifier.load_model():
        test_desc = "Develops robust algorithms for deep learning projects using PyTorch."
        predicted_cat = classifier.predict_category(test_desc)
        print(f"\nTest description: '{test_desc}'")
        print(f"Predicted category: {predicted_cat}")
    else:
        print("Failed to load model for testing.")