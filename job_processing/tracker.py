# job_processing/tracker.py
import json
import pandas as pd
import os
import logging
from datetime import datetime

class JobTracker:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self.tracked_jobs = {}
        self.logger = logging.getLogger(__name__)
        self._load_jobs()

    def _load_jobs(self):
        """Loads tracked jobs from the JSON file."""
        if os.path.exists(self.json_file_path):
            try:
                with open(self.json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.tracked_jobs = data.get('jobs', {})
                    self.logger.info(f"Loaded {len(self.tracked_jobs)} tracked jobs from {self.json_file_path}")
            except json.JSONDecodeError as e:
                self.logger.error(f"Error decoding JSON from {self.json_file_path}: {e}. Starting with empty tracker.")
                self.tracked_jobs = {}
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while loading {self.json_file_path}: {e}. Starting with empty tracker.")
                self.tracked_jobs = {}
        else:
            self.logger.info(f"Tracker file not found at {self.json_file_path}. Starting with empty tracker.")
            # Ensure the directory exists for saving later
            os.makedirs(os.path.dirname(self.json_file_path), exist_ok=True)


    def save_jobs(self):
        """Saves tracked jobs to the JSON file."""
        data_to_save = {
            'last_updated': datetime.now().isoformat(),
            'jobs': self.tracked_jobs
        }
        try:
            with open(self.json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4)
            self.logger.info(f"Saved {len(self.tracked_jobs)} tracked jobs to {self.json_file_path}")
        except Exception as e:
            self.logger.error(f"Error saving jobs to {self.json_file_path}: {e}")

    def get_new_jobs(self, current_jobs_df):
        """
        Compares current scraped jobs with tracked jobs and returns a DataFrame of new jobs.
        Assumes 'job_url' is the unique identifier.
        """
        if current_jobs_df.empty:
            return pd.DataFrame() # Return empty if no current jobs

        new_jobs = []
        for index, job in current_jobs_df.iterrows():
            job_url = job.get('job_url')
            # Check if job_url exists and is not 'N/A' and not already tracked
            if job_url and job_url != 'N/A' and job_url not in self.tracked_jobs:
                new_jobs.append(job.to_dict())

        return pd.DataFrame(new_jobs)

    def add_jobs(self, jobs_df):
        """Adds new jobs (and their classifications) to the tracker."""
        if jobs_df.empty:
            return

        for index, job in jobs_df.iterrows():
            job_url = job.get('job_url')
            if job_url and job_url != 'N/A':
                # Store relevant details. Ensure 'category' is included if classified.
                self.tracked_jobs[job_url] = {
                    'job_title': job.get('job_title'),
                    'company': job.get('company'),
                    'location': job.get('location'),
                    'skills': job.get('skills'),
                    'job_description': job.get('job_description'),
                    'category': job.get('category', 'Uncategorized'), # Store category
                    'first_seen': datetime.now().isoformat() # Track when it was first seen
                }
            else:
                self.logger.warning(f"Job without a valid URL skipped from tracking: {job.get('job_title')}")

# Example usage (for testing purposes)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Ensure 'data' directory exists for local testing
    os.makedirs('data', exist_ok=True)
    
    tracker_path = os.path.join("..", "data", "job_tracker_test.json")
    tracker = JobTracker(tracker_path)

    # Dummy current scraped jobs
    current_jobs = pd.DataFrame([
        {'job_title': 'Dev Ops Engineer', 'company': 'XYZ Corp', 'location': 'Kochi', 'job_url': 'http://example.com/job1', 'skills': 'AWS, Docker', 'job_description': 'Manages cloud infra', 'category': 'DevOps'},
        {'job_title': 'Jr. Python Dev', 'company': 'ABC Inc', 'location': 'Bangalore', 'job_url': 'http://example.com/job2', 'skills': 'Python, Django', 'job_description': 'Python dev', 'category': 'Software Development'},
        {'job_title': 'Marketing Associate', 'company': 'PQR Ltd', 'location': 'Chennai', 'job_url': 'http://example.com/job3', 'skills': 'Social Media', 'job_description': 'Marketing job', 'category': 'Marketing'}
    ])

    # Add a job that was already tracked
    tracker.tracked_jobs['http://example.com/job1'] = {
        'job_title': 'Dev Ops Engineer', 'company': 'XYZ Corp', 'location': 'Kochi', 'skills': 'AWS, Docker', 'job_description': 'Manages cloud infra', 'category': 'DevOps', 'first_seen': '2023-01-01'
    }
    tracker.save_jobs()

    new_jobs = tracker.get_new_jobs(current_jobs)
    print(f"\nNew jobs found:\n{new_jobs}")

    # Add new jobs to tracker
    tracker.add_jobs(new_jobs)
    tracker.save_jobs()