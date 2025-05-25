# scraper/karkidi_scraper.py
import requests
from bs4 import BeautifulSoup
import time
import random
import logging
from fake_useragent import UserAgent

# --- For relative imports to work when run from main.py ---
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# --- End of fix ---

from config import KARKIDI_JOB_LISTING_PATTERN, COMMON_SKILL_KEYWORDS

class KarkidiScraper:
    # Changed __init__ to accept base_url
    def __init__(self, base_url):
        self.base_url = base_url # Storing the base URL if needed for other methods
        self.base_url_pattern = KARKIDI_JOB_LISTING_PATTERN # This will be used for actual page URLs
        self.logger = logging.getLogger(__name__)
        self.ua = UserAgent() # For rotating user agents

    def _extract_skills_from_text(self, text):
        """
        Extracts skills from job description text using a predefined list of keywords.
        This is a basic keyword matching.
        """
        extracted_skills = []
        text_lower = text.lower()
        for skill in COMMON_SKILL_KEYWORDS:
            if skill.lower() in text_lower:
                extracted_skills.append(skill)
        return ", ".join(sorted(list(set(extracted_skills))))

    # scraper/karkidi_scraper.py
# ... (existing imports and class definition) ...

    def scrape_jobs(self):
        """
        Scrapes job listings from karkidi.com using requests and BeautifulSoup.
        """
        jobs_data = []
        MAX_PAGES_TO_SCRAPE = 10 # Adjust this if you need to scrape more or fewer pages

        self.logger.info(f"Starting scraping from {self.base_url} (Page 1) and then using pattern for subsequent pages.")

        for page_num in range(1, MAX_PAGES_TO_SCRAPE + 1):
            if page_num == 1:
                url = self.base_url # Use the base URL for the first page
            else:
                url = self.base_url_pattern.format(page_num) # Use the pattern for page 2 onwards

            self.logger.info(f"Visiting page: {url}")

            try:
                headers = {'User-Agent': self.ua.random}
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")

                # Find all job blocks (main container for each job listing)
                job_blocks = soup.find_all("div", class_="ads-details")

                if not job_blocks:
                    self.logger.info(f"No job blocks found on page {page_num}. Ending scrape.")
                    break # No more jobs on this page or last page reached

                self.logger.info(f"Found {len(job_blocks)} job elements on page {page_num}.")

                for job_block in job_blocks:
                    job = {}
                    try:
                        # Extract Job Title and URL
                        title_tag = job_block.find("h4")
                        job['job_title'] = title_tag.get_text(strip=True) if title_tag else 'N/A'

                        job_link_tag = title_tag.find('a') if title_tag else None
                        job_url = job_link_tag['href'] if job_link_tag and 'href' in job_link_tag.attrs else 'N/A'
                        if job_url and not job_url.startswith('http'):
                            job['job_url'] = "https://www.karkidi.com" + job_url # Ensure job_url is absolute
                        else:
                            job['job_url'] = job_url

                        # Extract Company
                        company_tag = job_block.find("a", href=lambda x: x and "Employer-Profile" in x)
                        job['company'] = company_tag.get_text(strip=True) if company_tag else 'N/A'

                        # Extract Location
                        location_tag = job_block.find("p") # This selector might still be too generic, confirm from HTML
                        job['location'] = location_tag.get_text(strip=True) if location_tag else 'N/A'

                        # Extract Key Skills from the main listing block
                        # You might need to inspect the HTML of the job listing itself
                        # to confirm the exact span text or class for "Key Skills"
                        key_skills_span = job_block.find("span", string="Key Skills")
                        skills_text = key_skills_span.find_next("p").get_text(strip=True) if key_skills_span else ""
                        job['skills'] = self._extract_skills_from_text(skills_text)

                        # For job_description, for now, we'll use the skills text or extend it.
                        job['job_description'] = skills_text # Placeholder, improve if needed

                        jobs_data.append(job)

                    except Exception as e:
                        self.logger.warning(f"Parsing error on page {page_num} for a job block: {e}", exc_info=True)

                time.sleep(random.uniform(2, 5))

            except requests.exceptions.Timeout:
                self.logger.error(f"Request timed out after 30 seconds for page {page_num}: {url}. Skipping this page.")
                continue
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"HTTP Error for page {page_num}: {e}. Status code: {e.response.status_code}. Skipping this page.")
                # IMPORTANT: If you keep getting 404s after fixing the URL, the pattern is still wrong.
                # If a 404 means no more pages, you might consider 'break' instead of 'continue' here.
                # For now, 'continue' is safer.
                continue
            except requests.exceptions.RequestException as e:
                self.logger.error(f"General Request Error for page {page_num}: {e}. Skipping this page.")
                continue
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while processing page {page_num}: {e}", exc_info=True)
                continue

        self.logger.info("Scraping complete.")
        return jobs_data

# Example of how to use for testing (will be called by main.py usually)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # When running directly, provide a dummy base_url
    scraper = KarkidiScraper("https://www.karkidi.com")
    scraped_jobs = scraper.scrape_jobs()
    if scraped_jobs:
        import pandas as pd
        df = pd.DataFrame(scraped_jobs)
        print(f"Scraped {len(df)} jobs.")
        print(df.head())
        # Ensure 'data' directory exists for local testing
        os.makedirs('data', exist_ok=True)
        df.to_csv(os.path.join("data", "scraped_karkidi_jobs_test.csv"), index=False)
        print("Scraped data saved to data/scraped_karkidi_jobs_test.csv")
    else:
        print("No jobs scraped.")