# app.py
import streamlit as st
import pandas as pd
import os
from datetime import datetime

# Adjust this path if your scraped data CSV is saved elsewhere
SCRAPED_JOBS_CSV_PATH = "data/latest_jobs.csv"
TRACKED_JOBS_JSON_PATH = "data/job_tracker.json"

st.set_page_config(layout="wide", page_title="Karkidi Job Finder")

@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid reloading on every interaction
def load_job_data(file_path):
    """Loads job data from a CSV file."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Ensure 'job_url' column exists, if not, add it with 'N/A'
        if 'job_url' not in df.columns:
            df['job_url'] = 'N/A'
        # Ensure 'category' column exists, if not, add it with 'Uncategorized'
        if 'category' not in df.columns:
            df['category'] = 'Uncategorized'
        return df
    return pd.DataFrame() # Return empty DataFrame if file not found

@st.cache_data(ttl=3600)
def load_tracked_job_data(file_path):
    """Loads tracked job data from JSON for last update time."""
    if os.path.exists(file_path):
        try:
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # The JSON might store last_updated in a specific format
                return data.get('last_updated', 'N/A')
        except json.JSONDecodeError:
            return 'N/A'
    return 'N/A'


# --- Streamlit App UI ---
st.title("CareerScope Kerala (powered by Karkidi.com data)")

# Load data
jobs_df = load_job_data(SCRAPED_JOBS_CSV_PATH)
last_update_time = load_tracked_job_data(TRACKED_JOBS_JSON_PATH)

if not jobs_df.empty:
    st.sidebar.header("Filter Jobs")

    # Search box
    search_query = st.sidebar.text_input("Search (Title, Company, Skills, Location)", "").lower()

    # Category filter
    categories = ['All'] + sorted(jobs_df['category'].dropna().unique().tolist())
    selected_category = st.sidebar.selectbox("Filter by Category", categories)

    # Apply filters
    filtered_df = jobs_df.copy()
    if search_query:
        filtered_df = filtered_df[
            filtered_df['job_title'].str.lower().str.contains(search_query) |
            filtered_df['company'].str.lower().str.contains(search_query) |
            filtered_df['skills'].str.lower().str.contains(search_query) |
            filtered_df['location'].str.lower().str.contains(search_query) |
            filtered_df['job_description'].str.lower().str.contains(search_query)
        ]

    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]

    st.subheader("Latest Job Listings")
    st.info(f"Data last updated: {last_update_time}")

    if filtered_df.empty:
        st.warning("No jobs found matching your criteria.")
    else:
        st.write(f"Displaying {len(filtered_df)} jobs.")
        # Display jobs using expanders for better readability
        for index, row in filtered_df.iterrows():
            with st.expander(f"**{row['job_title']}** at **{row['company']}** ({row['location']}) - Category: {row['category'] if pd.notna(row['category']) else 'Uncategorized'}"):
                st.write(f"**Company:** {row['company']}")
                st.write(f"**Location:** {row['location']}")
                st.write(f"**Skills:** {row['skills']}")
                st.write(f"**Description:** {row['job_description']}")
                if row['job_url'] and row['job_url'] != 'N/A':
                    st.markdown(f"[View Job on Karkidi.com]({row['job_url']})")
else:
    st.warning("No job data available. The scraper might not have run yet or failed. Please check the logs.")
    st.info("The scheduler will run the scraper periodically to fetch new data.")
