# Job Posting Classification Based on Required Skills (Hierarchical Clustering)

This project scrapes job listings from [karkidi.com](https://www.karkidi.com) and automatically categorizes them based on required skills using **Hierarchical Clustering** (Agglomerative Clustering). The system clusters jobs by skills and saves the model for later use in classification or alerting users.

---

## Features

- Scrapes job titles, companies, locations, experience, summary, and required skills from karkidi.com.
- Preprocesses the skills text (cleaning, tokenization).
- Uses TF-IDF vectorization on skills text.
- Performs unsupervised clustering with **Hierarchical Clustering** (Agglomerative Clustering).
- Saves the trained clustering model, vectorizer, and clustered job data.
- (Optional) Can be extended to notify users based on their skill preferences.
- (Optional) Automate daily scraping and clustering using scheduling.

---

## Setup & Usage

### Requirements

- Python 3.7+
- Packages:
  - requests
  - beautifulsoup4
  - pandas
  - scikit-learn
  - joblib
  - streamlit (optional, for UI)

Install dependencies via:

```bash
pip install -r requirements.txt
Running the scraper and clustering
python
Copy
Edit
python daily_scrape_and_cluster.py
This script will:

Scrape job listings from karkidi.com (default 2 pages, configurable)

Preprocess skills column

Vectorize and cluster jobs using Hierarchical Clustering

Save model (model/karkidi_model.pkl), vectorizer (model/karkidi_vectorizer.pkl), and clustered data (clustered_jobs.csv)

Deploying the Streamlit app
The Streamlit app (app.py) loads the saved model and clustered data to:

Display job listings grouped by clusters

Allow filtering by cluster

(Optional) Notify users on new jobs matching their preferred skills

Run locally:

bash
Copy
Edit
streamlit run app.py
Automation (Optional)
To automate scraping and clustering daily:

Schedule the daily_scrape_and_cluster.py script using cron (Linux/macOS) or Task Scheduler (Windows)

Or use GitHub Actions workflow to run the script and push updates

Streamlit app will then reflect updated clusters on reload

Project Structure
bash
Copy
Edit
your-repo/
├── daily_scrape_and_cluster.py   # Main scraping & clustering script
├── app.py                        # Streamlit app UI
├── clustered_jobs.csv            # Scraped & clustered job data (updated regularly)
├── model/
│   ├── karkidi_model.pkl         # Trained hierarchical clustering model
│   └── karkidi_vectorizer.pkl    # TF-IDF vectorizer
├── requirements.txt              # Python dependencies
└── README.md                    # This documentation
