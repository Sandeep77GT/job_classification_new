Job Posting Classification Based on Required Skills Using Hierarchical Clustering
This project scrapes job postings from Karkidi.com, extracts relevant information including required skills, and classifies the jobs using hierarchical clustering based on those skills. A Streamlit frontend is provided to explore the clustered job postings interactively.

Features
Scrapes job postings related to a specific keyword from multiple pages.

Extracts job title, company, location, experience, skills, and summary.

Preprocesses the extracted skills into a format suitable for text vectorization.

Uses TF-IDF vectorization and Agglomerative (Hierarchical) Clustering to group jobs by skill similarity.

Saves clustered data and models for later use.

Visualizes and filters job clusters via an interactive Streamlit web application.

Project Structure
bash
Copy
Edit
.
├── hierarchical_clustering_jobs.py        # Main script to scrape, preprocess, and cluster jobs
├── job_cluster_viewer.py                  # Streamlit app to display clustered job postings
├── model/
│   ├── karkidi_model.pkl                  # Saved clustering model
│   └── karkidi_vectorizer.pkl             # Saved TF-IDF vectorizer
├── clustered_jobs.csv                     # CSV output with clustered job data
└── requirements.txt                       # Required Python packages
Setup Instructions
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/job-posting-clustering.git
cd job-posting-clustering
2. Install Required Packages
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Scraper and Clustering Script
bash
Copy
Edit
python hierarchical_clustering_jobs.py
This will generate:

clustered_jobs.csv

TF-IDF vectorizer and clustering model in the model/ directory

4. Launch the Streamlit App
bash
Copy
Edit
streamlit run job_cluster_viewer.py
Requirements
Python 3.7+

joblib

beautifulsoup4

scikit-learn

pandas

requests

streamlit

Install them with:

bash
Copy
Edit
pip install -r requirements.txt
Customization
Modify the keyword and pages parameters in scrape_karkidi_jobs() to customize search queries.

Adjust n_clusters in cluster_jobs() to experiment with different numbers of clusters.

License
This project is licensed under the MIT License.
