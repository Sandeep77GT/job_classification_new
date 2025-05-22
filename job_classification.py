import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestCentroid
import joblib

# --- Web Scraper ---
def scrape_karkidi_jobs(keyword="data science", pages=1):
    headers = {'User-Agent': 'Mozilla/5.0'}
    base_url = "https://www.karkidi.com/Find-Jobs/{page}/all/India?search={query}"
    jobs_list = []

    for page in range(1, pages + 1):
        url = base_url.format(page=page, query=keyword.replace(' ', '%20'))
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        job_blocks = soup.find_all("div", class_="ads-details")
        for job in job_blocks:
            try:
                title = job.find("h4").get_text(strip=True)
                company = job.find("a", href=lambda x: x and "Employer-Profile" in x).get_text(strip=True)
                location = job.find("p").get_text(strip=True)
                experience = job.find("p", class_="emp-exp").get_text(strip=True)
                key_skills_tag = job.find("span", string="Key Skills")
                skills = key_skills_tag.find_next("p").get_text(strip=True) if key_skills_tag else ""
                summary_tag = job.find("span", string="Summary")
                summary = summary_tag.find_next("p").get_text(strip=True) if summary_tag else ""

                jobs_list.append({
                    "Title": title,
                    "Company": company,
                    "Location": location,
                    "Experience": experience,
                    "Summary": summary,
                    "Skills": skills
                })
            except Exception as e:
                print(f"Error parsing job block: {e}")
                continue

        time.sleep(1)

    return pd.DataFrame(jobs_list)

# --- Text Processing and ML ---
def split_skills_tokenizer(x):
    return x.split(',')

def preprocess_skills(df):
    df = df.copy()
    df['Skills'] = df['Skills'].fillna("").str.lower().str.strip()
    return df

def vectorize_skills(df):
    vectorizer = TfidfVectorizer(tokenizer=split_skills_tokenizer, lowercase=True)
    X = vectorizer.fit_transform(df['Skills'])
    return X, vectorizer

def cluster_skills(X, n_clusters=5):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X.toarray())
    return model, labels

def train_centroid_classifier(X, labels):
    clf = NearestCentroid()
    clf.fit(X.toarray(), labels)
    return clf

def classify_new_jobs(df_new, vectorizer, clf):
    df_new = preprocess_skills(df_new)
    X_new = vectorizer.transform(df_new['Skills'])
    df_new['Cluster'] = clf.predict(X_new.toarray())
    return df_new

def notify_user(df_classified, user_cluster_id):
    matched = df_classified[df_classified['Cluster'] == user_cluster_id]
    if not matched.empty:
        st.subheader("🎯 New job(s) matching your interest:")
        st.dataframe(matched[['Title', 'Company', 'Skills']])
    else:
        st.info("No new matching jobs today.")

# --- Streamlit App ---
st.title("💼 Karkidi Job Recommender")
keyword = st.text_input("Enter job keyword:", "data science")
pages = st.slider("Number of pages to scrape:", 1, 5, 2)
cluster_choice = st.slider("Select your interest cluster:", 0, 4, 2)

if st.button("🔍 Scrape and Analyze Jobs"):
    with st.spinner("Scraping jobs and analyzing..."):
        df_jobs = scrape_karkidi_jobs(keyword, pages)
        df_jobs = preprocess_skills(df_jobs)
        X, vectorizer = vectorize_skills(df_jobs)
        model, labels = cluster_skills(X, n_clusters=5)
        df_jobs['Cluster'] = labels

        clf = train_centroid_classifier(X, labels)

        df_new_jobs = scrape_karkidi_jobs(keyword, pages=1)
        df_classified = classify_new_jobs(df_new_jobs, vectorizer, clf)

    notify_user(df_classified, user_cluster_id=cluster_choice)
