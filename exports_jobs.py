import requests
import pandas as pd

app_id = "20e7d40c"
app_key = "f027eeb1600bcf2003d767b5cdf9dc7e"

countries = ["gb", "us", "in", "pk"]  # UK, US, India, Pakistan
all_jobs = []

for country in countries:
    for page in range(1, 6):  # fetch first 5 pages
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "results_per_page": 50,
            "what": "software engineer"  # change keyword if needed
        }
        response = requests.get(url, params=params).json()
        jobs = response.get("results", [])
        all_jobs.extend(jobs)

df = pd.DataFrame([{
    "title": job["title"],
    "company": job["company"]["display_name"],
    "location": job["location"]["display_name"],
    "description": job["description"],
    "category": job["category"]["label"]
} for job in all_jobs])

df.to_csv("global_jobs_dataset.csv", index=False)
print("Saved global_jobs_dataset.csv with", len(df), "rows")
