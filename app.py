from flask import Flask, request, render_template
import pdfplumber
import requests
import joblib

app = Flask(__name__)

# --- Load trained ML model and vectorizer ---
model = joblib.load("job_recommender.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_category(cv_text):
    """Predict job category from CV text using trained model"""
    X = vectorizer.transform([cv_text])
    return model.predict(X)[0]

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['cvfile']
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    # --- Predict category using ML model ---
    category = predict_category(text)

    # --- Query Adzuna API with predicted category ---
    app_id = "20e7d40c"   # replace with your Adzuna App ID
    app_key = "f027eeb1600bcf2003d767b5cdf9dc7e" # replace with your Adzuna App Key
    url = "https://api.adzuna.com/v1/api/jobs/gb/search/1"  # change country code if needed
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": 5,
        "what": category
    }

    response = requests.get(url, params=params).json()
    jobs = response.get("results", [])

    # --- Format job listings ---
    job_listings = f"<h3>Recommended Jobs for {category}:</h3><ul>"
    for job in jobs:
        job_listings += f"<li><a href='{job['redirect_url']}' target='_blank'>{job['title']}</a> at {job['company']['display_name']} – {job['location']['display_name']}</li>"
    job_listings += "</ul>"

    return job_listings + "<hr><h3>Extracted CV Text:</h3><pre>" + text + "</pre>"

if __name__ == '__main__':
    app.run(debug=True)
