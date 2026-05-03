from flask import Flask, request, render_template
import pdfplumber
import requests
import joblib

app = Flask(__name__)

# --- Load trained ML model and vectorizer ---
model = joblib.load("job_recommender.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_category(cv_text):
    X = vectorizer.transform([cv_text])
    return model.predict(X)[0]

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('cvfile')

    # ✅ 1. Check if file exists
    if not file or file.filename == '':
        return render_template('upload.html', error="No file selected")

    # ✅ 2. Check if file is PDF
    if not file.filename.lower().endswith('.pdf'):
        return render_template('upload.html', error="Only PDF files are allowed!")

    # ✅ 3. Get file info (name + size)
    file_name = file.filename
    file.seek(0, 2)  # move to end of file
    file_size = file.tell() / 1024  # size in KB
    file.seek(0)  # reset pointer

    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    # --- Predict category ---
    category = predict_category(text)

    # --- API Call ---
    app_id = "20e7d40c"
    app_key = "f027eeb1600bcf2003d767b5cdf9dc7e"

    url = "https://api.adzuna.com/v1/api/jobs/gb/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": 5,
        "what": category
    }

    response = requests.get(url, params=params).json()

    jobs = []
    for job in response.get("results", []):
        jobs.append({
            "title": job.get("title"),
            "company": job.get("company", {}).get("display_name"),
            "location": job.get("location", {}).get("display_name"),
            "redirect_url": job.get("redirect_url")
        })

    # ✅ 4. Send file info to results page
    return render_template(
        "results.html",
        jobs=jobs,
        category=category,
        cv_text=text,
        file_name=file_name,
        file_size=round(file_size, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)