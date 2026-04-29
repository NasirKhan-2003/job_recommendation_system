import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# =========================
# Load model & vectorizer
# =========================
model = joblib.load("job_recommender.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# =========================
# Load dataset
# =========================
df = pd.read_csv("global_jobs_dataset.csv")

# Features & labels
X = vectorizer.transform(df["description"])
y = df["category"]

# =========================
# Train/Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Predictions
# =========================
y_pred = model.predict(X_test)

# =========================
# Evaluation Metrics
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# =========================
# Bar Graph Plot
# =========================
# =========================
# Bar Graph Plot (Styled)
# =========================
labels = ["Accuracy", "Precision", "Recall"]
values = [accuracy, precision, recall]

plt.figure(figsize=(6,5))

# Custom colors
colors = ["#4CAF50", "#2196F3", "#FF9800"]

# Narrow bars
bars = plt.bar(labels, values, color=colors, width=0.4)

plt.ylim(0, 1)
plt.ylabel("Score", fontsize=12)
plt.title("Model Performance", fontsize=14, fontweight='bold')

# Add value labels inside bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height - 0.08,
        f"{height:.2f}",
        ha='center',
        va='center',
        color='white',
        fontsize=11,
        fontweight='bold'
    )

# Clean up look
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.xticks(fontsize=11)
plt.yticks(fontsize=10)

plt.tight_layout()
plt.show()
