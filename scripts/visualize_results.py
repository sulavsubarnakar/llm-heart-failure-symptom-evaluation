import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data = pd.read_csv(os.path.join(BASE_DIR, "evaluation", "scored_results.csv"))

# -----------------------------
# MODEL ACCURACY
# -----------------------------
model_acc = data.groupby("model")["correct_prediction"].mean()

plt.figure()
model_acc.plot(kind="bar")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=0)
plt.savefig(os.path.join(BASE_DIR, "results", "model_accuracy.png"))
plt.close()

# -----------------------------
# CATEGORY ACCURACY
# -----------------------------
category_acc = data.groupby("category")["correct_prediction"].mean()

plt.figure(figsize=(10,5))
category_acc.plot(kind="bar")
plt.title("Accuracy by Symptom Category")
plt.ylabel("Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "results", "category_accuracy.png"))
plt.close()

# -----------------------------
# ERROR DISTRIBUTION
# -----------------------------
error_dist = data["error_type"].value_counts()

plt.figure()
plt.pie(error_dist, labels=error_dist.index, autopct="%1.1f%%")
plt.title("Error Distribution")
plt.savefig(os.path.join(BASE_DIR, "results", "error_distribution.png"))
plt.close()

print("Visualizations saved.")