import pandas as pd
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import os

# -----------------------------
# PATH SETUP (ROBUST)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESPONSES_PATH = os.path.join(BASE_DIR, "responses", "model_responses.csv")
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "hf_symptom_dataset.csv")

OUTPUT_EVAL = os.path.join(BASE_DIR, "evaluation", "scored_results.csv")
OUTPUT_SUMMARY = os.path.join(BASE_DIR, "results", "evaluation_summary.csv")

# -----------------------------
# LOAD DATA
# -----------------------------
responses = pd.read_csv(RESPONSES_PATH)
dataset = pd.read_csv(DATASET_PATH)

print("Responses loaded:", len(responses))

# -----------------------------
# EXTRACT LABEL FROM JSON
# -----------------------------
def extract_label(response):
    try:
        data = json.loads(response)
        return data.get("hf_relevance", "uncertain")
    except:
        return "uncertain"

responses["predicted_label"] = responses["model_response"].apply(extract_label)

# -----------------------------
# MERGE DATA
# -----------------------------
merged = responses.merge(dataset, on="id")

# -----------------------------
# CORRECTNESS
# -----------------------------
merged["correct_prediction"] = (
    merged["hf_relevance"] == merged["predicted_label"]
)

# -----------------------------
# OVERALL METRICS
# -----------------------------
true_labels = merged["hf_relevance"]
pred_labels = merged["predicted_label"]

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, average="weighted", zero_division=0)
recall = recall_score(true_labels, pred_labels, average="weighted", zero_division=0)
f1 = f1_score(true_labels, pred_labels, average="weighted", zero_division=0)

from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(true_labels, pred_labels)

print("\n===== OVERALL PERFORMANCE =====")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Cohen Kappa: {kappa:.4f}")

# -----------------------------
# CLASSIFICATION REPORT (VERY IMPORTANT)
# -----------------------------
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(true_labels, pred_labels))

# -----------------------------
# MODEL-LEVEL PERFORMANCE
# -----------------------------
model_accuracy = merged.groupby("model")["correct_prediction"].mean()

print("\n===== MODEL PERFORMANCE =====")
print(model_accuracy)

# -----------------------------
# CATEGORY-LEVEL PERFORMANCE
# -----------------------------
category_accuracy = merged.groupby("category")["correct_prediction"].mean()

print("\n===== CATEGORY PERFORMANCE =====")
print(category_accuracy)

# -----------------------------
# ERROR ANALYSIS (HIGH MARKS)
# -----------------------------
merged["error_type"] = "correct"

merged.loc[
    (merged["hf_relevance"] != merged["predicted_label"]) &
    (merged["predicted_label"] == "HF_related"),
    "error_type"
] = "false_positive"

merged.loc[
    (merged["hf_relevance"] != merged["predicted_label"]) &
    (merged["hf_relevance"] == "HF_related"),
    "error_type"
] = "false_negative"

merged.loc[
    (merged["predicted_label"] == "uncertain"),
    "error_type"
] = "uncertain"

# -----------------------------
# UNCERTAINTY ANALYSIS (DISTINCTION LEVEL)
# -----------------------------
merged["is_uncertain"] = merged["predicted_label"] == "uncertain"

uncertainty_rate = merged.groupby("model")["is_uncertain"].mean()

print("\n===== UNCERTAINTY RATE =====")
print(uncertainty_rate)

error_distribution = merged["error_type"].value_counts()

print("\n===== ERROR DISTRIBUTION =====")
print(error_distribution)

# -----------------------------
# SAVE RESULTS
# -----------------------------
os.makedirs(os.path.join(BASE_DIR, "evaluation"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "results"), exist_ok=True)

merged.to_csv(OUTPUT_EVAL, index=False)

summary = pd.DataFrame([{
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "cohen_kappa": kappa
}])

uncertainty_rate.to_csv(os.path.join(BASE_DIR, "results", "uncertainty_by_model.csv"))
summary.to_csv(OUTPUT_SUMMARY, index=False)

print("\nResults saved successfully.")