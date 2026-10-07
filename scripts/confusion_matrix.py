import pandas as pd
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# -----------------------------
# LOAD DATA
# -----------------------------
responses = pd.read_csv("responses/model_responses.csv")
dataset = pd.read_csv("dataset/hf_symptom_dataset.csv")

# -----------------------------
# EXTRACT PREDICTIONS
# -----------------------------
def extract_label(response):
    try:
        data = json.loads(response)
        return data.get("hf_relevance", "uncertain")
    except:
        return "uncertain"

responses["predicted_label"] = responses["model_response"].apply(extract_label)

# -----------------------------
# MERGE WITH TRUE LABELS
# -----------------------------
merged = responses.merge(dataset, on="id")

true_labels = merged["hf_relevance"]
pred_labels = merged["predicted_label"]

# -----------------------------
# DEFINE LABEL ORDER (IMPORTANT)
# -----------------------------
labels = ["HF_related", "possible_HF", "uncertain", "not_HF"]

# -----------------------------
# COMPUTE CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(true_labels, pred_labels, labels=labels)

print("Confusion Matrix:")
print(cm)

# -----------------------------
# VISUALIZE (OPTIONAL)
# -----------------------------
plt.figure(figsize=(6,5))
plt.imshow(cm)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)

# Add numbers inside cells
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.show()