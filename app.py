import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

from scripts.llm_clients import run_model

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_file(path):
    try:
        return pd.read_csv(path)
    except:
        return None

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Heart Failure LLM Evaluation System",
    page_icon="❤️",
    layout="wide"
)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Symptom Analyzer",
        "Evaluation Dashboard",
        "Dataset Explorer",
        "System Overview"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("CN6000 Dissertation Project")

# -----------------------------
# LABEL MAP
# -----------------------------
label_map = {
    "HF_related": "Heart Failure Related",
    "possible_HF": "Possible Heart Failure",
    "uncertain": "Uncertain",
    "not_HF": "Not Heart Failure"
}

labels = list(label_map.keys())

# -----------------------------

# -----------------------------
# INPUT VALIDATION
# -----------------------------
def validate_input(text):
    if not text or len(text.strip()) < 5:
        return False
    if text.strip().isdigit():
        return False
    return True

# -----------------------------
# ANALYSIS FUNCTION
# -----------------------------
def analyze_symptom(symptom, model_choice):

    # -----------------------------
    # REAL LLM (NEW ADDITION)
    # -----------------------------
    if model_choice == "Real_LLM":

        prompt = f"""
You are a clinical assistant.

Classify the symptom into one of:
HF_related, possible_HF, uncertain, not_HF

Return format STRICTLY:
Label: one of [HF_related, possible_HF, uncertain, not_HF]
Explanation: one short sentence

Symptom: {symptom}
"""

        result = run_model("Real_LLM", prompt, None)

    # -----------------------------
    # SIMULATED MODELS (UNCHANGED)
    # -----------------------------
    else:
        if "breath" in symptom.lower() or "swelling" in symptom.lower():
            base_label = "HF_related"
        elif "tired" in symptom.lower() or "fatigue" in symptom.lower():
            base_label = "possible_HF"
        elif "cough" in symptom.lower():
            base_label = "uncertain"
        else:
            base_label = "not_HF"

        result = run_model(model_choice, symptom, base_label)

    # -----------------------------
    # COMMON POST-PROCESSING
    # -----------------------------
    # SAFETY CHECK (VERY IMPORTANT)
    if "hf_relevance" not in result:
        result["hf_relevance"] = "uncertain"

    if "explanation" not in result:
        result["explanation"] = "Model returned incomplete response."

    result["identified_symptom"] = symptom

    return result


# =============================
# PAGE 1 - SYMPTOM ANALYZER
# =============================
if page == "Symptom Analyzer":

    st.title("❤️ Heart Failure Symptom Interpretation System")
    st.warning("Research demo only. Not medical advice. Three of the four models are simulated.")
    st.markdown("### AI-assisted clinical text classification using LLMs")

    st.write("Enter a realistic symptom description.")

    st.info("💡 Example: I feel breathless when walking upstairs")

    model_choice = st.selectbox(
        "Select Model",
        ["Model_A", "Model_B", "Model_C", "Real_LLM"]
    )

    if model_choice == "Real_LLM":
        st.success("🔵 Using Real LLM (API-based model)")
        st.caption("This model performs real-time inference using an external API")
    else:
        st.info("⚪ Using Simulated Model")

    symptom = st.text_area(
        "Symptom description",
        placeholder="Describe symptoms in natural language..."
    )

    if st.button("Analyze Symptom"):

        if not validate_input(symptom):
            st.error("⚠️ Please enter a valid symptom description.")
            st.stop()

        result = analyze_symptom(symptom, model_choice)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Prediction Output")

            readable = label_map.get(result["hf_relevance"], "Unknown")

            if result["hf_relevance"] == "HF_related":
                st.success("🟢 " + readable)
            elif result["hf_relevance"] == "possible_HF":
                st.warning("🟡 " + readable)
            elif result["hf_relevance"] == "not_HF":
                st.error("🔴 " + readable)
            else:
                st.info("⚪ " + readable)


        with col2:
            st.subheader("Model Explanation")
            st.write(result["explanation"])

        st.markdown("---")

        st.subheader("Clinical Interpretation Summary")

        if result["hf_relevance"] == "HF_related":
            st.error("🚨 High Risk of Heart Failure")
        elif result["hf_relevance"] == "possible_HF":
            st.warning("⚠️ Moderate Risk")
        elif result["hf_relevance"] == "uncertain":
            st.info("❓ Uncertain")
        else:
            st.success("✅ Low Risk")

        st.markdown(f"""
### Structured Interpretation

**Symptom Description:**  
{result["identified_symptom"]}

**Model Interpretation:**  
{label_map[result["hf_relevance"]]}


**Clinical Explanation:**  
{result["explanation"]}
""")


# =============================
# PAGE 2 - DASHBOARD
# =============================
elif page == "Evaluation Dashboard":

    st.title("📊 Model Evaluation Dashboard")
    st.warning("Research demo only. Not medical advice. Three of the four models are simulated.")

    summary = load_file(os.path.join(BASE_DIR, "results", "evaluation_summary.csv"))
    dataset = load_file(os.path.join(BASE_DIR, "dataset", "hf_symptom_dataset.csv"))
    responses = load_file(os.path.join(BASE_DIR, "responses", "model_responses.csv"))
    scored = load_file(os.path.join(BASE_DIR, "evaluation", "scored_results.csv"))

    st.subheader("System Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Size", len(dataset) if dataset is not None else 0)
    col2.metric("Models Tested", responses["model"].nunique() if responses is not None else 0)
    col3.metric("Total Predictions", len(responses) if responses is not None else 0)

    st.divider()

    if summary is not None:
        st.subheader("Evaluation Metrics")
        st.caption("These numbers combine all four models (three simulated, one real). They are not the real model's score.")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Accuracy", f"{summary['accuracy'][0]*100:.2f}%")
        col2.metric("Precision", f"{summary['precision'][0]*100:.2f}%")
        col3.metric("Recall", f"{summary['recall'][0]*100:.2f}%")
        col4.metric("F1 Score", f"{summary['f1_score'][0]*100:.2f}%")

    st.divider()

    if scored is not None and summary is not None:

        st.subheader("Key Insights")

        best_model = scored.groupby("model")["correct_prediction"].mean().idxmax()
        worst_model = scored.groupby("model")["correct_prediction"].mean().idxmin()
        common_error = scored["error_type"].value_counts().idxmax()

        st.markdown(f"""
- **Best Performing Model (simulated models included):** {best_model}  
- **Lowest Performing Model (simulated models included):** {worst_model}  
- **Accuracy (all 4 models combined):** {summary['accuracy'][0]*100:.2f}%  
- **Most Common Error Type:** {common_error}  
""")

        st.divider()

        # MODEL GRAPH
        st.subheader("Model Performance Comparison")

        model_accuracy = scored.groupby("model")["correct_prediction"].mean()

        fig, ax = plt.subplots()
        model_accuracy.plot(kind="bar", ax=ax)

        ax.set_title("Accuracy by Model")
        ax.set_ylim(0, 1)

        for i, v in enumerate(model_accuracy):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

        st.pyplot(fig)

        # CATEGORY GRAPH
        st.subheader("Performance by Symptom Category")

        category_accuracy = scored.groupby("category")["correct_prediction"].mean()

        fig, ax = plt.subplots(figsize=(10,5))
        category_accuracy.plot(kind="bar", ax=ax)

        plt.xticks(rotation=45)
        ax.set_ylim(0, 1)

        st.pyplot(fig)

        # ERROR PIE
        st.subheader("Error Distribution")

        fig, ax = plt.subplots()
        ax.pie(scored["error_type"].value_counts(), autopct="%1.1f%%")
        st.pyplot(fig)

        # -----------------------------
        # CONFUSION MATRIX (IMPROVED)
        # -----------------------------
        st.subheader("Confusion Matrix Analysis")

        cm = confusion_matrix(
            scored["hf_relevance"],
            scored["predicted_label"],
            labels=labels
        )

        fig, ax = plt.subplots(figsize=(7,6))

        im = ax.imshow(cm)

        cbar = plt.colorbar(im)
        cbar.ax.set_ylabel("Number of Predictions", rotation=270, labelpad=15)

        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))

        ax.set_xticklabels([label_map[l] for l in labels], rotation=45, ha="right")
        ax.set_yticklabels([label_map[l] for l in labels])

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix for Heart Failure Classification")

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        plt.tight_layout()

        st.pyplot(fig)
        


# =============================
# PAGE 3 - DATASET
# =============================
elif page == "Dataset Explorer":

    st.title("📂 Dataset Explorer")

    dataset = load_file(os.path.join(BASE_DIR, "dataset", "hf_symptom_dataset.csv"))

    if dataset is None:
        st.error("Dataset not found.")
        st.stop()

    search = st.text_input("Search")

    if search:
        dataset = dataset[
            dataset["symptom_description"].str.contains(search, case=False)
        ]

    category = st.selectbox("Filter", ["All"] + sorted(dataset["category"].unique()))

    if category != "All":
        dataset = dataset[dataset["category"] == category]

    st.dataframe(dataset, use_container_width=True)


# =============================
# PAGE 4 - SYSTEM OVERVIEW
# =============================
elif page == "System Overview":

    st.title("⚙ System Architecture")

    st.markdown("""
### Evaluation Pipeline

1 Dataset Construction  
2 Prompt Generation  
3 Model Inference  
4 Response Storage  
5 Automated Scoring  
6 Metrics Calculation  
7 Visualisation Dashboard  
""")

    st.info("Research system only — not for medical use.")