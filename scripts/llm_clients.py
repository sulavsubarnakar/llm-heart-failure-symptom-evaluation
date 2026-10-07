import random
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# -----------------------------
# OPENAI CLIENT SETUP
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# GLOBAL SETTINGS
# -----------------------------
LABELS = ["HF_related", "possible_HF", "uncertain", "not_HF"]

# Set seed for reproducibility
random.seed(42)


# -----------------------------
# MODEL A (HIGH PERFORMANCE)
# -----------------------------
def simulate_model_A(true_label):
    if random.random() < 0.90:
        return true_label
    return introduce_structured_error(true_label)


# -----------------------------
# MODEL B (MEDIUM PERFORMANCE)
# -----------------------------
def simulate_model_B(true_label):
    if random.random() < 0.75:
        return true_label
    return introduce_structured_error(true_label)


# -----------------------------
# MODEL C (CONSERVATIVE MODEL)
# -----------------------------
def simulate_model_C(true_label):
    if random.random() < 0.60:
        return true_label

    if random.random() < 0.50:
        return "uncertain"

    return introduce_structured_error(true_label)


# -----------------------------
# STRUCTURED ERROR FUNCTION
# -----------------------------
def introduce_structured_error(true_label):

    if true_label == "HF_related":
        return random.choice(["possible_HF", "uncertain"])

    elif true_label == "possible_HF":
        return random.choice(["HF_related", "uncertain"])

    elif true_label == "not_HF":
        return random.choice(["uncertain", "possible_HF"])

    else:
        return random.choice(LABELS)


# -----------------------------
# REAL LLM FUNCTION
# -----------------------------
def run_real_llm(prompt):

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # cheap + good
            messages=[
                {"role": "system", "content": "You are a clinical assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        text = response.choices[0].message.content

        return parse_llm_output(text)

    except Exception as e:
        return {
            "identified_symptom": "error",
            "hf_relevance": "uncertain",
            "explanation": str(e)
        }


# -----------------------------
# PARSE LLM OUTPUT
# -----------------------------
def parse_llm_output(text):

    text_lower = text.lower()

    if "hf_related" in text_lower:
        label = "HF_related"
    elif "possible_hf" in text_lower:
        label = "possible_HF"
    elif "uncertain" in text_lower:
        label = "uncertain"
    else:
        label = "not_HF"

    return {
        "identified_symptom": "real_llm",
        "hf_relevance": label,
        "explanation": text[:200]
    }


# -----------------------------
# MAIN MODEL INTERFACE
# -----------------------------
def run_model(model_name, prompt, true_label):

    if model_name == "Model_A":
        prediction = simulate_model_A(true_label)

    elif model_name == "Model_B":
        prediction = simulate_model_B(true_label)

    elif model_name == "Model_C":
        prediction = simulate_model_C(true_label)

    elif model_name == "Real_LLM":
        return run_real_llm(prompt)

    else:
        prediction = "uncertain"

    return {
        "identified_symptom": "simulated_symptom",
        "hf_relevance": prediction,
        "explanation": f"{model_name} prediction based on simulated reasoning"
    }