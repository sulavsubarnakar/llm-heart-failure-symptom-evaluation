import pandas as pd
import json
from datetime import datetime
from prompt_generator import load_prompt_template, generate_prompt
from llm_clients import run_model

# -----------------------------
# CONFIGURATION
# -----------------------------
DATASET_PATH = "dataset/hf_symptom_dataset.csv"
PROMPT_PATH = "prompts/prompt_template.txt"
OUTPUT_PATH = "responses/model_responses.csv"

# Models to evaluate
MODELS = ["Model_A", "Model_B", "Model_C", "Real_LLM"]

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv(DATASET_PATH)

print("Dataset loaded:", len(df), "entries")

# OPTIONAL: limit for testing
# df = df.head(50)

# -----------------------------
# LOAD PROMPT TEMPLATE
# -----------------------------
prompt_template = load_prompt_template(PROMPT_PATH)

responses = []

# -----------------------------
# MAIN EVALUATION LOOP
# -----------------------------
for index, row in df.iterrows():

    symptom = row["symptom_description"]
    true_label = row["hf_relevance"]

    # Generate prompt FIRST
    prompt = generate_prompt(prompt_template, symptom)

    for model in MODELS:

        # Handle real LLM differently
        if model == "Real_LLM":
            result = run_model(model, prompt, None)
        else:
            result = run_model(model, prompt, true_label)

        responses.append({
            "id": row["id"],
            "model": model,
            "symptom_description": symptom,
            "true_label": true_label,
            "model_response": json.dumps(result),
            "timestamp": datetime.now().isoformat()
        })

    if index % 50 == 0:
        print(f"Processed {index} entries...")
        
# -----------------------------
# SAVE RESULTS
# -----------------------------
responses_df = pd.DataFrame(responses)

responses_df.to_csv(OUTPUT_PATH, index=False)

print("\nEvaluation complete.")
print("Total responses:", len(responses_df))
print("Saved to:", OUTPUT_PATH)