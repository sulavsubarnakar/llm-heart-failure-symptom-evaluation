# Evaluating LLMs in Heart Failure Symptom Recognition

A Python pipeline and Streamlit dashboard that tests how well AI models classify
plain-language symptom descriptions as related to heart failure or not.

Final-year project, BSc (Hons) Computer Science, University of East London (2026).

> Research project only. Not medical advice. Do not use it to diagnose anyone.

## What it does
1. Takes 500 plain-language symptom descriptions (`dataset/hf_symptom_dataset.csv`).
2. Sends each one to four models and saves their answers (`responses/`).
3. Scores the answers against the correct labels (`evaluation/`, `results/`).
4. Shows the results in a Streamlit dashboard.

Each description has one of four labels: `HF_related`, `possible_HF`, `uncertain`, `not_HF`.

## Important limits (please read)
- **Three of the four models are simulated.** Model_A, Model_B and Model_C are not
  AI models. They are small functions that receive the correct label and return it
  with a fixed chance (about 90%, 75% and 60%), otherwise a planned wrong label.
  They exist to test the scoring pipeline and the dashboard. **Their accuracy is
  set by the code, so it is not a finding.**
- **Only one model is a real LLM:** Real_LLM uses OpenAI `gpt-4o-mini`
  (temperature 0.2).
- The dataset is synthetic: 500 descriptions written from clinical sources and
  reworded to sound like patients. It is not real patient data.
- The code reads the model's answer by searching for label words in the text. This
  is a weak method. Unreadable answers become `not_HF`, and failed API calls are
  saved as `uncertain`, which may affect the score.
- One real model on 500 examples cannot show how LLMs in general perform.

## Results
Real model result (OpenAI gpt-4o-mini): **68.6% accuracy** on 500 descriptions.

| Model | Type | Accuracy | Meaning |
|---|---|---|---|
| Real_LLM | OpenAI gpt-4o-mini | 68.6% | Measured result |
| Model_A | Simulated | 91.2% | Set by design (about 90%) |
| Model_B | Simulated | 76.8% | Set by design (about 75%) |
| Model_C | Simulated | 66.2% | Set by design (about 60%) |

- The dashboard's overall accuracy (about 75.7%) is an average over all four models
  (three simulated, one real). It is not the real model's score. The real model's
  score is 68.6%.
- The real model answered "uncertain" in only 4.0% of cases.
- Accuracy is higher for clear symptoms such as breathlessness and lower for
  vague or mixed descriptions.

## Project structure
- `app.py`: Streamlit dashboard
- `scripts/run_llm_evaluation.py`: runs all four models on the dataset
- `scripts/scoring_engine.py`: scores the answers
- `scripts/visualize_results.py`: makes the charts
- `scripts/llm_clients.py`: simulated models and the OpenAI call
- `scripts/prompt_generator.py`: builds the prompt for each description
- `dataset/`, `prompts/`, `responses/`, `evaluation/`, `results/`

## How to run
1. Install Python 3.8 or higher.
2. Install the packages: `pip install -r requirements.txt`
3. Create a file called `.env` in the main folder and add your own key:
   `OPENAI_API_KEY=your_key_here`
   (never share or upload this file).
4. Run these from the main folder, in order:

       python scripts/run_llm_evaluation.py
       python scripts/scoring_engine.py
       python scripts/visualize_results.py
       streamlit run app.py

   Note: the first step calls the OpenAI API 500 times and overwrites
   `responses/model_responses.csv`. The saved results from the original run are
   already included, so you can skip to `streamlit run app.py` to view them.

## Possible improvements
- Read the `Label:` line of the model's answer directly instead of searching the text.
- Count failed API calls separately from "uncertain" answers.
- Test more than one real model.

## Author
Sulav Subarnakar | [LinkedIn](https://www.linkedin.com/in/sulav-subarnakar-8b81a1259/)