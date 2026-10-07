from llm_clients import run_model

print(run_model("Model_A", "test prompt", "HF_related"))
print(run_model("Model_B", "test prompt", "not_HF"))
print(run_model("Model_C", "test prompt", "possible_HF"))