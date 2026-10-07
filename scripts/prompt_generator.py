def load_prompt_template(path):
    with open(path, "r") as f:
        return f.read()


def generate_prompt(template, symptom):
    return template.replace("{symptom}", symptom)