from resume_semantic_scoring_engine import load_resume, generate_tailored_resume
from jinja2 import Environment, FileSystemLoader
import os
import openai
from dotenv import load_dotenv
from resume_template_config import load_resume_templates, get_template_path_by_id, get_default_template_id

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load resume data
resume = load_resume("modular_resume_full.yaml")

# Example job description
job_description = """
Looking for an Engineering Manager with strong CI/CD, AWS, and developer productivity experience.
Should have a background in SaaS platform scaling, observability, and cross-functional team leadership.
"""

# Tailor resume
tailored = generate_tailored_resume(resume, job_description, top_n=3, use_embeddings=False)

# Load available templates
templates = load_resume_templates()
current_template_id = get_default_template_id(templates)

# Function to switch template dynamically
def switch_template(new_template_id):
    global current_template_id
    current_template_id = new_template_id
    template_path = get_template_path_by_id(templates, current_template_id)
    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template(template_path)
    html_out = template.render(
        name="Travis Karr",
        contact="Axton, VA | (214) 207-7182 | travisekarr@gmail.com",
        resume=tailored
    )
    with open("tailored_resume_output.html", "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"✅ Template switched to {current_template_id}: tailored_resume_output.html")

# Render to HTML using the default template
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template("tailored_resume_template.html")
html_out = template.render(
    name="Travis Karr",
    contact="Axton, VA | (214) 207-7182 | travisekarr@gmail.com",
    resume=tailored
)

with open("tailored_resume_output.html", "w", encoding="utf-8") as f:
    f.write(html_out)

print("✅ Tailored resume generated: tailored_resume_output.html")

# Example usage
switch_template("modern_template")
