import streamlit as st
from resume_semantic_scoring_engine import load_resume, generate_tailored_resume
from jinja2 import Environment, FileSystemLoader
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()
if os.getenv("OPENAI_API_KEY"):
    openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Tailored Summary Generator ---
def generate_tailored_summary(resume, job_description, use_gpt=False):
    """
    Generate a tailored summary for the resume.
    If use_gpt=True, uses OpenAI API (costs credits).
    Otherwise, uses offline keyword matching.
    """
    if use_gpt:
        # Build combined context from resume tags
        all_tags = []
        for section in resume:
            if "tags" in section:
                all_tags.extend(section["tags"])
        tag_text = ", ".join(sorted(set(all_tags)))

        prompt = f"""
        Job Description:
        {job_description}

        Candidate Experience Tags:
        {tag_text}

        Write a concise 2–3 sentence professional summary highlighting why this candidate is a great match for the role.
        Avoid restating the entire resume — focus on relevancy.
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are an expert technical recruiter."},
                      {"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message["content"].strip()

    else:
        # Offline keyword-based version
        job_keywords = set(word.lower() for word in job_description.split() if len(word) > 2)
        matching_tags = set()
        for section in resume:
            if "tags" in section:
                matching_tags.update(tag.lower() for tag in section["tags"])
        overlap = job_keywords & matching_tags
        if overlap:
            relevant_skills = ", ".join(sorted(list(overlap))[:6])
            return f"Engineering leader with extensive experience in {relevant_skills}, well-suited to meet the needs of this role."
        else:
            return "Engineering leader with deep experience across cloud platforms, CI/CD, and cross-functional leadership."


# --- Streamlit Page Config ---
st.set_page_config(page_title="Tailored Resume Builder", layout="centered")
st.title("🎯 Tailored Resume Generator")
st.markdown(
    "Paste a job description below and click **Generate Resume** to get a customized resume based on your experience."
)

# Load modular resume
resume = load_resume("modular_resume_full.yaml")

# --- User Inputs ---
job_description = st.text_area("📝 Job Description", height=300)

summary_mode = st.radio(
    "Select Summary Mode",
    ("Offline (free)", "GPT-powered (API cost)"),
    index=0
)

ordering_mode = st.radio(
    "Experience Ordering",
    ("Relevancy First", "Chronological", "Hybrid"),
    index=0
)

ordering_map = {
    "Relevancy First": "relevancy",
    "Chronological": "chronological",
    "Hybrid": "hybrid"
}

use_embeddings = st.checkbox("Use semantic matching (OpenAI API required)", value=False)

# --- Generate Resume ---
if st.button("Generate Tailored Resume"):
    if not job_description.strip():
        st.warning("Please enter a job description first.")
    else:
        tailored = generate_tailored_resume(
            resume,
            job_description,
            use_embeddings=use_embeddings,
            ordering=ordering_map[ordering_mode]
        )
        tailored_summary = generate_tailored_summary(
            resume,
            job_description,
            use_gpt=(summary_mode == "GPT-powered (API cost)")
        )

        # Load HTML template
        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("tailored_resume_template.html")

        html = template.render(
            name="Travis Karr",
            contact="Axton, VA | (214) 207-7182 | travisekarr@gmail.com",
            tailored_summary=tailored_summary,
            resume=tailored
        )

        st.success("✅ Resume generated!")
        st.download_button(
            label="📥 Download Resume as HTML",
            data=html,
            file_name="tailored_resume.html",
            mime="text/html"
        )
