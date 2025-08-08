
import os
from openai import OpenAI
import streamlit as st
from resume_semantic_scoring_engine import load_resume, generate_tailored_resume
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
import weasyprint
import re
import base64
from io import BytesIO

# Load environment variables
load_dotenv()
client = OpenAI()

# Supported GPT models and pricing
GPT_MODELS = {
    "gpt-3.5-turbo": "$0.0015 / 1K tokens",
    "gpt-4": "$0.03 / 1K tokens (input), $0.06 / 1K (output)",
    "gpt-4o": "$0.005 / 1K tokens"
}

def generate_tailored_summary(resume, job_description, use_gpt=False, model="gpt-3.5-turbo"):
    header = next((sec for sec in resume if sec["type"] == "header"), {})
    title = header.get("title", "Experienced Professional")
    years_exp = header.get("years_experience", None)

    if years_exp:
        opening_line = f"{title} with {years_exp}+ years of proven expertise"
    else:
        opening_line = f"{title} with proven expertise"

    job_keywords = set(word.lower() for word in job_description.split() if len(word) > 2)
    all_resume_tags = set()
    for section in resume:
        if "tags" in section:
            all_resume_tags.update(tag.lower() for tag in section["tags"])

    matched_skills = sorted(list(job_keywords & all_resume_tags))
    top_skills_str = ", ".join(matched_skills[:4]) if matched_skills else None

    achievement = None
    for section in resume:
        if section.get("type") == "experience":
            for contrib in section.get("contributions", []):
                desc = contrib.get("description", "")
                if any(char.isdigit() for char in desc):
                    achievement = desc
                    break
            if achievement:
                break

    if not achievement:
        summary_section = next((sec for sec in resume if sec["type"] == "summary"), {})
        if "achievements" in summary_section and summary_section["achievements"]:
            achievement = summary_section["achievements"][0]

    if use_gpt:
        prompt = f'''
        Job Description:
        {job_description}

        Candidate Info:
        Title: {title}
        Years Experience: {years_exp or "Not specified"}
        Top Skills: {top_skills_str or "None matched"}
        Achievement: {achievement or "None found"}

        Write a 2-4 sentence professional summary (max 300 characters) starting with the title and years of experience.
        Mention the top skills naturally. Include the achievement at the end.
        Keep it concise, professional, and tailored to the job description.
        '''

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert technical recruiter skilled in writing concise, tailored professional summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()

    summary_parts = [opening_line]
    if top_skills_str:
        summary_parts.append(f"specializing in {top_skills_str}")
    if achievement:
        summary_parts.append(f"Notable achievement: {achievement}")
    return ". ".join(summary_parts) + "."

st.set_page_config(page_title="Tailored Resume Builder", layout="centered")
st.title("🎯 Tailored Resume Generator")
st.markdown(
    "Paste a job description below and click **Generate Resume** to get a customized resume based on your experience."
)

resume = load_resume("modular_resume_full.yaml")
header = next((sec for sec in resume if sec["type"] == "header"), None)

job_description = st.text_area("📝 Job Description", height=300)

summary_mode = st.radio(
    "Select Summary Mode",
    ("Offline (free)", "GPT-powered (API cost)"),
    index=0
)

selected_model = st.selectbox(
    "Choose GPT Model for Tailored Summary (API cost per 1K tokens):",
    options=list(GPT_MODELS.keys()),
    format_func=lambda model: f"{model} — {GPT_MODELS[model]}",
    index=0
)

ordering_mode = st.radio(
    "Experience Ordering",
    ("Relevancy First", "Chronological", "Hybrid"),
    index=0
)

top_n_hybrid = 3
if ordering_mode == "Hybrid":
    top_n_hybrid = st.slider(
        "Number of most relevant experiences to show before chronological ordering:",
        min_value=1,
        max_value=10,
        value=3
    )

use_embeddings = st.checkbox(
    "Use semantic matching (OpenAI API required)",
    value=False
)

ordering_map = {
    "Relevancy First": "relevancy",
    "Chronological": "chronological",
    "Hybrid": "hybrid"
}

if st.button("Generate Tailored Resume"):
    if not job_description.strip():
        st.warning("Please enter a job description first.")
    else:
        tailored = generate_tailored_resume(
            resume,
            job_description,
            top_n=top_n_hybrid,
            use_embeddings=use_embeddings,
            ordering=ordering_map[ordering_mode]
        )
        tailored_summary = generate_tailored_summary(
            resume,
            job_description,
            use_gpt=(summary_mode == "GPT-powered (API cost)"),
            model=selected_model
        )

        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("tailored_resume_template.html")
        html = template.render(
            header=header,
            tailored_summary=tailored_summary,
            resume=tailored
        )

        st.session_state.generated_html = html

if "generated_html" in st.session_state:
    html = st.session_state.generated_html

    st.success("✅ Resume generated!")

    st.download_button(
        label="📥 Download Resume as HTML",
        data=html,
        file_name="tailored_resume.html",
        mime="text/html"
    )

    preview_mode = st.radio(
        "Preview Mode",
        ("Formatted (HTML)", "Plain Text", "PDF Preview"),
        index=0
    )

    st.markdown("### 📄 Resume Preview")

    if preview_mode == "Formatted (HTML)":
        st.components.v1.html(html, height=800, scrolling=True)

    elif preview_mode == "Plain Text":
        plain_text = re.sub(r"<[^>]+>", "", html)
        st.text_area("Plain Text Resume", plain_text, height=800)

    elif preview_mode == "PDF Preview":
        try:
            from weasyprint import HTML as WPHTML
            pdf_buffer = BytesIO()
            WPHTML(string=html).write_pdf(pdf_buffer)
            pdf_data = pdf_buffer.getvalue()
            b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.download_button(
                label="📥 Download Resume as PDF",
                data=pdf_data,
                file_name="tailored_resume.pdf",
                mime="application/pdf"
            )
        except ImportError:
            st.error("WeasyPrint is not installed. Run `pip install weasyprint` to enable PDF preview.")
