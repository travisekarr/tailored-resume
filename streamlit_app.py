import os

# Suppress harmless GLib/GTK warnings from WeasyPrint on Windows
# These appear when GTK enumerates UWP apps with incomplete file associations
os.environ["G_MESSAGES_DEBUG"] = "none"

import streamlit as st
from resume_semantic_scoring_engine import load_resume, generate_tailored_resume
from jinja2 import Environment, FileSystemLoader
from dotenv import load_dotenv
import openai
import weasyprint

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
header = next((sec for sec in resume if sec["type"] == "header"), None)

# --- User Inputs ---
job_description = st.text_area("📝 Job Description", height=300)

summary_mode = st.radio(
    "Select Summary Mode",
    ("Offline (free)", "GPT-powered (API cost)"),
    index=0,
    help=(
        "Offline (free): Generates a concise summary using keyword matches from the job description and your resume. "
        "No API usage or cost.\n\n"
        "GPT-powered: Uses OpenAI GPT to generate a professional, recruiter-style summary tailored to the job description. "
        "Requires API key and small per-use cost."
    )
)

ordering_mode = st.radio(
    "Experience Ordering",
    ("Relevancy First", "Chronological", "Hybrid"),
    index=0,
    help=(
        "Relevancy First: Orders experiences by match strength to the job description, regardless of date. Best for ATS optimization.\n\n"
        "Chronological: Orders experiences in reverse chronological order (most recent first). Best for human readability.\n\n"
        "Hybrid: Shows the top N most relevant experiences first, then the rest in reverse chronological order."
    )
)

top_n_hybrid = 3
if ordering_mode == "Hybrid":
    top_n_hybrid = st.slider(
        "Number of most relevant experiences to show before chronological ordering:",
        min_value=1,
        max_value=10,
        value=3,
        help="Determines how many of your top matching experiences will be shown first before chronological ordering."
    )

ordering_map = {
    "Relevancy First": "relevancy",
    "Chronological": "chronological",
    "Hybrid": "hybrid"
}

use_embeddings = st.checkbox(
    "Use semantic matching (OpenAI API required)",
    value=False,
    help=(
        "When enabled, uses OpenAI embeddings to semantically match your experience to the job description. "
        "Produces more intelligent matches but will use API credits."
    )
)

# --- Generate Resume ---
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
            use_gpt=(summary_mode == "GPT-powered (API cost)")
        )

        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("tailored_resume_template.html")

        html = template.render(
            header=header,
            tailored_summary=tailored_summary,
            resume=tailored
        )

        # Store HTML so preview mode changes don't reset
        st.session_state.generated_html = html

# --- Preview & Download ---
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
        index=0,
        help="Choose between a formatted HTML preview, a plain text ATS-friendly view, or a PDF preview."
    )

    st.markdown("### 📄 Resume Preview")

    if preview_mode == "Formatted (HTML)":
        st.components.v1.html(
            html,
            height=800,
            scrolling=True
        )

    elif preview_mode == "Plain Text":
        import re
        plain_text = re.sub(r"<[^>]+>", "", html)
        st.text_area(
            "Plain Text Resume",
            plain_text,
            height=800
        )

    elif preview_mode == "PDF Preview":
        try:
            from weasyprint import HTML
            import base64
            from io import BytesIO

            pdf_buffer = BytesIO()
            HTML(string=html).write_pdf(pdf_buffer)
            pdf_data = pdf_buffer.getvalue()

            b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
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
