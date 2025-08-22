import os
import warnings
# Suppress harmless GLib/GTK warnings from WeasyPrint on Windows
os.environ["G_MESSAGES_DEBUG"] = "none"
warnings.filterwarnings("ignore")

import re
import math
import base64
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
from resume_template_config import load_resume_templates, get_default_template_id, get_template_path_by_id
# Model config helpers
from models_config import load_models_cfg, ui_default
_model_cfg = load_models_cfg()
from jinja2 import Environment, FileSystemLoader

from resume_semantic_scoring_engine import (
    load_resume,
    generate_tailored_resume,
    enhance_experience_with_impact,
    clear_embeddings_cache,
    section_text,   # used for keyword chip matching
    generate_tailored_summary,  # <-- import shared summary
)
from resume_utils import base_resume_name_from_jd as _ru_base_resume_name
from resume_utils import build_keywords as _ru_build_keywords
from resume_utils import pick_best_achievement_overlap as _ru_pick_best_achievement_overlap
from resume_utils import extract_company_name as _ru_extract_company
from resume_utils import extract_role_title as _ru_extract_role
from resume_utils import highlight_html as _ru_highlight_html
from resume_utils import clean_for_filename as _ru_clean_for_filename
from resume_ui_controls import render_resume_sidebar_controls, select_preview_mode, show_preview_and_download

# ==============================
# CONFIG
# ==============================
RESUME_PATH = "modular_resume_full.yaml"
# Load template config
_resume_templates = load_resume_templates()
_default_template_id = get_default_template_id(_resume_templates)

# Feature flags
# Temporarily disable Cover Letter UI until refactor/fix is complete
ENABLE_COVER_LETTER = False

# --- Setup ---
load_dotenv()
#print("OPENAI_API_KEY:", os.environ.get("OPENAI_API_KEY"))
client = OpenAI()

# Configure Streamlit early (must be before any other st.* calls)
st.set_page_config(page_title="Tailored Resume Builder", layout="wide")

# ----- Model/pricing directories -----
GPT_MODELS = {
    "gpt-3.5-turbo": "$0.0015 / 1K tokens",
    "gpt-4": "$0.03 / 1K (in), $0.06 / 1K (out)",
    "gpt-4o": "$0.005 / 1K tokens",
}

EMBED_MODELS = {
    "text-embedding-3-small": "$0.02 / 1M tokens",
    "text-embedding-3-large": "$0.13 / 1M tokens",
}

# ----- Sidebar: Usage & Models -----
with st.sidebar:
	# Replace inline controls with shared controls; preserve the Tools section below
	_ctrl = render_resume_sidebar_controls(key_prefix="app")
	summary_mode = _ctrl["summary_mode"]
	strict_mode = _ctrl["strict_mode"]
	selected_model = _ctrl["selected_model"]
	add_impact = _ctrl["add_impact"]
	bullets_per_role = _ctrl["bullets_per_role"]
	show_generated = _ctrl["show_generated"]
	use_embeddings = _ctrl["use_embeddings"]
	embedding_model = _ctrl["embedding_model"]

	st.markdown("---")
	st.markdown("### Tools")
	if st.button("Clear embeddings cache", key="btn_clear_cache"):
		try:
			clear_embeddings_cache()
			st.success("Embeddings cache cleared.")
		except Exception as e:
			print(f"[streamlit_app] Error clearing embeddings cache: {e}")
			st.info("No cache file found or unable to clear.")

# Add some common noise/HTML/tag words you never want highlighted
STOPWORDS = {
    "and","the","with","for","your","you","our","their","this","that",
    "skills","experience","years","team","work","ability","in","to","of",
    # HTML/tag-ish words to avoid breaking markup:
    "strong","em","span","div","class","style","script","http","https","href","mark"
}

# ----- Achievement selection helpers -----

def _collect_candidate_achievements(resume):
    candidates = []
    for sec in resume:
        if sec.get("type") == "experience":
            for c in (sec.get("contributions") or []):
                desc = (c or {}).get("description")
                if isinstance(desc, str) and desc.strip():
                    candidates.append(desc.strip())
    summary_sec = next((sec for sec in resume if sec.get("type") == "summary"), {})
    for a in summary_sec.get("achievements") or []:
        if isinstance(a, str) and a.strip():
            candidates.append(a.strip())
    seen = set()
    uniq = []
    for c in candidates:
        key = c.lower()
        if key not in seen:
            uniq.append(c)
            seen.add(key)
    return uniq

def _cosine(a, b):
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    return (num / (da * db)) if da and db else 0.0

def _pick_best_achievement_embeddings(resume, job_description: str, embedding_model: str = None) -> str | None:
    candidates = _collect_candidate_achievements(resume)
    if not candidates:
        return None
    if embedding_model is None:
        embedding_model = st.session_state.get("app_embed_model") or ui_default(_model_cfg, "embeddings") or "text-embedding-3-small"
    resp = client.embeddings.create(model=embedding_model, input=[job_description] + candidates)
    jd_vec = resp.data[0].embedding
    best_idx, best_score = -1, -1.0
    for i, d in enumerate(resp.data[1:], start=0):
        score = _cosine(jd_vec, d.embedding)
        if score > best_score:
            best_idx, best_score = i, score
    return candidates[best_idx] if best_idx >= 0 else None

# ----- Tailored summary (uses embeddings if enabled) -----
# def generate_tailored_summary(
#     resume,
#     job_description,
#     use_gpt=False,
#     model="gpt-3.5-turbo",
#     use_embeddings=False,
#     embedding_model="text-embedding-3-small",
# ):
#     header = next((sec for sec in resume if sec.get("type") == "header"), {})
#     title = header.get("title", "Experienced Professional")
#     years_exp = header.get("years_experience")
#     opening_line = f"{title} with {years_exp}+ years of proven expertise" if years_exp else f"{title} with proven expertise"

#     job_keywords = set(word.lower() for word in job_description.split() if len(word) > 2)

#     # Flatten tags across sections, supporting {hard,soft} dicts and
#     # creating simple separator variants for robust matching ("ci_cd" <-> "ci/cd").
#     def _flatten_tags(tags):
#         if isinstance(tags, dict):
#             vals = []
#             vals.extend(tags.get("hard", []) or [])
#             vals.extend(tags.get("soft", []) or [])
#             return [str(t).strip().lower() for t in vals if t]
#         elif isinstance(tags, list):
#             return [str(t).strip().lower() for t in tags if t]
#         return []

#     def _tag_variants(t: str) -> set[str]:
#         v = {t}
#         # generate common separator/delimiter variants
#         v.add(t.replace("_", "/"))
#         v.add(t.replace("/", "_"))
#         v.add(t.replace("-", "_"))
#         v.add(t.replace("_", "-"))
#         v.add(t.replace("_", " "))  # space variant (e.g., sql_server -> sql server)
#         # dot variants (e.g., node.js -> nodejs, node js)
#         v.add(t.replace(".", ""))
#         v.add(t.replace(".", " "))
#         return {x for x in v if x}

#     all_resume_tags = set()
#     for section in resume:
#         tags = section.get("tags")
#         for t in _flatten_tags(tags):
#             all_resume_tags.update(_tag_variants(t))

#     matched_skills = sorted(list(job_keywords & all_resume_tags))
#     top_skills_str = ", ".join(matched_skills[:4]) if matched_skills else None

#     # Pick the most relevant achievement (embeddings when enabled)
#     if use_embeddings:
#         achievement = _pick_best_achievement_embeddings(resume, job_description, embedding_model)
#     else:
#         # was: achievement = _pick_best_achievement_overlap(resume, job_description)
#         achievement = _ru_pick_best_achievement_overlap(resume, job_description)

#     # Always-safe offline summary
#     def _offline():
#         parts = [opening_line]
#         if top_skills_str:
#             parts.append(f"specializing in {top_skills_str}")
#         if achievement:
#             parts.append(f"Notable achievement: {achievement}")
#         return ". ".join(parts) + "."

#     if not use_gpt:
#         return _offline()

#     # STRICT: GPT can only rephrase placeholders; we substitute facts after
#     outline = _build_summary_outline(title, years_exp, top_skills_str, achievement)
#     try:
#         templated = _safe_paraphrase_with_placeholders(outline, model)
#         final = _substitute_placeholders(templated, title, years_exp, top_skills_str, achievement)
#         return final
#     except Exception:
#         return _offline()

# ----- Cover Letter generation -----
def _collect_top_relevant_bullets(resume, scores_map, max_bullets=3):
    # Grab the highest-scoring experience bullets (if score map exists), else take first few bullets
    bullets = []
    # Try scored experiences first
    exp_sections = [s for s in resume if s.get("type") == "experience"]
    # Attach scores if we have them
    scored = []
    for s in exp_sections:
        sc = scores_map.get(id(s), s.get("_score", 0))
        scored.append((sc or 0, s))
    scored.sort(reverse=True, key=lambda x: x[0])
    for _, sec in scored:
        for c in sec.get("contributions", []):
            d = c.get("description")
            if isinstance(d, str) and d.strip():
                bullets.append(d.strip())
            if len(bullets) >= max_bullets:
                return bullets
    # Fallback: first available bullets
    if not bullets:
        for sec in exp_sections:
            for c in sec.get("contributions", []):
                d = c.get("description")
                if isinstance(d, str) and d.strip():
                    bullets.append(d.strip())
                if len(bullets) >= max_bullets:
                    return bullets
    return bullets[:max_bullets]

def render_resume(preview_mode, selected_template_id):
    """
    Render the resume based on the selected preview mode and template.
    """
    if "tailored" not in st.session_state:
        st.warning("No resume available. Please generate a resume first.")
        return

    try:
        template_path = get_template_path_by_id(_resume_templates, selected_template_id)
        env = Environment(loader=FileSystemLoader("templates"))
        template = env.get_template(os.path.basename(template_path))
        html = template.render(
            header=st.session_state.get("header", {}),
            tailored_summary=st.session_state.get("tailored_summary", ""),
            resume=st.session_state.get("tailored", []),
        )
        st.session_state.generated_html = html

        if preview_mode == "Formatted (HTML)":
            st.components.v1.html(html, height=800, scrolling=True)
        elif preview_mode == "Plain Text":
            plain_text = re.sub(r"<[^>]+>", "", html)
            st.text_area("Plain Text Resume", plain_text, height=800, key="ta_plain_text")
        elif preview_mode == "PDF Preview":
            try:
                import pdfkit
                import base64
                from io import BytesIO

                # Generate PDF using pdfkit
                pdf_data = pdfkit.from_string(html, False)

                # Encode PDF for inline display
                b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="800" type="application/pdf"></iframe>',
                    unsafe_allow_html=True
                )
            except ImportError as e:
                print(f"[streamlit_app] PDF preview error (pdfkit missing): {e}")
                st.error("pdfkit is not installed. Run `pip install pdfkit` to enable PDF preview.")
    except Exception as e:
        print(f"[streamlit_app] Error in render_resume: {e}")
        st.error(f"Error rendering resume template: {e}")

def _offline_cover_letter(header, company, role, tailored_summary, bullets):
    name = header.get("name", "")
    loc = header.get("location", "")
    email = header.get("email", "")
    phone = header.get("phone", "")

    greeting = f"Dear Hiring Manager{f' at {company}' if company else ''},"
    p1 = f"I’m excited to apply for the {role or 'open'} position{f' at {company}' if company else ''}. {tailored_summary}"
    if bullets:
        p2 = "Here are a few highlights that align with your needs:\n" + "\n".join(f"- {b}" for b in bullets)
    else:
        p2 = "I believe my experience closely aligns with your requirements and would welcome the opportunity to discuss how I can contribute."

    p3 = "Thank you for your time and consideration. I would welcome the opportunity to discuss how my background can help your team deliver results."

    footer = f"\nSincerely,\n{name}\n{loc}\n{email} | {phone}".strip()
    return "\n\n".join([greeting, p1, p2, p3, footer]).strip()

def _cover_letter_prompt(header, company, role, tailored_summary, bullets, job_description):
    name = header.get("name", "")
    contact = f"{header.get('email','')} | {header.get('phone','')}"
    facts_block = f"""
Candidate:
- Name: {name}
- Contact: {contact}
- Tailored Summary: {tailored_summary}
- Top Highlights:
{chr(10).join(f'- {b}' for b in bullets) if bullets else '- (none)'}
"""
    prompt = f"""
You write concise, professional cover letters. Use only the facts provided.
Do not invent employers, titles, or metrics. Keep to 180–250 words.

Company: {company or '(unspecified)'}
Role: {role or '(unspecified)'}

Job Description (for tone and alignment only):
{job_description}

Facts you may use verbatim:
{facts_block}

Write a 3-paragraph cover letter:
1) Short intro referencing the role/company.
2) Middle paragraph aligning 2–3 specific highlights to likely needs.
3) Brief closing with a polite call to action.

Return plain text only. No salutations beyond the greeting/sign-off.
Sign off as: {name}
"""
    return prompt.strip()

def generate_cover_letter(header, resume, scores_map, job_description, tailored_summary, use_gpt=False, model="gpt-3.5-turbo"):
    company = _ru_extract_company(job_description)
    role = _ru_extract_role(job_description)
    bullets = _collect_top_relevant_bullets(resume, scores_map or {}, max_bullets=3)

    if not use_gpt:
        return _offline_cover_letter(header, company, role, tailored_summary, bullets), company, role

    prompt = _cover_letter_prompt(header, company, role, tailored_summary, bullets, job_description)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":"You create professional, concise cover letters using only provided facts."},
            {"role":"user","content": prompt},
        ],
        temperature=0.4,
        max_tokens=500,
    )
    text = resp.choices[0].message.content.strip()
    # Soft sanity trim
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, company, role

tailored_summary = ""  # Initialize tailored_summary to an empty string to avoid NameError.

# ==== UI ====
st.title("🎯 Tailored Resume Generator")


st.markdown("Paste a job description below and click **Generate Resume** to get a customized resume based on your experience.")

# Load YAML
resume = load_resume(RESUME_PATH)
header = next((sec for sec in resume if sec.get("type") == "header"), None)

# Inputs
job_description = st.text_area("📝 Job Description", height=240, key="ta_job_desc")

col1, col2 = st.columns(2)
with col1:
    # Resume template selector
    template_ids = [t["id"] for t in _resume_templates]
    template_labels = {t["id"]: t["display"] for t in _resume_templates}
    def _fmt_template(x): return template_labels.get(x, x)
    selected_template_id = st.selectbox(
        "Resume template",
        template_ids,
        index=template_ids.index(_default_template_id) if _default_template_id in template_ids else 0,
        key="app_template_id",
        format_func=_fmt_template,
    )
    ordering_mode = st.radio("Experience Ordering", ("Relevancy First", "Chronological", "Hybrid"), index=0, key="rad_ordering")
    top_n_hybrid = 3
    if ordering_mode == "Hybrid":
        top_n_hybrid = st.slider("Top relevant items before chronological:", 1, 10, 3, key="sld_top_n_hybrid")
    highlight = st.checkbox("Highlight matches in preview", value=True, key="chk_highlight")

with col2:
    # Options moved to sidebar to match Generate From Jobs
    st.empty()

# Generate
if st.button("Generate Tailored Resume", use_container_width=True, key="btn_generate_resume"):

    if not job_description.strip():
        st.warning("Please enter a job description first.")
    else:
        try:
            tailored, scores_map = generate_tailored_resume(
                resume,
                job_description,
                top_n=top_n_hybrid,
                use_embeddings=use_embeddings,
                ordering=ordering_mode.lower(),   # 'relevancy'/'chronological'/'hybrid'
                embedding_model=embedding_model if use_embeddings else "text-embedding-3-small",
            )
        except Exception as e:
            print(f"[streamlit_app] Error generating tailored resume: {e}")
            st.error(f"Error generating tailored resume: {e}")
            tailored, scores_map = [], {}

        # Add tailored impact bullets if requested (requires GPT mode)
        if add_impact and (summary_mode == "GPT-powered (API cost)"):
            try:
                tailored = enhance_experience_with_impact(
                    tailored,
                    job_description,
                    use_gpt=True,
                    model=selected_model,
                    mark_generated=True,
                    bullets_per_role=bullets_per_role,
                )
            except Exception as e:
                print(f"[streamlit_app] Error adding impact statements: {e}")
                st.error(f"Error adding impact statements: {e}")

        # Hide generated impact if the user unchecks it
        if not show_generated:
            for sec in tailored:
                if sec.get("type") == "experience":
                    sec["contributions"] = [
                        c for c in sec.get("contributions", [])
                        if not (c.get("impact") and c.get("source") == "generated")
                    ]

        try:
            tailored_summary = generate_tailored_summary(
                resume,
                job_description,
                use_gpt=(summary_mode == "GPT-powered (API cost)"),
                model=selected_model,
                use_embeddings=use_embeddings,
                embedding_model=(embedding_model if use_embeddings else "text-embedding-3-small"),
            )
        except Exception as e:
            print(f"[streamlit_app] Error generating tailored summary: {e}")
            st.error(f"Error generating tailored summary: {e}")
            tailored_summary = ""

        try:
            template_path = get_template_path_by_id(_resume_templates, selected_template_id)
            env = Environment(loader=FileSystemLoader("templates"))
            template = env.get_template(os.path.basename(template_path))
            html = template.render(header=header, tailored_summary=tailored_summary, resume=tailored)
            if not html.strip():
                st.warning("Resume preview is empty. Please check your template and resume data.")
        except Exception as e:
            print(f"[streamlit_app] Error rendering template: {e}")
            st.error(f"Error rendering resume template: {e}")
            html = ""

        # Highlighted preview
        try:
            highlighted_html = _ru_highlight_html(html, job_description) if highlight else html
        except Exception as e:
            print(f"[streamlit_app] Error in highlight_html: {e}")
            st.error(f"Error highlighting resume preview: {e}")
            highlighted_html = html

        # Keyword chips (JD ∩ resume)
        try:
            jd_words = set(_ru_build_keywords(job_description))
            resume_text = " ".join((section_text(s) or "").lower() for s in resume)
            matched = sorted([w for w in jd_words if w in resume_text], key=len, reverse=True)[:50]
        except Exception as e:
            print(f"[streamlit_app] Error building keyword chips: {e}")
            matched = []

        # Company slug derived from JD (for filenames)
        try:
            # was:
            # company_slug = extract_company_name(job_description)
            # base_name = f"{company_slug}_full_resume" if company_slug else "full_resume"
            base_name = _ru_base_resume_name(job_description, default="full_resume")
        except Exception as e:
            print(f"[streamlit_app] Company slug extraction failed: {e}")
            base_name = "full_resume"

        st.session_state.generated_html = html
        st.session_state.highlighted_html = highlighted_html
        st.session_state.tailored = tailored
        st.session_state.keywords = matched
        st.session_state.scores_map = scores_map
        st.session_state.base_name = base_name
        st.session_state.tailored_summary = tailored_summary
        st.session_state.header = header
        st.session_state.job_description = job_description

# Only show preview/download if resume has been generated
if "generated_html" in st.session_state:
    html = st.session_state.generated_html
    highlighted_html = st.session_state.get("highlighted_html", html)
    tailored = st.session_state.get("tailored", [])
    matched = st.session_state.get("keywords", [])
    scores_map = st.session_state.get("scores_map", {})
    base_name = st.session_state.get("base_name", "full_resume")

    st.success("✅ Resume generated!")

    # Preview mode (shared)
    preview_mode = select_preview_mode(key_prefix="app_prev", default="Formatted (HTML)")
    st.markdown("### 📄 Resume Preview")

    # Render with selected template and chosen preview mode (keeps dynamic template updates)
    render_resume(preview_mode, selected_template_id)
    # html may be updated by render_resume
    html = st.session_state.get("generated_html", html)

    # Centralized download (avoid double preview; only build download)
    show_preview_and_download(
        html=html,
        base_name=base_name,
        mode=preview_mode,
        key_prefix="app_prev",
        display=False,   # preview already rendered by render_resume
    )

    # Keyword chips (dark-mode safe)
    if matched:
        chip_css = """
        <style>
        .chip{
          display:inline-block;
          padding:3px 8px;
          margin:2px;
          border-radius:999px;
          font-size:12px;
          background:#eef;
          border:1px solid #ccd;
          color:#000;
        }
        @media (prefers-color-scheme: dark) {
          .chip{
            background:#333;
            border:1px solid #555;
            color:#fff;
          }
        }
        </style>
        """
        st.markdown(chip_css, unsafe_allow_html=True)
        st.markdown("**Matched Keywords:**")
        chips = "".join(f"<span class='chip'>{w}</span>" for w in matched)
        st.markdown(chips, unsafe_allow_html=True)

    # Scores table (collapsible)
    with st.expander("Show section relevance scores"):
        # Fallback scorer using JD keyword overlap when no score is provided
        jd_text = st.session_state.get("job_description", "") or ""
        jd_tokens = set(_ru_build_keywords(jd_text)) if jd_text else set()

        def _cheap_overlap_score(sec) -> float | None:
            if not jd_tokens:
                return None
            txt = (section_text(sec) or "").lower()
            if not txt:
                return None
            toks = set(re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", txt))
            return len(jd_tokens & toks) / max(len(jd_tokens), 1)

        rows = []
        for sec in tailored:
            score = scores_map.get(id(sec), sec.get("_score"))
            if score is None:
                score = _cheap_overlap_score(sec)
            if score is None:
                continue
            title = sec.get("title", "")
            company = sec.get("company", "")
            typ = sec.get("type", "")
            label = f"[{typ}] {title} — {company}" if company else f"[{typ}] {title}"
            rows.append((label, f"{float(score)*100:.1f}%"))
        if rows:
            st.write("\n".join(f"- {lbl}: {s}" for lbl, s in rows))
        else:
            st.write("Scores not available for this ordering.")

# ========= Cover Letter =========
# NOTE: Disabled via feature flag until the cover letter flow is fixed/refactored.
if ENABLE_COVER_LETTER:
    st.markdown("---")
    st.subheader("✉️ Cover Letter")

    gen_cover = st.checkbox("Generate a cover letter", value=False, key="chk_gen_cover")
    if gen_cover:
        # Choose model re-use
        cl_use_gpt = (summary_mode == "GPT-powered (API cost)") and st.checkbox(
            "Use GPT for cover letter (API cost)", value=True, key="chk_cl_use_gpt"
        )
        cl_model = selected_model
        if cl_use_gpt:
            cl_model = st.selectbox(
                "Cover letter model",
                options=list(GPT_MODELS.keys()),
                format_func=lambda m: f"{m} — {GPT_MODELS[m]}",
                index=list(GPT_MODELS.keys()).index(selected_model) if selected_model in GPT_MODELS else 0,
                key="sel_cl_model"
            )

        # Generate (once) or Regenerate
        header_ss = st.session_state.get("header", {})
        tailored_ss = st.session_state.get("tailored", [])
        scores_map_ss = st.session_state.get("scores_map", {})
        job_desc_ss = st.session_state.get("job_description", "")
        tailored_summary_ss = st.session_state.get("tailored_summary", "")

        # Check if resume has been generated in this session
        has_resume = "tailored" in st.session_state and st.session_state.get("tailored")
        st.button("Generate Cover Letter", disabled=not has_resume, key="btn_cover_generate_gate")

        if not has_resume:
            st.info("Generate a tailored resume first, then create a cover letter.")

        if has_resume and st.button("Generate Cover Letter", key="btn_cover_generate"):
            # sanity fallback if user tries to generate a cover letter before generating resume
            if not tailored_summary_ss:
                # optionally regenerate a simple offline summary so we don't crash
                tailored_summary_ss = generate_tailored_summary(
                    resume,
                    job_desc_ss or (st.session_state.get("job_description") or ""),
                    use_gpt=False
                )

            cl_text, cl_company, cl_role = generate_cover_letter(
                header_ss,
                tailored_ss,                 # use tailored sections for bullets
                scores_map_ss,
                job_desc_ss,
                tailored_summary_ss,
                use_gpt=cl_use_gpt,
                model=cl_model
            )
            st.session_state.cover_letter_text = cl_text
            st.session_state.cl_company = cl_company
            st.session_state.cl_role = cl_role

        # Editable preview textarea
        if "cover_letter_text" in st.session_state:
            st.markdown("You can edit before downloading:")
            st.session_state.cover_letter_text = st.text_area(
                "Cover Letter (editable)",
                st.session_state.cover_letter_text,
                height=400,
                key="ta_cover_letter"
            )

            # Filenames
            cl_company = st.session_state.get("cl_company") or _ru_extract_company(st.session_state.get("job_description","") or "")
            cl_role = st.session_state.get("cl_role") or _ru_extract_role(st.session_state.get("job_description","") or "")
            name_bits = []
            if cl_company:
                name_bits.append(_ru_clean_for_filename(cl_company))
            if cl_role:
                name_bits.append(_ru_clean_for_filename(cl_role))
            base = "_".join(name_bits) + "_cover_letter" if name_bits else "cover_letter"

            # Download as TXT
            st.download_button(
                "📥 Download Cover Letter (TXT)",
                data=st.session_state.cover_letter_text,
                file_name=f"{base}.txt",
                mime="text/plain",
                key="dl_cover_txt"
            )

            # Download as PDF (simple HTML wrapper)
            try:
                from weasyprint import HTML as WPHTML
                cl_html = f"""<!doctype html><html><head>
                <meta charset="utf-8">
                <style>
                  body {{ font-family: Arial, sans-serif; font-size: 11pt; margin: 40px; }}
                  p {{ margin: 0 0 10px 0; }}
                  pre {{ white-space: pre-wrap; }}
                </style></head><body>
                <pre>{st.session_state.cover_letter_text}</pre>
                </body></html>"""
                buf = BytesIO()
                WPHTML(string=cl_html).write_pdf(buf)
                cl_pdf = buf.getvalue()
                st.download_button(
                    "📥 Download Cover Letter (PDF)",
                    data=cl_pdf,
                    file_name=f"{base}.pdf",
                    mime="application/pdf",
                    key="dl_cover_pdf"
                )
            except Exception:
                st.info("Install WeasyPrint to enable PDF cover letter download: `pip install weasyprint`")
else:
    # Hidden for now; will be re-enabled in a future refactor
    pass
 