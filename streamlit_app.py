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
from jinja2 import Environment, FileSystemLoader

from resume_semantic_scoring_engine import (
    load_resume,
    generate_tailored_resume,
    enhance_experience_with_impact,
    clear_embeddings_cache,
    section_text,   # used for keyword chip matching
)

# ==============================
# CONFIG
# ==============================
RESUME_PATH = "modular_resume_full.yaml"

# --- Setup ---
load_dotenv()
client = OpenAI()

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

# ----- Company name extraction from Job Description -----
def _clean_slug(s: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return slug[:80] if slug else ""

def extract_company_name(jd: str) -> str | None:
    """
    Best-effort company extractor from raw JD text.
    Order of heuristics:
      1) 'Company:' prefixed label
      2) 'About <Company>' header
      3) '<Company> is seeking|seeks|is hiring'
      4) 'at <Company>'
      5) Fallback: scan first lines for a proper-noun-ish line
    Returns a cleaned slug or None.
    """
    text = jd.strip()
    if not text:
        return None

    m = re.search(r"(?im)^\s*company\s*[:\-]\s*(.+)$", text)
    if m:
        cand = m.group(1).strip()
        cand = re.split(r"[|•\-\(\)\[\]\n\r]", cand)[0].strip()
        slug = _clean_slug(cand)
        if slug:
            return slug

    m = re.search(r"(?i)\babout\s+([A-Z][A-Za-z0-9&\.,\- ]{2,})", text)
    if m:
        cand = m.group(1).strip()
        cand = re.split(r"(?i)\s+(is|provides|offers|was|were|inc\.?|llc|ltd|plc|corp\.?)", cand)[0].strip()
        slug = _clean_slug(cand)
        if slug:
            return slug

    m = re.search(r"(?i)\b([A-Z][A-Za-z0-9&\.,\- ]{2,}?)\s+(?:is\s+seeking|seeks|is\s+hiring)\b", text)
    if m:
        slug = _clean_slug(m.group(1))
        if slug:
            return slug

    m = re.search(r"(?i)\bat\s+([A-Z][A-Za-z0-9&\.,\- ]{2,})(?:[\s,\.]|$)", text)
    if m:
        cand = m.group(1).strip()
        slug = _clean_slug(cand)
        if slug:
            return slug

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:6]:
        if re.match(r"(?i)^(we|our|the|you|role|position|responsibilities|requirements)\b", ln):
            continue
        if re.search(r"[A-Z][a-z]", ln) and len(ln) <= 60:
            ln = re.split(r"[|•\-–—:]", ln)[0].strip()
            slug = _clean_slug(ln)
            if slug:
                return slug
    return None

# ----- Role title extraction from JD (best-effort) -----
def extract_role_title(jd: str) -> str | None:
    text = jd.strip()
    if not text:
        return None
    # 1) "Title:" label
    m = re.search(r"(?im)^\s*(role|title)\s*[:\-]\s*(.+)$", text)
    if m:
        cand = m.group(2).strip()
        cand = re.split(r"[|•\-\(\)\[\]\n\r]", cand)[0].strip()
        return cand[:100]
    # 2) "We are hiring a/an X", "Seeking X", "X (Job Title)"
    m = re.search(r"(?i)\b(hiring|seek(?:ing)?|searching)\b.*?\b(for|as|a|an)\s+([A-Z][A-Za-z0-9\-/&\s]{2,})", text)
    if m:
        return re.sub(r"[\s\|•\-\(\)\[\]]+$", "", m.group(3)).strip()[:100]
    # 3) First heading-ish line
    for ln in [ln.strip() for ln in text.splitlines() if ln.strip()][:5]:
        if len(ln) <= 80 and re.search(r"[A-Za-z]", ln) and not ln.lower().startswith(("about","company","role","position")):
            return re.sub(r"[\|•\-–—:]+.*$", "", ln).strip()[:100]
    return None

def _clean_for_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:80]

# --- Highlighter (skips NOHL-marked blocks) ---
import re

# Add some common noise/HTML/tag words you never want highlighted
STOPWORDS = {
    "and","the","with","for","your","you","our","their","this","that",
    "skills","experience","years","team","work","ability","in","to","of",
    # HTML/tag-ish words to avoid breaking markup:
    "strong","em","span","div","class","style","script","http","https","href","mark"
}

def build_keywords(job_description: str):
    words = re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", job_description)
    uniq = {
        w.strip().lower()
        for w in words
        if w.strip() and w.strip().lower() not in STOPWORDS
    }
    # long-first to avoid partial overlaps
    return sorted(uniq, key=len, reverse=True)[:200]

def _highlight_fragment(fragment: str, keywords):
    out = fragment
    for w in keywords:
        # Do NOT match when inside tag names or closing tags:
        #   - previous char cannot be a word char, '>', '<', or '/'
        #   - next char cannot be a word char or '<'
        pattern = re.compile(
            rf"(?<![\w><\/])({re.escape(w)})(?![\w<])",
            flags=re.IGNORECASE,
        )
        out = pattern.sub(r"<mark>\g<1></mark>", out)
    return out

def highlight_html(html: str, job_description: str):
    keywords = build_keywords(job_description)
    if not keywords:
        return html
    # keep header/contact area safe from highlighting
    parts = re.split(r"(<!--NOHL_START-->.*?<!--NOHL_END-->)", html, flags=re.DOTALL)
    for i, part in enumerate(parts):
        if part.startswith("<!--NOHL_START-->"):
            continue
        parts[i] = _highlight_fragment(part, keywords)
    return "".join(parts)

# ----- Achievement selection helpers -----
def _tokenize(s: str):
    return {w.lower() for w in re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", s or "")}

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

def _score_sentence_vs_jd(sentence: str, jd_tokens: set[str]) -> float:
    if not sentence:
        return 0.0
    sent_tokens = _tokenize(sentence)
    if not sent_tokens:
        return 0.0
    overlap = jd_tokens & sent_tokens
    has_numbers = bool(re.search(r"(\d|%|\$)", sentence))
    return (len(overlap) / max(len(jd_tokens), 1)) + (0.05 if has_numbers else 0.0)

def _pick_best_achievement_overlap(resume, job_description: str) -> str | None:
    jd_tokens = _tokenize(job_description)
    candidates = _collect_candidate_achievements(resume)
    if not candidates:
        return None
    ranked = sorted(
        ((c, _score_sentence_vs_jd(c, jd_tokens)) for c in candidates),
        key=lambda x: x[1],
        reverse=True
    )
    best, best_score = ranked[0]
    return best if best_score >= 0.02 else None

def _cosine(a, b):
    num = sum(x*y for x, y in zip(a, b))
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(y*y for y in b))
    return (num / (da * db)) if da and db else 0.0

def _pick_best_achievement_embeddings(resume, job_description: str, embedding_model: str) -> str | None:
    candidates = _collect_candidate_achievements(resume)
    if not candidates:
        return None
    resp = client.embeddings.create(model=embedding_model, input=[job_description] + candidates)
    jd_vec = resp.data[0].embedding
    best_idx, best_score = -1, -1.0
    for i, d in enumerate(resp.data[1:], start=0):
        score = _cosine(jd_vec, d.embedding)
        if score > best_score:
            best_idx, best_score = i, score
    return candidates[best_idx] if best_idx >= 0 else None

def _build_summary_outline(title, years_exp, skills_str, achievement):
    parts = []
    parts.append("[TITLE] with [YEARS]+ years of proven expertise" if years_exp else "[TITLE] with proven expertise")
    if skills_str:
        parts.append("specializing in [SKILLS]")
    if achievement:
        parts.append("Notable achievement: [ACHIEVEMENT]")
    return ". ".join(parts) + "."

def _safe_paraphrase_with_placeholders(outline, model):
    prompt = f"""
Rewrite the following into a polished 2–4 sentence professional summary (<=300 characters),
keeping the placeholders [TITLE], [YEARS], [SKILLS], and [ACHIEVEMENT] EXACTLY AS WRITTEN.
Do NOT introduce any new facts, numbers, or achievements.

Outline:
{outline}
"""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You rewrite text concisely without adding facts. Preserve placeholders exactly."},
            {"role": "user", "content": prompt.strip()},
        ],
        max_tokens=180,
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def _substitute_placeholders(text, title, years_exp, skills_str, achievement):
    subs = {
        "[TITLE]": title,
        "[YEARS]": str(years_exp) if years_exp is not None else "",
        "[SKILLS]": skills_str or "",
        "[ACHIEVEMENT]": achievement or "",
    }
    for k, v in subs.items():
        text = text.replace(k, v)
    return re.sub(r"\s{2,}", " ", text).strip(" .") + "."

# ----- Tailored summary (uses embeddings if enabled) -----
def generate_tailored_summary(
    resume,
    job_description,
    use_gpt=False,
    model="gpt-3.5-turbo",
    use_embeddings=False,
    embedding_model="text-embedding-3-small",
):
    header = next((sec for sec in resume if sec.get("type") == "header"), {})
    title = header.get("title", "Experienced Professional")
    years_exp = header.get("years_experience")
    opening_line = f"{title} with {years_exp}+ years of proven expertise" if years_exp else f"{title} with proven expertise"

    job_keywords = set(word.lower() for word in job_description.split() if len(word) > 2)
    all_resume_tags = set()
    for section in resume:
        if "tags" in section:
            all_resume_tags.update(tag.lower() for tag in section["tags"])

    matched_skills = sorted(list(job_keywords & all_resume_tags))
    top_skills_str = ", ".join(matched_skills[:4]) if matched_skills else None

    # Pick the most relevant achievement (embeddings when enabled)
    if use_embeddings:
        achievement = _pick_best_achievement_embeddings(resume, job_description, embedding_model)
    else:
        achievement = _pick_best_achievement_overlap(resume, job_description)

    # Always-safe offline summary
    def _offline():
        parts = [opening_line]
        if top_skills_str:
            parts.append(f"specializing in {top_skills_str}")
        if achievement:
            parts.append(f"Notable achievement: {achievement}")
        return ". ".join(parts) + "."

    if not use_gpt:
        return _offline()

    # STRICT: GPT can only rephrase placeholders; we substitute facts after
    outline = _build_summary_outline(title, years_exp, top_skills_str, achievement)
    try:
        templated = _safe_paraphrase_with_placeholders(outline, model)
        final = _substitute_placeholders(templated, title, years_exp, top_skills_str, achievement)
        return final
    except Exception:
        return _offline()

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
    company = extract_company_name(job_description)
    role = extract_role_title(job_description)
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

# ==== UI ====
st.set_page_config(page_title="Tailored Resume Builder", layout="wide")
st.title("🎯 Tailored Resume Generator")

# Sidebar: cache controls
with st.sidebar:
    st.markdown("### Tools")
    if st.button("Clear embeddings cache"):
        try:
            clear_embeddings_cache()
            st.success("Embeddings cache cleared.")
        except Exception:
            st.info("No cache file found or unable to clear.")

st.markdown("Paste a job description below and click **Generate Resume** to get a customized resume based on your experience.")

# Load YAML
resume = load_resume(RESUME_PATH)
header = next((sec for sec in resume if sec.get("type") == "header"), None)

# Inputs
job_description = st.text_area("📝 Job Description", height=240)

col1, col2 = st.columns(2)
with col1:
    summary_mode = st.radio("Summary Mode", ("Offline (free)", "GPT-powered (API cost)"), index=0)
    strict_mode = summary_mode == "GPT-powered (API cost)" and st.checkbox(
        "Strict factual mode (no new claims)", value=True,
        help="Model may only rephrase supplied facts. No new achievements or metrics."
    )
    selected_model = st.selectbox(
        "GPT Model (pricing per 1K tokens):",
        options=list(GPT_MODELS.keys()),
        format_func=lambda m: f"{m} — {GPT_MODELS[m]}",
        index=0,
    )
    add_impact = st.checkbox("Add tailored impact statements (per role)", value=False)
    bullets_per_role = 1
    if add_impact:
        bullets_per_role = st.slider("Impact bullets per role", 1, 3, 1)
    show_generated = st.checkbox("Show generated impact statements", value=True)

with col2:
    ordering_mode = st.radio("Experience Ordering", ("Relevancy First", "Chronological", "Hybrid"), index=0)
    top_n_hybrid = 3
    if ordering_mode == "Hybrid":
        top_n_hybrid = st.slider("Top relevant items before chronological:", 1, 10, 3)
    # Embeddings options
    use_embeddings = st.checkbox("Use semantic matching (OpenAI embeddings)", value=False)
    embedding_model = st.selectbox(
        "Embeddings model (pricing per 1M tokens):",
        options=list(EMBED_MODELS.keys()),
        format_func=lambda m: f"{m} — {EMBED_MODELS[m]}",
        index=0,
        disabled=not use_embeddings
    )
    highlight = st.checkbox("Highlight matches in preview", value=True)

# Generate
if st.button("Generate Tailored Resume", use_container_width=True):
    if not job_description.strip():
        st.warning("Please enter a job description first.")
    else:
        tailored, scores_map = generate_tailored_resume(
            resume,
            job_description,
            top_n=top_n_hybrid,
            use_embeddings=use_embeddings,
            ordering=ordering_mode.lower(),   # 'relevancy'/'chronological'/'hybrid'
            embedding_model=embedding_model if use_embeddings else "text-embedding-3-small",
        )

        # Add tailored impact bullets if requested (requires GPT mode)
        if add_impact and (summary_mode == "GPT-powered (API cost)"):
            tailored = enhance_experience_with_impact(
                tailored,
                job_description,
                use_gpt=True,
                model=selected_model,
                mark_generated=True,
                bullets_per_role=bullets_per_role,
            )

        # Hide generated impact if the user unchecks it
        if not show_generated:
            for sec in tailored:
                if sec.get("type") == "experience":
                    sec["contributions"] = [
                        c for c in sec.get("contributions", [])
                        if not (c.get("impact") and c.get("source") == "generated")
                    ]

        tailored_summary = generate_tailored_summary(
            resume,
            job_description,
            use_gpt=(summary_mode == "GPT-powered (API cost)"),
            model=selected_model,
            use_embeddings=use_embeddings,
            embedding_model=(embedding_model if use_embeddings else "text-embedding-3-small"),
        )

        env = Environment(loader=FileSystemLoader("."))
        template = env.get_template("tailored_resume_template.html")
        html = template.render(header=header, tailored_summary=tailored_summary, resume=tailored)

        # Highlighted preview
        highlighted_html = highlight_html(html, job_description) if highlight else html

        # Keyword chips (JD ∩ resume)
        jd_words = set(build_keywords(job_description))
        resume_text = " ".join((section_text(s) or "").lower() for s in resume)
        matched = sorted([w for w in jd_words if w in resume_text], key=len, reverse=True)[:50]

        # Company slug derived from JD (for filenames)
        company_slug = extract_company_name(job_description)
        base_name = f"{company_slug}_full_resume" if company_slug else "full_resume"

        st.session_state.generated_html = html
        st.session_state.highlighted_html = highlighted_html
        st.session_state.tailored = tailored
        st.session_state.keywords = matched
        st.session_state.scores_map = scores_map
        st.session_state.base_name = base_name
        st.session_state.tailored_summary = tailored_summary
        st.session_state.header = header
        st.session_state.job_description = job_description


# Preview & Download + Scores & Keywords
if "generated_html" in st.session_state:
    html = st.session_state.generated_html
    highlighted_html = st.session_state.get("highlighted_html", html)
    tailored = st.session_state.get("tailored", [])
    matched = st.session_state.get("keywords", [])
    scores_map = st.session_state.get("scores_map", {})
    base_name = st.session_state.get("base_name", "full_resume")

    st.success("✅ Resume generated!")

    # Downloads use the clean (non-highlighted) HTML
    st.download_button(
        "📥 Download Resume as HTML",
        data=html,
        file_name=f"{base_name}.html",
        mime="text/html"
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
        rows = []
        for sec in tailored:
            score = scores_map.get(id(sec), sec.get("_score"))
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

    # Preview mode
    preview_mode = st.radio("Preview Mode", ("Formatted (HTML)", "Plain Text", "PDF Preview"), index=0)
    st.markdown("### 📄 Resume Preview")

    if preview_mode == "Formatted (HTML)":
        st.components.v1.html(highlighted_html, height=800, scrolling=True)

    elif preview_mode == "Plain Text":
        plain_text = re.sub(r"<[^>]+>", "", html)
        st.text_area("Plain Text Resume", plain_text, height=800)

    elif preview_mode == "PDF Preview":
        try:
            from weasyprint import HTML as WPHTML
            pdf_buffer = BytesIO()
            WPHTML(string=highlighted_html).write_pdf(pdf_buffer)  # preview reflects highlighting
            pdf_data = pdf_buffer.getvalue()
            b64_pdf = base64.b64encode(pdf_data).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{b64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.download_button(
                "📥 Download Resume as PDF",
                data=pdf_data,
                file_name=f"{base_name}.pdf",
                mime="application/pdf"
            )
        except ImportError:
            st.error("WeasyPrint is not installed. Run `pip install weasyprint` to enable PDF preview.")

    # ========= Cover Letter =========
    st.markdown("---")
    st.subheader("✉️ Cover Letter")

    gen_cover = st.checkbox("Generate a cover letter", value=False)
    if gen_cover:
        # Choose model re-use
        cl_use_gpt = (summary_mode == "GPT-powered (API cost)") and st.checkbox("Use GPT for cover letter (API cost)", value=True)
        cl_model = selected_model
        if cl_use_gpt:
            cl_model = st.selectbox(
                "Cover letter model",
                options=list(GPT_MODELS.keys()),
                format_func=lambda m: f"{m} — {GPT_MODELS[m]}",
                index=list(GPT_MODELS.keys()).index(selected_model) if selected_model in GPT_MODELS else 0
            )

        # Generate (once) or Regenerate
        # pull from session_state so it works across reruns
        header_ss = st.session_state.get("header", {})
        tailored_ss = st.session_state.get("tailored", [])
        scores_map_ss = st.session_state.get("scores_map", {})
        job_desc_ss = st.session_state.get("job_description", "")
        tailored_summary_ss = st.session_state.get("tailored_summary", "")

        # Check if resume has been generated in this session
        has_resume = "tailored" in st.session_state and st.session_state.get("tailored")
        st.button("Generate Cover Letter", disabled=not has_resume)

        if not has_resume:
            st.info("Generate a tailored resume first, then create a cover letter.")

        if has_resume and st.button("Generate Cover Letter"):
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
                height=400
            )

            # Filenames
            cl_company = st.session_state.get("cl_company") or extract_company_name(job_description)
            cl_role = st.session_state.get("cl_role") or extract_role_title(job_description)
            name_bits = []
            if cl_company:
                name_bits.append(_clean_for_filename(cl_company))
            if cl_role:
                name_bits.append(_clean_for_filename(cl_role))
            base = "_".join(name_bits) + "_cover_letter" if name_bits else "cover_letter"

            # Download as TXT
            st.download_button(
                "📥 Download Cover Letter (TXT)",
                data=st.session_state.cover_letter_text,
                file_name=f"{base}.txt",
                mime="text/plain"
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
                    mime="application/pdf"
                )
            except Exception:
                st.info("Install WeasyPrint to enable PDF cover letter download: `pip install weasyprint`")
