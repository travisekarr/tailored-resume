# pages/04_Generate_From_Jobs.py
import os
import re
import traceback
import urllib.parse
from io import BytesIO
from datetime import datetime, timezone
import streamlit as st
import yaml
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from resume_semantic_scoring_engine import load_resume, generate_tailored_resume


# Core job DB helpers
from job_store import (
    query_top_matches,
    set_job_status,
    set_job_resume,
    log_api_usage,            # usage logging
)

# Optional actions (graceful if missing in job_store)
try:
    from job_store import mark_not_suitable as _mark_not_suitable_db
except Exception:
    _mark_not_suitable_db = None
try:
    from job_store import mark_submitted as _mark_submitted_db
except Exception:
    _mark_submitted_db = None
try:
    from job_store import query_submitted as _query_submitted_db
except Exception:
    _query_submitted_db = None

# ==============================
# CONFIG
# ==============================
RESUME_PATH = "modular_resume_full.yaml"
TEMPLATE_FILE = "tailored_resume_template.html"
DEFAULTS_PATH = "report_defaults.yaml"
MODEL_SETTINGS_PATH = "api_models.yaml"   # <- unified model settings YAML

# Environment / OpenAI
load_dotenv()
_openai_client = OpenAI()  # safe if key missing; we gate GPT usage behind toggles


# ==============================
# Model settings (YAML-driven)
# ==============================
_MODEL_FALLBACK = {
    "models": {
        "chat": {
            "default": "gpt-5.0",
            "options": [
                {"id": "gpt-5.0", "display": "GPT 5.0 (Thinking)"},
                {"id": "gpt-5o",  "display": "GPT-5o"},
                {"id": "gpt-4.1", "display": "GPT-4.1"},
                {"id": "gpt-4o",  "display": "GPT-4o"},
                {"id": "gpt-4",   "display": "GPT-4"},
                {"id": "gpt-3.5-turbo", "display": "GPT-3.5 Turbo"},
            ],
        },
        "embeddings": {
            "default": "text-embedding-3-small",
            "options": [
                {"id": "text-embedding-3-small", "display": "Text Embedding 3 Small"},
                {"id": "text-embedding-3-large", "display": "Text Embedding 3 Large"},
            ],
        },
    }
}

def _load_model_settings(path: str = MODEL_SETTINGS_PATH) -> dict:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # minimal validation
            if "models" in data and "chat" in data["models"] and "embeddings" in data["models"]:
                return data
    except Exception:
        pass
    return _MODEL_FALLBACK

def _model_selectbox(label: str, group: str, *, key: str, disabled: bool = False):
    """
    Render a selectbox using YAML-backed model options.
    Stores the selected **model id** in session (so the rest of the code can
    pass it directly to OpenAI). Displays the **display** label.
    """
    settings = _load_model_settings()
    grp = settings["models"][group]
    opts = grp.get("options", [])
    default_id = grp.get("default") or (opts[0]["id"] if opts else "")
    ids = [o["id"] for o in opts]
    id_to_label = {o["id"]: o.get("display", o["id"]) for o in opts}

    def _fmt(x): return id_to_label.get(x, x)
    try:
        default_idx = ids.index(default_id) if default_id in ids else 0
    except Exception:
        default_idx = 0

    return st.selectbox(
        label,
        ids,
        index=default_idx,
        key=key,
        disabled=disabled,
        format_func=_fmt,
    )


# ==============================
# Defaults loader (same shape as Jobs Report), with resume_save_dir
# ==============================
DEFAULTS_FALLBACK = {
    "db": "jobs.db",
    "since": "24h",
    "min_score": 0.0,
    "hide_stale_days": None,
    "search": "",
    "top_count": 15,
    "new_limit": 100,
    "changed_limit": 200,
    "resume_save_dir": "kept_resumes",
    "filters": {
        "title_contains": "",
        "company_contains": "",
        "location_contains": "",
        "description_contains": "",
        "sources": [],
        "remote_only": False,
        "posted_after": "",
        "changed_fields": [],
        "statuses": [],
        "starred_only": False,
        "use_location_allowlist": True,
        "allowed_locations": ["USA", "VA", "North America", "Remote", "Worldwide"],
        "allow_empty_location": True,
    },
}

def load_defaults(path=DEFAULTS_PATH):
    if not os.path.exists(path):
        return DEFAULTS_FALLBACK.copy()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = {**DEFAULTS_FALLBACK, **data}
    merged["filters"] = {**DEFAULTS_FALLBACK["filters"], **(merged.get("filters") or {})}
    if "resume_save_dir" not in merged or not merged["resume_save_dir"]:
        merged["resume_save_dir"] = "kept_resumes"
    return merged


# ==============================
# Persistent state
# ==============================
if "gen_debug" not in st.session_state:
    st.session_state.gen_debug = []
if "gen_cache" not in st.session_state:
    # per-row cache: { row_key: {"html": str, "pdf": bytes|None, "base_stem": str, "job_id": str, "out_score": float} }
    st.session_state.gen_cache = {}

def log(msg: str):
    st.session_state.gen_debug.append(str(msg))


# ==============================
# Helpers: extraction, formatting, scoring
# ==============================
def _clean_slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", (s or "")).strip("_")[:80]

def extract_company_name(jd: str) -> str | None:
    text = (jd or "").strip()
    if not text:
        return None
    m = re.search(r"(?im)^\s*company\s*[:\-]\s*(.+)$", text)
    if m:
        cand = re.split(r"[|•\-\(\)\[\]\n\r]", m.group(1).strip())[0].strip()
        slug = _clean_slug(cand)
        if slug:
            return slug
    m = re.search(r"(?i)\babout\s+([A-Z][A-Za-z0-9&\.,\- ]{2,})", text)
    if m:
        cand = re.split(r"(?i)\s+(is|provides|offers|was|were|inc\.?|llc|ltd|plc|corp\.?)", m.group(1).strip())[0].strip()
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
        slug = _clean_slug(m.group(1).strip())
        if slug:
            return slug
    for ln in [ln.strip() for ln in text.splitlines() if ln.strip()][:6]:
        if re.match(r"(?i)^(we|our|role|position|responsibil|about|company)\b", ln):
            continue
        if re.search(r"[A-Z][a-z]", ln) and len(ln) <= 60:
            slug = _clean_slug(re.split(r"[|•\-–—:]", ln)[0].strip())
            if slug:
                return slug
    return None

def html_to_text(html: str) -> str:
    return re.sub(r"<[^>]+>", " ", html or "").replace("\xa0", " ").strip()

def tfidf_score(a_text: str, b_text: str) -> float:
    corpus = [a_text or "", b_text or ""]
    vec = TfidfVectorizer().fit_transform(corpus)
    sim = cosine_similarity(vec[0:1], vec[1:2]).flatten()[0]
    return float(sim or 0.0)

def path_to_file_url(path: str) -> str:
    abs_path = os.path.abspath(path).replace("\\", "/")
    return "file:///" + urllib.parse.quote(abs_path, safe="/:._-")

def location_allowed(loc_value: str | None, allowed: list[str], allow_empty: bool) -> bool:
    if not (allowed or allow_empty):
        return True
    loc = (loc_value or "").strip()
    if not loc:
        return bool(allow_empty)
    low = loc.lower()
    for tok in (allowed or []):
        if tok and tok.lower() in low:
            return True
    return False

def _contains(val, needle):
    return (needle.lower() in (val or "").lower()) if needle else True

def apply_saved_filters(rows: list[dict], fdefs: dict) -> list[dict]:
    if not rows:
        return rows

    title_q   = fdefs.get("title_contains", "")
    company_q = fdefs.get("company_contains", "")
    loc_q     = fdefs.get("location_contains", "")
    desc_q    = fdefs.get("description_contains", "")
    sources   = fdefs.get("sources") or []
    remote_only = bool(fdefs.get("remote_only"))
    statuses  = fdefs.get("statuses") or []
    starred_only = bool(fdefs.get("starred_only"))

    use_location_allowlist = bool(fdefs.get("use_location_allowlist", True))
    allowed_locations = fdefs.get("allowed_locations") or []
    allow_empty_location = bool(fdefs.get("allow_empty_location", True))

    out = []
    for r in rows:
        if not _contains(r.get("title",""), title_q):      continue
        if not _contains(r.get("company",""), company_q):  continue
        if not _contains(r.get("location",""), loc_q):     continue
        if not _contains(r.get("description",""), desc_q): continue
        if sources and (r.get("source") not in sources):   continue
        if remote_only and not bool(r.get("remote")):      continue
        if statuses and (r.get("status") or "new") not in statuses: continue
        if starred_only and int(r.get("starred") or 0) != 1: continue
        if use_location_allowlist:
            if not location_allowed(r.get("location"), allowed_locations, allow_empty_location):
                continue
        out.append(r)
    return out

def make_arrow_friendly(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for col in df.columns:
        low = col.lower()
        if low in ("id", "job_id"):
            df[col] = df[col].astype("string")
        elif low in ("posted_at","first_seen","last_seen","pulled_at","created_at",
                     "updated_at","changed_at","submitted_at"):
            df[col] = df[col].astype("string")
        elif low in ("score", "resume_score"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string")
    return df

def _log_chat_usage(resp, model: str, context: str):
    try:
        u = resp.usage
        it = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None) or 0
        ot = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None) or 0
        tt = getattr(u, "total_tokens", None) or (it + ot)
        rid = getattr(resp, "id", "") or getattr(resp, "request_id", "")
        log_api_usage(
            db_path=st.session_state.get("gen_db_path") or "jobs.db",
            endpoint="chat.completions",
            model=model,
            input_tokens=int(it or 0),
            output_tokens=int(ot or 0),
            total_tokens=int(tt or 0),
            request_id=rid,
            context=context,
            meta={}
        )
    except Exception:
        pass


# ==============================
# Strict rephrase helpers (optional GPT)
# ==============================
def _tokenize(s: str):
    return {w.lower() for w in re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", s or "")}

def _collect_candidate_achievements(resume):
    cands = []
    for sec in resume:
        if sec.get("type") == "experience":
            for c in (sec.get("contributions") or []):
                d = (c or {}).get("description")
                if isinstance(d, str) and d.strip():
                    cands.append(d.strip())
    summary_sec = next((sec for sec in resume if sec.get("type") == "summary"), {})
    for a in summary_sec.get("achievements") or []:
        if isinstance(a, str) and a.strip():
            cands.append(a.strip())
    out, seen = [], set()
    for c in cands:
        k = c.lower()
        if k not in seen:
            out.append(c); seen.add(k)
    return out

def _score_sentence_vs_jd(sentence: str, jd_tokens: set[str]) -> float:
    if not sentence: return 0.0
    sent_tokens = _tokenize(sentence)
    if not sent_tokens: return 0.0
    overlap = jd_tokens & sent_tokens
    has_numbers = bool(re.search(r"(\d|%|\$)", sentence))
    return (len(overlap) / max(len(jd_tokens), 1)) + (0.05 if has_numbers else 0.0)

def _pick_best_achievement_overlap(resume, job_description: str) -> str | None:
    jd_tokens = _tokenize(job_description)
    cands = _collect_candidate_achievements(resume)
    if not cands: return None
    ranked = sorted(((c, _score_sentence_vs_jd(c, jd_tokens)) for c in cands),
                    key=lambda x: x[1], reverse=True)
    best, score = ranked[0]
    return best if score >= 0.02 else None

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
    try:
        resp = _openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You rewrite text concisely without adding facts. Preserve placeholders exactly."},
                {"role": "user", "content": prompt.strip()},
            ],
            max_tokens=180,
            temperature=0.2,
        )
        _log_chat_usage(resp, model, "ui:04_Generate_From_Jobs:summary")
        return resp.choices[0].message.content.strip()
    except Exception as e:
        if "invalid model" in str(e).lower():
            fallback = "gpt-4o"
            st.warning(f"Model '{model}' not available; using {fallback} instead.")
            resp = _openai_client.chat.completions.create(
                model=fallback,
                messages=[
                    {"role": "system", "content": "You rewrite text concisely without adding facts. Preserve placeholders exactly."},
                    {"role": "user", "content": prompt.strip()},
                ],
                max_tokens=180,
                temperature=0.2,
            )
            _log_chat_usage(resp, fallback, "ui:04_Generate_From_Jobs:summary")
            return resp.choices[0].message.content.strip()
        # Any other error: just return the offline outline
        return outline

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

def build_allowed_vocab(resume, job_description: str) -> list[str]:
    jd_tokens = {w.lower() for w in re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", job_description or "")}
    resume_terms = set()
    for sec in resume:
        for t in (sec.get("tags") or []):
            resume_terms.add(t.lower())
        if sec.get("type") == "experience":
            for c in sec.get("contributions") or []:
                for s in (c.get("skills_used") or []):
                    resume_terms.add(s.lower())
    allowed = sorted(list(jd_tokens & resume_terms))
    return allowed[:200]

def paraphrase_bullet_strict(bullet: str, allowed_vocab: list[str], model: str) -> str:
    prompt = f"""
Rephrase this resume bullet to better match the job's terminology.
Rules:
- Use ONLY these extra words if you add/substitute terms: {", ".join(allowed_vocab) or "(none)"}.
- Do NOT add any new achievements, numbers, employers, or tools not present in the original or allowed list.
- Keep the original meaning and scope. <= 28 words.

Original:
{bullet}
"""
    try:
        resp = _openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You rephrase text concisely without adding facts. Obey constraints exactly."},
                {"role": "user", "content": prompt.strip()},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        _log_chat_usage(resp, model, "ui:04_Generate_From_Jobs:bullets")
        text = resp.choices[0].message.content.strip()
        return re.sub(r"\s+", " ", text)
    except Exception as e:
        if "invalid model" in str(e).lower():
            fallback = "gpt-4o"
            st.warning(f"Model '{model}' not available; using {fallback} instead.")
            resp = _openai_client.chat.completions.create(
                model=fallback,
                messages=[
                    {"role": "system", "content": "You rephrase text concisely without adding facts. Obey constraints exactly."},
                    {"role": "user", "content": prompt.strip()},
                ],
                temperature=0.2,
                max_tokens=120,
            )
            _log_chat_usage(resp, fallback, "ui:04_Generate_From_Jobs:bullets")
            text = resp.choices[0].message.content.strip()
            return re.sub(r"\s+", " ", text)
        # Any other error → keep original bullet
        return bullet

def maybe_paraphrase_experience_bullets(sections, job_description, use_gpt: bool, model: str):
    if not use_gpt:
        return sections
    allowed = build_allowed_vocab(sections, job_description)
    out = []
    for sec in sections:
        if sec.get("type") != "experience":
            out.append(sec); continue
        new_sec = dict(sec)
        new_bullets = []
        for c in sec.get("contributions") or []:
            desc = c.get("description", "")
            if not isinstance(desc, str) or not desc.strip():
                new_bullets.append(c); continue
            new_desc = paraphrase_bullet_strict(desc, allowed, model=model)
            new_bullets.append({**c, "description": new_desc})
        new_sec["contributions"] = new_bullets
        out.append(new_sec)
    return out

def generate_tailored_summary_strict(resume, job_description, use_gpt=False, model="gpt-5.0"):
    header = next((sec for sec in resume if sec.get("type") == "header"), {})
    title = header.get("title", "Experienced Professional")
    years_exp = header.get("years_experience")

    job_keywords = set(word.lower() for word in job_description.split() if len(word) > 2)
    all_resume_tags = set()
    for section in resume:
        for t in (section.get("tags") or []):
            all_resume_tags.add(t.lower())
    matched_skills = sorted(list(job_keywords & all_resume_tags))
    top_skills_str = ", ".join(matched_skills[:4]) if matched_skills else None

    achievement = _pick_best_achievement_overlap(resume, job_description)

    opening = f"{title} with {years_exp}+ years of proven expertise" if years_exp else f"{title} with proven expertise"
    offline = opening
    if top_skills_str:
        offline += f". Specializing in {top_skills_str}"
    if achievement:
        offline += f". Notable achievement: {achievement}"
    offline += "."

    if not use_gpt:
        return offline

    outline = _build_summary_outline(title, years_exp, top_skills_str, achievement)
    try:
        templated = _safe_paraphrase_with_placeholders(outline, model)
        final = _substitute_placeholders(templated, title, years_exp, top_skills_str, achievement)
        return final
    except Exception:
        return offline


# ==============================
# UI
# ==============================
st.set_page_config(page_title="Generate Resume from Job", layout="wide")
st.title("🧩 Generate Resume for a Specific Job")

# Load resume once
resume = load_resume(RESUME_PATH)
header = next((s for s in resume if s.get("type") == "header"), {})

# Load saved report defaults (so filters match Jobs Report page)
defaults = load_defaults()
fdefs = defaults.get("filters", {})

# Configurable save dir (defaults + ensure exists)
with st.sidebar:
    st.header("Options")
    db_path = st.text_input("Database file", value=defaults.get("db", "jobs.db"), key="gen_db_path")

    # configurable resume save directory
    resume_save_dir = st.text_input("Resume save folder", value=defaults.get("resume_save_dir", "kept_resumes"))
    resolved_dir = os.path.abspath(resume_save_dir)
    try:
        os.makedirs(resolved_dir, exist_ok=True)
        # quick write permission check
        test_path = os.path.join(resolved_dir, ".write_test.tmp")
        with open(test_path, "w", encoding="utf-8") as _f:
            _f.write("ok")
        os.remove(test_path)
        st.caption(f"Save directory: {resolved_dir}")
        log(f"[save-dir-ok] {resolved_dir}")
    except Exception as e:
        st.error(f"Cannot create or write to folder: {resolved_dir}\n\n{e}")
        log(f"[save-dir-error] {resolved_dir} :: {e}")

    threshold = st.slider("Min job score to show", 0.0, 1.0, float(defaults.get("min_score", 0.25)), 0.01, key="gen_threshold")
    limit = st.slider("Max jobs listed", 5, 200, int(defaults.get("top_count", 50)), key="gen_limit")

    st.markdown("---")
    st.subheader("Tailoring Options")
    ordering_mode = st.radio("Experience Ordering", ("Relevancy First", "Chronological", "Hybrid"), index=1, key="gen_order")
    top_n_hybrid = 3
    if ordering_mode == "Hybrid":
        top_n_hybrid = st.slider("Top relevant items before chronological", 1, 10, 3, key="gen_topN")

    use_embeddings = st.checkbox("Use semantic matching (embeddings)", value=True, key="gen_use_embed")
    embed_model = _model_selectbox(
        "Embeddings model",
        group="embeddings",
        key="gen_embed_model",
        disabled=not use_embeddings,
    )

    st.markdown("---")
    st.subheader("Rephrasing (no new facts)")
    gpt_paraphrase_bullets = st.checkbox("Paraphrase bullets to match JD wording (GPT)", value=True, key="gen_gp_bul")
    gpt_model = _model_selectbox(
        "GPT model for rephrasing",
        group="chat",
        key="gen_gpt_model",
        disabled=not gpt_paraphrase_bullets,
    )

    st.markdown("---")
    st.subheader("Tailored Summary (top of resume)")
    add_tailored_summary = st.checkbox("Include tailored summary section", value=True, key="gen_add_sum")
    gpt_paraphrase_summary = st.checkbox(
        "Use GPT to paraphrase summary (strict placeholders, no new facts)",
        value=True,
        key="gen_sum_gpt",
        disabled=not add_tailored_summary
    )
    gpt_summary_model = _model_selectbox(
        "GPT model for summary",
        group="chat",
        key="gen_sum_model",
        disabled=not (add_tailored_summary and gpt_paraphrase_summary),
    )

    st.caption("Filters below are loaded from report_defaults.yaml and applied here.")
    st.markdown("---")
    st.write("**Active location filter:**",
             fdefs.get("allowed_locations", ["USA","VA","North America","Remote","Worldwide"]))

# Persist save dir in session so button callbacks see it
st.session_state["resume_save_dir"] = resolved_dir

# Fetch candidates ≥ threshold (exclude not_suitable/submitted early if columns exist)
# ===== Fetch a generous pool, then filter & finally cap to the UI limit =====
requested = int(limit)
FETCH_CAP = max(requested * 6, 300)  # pull plenty so filters don't starve the list

all_candidates = query_top_matches(
    defaults.get("db", "jobs.db"),
    limit=FETCH_CAP,
    min_score=float(threshold),
    hide_stale_days=None,
) or []
raw_count = len(all_candidates)

# Remove already-processed jobs early
all_candidates = [
    c for c in all_candidates
    if int(c.get("not_suitable") or 0) == 0 and int(c.get("submitted") or 0) == 0
]
after_flags = len(all_candidates)

# Apply the SAME saved filters as Jobs Report (incl. location allowlist)
filtered = apply_saved_filters(all_candidates, fdefs)

# Belt & suspenders—don’t show flagged rows even if filters change later
filtered = [
    c for c in filtered
    if int(c.get("not_suitable") or 0) != 1 and int(c.get("submitted") or 0) != 1
]
after_filters = len(filtered)

# Keep score ordering, then cap to the user’s requested amount
filtered.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
DISPLAY_JOBS = filtered[:requested]

# UI + logs
st.subheader(
    f"Actionable jobs ≥ threshold — showing {len(DISPLAY_JOBS)} of {after_filters} "
    f"(from {raw_count} pulled)"
)
log(f"[candidates] pulled={raw_count} after_flags={after_flags} "
    f"after_filters={after_filters} requested={requested} shown={len(DISPLAY_JOBS)}")

# Jinja2 template (once)
env = Environment(loader=FileSystemLoader("."))
template = env.get_template(TEMPLATE_FILE)

# Helper: unique file name in configured dir
def unique_path(stem: str, ext: str) -> str:
    base = os.path.join(st.session_state["resume_save_dir"], f"{stem}.{ext}")
    if not os.path.exists(base):
        return base
    i = 1
    while True:
        cand = os.path.join(st.session_state["resume_save_dir"], f"{stem}_{i}.{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1

def path_to_file_url(path: str) -> str:
    abs_path = os.path.abspath(path).replace("\\", "/")
    return "file:///" + urllib.parse.quote(abs_path, safe="/:._-")

# Small wrappers against job_store actions (stay no-op if functions missing)
def _mark_not_suitable(db_path: str, job_id: str, reasons=None, note=None) -> bool:
    if callable(_mark_not_suitable_db):
        try:
            # Newer signature (keyword, job_id)
            return _mark_not_suitable_db(db_path=db_path, job_id=job_id, reasons=reasons or [], note=(note or ""))
        except TypeError:
            # Try alternate keyword (job_key)
            try:
                return _mark_not_suitable_db(db_path=db_path, job_key=job_id, reasons=reasons or [], note=(note or ""))
            except TypeError:
                # Oldest signature: (db_path, job_id)
                try:
                    return _mark_not_suitable_db(db_path, job_id)
                except Exception:
                    pass
        except Exception:
            pass
    # Fallback: at least set status so it leaves actionable list; stash note in user_notes if supported
    try:
        user_notes = (note or ", ".join([str(x) for x in (reasons or []) if x])) or "not suitable"
        set_job_status(db_path, job_id, status="not_suitable", user_notes=user_notes)
        return True
    except Exception:
        return False

def _mark_submitted(db_path: str, job_id: str) -> bool:
    if callable(_mark_submitted_db):
        now_iso = datetime.now(timezone.utc).isoformat()
        try:
            # Newer signature (keyword, job_id)
            return _mark_submitted_db(db_path=db_path, job_id=job_id, submitted=True, submitted_at=now_iso)
        except TypeError:
            # Try alternate keyword (job_key)
            try:
                return _mark_submitted_db(db_path=db_path, job_key=job_id, submitted=True, submitted_at=now_iso)
            except TypeError:
                # Oldest signature: (db_path, job_id)
                try:
                    return _mark_submitted_db(db_path, job_id)
                except Exception:
                    pass
        except Exception:
            pass
    # Fallback: set status so it leaves actionable list
    try:
        set_job_status(db_path, job_id, status="submitted")
        return True
    except Exception:
        return False

def _list_submitted(db_path: str, limit: int = 500) -> list[dict]:
    if callable(_query_submitted_db):
        try:
            return _query_submitted_db(db_path=db_path, limit=limit)
        except TypeError:
            # Very old signature: (db_path, limit)
            try:
                return _query_submitted_db(db_path, limit)
            except Exception:
                return []
        except Exception:
            return []
    return []

# Table with resume link / score if already saved
if DISPLAY_JOBS:
    def row_to_dict(c):
        resume_path = c.get("resume_path") or ""
        resume_url = path_to_file_url(resume_path) if resume_path else ""
        resume_score = c.get("resume_score")
        return {
            "job_id": c.get("job_id") or c.get("id", ""),
            "score": round(float(c.get("score", 0.0) or 0.0), 3),
            "resume_score": (round(float(resume_score), 3) if resume_score is not None else ""),
            "title": c.get("title", ""),
            "company": c.get("company", ""),
            "location": c.get("location", ""),
            "source": c.get("source", ""),
            "posted_at": c.get("posted_at", ""),
            "url": c.get("url", ""),
            "resume": resume_url,
        }

    df_list = [row_to_dict(c) for c in DISPLAY_JOBS]
    df = pd.DataFrame(df_list)
    df = make_arrow_friendly(df)

    if not df.empty:
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "url": st.column_config.LinkColumn("URL", display_text="Open"),
                "resume": st.column_config.LinkColumn("Resume", display_text="View"),
            },
            height=360,
        )

    st.markdown("---")
    st.markdown("### Generate / Keep")

    # Per-row UI
    for idx, c in enumerate(DISPLAY_JOBS):
        job_id = c.get("job_id") or c.get("id") or ""
        title = c.get("title", "")
        company = c.get("company", "")
        score = round(float(c.get("score", 0.0) or 0.0), 3)
        url = c.get("url", "")
        jd_text = c.get("description", "") or ""

        row_key = _clean_slug(f"{job_id or url or title}_{idx}") or f"row_{idx}"
        cache = st.session_state.gen_cache.get(row_key)  # None or dict

        with st.container():
            st.write(f"**{title} — {company}**  |  score: `{score:.3f}`  |  [open]({url})")
            colA, colB, colC, colD = st.columns([1,1,1,2])
            with colA:
                gen_btn = st.button("Generate Resume", key=f"gen_{row_key}")
            with colB:
                preview_exp = st.checkbox("Preview", value=True, key=f"prev_{row_key}")

            # Not suitable → reason + note; separate mini-form (no 'continue' here)
            with colC:
                with st.expander("Not suitable", expanded=False):
                    ns_reason = st.selectbox(
                        "Reason",
                        [
                            "Seniority mismatch",
                            "Tech stack mismatch",
                            "Location/onsite requirement",
                            "Comp/level misaligned",
                            "Domain mismatch",
                            "Job function mismatch",
                            "Security clearance/eligibility",
                            "Contract only",
                            "Company fit",
                            "Spam/duplicate",
                            "Not interested",
                            "Job posting age",
                            "Job no longer open",
                            "Other",
                        ],
                        key=f"ns_reason_{row_key}",
                    )
                    ns_note = st.text_input(
                        "Notes (optional)",
                        key=f"ns_note_{row_key}",
                        placeholder="Add any quick context…",
                    )
                    ns_save = st.button("Save not suitable", key=f"ns_save_{row_key}")

            with colD:
                sub_btn = st.button("Mark Submitted", key=f"sub_{row_key}")

            # Handle Not suitable
            if ns_save:
                reasons = [ns_reason] + ([ns_note] if ns_note.strip() else [])
                ok = _mark_not_suitable(db_path, job_id, reasons=reasons, note=ns_note.strip() or ns_reason)
                log(f"[not-suitable] job={job_id} ok={ok} reasons={reasons}")
                if ok:
                    st.success("Marked as not suitable.")
                    st.rerun()
                else:
                    st.error("Could not mark as not suitable. Ensure job_store.mark_not_suitable() exists.")

            # Handle Submitted
            if sub_btn:
                ok = _mark_submitted(db_path, job_id)
                log(f"[submitted] job={job_id} ok={ok}")
                if ok:
                    st.success("Marked as submitted.")
                    st.rerun()
                else:
                    st.error("Could not mark as submitted. Ensure job_store.mark_submitted() exists.")

            # Generate → cache it
            if gen_btn:
                log(f"[generate-click] job_id={job_id} company={company} key={row_key}")
                try:
                    tailored, scores_map = generate_tailored_resume(
                        resume,
                        jd_text,
                        top_n=(3 if st.session_state.get("gen_order") == "Hybrid" else None),
                        use_embeddings=st.session_state.get("gen_use_embed", False),
                        ordering=st.session_state.get("gen_order","Relevancy First").lower(),
                        embedding_model=(st.session_state.get("gen_embed_model") or "text-embedding-3-small"),
                    )
                except Exception as e:
                    st.error(f"Tailoring failed: {e}")
                    log("Tailoring failed:\n" + traceback.format_exc())
                    continue

                # Optional tailored summary (strict)
                tailored_summary = None
                try:
                    if st.session_state.get("gen_add_sum", True):
                        tailored_summary = generate_tailored_summary_strict(
                            resume,
                            jd_text,
                            use_gpt=st.session_state.get("gen_sum_gpt", False),
                            model=st.session_state.get("gen_sum_model") or "gpt-5.0"
                        )
                except Exception as e:
                    st.warning(f"Summary generation failed, omitting summary: {e}")
                    log("Summary failed:\n" + traceback.format_exc())

                # Optional GPT paraphrase of bullets with strict guardrails
                try:
                    if st.session_state.get("gen_gp_bul", False):
                        tailored = maybe_paraphrase_experience_bullets(
                            tailored, jd_text, use_gpt=True, model=st.session_state.get("gen_gpt_model") or "gpt-5.0"
                        )
                except Exception as e:
                    st.warning(f"Bullet paraphrase failed (using unmodified bullets): {e}")
                    log("Paraphrase failed:\n" + traceback.format_exc())

                # Render clean HTML
                try:
                    html = template.render(header=header, tailored_summary=tailored_summary, resume=tailored)
                except Exception as e:
                    st.error(f"Template render failed: {e}")
                    log("Template render failed:\n" + traceback.format_exc())
                    continue

                # Score generated resume vs JD (TF-IDF)
                try:
                    gen_text = html_to_text(html)
                    out_score = tfidf_score(gen_text, jd_text)
                except Exception as e:
                    out_score = 0.0
                    st.warning(f"Scoring failed, setting score=0.0: {e}")
                    log("Scoring failed:\n" + traceback.format_exc())

                # Base name for files — prefer the job record's company if present
                company_slug = _clean_slug(company or "")
                if not company_slug:
                    extracted = extract_company_name(jd_text)
                    company_slug = _clean_slug(extracted) if extracted else ""
                base_stem = f"{company_slug}_full_resume" if company_slug else "full_resume"

                # Try to build PDF
                pdf_bytes = None
                try:
                    from weasyprint import HTML as WPHTML
                    buf = BytesIO()
                    WPHTML(string=html).write_pdf(buf)
                    pdf_bytes = buf.getvalue()
                except Exception:
                    log("WeasyPrint failed:\n" + traceback.format_exc())

                # Cache it for Keep/Preview on next reruns
                st.session_state.gen_cache[row_key] = {
                    "html": html,
                    "pdf": pdf_bytes,
                    "base_stem": base_stem,
                    "job_id": job_id,
                    "out_score": float(out_score),
                }
                cache = st.session_state.gen_cache[row_key]
                st.success(f"Generated. Similarity score vs JD: **{out_score:.3f}**")

            # If we have cached output, show actions OUTSIDE the generate branch
            if cache:
                html = cache["html"]
                pdf_bytes = cache["pdf"]
                base_stem = cache["base_stem"]
                out_score = cache["out_score"]

                # Downloads
                st.download_button(
                    "Download HTML",
                    data=html,
                    file_name=f"{base_stem}.html",
                    mime="text/html",
                    key=f"dl_html_{row_key}"
                )
                if pdf_bytes:
                    st.download_button(
                        "Download PDF",
                        data=pdf_bytes,
                        file_name=f"{base_stem}.pdf",
                        mime="application/pdf",
                        key=f"dl_pdf_{row_key}"
                    )
                else:
                    st.info("Install WeasyPrint for PDF (`pip install weasyprint`).")

                # Keep Resume (now works across reruns)
                if st.button("Keep Resume", key=f"keep_{row_key}"):
                    save_dir = st.session_state.get("resume_save_dir") or os.path.abspath(defaults.get("resume_save_dir", "kept_resumes"))
                    try:
                        os.makedirs(save_dir, exist_ok=True)
                        log(f"[keep-click] dir={save_dir} key={row_key}")
                    except Exception as e:
                        st.error(f"Cannot create save folder: {save_dir}\n\n{e}")
                        log("Create dir failed:\n" + traceback.format_exc())
                        st.stop()

                    # Save HTML
                    try:
                        html_path = unique_path(base_stem, "html")
                        with open(html_path, "w", encoding="utf-8") as f:
                            f.write(html)
                        if not os.path.exists(html_path):
                            raise RuntimeError(f"HTML save did not create file: {html_path}")
                        log(f"[save-html] {html_path}")
                    except Exception as e:
                        st.error(f"Failed saving HTML to {save_dir}\n\n{e}")
                        log("Save HTML failed:\n" + traceback.format_exc())
                        st.stop()

                    # Save PDF if available; otherwise keep HTML path
                    saved_path = html_path
                    if pdf_bytes:
                        try:
                            pdf_path = unique_path(base_stem, "pdf")
                            with open(pdf_path, "wb") as f:
                                f.write(pdf_bytes)
                            if not os.path.exists(pdf_path):
                                raise RuntimeError(f"PDF save did not create file: {pdf_path}")
                            saved_path = pdf_path
                            log(f"[save-pdf] {pdf_path}")
                        except Exception as e:
                            st.warning(f"PDF save failed, keeping HTML only:\n\n{e}")
                            log("Save PDF failed:\n" + traceback.format_exc())

                    # Update DB with path + score (tries job_id, then id)
                    try:
                        ok = set_job_resume(
                            db_path=db_path,
                            job_id=(cache.get("job_id") or job_id),
                            resume_path=saved_path,
                            resume_score=float(out_score),
                        )
                        if ok:
                            st.success(f"Saved → {saved_path}\nScore stored: {out_score:.3f}")
                            log(f"[db-update] job_id={(cache.get('job_id') or job_id)} path={saved_path} score={out_score:.3f}")
                            try:
                                set_job_status(db_path, (cache.get("job_id") or job_id), status="saved")
                            except Exception:
                                pass
                            st.rerun()
                        else:
                            st.warning("Saved files, but DB update returned 0 rows (job not found by job_id or id).")
                            log("[db-update] 0 rows updated")
                    except Exception as e:
                        st.error(f"Saved files, but DB update failed:\n\n{e}")
                        log("DB update failed:\n" + traceback.format_exc())

                # Preview
                if preview_exp:
                    st.markdown("**Preview (HTML render):**")
                    st.components.v1.html(html, height=700, scrolling=True)

# ===== Submitted list (bottom) =====
st.markdown("---")
st.subheader("📬 Submitted")
submitted_rows = _list_submitted(defaults.get("db", "jobs.db"), limit=500)
submitted_rows = apply_saved_filters(submitted_rows, fdefs)

def _row_sub(c):
    resume_path = c.get("resume_path") or ""
    resume_url = path_to_file_url(resume_path) if resume_path else ""
    rs = c.get("resume_score")
    return {
        "job_id": c.get("job_id") or c.get("id",""),
        "title": c.get("title",""),
        "company": c.get("company",""),
        "location": c.get("location",""),
        "source": c.get("source",""),
        "posted_at": c.get("posted_at",""),
        "submitted_at": c.get("submitted_at",""),
        "resume_score": (round(float(rs),3) if rs is not None else ""),
        "url": c.get("url",""),
        "resume": resume_url,
    }

if submitted_rows:
    df_sub = pd.DataFrame([_row_sub(r) for r in submitted_rows])
    df_sub = make_arrow_friendly(df_sub)
    st.dataframe(
        df_sub,
        use_container_width=True,
        column_config={
            "url": st.column_config.LinkColumn("URL", display_text="Open"),
            "resume": st.column_config.LinkColumn("Resume", display_text="View"),
        },
        height=360,
    )
else:
    st.info("No submitted jobs yet.")

# Debug block
with st.expander("Debug / Save Logs"):
    if st.session_state.gen_debug:
        st.code("\n".join(st.session_state.gen_debug[-400:]), language="text")
    else:
        st.write("No debug output yet.")
