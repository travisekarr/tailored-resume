# pages/04_Generate_From_Jobs.py
import os
import re
import traceback
import urllib.parse
from io import BytesIO
from datetime import datetime, timezone
import pytz
from dotenv import load_dotenv
import streamlit as st
from resume_template_config import load_resume_templates, get_default_template_id, get_template_path_by_id
import yaml
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai_utils import chat_completion
from resume_semantic_scoring_engine import load_resume, generate_tailored_resume, enhance_experience_with_impact, generate_tailored_summary
from resume_ui_controls import render_storage_controls
from resume_ui_controls import select_preview_mode, show_preview_and_download

# Model config helpers
from models_config import load_models_cfg, ui_choices, ui_default, model_pricing

# Core job DB helpers
from job_store import (
    query_top_matches,
    set_job_status,
    set_job_resume,
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
# Load template config
_resume_templates = load_resume_templates()
_default_template_id = get_default_template_id(_resume_templates)
DEFAULTS_PATH = "report_defaults.yaml"
MODEL_SETTINGS_PATH = "api_models.yaml"   # <- unified model settings YAML

# Environment / OpenAI
load_dotenv()
TZ = os.getenv("TIMEZONE", "America/New_York")
DT_FMT = os.getenv("DATETIME_DISPLAY_FORMAT", "%Y-%m-%d:%I-%M %p")

def format_dt(val):
    if not val:
        return ""
    try:
        dt = pd.to_datetime(val, utc=True)
        tz = pytz.timezone(TZ)
        dt = dt.tz_convert(tz)
        return dt.strftime(DT_FMT)
    except Exception as e:
        print(f"[04_Generate_From_Jobs] format_dt error: {e}")
        return str(val)
# _openai_client = OpenAI()  # replaced by openai_utils



# ==============================
# Model selection using models_config.py and openai_pricing.yaml
# ==============================
_model_cfg = load_models_cfg()

def _model_selectbox(label: str, group: str, *, key: str, disabled: bool = False):
    # Map legacy group names to openai_pricing.yaml group names
    group_map = {
        "chat": "rephrasing",
        "embeddings": "embeddings",
        "summary": "summary",
    }
    actual_group = group_map.get(group, group)
    choices = ui_choices(_model_cfg, actual_group)
    default_id = ui_default(_model_cfg, actual_group)
    ids = [id for _, id in choices]
    labels = {id: display for display, id in choices}
    def _fmt(x): return labels.get(x, x)
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

## Usage logging is now handled in openai_utils.py; this helper is deprecated.


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

# === NEW: Boost relevance using JD DS/BI terms ===
def _jd_terms(text: str) -> set[str]:
    base = {w.lower() for w in re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", text or "")}
    synonyms = {
        "data science": {"data", "science", "datascience"},
        "machine learning": {"ml", "machine", "learning", "modeling", "modelling"},
        "deep learning": {"deep", "learning", "dl"},
        "nlp": {"nlp", "language", "text"},
        "analytics": {"analytics", "analysis", "analyst", "insights"},
        "business intelligence": {"bi", "business", "intelligence"},
        "power bi": {"power", "bi", "powerbi"},
        "tableau": {"tableau"},
        "looker": {"looker", "lookerstudio"},
        "sql": {"sql"},
        "python": {"python", "py", "pandas", "numpy", "sklearn", "scikit", "scikit-learn"},
        "r": {"r"},
        "spark": {"spark", "pyspark"},
        "databricks": {"databricks"},
        "snowflake": {"snowflake"},
        "redshift": {"redshift"},
        "bigquery": {"bigquery"},
        "dbt": {"dbt"},
        "etl": {"etl", "elt", "pipeline", "pipelines"},
        "warehouse": {"warehouse", "warehousing", "dwh"},
        "experiment": {"ab", "a/b", "a-b", "test", "testing", "experimentation"},
        "forecasting": {"forecast", "forecasting", "timeseries", "time", "series", "arima", "prophet"},
        "dashboard": {"dashboard", "dashboards", "report", "reports", "reporting"},
        "metrics": {"kpi", "kpis", "metric", "metrics"},
        "azure": {"azure", "synapse", "fabric"},
        "aws": {"aws", "sagemaker"},
        "gcp": {"gcp", "vertex", "vertexai"},
    }
    out = set(base)
    for _, vs in synonyms.items():
        out |= {v.lower() for v in vs}
    return out

def _score_against_terms(text: str, terms: set[str]) -> int:
    toks = {w.lower() for w in re.findall(r"[A-Za-z0-9+\-/#\.]{2,}", text or "")}
    return sum(1 for t in toks if t in terms)

def boost_resume_relevance(sections: list[dict], job_description: str, top_k_per_role: int = 3) -> list[dict]:
    terms = _jd_terms(job_description)
    boosted = []
    for sec in sections:
        if sec.get("type") != "experience":
            boosted.append(sec)
            continue
        new_sec = dict(sec)
        scored = []
        for c in (sec.get("contributions") or []):
            desc = (c.get("description") or "")
            skills = ", ".join(c.get("skills_used") or [])
            s = _score_against_terms(f"{desc} {skills}", terms)
            scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        new_sec["contributions"] = [c for _, c in scored]
        new_sec["_jd_score"] = sum(s for s, _ in scored[:max(1, min(top_k_per_role, len(scored)))])
        boosted.append(new_sec)
    exp = [s for s in boosted if s.get("type") == "experience"]
    other = [s for s in boosted if s.get("type") != "experience"]
    exp.sort(key=lambda s: float(s.get("_jd_score") or 0), reverse=True)
    return other[:1] + exp + other[1:]
# === /NEW ===

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

## Usage logging is now handled in openai_utils.py; this helper is deprecated.


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

# === NEW: Boost relevance using JD DS/BI terms ===
def _jd_terms(text: str) -> set[str]:
    base = {w.lower() for w in re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", text or "")}
    synonyms = {
        "data science": {"data", "science", "datascience"},
        "machine learning": {"ml", "machine", "learning", "modeling", "modelling"},
        "deep learning": {"deep", "learning", "dl"},
        "nlp": {"nlp", "language", "text"},
        "analytics": {"analytics", "analysis", "analyst", "insights"},
        "business intelligence": {"bi", "business", "intelligence"},
        "power bi": {"power", "bi", "powerbi"},
        "tableau": {"tableau"},
        "looker": {"looker", "lookerstudio"},
        "sql": {"sql"},
        "python": {"python", "py", "pandas", "numpy", "sklearn", "scikit", "scikit-learn"},
        "r": {"r"},
        "spark": {"spark", "pyspark"},
        "databricks": {"databricks"},
        "snowflake": {"snowflake"},
        "redshift": {"redshift"},
        "bigquery": {"bigquery"},
        "dbt": {"dbt"},
        "etl": {"etl", "elt", "pipeline", "pipelines"},
        "warehouse": {"warehouse", "warehousing", "dwh"},
        "experiment": {"ab", "a/b", "a-b", "test", "testing", "experimentation"},
        "forecasting": {"forecast", "forecasting", "timeseries", "time", "series", "arima", "prophet"},
        "dashboard": {"dashboard", "dashboards", "report", "reports", "reporting"},
        "metrics": {"kpi", "kpis", "metric", "metrics"},
        "azure": {"azure", "synapse", "fabric"},
        "aws": {"aws", "sagemaker"},
        "gcp": {"gcp", "vertex", "vertexai"},
    }
    out = set(base)
    for _, vs in synonyms.items():
        out |= {v.lower() for v in vs}
    return out

def _score_against_terms(text: str, terms: set[str]) -> int:
    toks = {w.lower() for w in re.findall(r"[A-Za-z0-9+\-/#\.]{2,}", text or "")}
    return sum(1 for t in toks if t in terms)

def boost_resume_relevance(sections: list[dict], job_description: str, top_k_per_role: int = 3) -> list[dict]:
    terms = _jd_terms(job_description)
    boosted = []
    for sec in sections:
        if sec.get("type") != "experience":
            boosted.append(sec)
            continue
        new_sec = dict(sec)
        scored = []
        for c in (sec.get("contributions") or []):
            desc = (c.get("description") or "")
            skills = ", ".join(c.get("skills_used") or [])
            s = _score_against_terms(f"{desc} {skills}", terms)
            scored.append((s, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        new_sec["contributions"] = [c for _, c in scored]
        new_sec["_jd_score"] = sum(s for s, _ in scored[:max(1, min(top_k_per_role, len(scored)))])
        boosted.append(new_sec)
    exp = [s for s in boosted if s.get("type") == "experience"]
    other = [s for s in boosted if s.get("type") != "experience"]
    exp.sort(key=lambda s: float(s.get("_jd_score") or 0), reverse=True)
    return other[:1] + exp + other[1:]
# === /NEW ===

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
    st.markdown("---")
    st.subheader("Resume Template")
    template_ids = [t["id"] for t in _resume_templates]
    template_labels = {t["id"]: t["display"] for t in _resume_templates}
    def _fmt_template(x): return template_labels.get(x, x)
    selected_template_id = st.selectbox(
        "Resume template",
        template_ids,
        index=template_ids.index(_default_template_id) if _default_template_id in template_ids else 0,
        key="gen_template_id",
        format_func=_fmt_template,
    )
    # Update preview dynamically when template changes
    if "gen_cache" in st.session_state:
        try:
            template_path = get_template_path_by_id(_resume_templates, selected_template_id)
            env = Environment(loader=FileSystemLoader("."))
            template = env.get_template(template_path)
            for row_key, cache in st.session_state.gen_cache.items():
                html = template.render(
                    header=st.session_state.get("header", {}),
                    tailored_summary=cache.get("tailored_summary", ""),
                    resume=cache.get("tailored", []),
                )
                # Render preview in the correct panel
                st.markdown("### 📄 Resume Preview")
                st.components.v1.html(html, height=700, scrolling=True)
        except Exception as e:
            print(f"[04_Generate_From_Jobs] Error rendering resume template in sidebar preview: {e}")
            st.error(f"Error rendering resume template: {e}")

    debug_usage_log = st.checkbox("Debug: print OpenAI usage records to console", value=False, key="gen_debug_usage")
    st.header("Options")
    storage = render_storage_controls(defaults=defaults, key_prefix="gen")
    db_path = storage["db_path"]
    resolved_dir = storage["resume_save_dir"]

    threshold = st.slider("Min job score to show", 0.0, 1.0, float(defaults.get("min_score", 0.25)), 0.01, key="gen_threshold")
    limit = st.slider("Max jobs listed", 5, 200, int(defaults.get("top_count", 50)), key="gen_limit")

    st.markdown("---")
    st.subheader("Tailoring Options")
    ordering_mode = st.radio("Experience Ordering", ("Relevancy First", "Chronological", "Hybrid"), index=1, key="gen_order")
    top_n_hybrid = 3
    if ordering_mode == "Hybrid":
        top_n_hybrid = st.slider("Top relevant items before chronological", 1, 10, 3, key="gen_topN")
    # NEW: JD booster controls
    boost_jd = st.checkbox("Boost JD alignment (prioritize DS/BI bullets)", value=True, key="gen_boost_jd")
    boost_topk = st.slider("Top bullets prioritized (per role)", 1, 5, 3, key="gen_boost_topk", disabled=not boost_jd)

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
        group="rephrasing",
        key="gen_gpt_model",
        disabled=not gpt_paraphrase_bullets,
    )

    st.markdown("---")
    st.subheader("Tailored Summary (top of resume)")
    # Always include tailored summary in this workflow
    add_tailored_summary = True
    gpt_paraphrase_summary = st.checkbox(
        "Use GPT to paraphrase summary (strict placeholders, no new facts)",
        value=True,
        key="gen_sum_gpt"
    )
    gpt_summary_model = _model_selectbox(
        "GPT model for summary",
        group="summary",
        key="gen_sum_model",
        disabled=not gpt_paraphrase_summary,
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
    db_path,
    limit=FETCH_CAP,
    min_score=float(threshold),
    hide_stale_days=None,
)
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
template_path = get_template_path_by_id(_resume_templates, st.session_state.get("gen_template_id", _default_template_id))
env = Environment(loader=FileSystemLoader("."))
template = env.get_template(template_path)

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
        # Calculate Age (days since posted_at)
        posted_raw = c.get("posted_at", "")
        age_days = ""
        if posted_raw:
            try:
                dt_posted = pd.to_datetime(posted_raw, utc=True)
                now = pd.Timestamp.now(tz=dt_posted.tz)
                age_days = int((now - dt_posted).days)
            except Exception:
                age_days = ""
        return {
            "job_id": c.get("job_id") or c.get("id", ""),
            "score": round(float(c.get("score", 0.0) or 0.0), 3),
            "resume_score": (round(float(resume_score), 3) if resume_score is not None else ""),
            "title": c.get("title", ""),
            "company": c.get("company", ""),
            "location": c.get("location", ""),
            "source": c.get("source", ""),
            "posted_at": format_dt(posted_raw),
            "Age": age_days,
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
            cols = st.columns([1,1,1,2])
            colA, colB, colC, colD = cols
            with colA:
                gen_btn = st.button("Generate Resume", key=f"gen_{row_key}")
            with colB:
                preview_exp = st.checkbox("Preview", value=True, key=f"prev_{row_key}")

            # Not suitable → reason + note; separate row for clarity
            if int(c.get("not_suitable") or 0) == 1:
                reason = c.get("user_notes") or "Not suitable"
                st.error(f"❌ {reason}", icon="🚫")
                continue

            # Resume link (if already saved)
            if cache:
                resume_path = cache.get("resume_path") or ""
                if resume_path and os.path.exists(resume_path):
                    resume_url = path_to_file_url(resume_path)
                    st.markdown(f"✅ Resume saved: [View]({resume_url})")
                else:
                    st.markdown("⚠️ Resume not found (file may have been moved or deleted)")

            # Preview section
            if preview_exp and cache:
                html = cache.get("html")
                if html:
                    st.markdown("### 📄 Resume Preview")
                    # Shared preview selector + preview + download
                    mode = select_preview_mode(key_prefix=f"jobprev_{row_key}", default="Formatted (HTML)")
                    base_name = cache.get("base_stem") or f"resume_{job_id}"
                    show_preview_and_download(
                         html=html,
                         base_name=base_name,
                         mode=mode,
                         key_prefix=f"jobprev_{row_key}",
                         display=True,
                         db_path=db_path,
                         job_id=job_id,
                     )
                else:
                    st.warning("No HTML preview available.")

            # Generate button action
            if gen_btn:
                with st.spinner(f"Generating resume for {title}..."):
                    try:
                        # --- 1. Tailor sections using the engine ---
                        tailored, scores_map = generate_tailored_resume(
                            resume,
                            jd_text,
                            top_n=top_n_hybrid,
                            use_embeddings=use_embeddings,
                            ordering=ordering_mode.lower(),  # 'relevancy'/'chronological'/'hybrid'
                            embedding_model=(embed_model if use_embeddings else "text-embedding-3-small"),
                        )

                        # Optional JD booster (offline prioritization)
                        if boost_jd:
                            tailored = boost_resume_relevance(tailored, jd_text, top_k_per_role=boost_topk)

                        # Optional GPT-based bullet rephrasing/impact
                        if gpt_paraphrase_bullets:
                            tailored = enhance_experience_with_impact(
                                tailored,
                                jd_text,
                                use_gpt=True,
                                model=gpt_model,
                                mark_generated=True,
                                bullets_per_role=1,
                            )

                        # --- 2. Tailored summary (always include) ---
                        tailored_summary = generate_tailored_summary(
                            resume,
                            jd_text,
                            use_gpt=bool(gpt_paraphrase_summary),
                            model=gpt_summary_model,
                            use_embeddings=bool(use_embeddings),
                            embedding_model=(embed_model if use_embeddings else "text-embedding-3-small"),
                        )

                        # --- 3. Render HTML with the selected template ---
                        try:
                            html = template.render(header=header, tailored_summary=tailored_summary, resume=tailored)
                        except Exception as e:
                            st.error(f"Error rendering resume template: {e}")
                            html = ""

                        # --- 4. Preview only (no auto-save); downloads are offered via preview controls ---

                        log(f"[generated] {job_id}")

                        # --- 5. Cache the result for preview ---
                        st.session_state.gen_cache[row_key] = {
                            "html": html,
                            "pdf": None,
                            "base_stem": f"resume_{job_id}",
                            "job_id": job_id,
                            "out_score": None,
                            "tailored_summary": tailored_summary,
                            "tailored": tailored,
                            "resume_path": "",
                        }

                        # --- 6. Do not auto-mark submitted; user reviews first via preview ---
                    except Exception as e:
                        st.error(f"Error generating resume: {e}")
                        log(f"[gen-error] {job_id} :: {e}")
                        traceback.print_exc()

    # Footer: links to other tools or documentation
    st.markdown("---")
    st.markdown("### Need more help?")
    st.markdown(
        "- Check out the [User Guide](https://example.com/user-guide) for detailed instructions."
        "- Visit our [FAQ](https://example.com/faq) for common questions."
        "- Contact support if you need further assistance."
    )