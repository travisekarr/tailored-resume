"""
Shared resume/job-description parsing helpers.
Pure-stdlib utilities safe to import from Streamlit pages.
"""

from __future__ import annotations
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

# -------------------------
# Slug helpers
# -------------------------
def clean_slug(s: str, *, max_len: int = 80) -> str:
    """
    Safe slug for file names or identifiers.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "_", s or "").strip("_")
    return slug[:max_len] if slug else ""

def clean_for_filename(s: str, *, max_len: int = 80) -> str:
    return clean_slug(s, max_len=max_len)


# -------------------------
# JD parsing (company/title)
# -------------------------
def extract_company_name(jd: str) -> Optional[str]:
    """
    Best-effort company extractor from raw JD text.
    Heuristics:
      1) 'Company:' prefixed label
      2) 'About <Company>' header
      3) '<Company> is seeking|seeks|is hiring'
      4) 'at <Company>'
      5) Fallback: scan first lines for a proper-noun-ish line
    Returns a cleaned slug or None.
    """
    text = (jd or "").strip()
    if not text:
        return None

    m = re.search(r"(?im)^\s*company\s*[:\-]\s*(.+)$", text)
    if m:
        cand = re.split(r"[|•\-\(\)\[\]\n\r]", m.group(1).strip())[0].strip()
        slug = clean_slug(cand)
        if slug:
            return slug

    m = re.search(r"(?i)\babout\s+([A-Z][A-Za-z0-9&\.,\- ]{2,})", text)
    if m:
        cand = re.split(r"(?i)\s+(is|provides|offers|was|were|inc\.?|llc|ltd|plc|corp\.?)", m.group(1).strip())[0].strip()
        slug = clean_slug(cand)
        if slug:
            return slug

    m = re.search(r"(?i)\b([A-Z][A-Za-z0-9&\.,\- ]{2,}?)\s+(?:is\s+seeking|seeks|is\s+hiring)\b", text)
    if m:
        slug = clean_slug(m.group(1))
        if slug:
            return slug

    m = re.search(r"(?i)\bat\s+([A-Z][A-Za-z0-9&\.,\- ]{2,})(?:[\s,\.]|$)", text)
    if m:
        slug = clean_slug(m.group(1).strip())
        if slug:
            return slug

    # Fallback: scan first few lines for proper-noun-ish single line
    for ln in [ln.strip() for ln in text.splitlines() if ln.strip()][:6]:
        if re.match(r"(?i)^(we|our|the|you|role|position|responsibilities|requirements)\b", ln):
            continue
        if re.search(r"[A-Z][a-z]", ln) and len(ln) <= 60:
            slug = clean_slug(re.split(r"[|•\-–—:]", ln)[0].strip())
            if slug:
                return slug
    return None

def extract_role_title(jd: str) -> Optional[str]:
    """
    Extract a plausible role title string from the job description.
    """
    text = (jd or "").strip()
    if not text:
        return None

    m = re.search(r"(?im)^\s*(role|title)\s*[:\-]\s*(.+)$", text)
    if m:
        cand = re.split(r"[|•\-\(\)\[\]\n\r]", m.group(2).strip())[0].strip()
        return cand[:100] if cand else None

    m = re.search(r"(?i)\b(hiring|seek(?:ing)?|searching)\b.*?\b(for|as|a|an)\s+([A-Z][A-Za-z0-9\-/&\s]{2,})", text)
    if m:
        return re.sub(r"[\s\|•\-\(\)\[\]]+$", "", m.group(3)).strip()[:100]

    for ln in [ln.strip() for ln in text.splitlines() if ln.strip()][:5]:
        if len(ln) <= 80 and re.search(r"[A-Za-z]", ln) and not ln.lower().startswith(("about","company","role","position")):
            return re.sub(r"[\|•\-–—:]+.*$", "", ln).strip()[:100]
    return None


# -------------------------
# JD keyword builder + HTML highlighter
# -------------------------
DEFAULT_STOPWORDS: Set[str] = {
    "and","the","with","for","your","you","our","their","this","that",
    "skills","experience","years","team","work","ability","in","to","of",
    # HTML/tag-ish words to avoid breaking markup:
    "strong","em","span","div","class","style","script","http","https","href","mark"
}

def build_keywords(job_description: str, stopwords: Optional[Set[str]] = None, limit: int = 200) -> List[str]:
    """
    Extract distinct alphanumeric-ish tokens from JD, minus stopwords.
    Longest-first ordering to help avoid partial overlaps when highlighting.
    """
    words = re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", job_description or "")
    sw = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    uniq = {
        w.strip().lower()
        for w in words
        if w.strip() and w.strip().lower() not in sw
    }
    return sorted(uniq, key=len, reverse=True)[:limit]

def _highlight_fragment(fragment: str, keywords: Iterable[str]) -> str:
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

def highlight_html(html: str, job_description: str) -> str:
    """
    Highlight matches in HTML using <mark>, skipping the header/contact area
    delimited by <!--NOHL_START--> ... <!--NOHL_END-->.
    """
    keywords = build_keywords(job_description)
    if not keywords:
        return html or ""
    parts = re.split(r"(<!--NOHL_START-->.*?<!--NOHL_END-->)", html or "", flags=re.DOTALL)
    for i, part in enumerate(parts):
        if part.startswith("<!--NOHL_START-->"):
            continue
        parts[i] = _highlight_fragment(part, keywords)
    return "".join(parts)


# -------------------------
# Achievement/term utilities
# -------------------------
def tokenize(text: str) -> Set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", text or "")}

def collect_candidate_achievements(resume: List[Dict[str, Any]]) -> List[str]:
    """
    Collect bullets from experience sections and achievements from the summary section.
    """
    candidates: List[str] = []
    for sec in resume or []:
        if sec.get("type") == "experience":
            for c in (sec.get("contributions") or []):
                desc = (c or {}).get("description")
                if isinstance(desc, str) and desc.strip():
                    candidates.append(desc.strip())
    summary_sec = next((sec for sec in resume or [] if sec.get("type") == "summary"), {})
    for a in summary_sec.get("achievements") or []:
        if isinstance(a, str) and a.strip():
            candidates.append(a.strip())
    # de-duplicate
    out, seen = [], set()
    for c in candidates:
        k = c.lower()
        if k not in seen:
            out.append(c); seen.add(k)
    return out

def pick_best_achievement_overlap(resume: List[Dict[str, Any]], job_description: str, *, min_score: float = 0.02) -> Optional[str]:
    """
    Pick the achievement sentence with the highest token overlap score vs JD.
    """
    jd_tokens = tokenize(job_description)
    cands = collect_candidate_achievements(resume)
    if not cands:
        return None
    def _score_sentence_vs_jd(sentence: str, jd_tok: Set[str]) -> float:
        if not sentence:
            return 0.0
        sent_tokens = tokenize(sentence)
        if not sent_tokens:
            return 0.0
        overlap = jd_tok & sent_tokens
        has_numbers = bool(re.search(r"(\d|%|\$)", sentence))
        return (len(overlap) / max(len(jd_tok), 1)) + (0.05 if has_numbers else 0.0)
    ranked = sorted(((c, _score_sentence_vs_jd(c, jd_tokens)) for c in cands), key=lambda x: x[1], reverse=True)
    best, score = ranked[0]
    return best if score >= min_score else None


# -------------------------
# Tags/vocab helpers (for future JD-aligned paraphrasing)
# -------------------------
def _normalize_term(t: str) -> Set[str]:
    t = (t or "").strip().lower()
    variants = {
        t,
        t.replace("_", " "), t.replace("-", " "), t.replace("/", " "),
        t.replace(".", ""), t.replace(".", " "),
    }
    if "postgresql" in t: variants.update({"postgres", "psql"})
    if "react.js" in t or "reactjs" in t: variants.update({"react"})
    if "node.js" in t or "nodejs" in t: variants.update({"node"})
    if "ci/cd" in t or "ci cd" in t: variants.update({"ci_cd", "cicd"})
    return {v for v in variants if v}

def flatten_resume_tags(resume: List[Dict[str, Any]]) -> Set[str]:
    """
    Flatten tags across sections; supports dict {hard, soft} or simple list.
    Also includes skills_used from experience bullets.
    """
    out: Set[str] = set()
    for sec in resume or []:
        tags = sec.get("tags")
        if isinstance(tags, dict):
            for k in ("hard","soft"):
                for t in (tags.get(k) or []):
                    out |= _normalize_term(str(t))
        elif isinstance(tags, list):
            for t in tags:
                out |= _normalize_term(str(t))
        if sec.get("type") == "experience":
            for c in sec.get("contributions") or []:
                for s in (c.get("skills_used") or []):
                    out |= _normalize_term(str(s))
    return out

def build_allowed_vocab(resume: List[Dict[str, Any]], job_description: str, limit: int = 200) -> List[str]:
    """
    Intersection of JD tokens and resume tags/skills; capped to a safe size.
    """
    jd_tokens = tokenize(job_description)
    resume_terms = flatten_resume_tags(resume)
    allowed = sorted(list(jd_tokens & resume_terms))
    return allowed[:limit]


# -------------------------
# Filename helper
# -------------------------
def base_resume_name_from_jd(job_description: str, *, default: str = "full_resume") -> str:
    """
    Build a base filename like '<company>_full_resume' or fallback to default.
    """
    company_slug = extract_company_name(job_description) or ""
    return f"{company_slug}_full_resume" if company_slug else default


__all__ = [
    "clean_slug",
    "clean_for_filename",
    "extract_company_name",
    "extract_role_title",
    "build_keywords",
    "highlight_html",
    "tokenize",
    "collect_candidate_achievements",
    "pick_best_achievement_overlap",
    "flatten_resume_tags",
    "build_allowed_vocab",
    "base_resume_name_from_jd",
]
