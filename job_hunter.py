# job_hunter.py
import os
import re
from dotenv import load_dotenv
import dateutil.parser
import pytz
_re = re
import json
import time
import math
import hashlib
import requests
import feedparser
from datetime import datetime, timezone
load_dotenv()
_TZ_NAME = os.getenv("TIMEZONE", "America/New_York")
_DT_FORMAT = os.getenv("DATETIME_DISPLAY_FORMAT", "%Y-%m-%d:%I-%M %p")
try:
    _TZ = pytz.timezone(_TZ_NAME)
except Exception:
    _TZ = pytz.timezone("America/New_York")
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from openai_utils import get_embedding

# Optional content extractors (graceful fallback)
try:
    import trafilatura
except Exception as e:
    print(f"Error importing trafilatura: {e}")
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except Exception as e:
    print(f"Error importing BeautifulSoup: {e}")
    BeautifulSoup = None

import html as _html
import math as _math
try:
    import tiktoken as _tiktoken  # optional, improves token estimates if installed
except Exception as e:
    print(f"Error importing tiktoken: {e}")
    _tiktoken = None

# -----------------------------
# Utilities & logging
# -----------------------------
def utcnow() -> str:
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(_TZ)
    return now_local.strftime(_DT_FORMAT)

def _clean_text(x: Optional[str]) -> str:
    return (x or "").strip()

def _hash_id(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

def _cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return (num / (da * db)) if da and db else 0.0

def _log(logs: list, msg: str, data: Optional[dict] = None):
    if "enrich.skip" not in msg:
        log_entry = {"ts": utcnow(), "msg": msg, "data": data or {}}
        if "error" in msg.lower():
            print(f"\033[91m{log_entry}\033[0m")
        else:
            print(log_entry)
    logs.append({"ts": utcnow(), "msg": msg, "data": data or {}})

def _getenv_or(cfg: dict, key: str, env: str) -> Optional[str]:
    return (cfg.get(key) if isinstance(cfg, dict) else None) or os.getenv(env)

# HTTP helpers (log every call)
def _http_get(url: str, logs: list, headers: Optional[dict] = None, params: Optional[dict] = None, timeout: int = 20):
    try:
        r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
        _log(logs, "http.get", {
            "url": r.url if hasattr(r, "url") else url,
            "status": getattr(r, "status_code", None),
            "reason": getattr(r, "reason", None),
            "len": len(getattr(r, "content", b"") or b"")
        })
        return r, None
    except Exception as e:
        _log(logs, "http.error", {"url": url, "error": str(e)})
        return None, str(e)

def _http_post_json(url: str, payload: dict, logs: list, headers: Optional[dict] = None, timeout: int = 20):
    try:
        h = {"Content-Type": "application/json"}
        if headers:
            h.update(headers)
        r = requests.post(url, headers=h, json=payload, timeout=timeout)
        _log(logs, "http.post", {
            "url": url,
            "status": getattr(r, "status_code", None),
            "reason": getattr(r, "reason", None),
            "req_len": len(json.dumps(payload)),
            "res_len": len(getattr(r, "content", b"") or b""),
        })
        return r, None
    except Exception as e:
        _log(logs, "http.error", {"url": url, "error": str(e)})
        return None, str(e)

def _stringify(x) -> str:
    """Safely turn any nested structure into plain text."""
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", "ignore")
        except Exception:
            return ""
    if isinstance(x, (int, float, bool)):
        return str(x)
    if isinstance(x, dict):
        return " ".join(_stringify(v) for v in x.values())
    if isinstance(x, (list, tuple, set)):
        return " ".join(_stringify(v) for v in x)
    return str(x)

def _is_remote(it: dict) -> bool:
    """Best-effort remote detector across noisy feeds."""
    val = str(it.get("remote", "")).strip().lower()
    if val in ("true", "1", "yes", "y"):  # explicit truthy
        return True
    # fallbacks
    loc  = (it.get("location") or "").lower()
    tit  = (it.get("title") or "").lower()
    desc = (it.get("description") or "").lower()
    needles = ("remote", "anywhere", "worldwide", "work from home")
    if any(n in loc for n in needles): return True
    if "remote" in tit: return True
    if "remote" in desc: return True
    return False

# -----------------------------
# Date/time normalization utility
# -----------------------------
def normalize_datetime(val) -> Optional[datetime]:
    """
    Convert any date value (int, float, ISO, RFC2822, etc.) to a timezone-aware datetime (UTC).
    Returns None if conversion fails.
    """
    if not val:
        return None
    try:
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.strip().isdigit()):
            return datetime.fromtimestamp(float(val), tz=timezone.utc)
        dt = dateutil.parser.parse(str(val))
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None

# -----------------------------
# YAML loader (with legacy support)
# -----------------------------
def load_sources_yaml(path: str, debug: bool = False) -> Tuple[Optional[dict], List[str]]:
    errs: List[str] = []
    if not path or not os.path.exists(path):
        errs.append(f"file_not_found: {path}")
        return None, errs

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        print(f"Error reading YAML file: {e}")
        errs.append(f"read_error: {e}")
        return None, errs

    try:
        data = yaml.safe_load(raw) or {}
    except Exception as e:
        print(f"Error parsing YAML: {e}")
        errs.append(f"yaml_parse_error: {e}")
        return None, errs

    # Legacy compatibility: allow "sources" root instead of "aggregators"
    if "aggregators" not in data and "sources" in data and isinstance(data["sources"], dict):
        data["aggregators"] = data["sources"]

    aggs = data.get("aggregators")
    if not isinstance(aggs, dict) or not aggs:
        errs.append("validation_error: top-level 'aggregators' missing or not a mapping")

    enabled_count = 0
    if isinstance(aggs, dict):
        for _, cfg in aggs.items():
            if isinstance(cfg, dict) and cfg.get("enabled") is True:
                enabled_count += 1
    if enabled_count == 0:
        errs.append("validation_error: no enabled aggregators found (set `enabled: true`)")

    return data, errs


# -----------------------------
# Resume text builder (TF-IDF)
# -----------------------------
def _build_resume_text(resume_yaml_path: str) -> str:
    import yaml as _y
    try:
        with open(resume_yaml_path, "r", encoding="utf-8") as f:
            sections = _y.safe_load(f)
    except Exception as e:
        print(f"Error reading resume YAML: {e}")
        return ""
    if not isinstance(sections, list):
        return ""

    def _sec_text(sec: dict) -> str:
        parts = []
        if isinstance(sec, dict):
            if sec.get("summary"):
                parts.append(str(sec.get("summary")))
            if "tags" in sec and isinstance(sec["tags"], list):
                parts.extend([str(t) for t in sec["tags"] if t])
            if sec.get("type") == "experience":
                for c in sec.get("contributions") or []:
                    if isinstance(c, dict):
                        if c.get("description"):
                            parts.append(str(c["description"]))
                        for k in ("skills_used", "impact_tags"):
                            if isinstance(c.get(k), list):
                                parts.extend([str(x) for x in c[k] if x])
            elif "entries" in sec and isinstance(sec["entries"], list):
                for entry in sec["entries"]:
                    if isinstance(entry, dict):
                        for v in entry.values():
                            if v:
                                parts.append(str(v))
        return " ".join(parts)

    buf = []
    for s in sections:
        try:
            buf.append(_sec_text(s))
        except Exception as e:
            print(f"Error processing resume section: {e}")
            continue
    return " ".join(buf)

# -----------------------------
# Scoring helpers
# -----------------------------
def _score_tfidf(resume_text: str, job_texts: list[str]) -> list[float]:
    """
    Stronger TF-IDF:
    - unigrams + bigrams
    - English stopwords removed
    - sublinear tf (log-scaled)
    - cap very-common terms
    """
    corpus = [resume_text] + job_texts
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
        sublinear_tf=True,
        max_df=0.95,
        min_df=1,
    )
    tfidf = vec.fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return [float(x) for x in sims]


def _tokenize_words(text: str) -> set:
    """Simple tokenizer that lowercases, strips non-alphanumerics, and returns a set of tokens."""
    if not text:
        return set()
    txt = re.sub(r"[^a-z0-9\s]", " ", str(text).lower())
    return set(w for w in txt.split() if len(w) > 1)


def _extract_resume_keywords(resume_yaml_path: str) -> set:
    """Extract a set of keyword tokens from a resume YAML file."""
    try:
        txt = _build_resume_text(resume_yaml_path)
        return _tokenize_words(txt)
    except Exception:
        return set()


def _score_embeddings(resume_text: str, texts: List[str], model: str, logs: list) -> List[float]:
    """Obtain embeddings for a list of texts and score each item against the resume (cosine similarity).
    Falls back to TF-IDF if embeddings fail.
    """
    try:
        embs = []
        for t in texts:
            e = get_embedding(t, model)
            # Normalise common return shapes
            vec = None
            if isinstance(e, dict):
                if "data" in e and isinstance(e["data"], (list, tuple)) and e["data"]:
                    vec = e["data"][0].get("embedding") or e["data"][0].get("vector")
                elif "embedding" in e:
                    vec = e.get("embedding")
                else:
                    # unknown dict shape; try to stringify
                    vec = e
            else:
                vec = e
            embs.append(vec)

        base = embs[0]
        scores = []
        for e in embs[1:]:
            try:
                if base and e:
                    scores.append(_cosine(base, e))
                else:
                    scores.append(0.0)
            except Exception:
                scores.append(0.0)
        return scores
    except Exception as ex:
        _log(logs, "score.embeddings.error", {"error": str(ex)})
        # fallback
        try:
            return _score_tfidf(resume_text, texts[1:])
        except Exception:
            return [0.0] * max(0, len(texts) - 1)

# -----------------------------
# Pay parsing & default filters
# -----------------------------
def _parse_pay_info(it: dict) -> dict:
    """Best-effort extraction of pay amounts and period (hourly vs annual).
    Returns {'min': float|None, 'max': float|None, 'period': 'hourly'|'annual'|None}
    """
    text = " ".join([str(it.get(k, "")) for k in ("title", "description", "location", "salary", "pay") if it.get(k)]).lower()
    # find dollar amounts like $120,000 or 120k or 120000
    amounts = []
    for m in re.finditer(r"\$?([0-9]{1,3}(?:[\,0-9]*)(?:\.[0-9]+)?)(k)?", text, flags=re.I):
        num = m.group(1).replace(",", "")
        mult = 1000.0 if (m.group(2) or "").lower() == "k" else 1.0
        try:
            amounts.append(float(num) * mult)
        except Exception:
            continue
    # also handle ranges like 50-60k or 50 to 60
    for m in re.finditer(r"([0-9]{1,3}(?:[\,0-9]*)(?:\.[0-9]+)?)\s*(k)?\s*(?:-|to)\s*([0-9]{1,3}(?:[\,0-9]*)(?:\.[0-9]+)?)(k)?", text, flags=re.I):
        a = float(m.group(1).replace(",", "")) * (1000.0 if (m.group(2) or "").lower() == "k" else 1.0)
        b = float(m.group(3).replace(",", "")) * (1000.0 if (m.group(4) or "").lower() == "k" else 1.0)
        amounts.extend([a, b])

    if not amounts:
        return {"min": None, "max": None, "period": None}

    max_amt = max(amounts)
    min_amt = min(amounts)

    # Decide period: look for hourly hints
    period = None
    if re.search(r"per\s+hour|/hr|hourly|hr\b", text):
        period = "hourly"
    elif re.search(r"per\s+year|/yr|yearly|annual|pa\b|per\s+annum", text):
        period = "annual"
    else:
        # Heuristic: if amounts < 200 (and likely >5), treat as hourly
        if max_amt and 5.0 < max_amt < 200.0:
            period = "hourly"
        else:
            period = "annual"

    return {"min": min_amt, "max": max_amt, "period": period}

def _is_archived_item(it: dict) -> bool:
    # generic archived detection across sources
    if isinstance(it.get("is_archived"), bool) and it.get("is_archived"):
        return True
    if isinstance(it.get("archived"), bool) and it.get("archived"):
        return True
    st = it.get("status")
    if st and str(st).strip().lower() in ("archived", "closed", "expired", "deleted"):
        return True
    txt = " ".join([str(it.get(k, "")) for k in ("title", "description") if it.get(k)]).lower()
    if any(x in txt for x in ("archived", "expired", "no longer open", "no longer available", "position closed", "closed")):
        return True
    return False

def _is_hybrid(it: dict) -> bool:
    """Detect hybrid postings (partial onsite + remote) using common hints."""
    title = (it.get("title") or "").lower()
    desc = (it.get("description") or "").lower()
    loc = (it.get("location") or "").lower()
    needles = ("hybrid", "on-site/remote", "on site/remote", "on-site or remote", "in office" ,"office+remote")
    if any(n in title for n in needles):
        return True
    if any(n in desc for n in needles):
        return True
    if any(n in loc for n in needles):
        return True
    return False

# -----------------------------
# Resume enrichment (optional)
# -----------------------------
def enrich_resume(resume_path: str, defaults: dict, logs: list) -> None:
    """
    Enrich the resume YAML with additional details:
    - Extract skills/tags/tech from the resume sections
    - Optionally fetch and enrich description fields from the web
    """
    enrich_opts = defaults.get("enrichment") or {}
    max_fetch = int(enrich_opts.get("max_fetch", 20))
    only_if_shorter_than = int(enrich_opts.get("only_if_shorter_than", 400))
    min_chars = int(enrich_opts.get("min_chars", 600))
    rate_limit_s = float(enrich_opts.get("rate_limit_s", 1.0))
    timeout = int(enrich_opts.get("timeout_s", 20))
    user_agent = enrich_opts.get("user_agent", "Mozilla/5.0 (job-hunter)")

    sources_pref = enrich_opts.get("sources")
    if sources_pref is None:
        sources_pref = ["remoteok", "weworkremotely", "hnrss", "greenhouse", "lever", "workable", "smartrecruiters", "recruitee", "personio"]

    logs = []
    _log(logs, "yaml.read", {"path": resume_path})
    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        print(f"Error reading resume YAML: {e}")
        return

    try:
        sections = yaml.safe_load(raw) or []
    except Exception as e:
        print(f"Error parsing YAML: {e}")
        return

    buf = []
    for s in sections:
        try:
            if isinstance(s, dict) and s.get("type") == "experience":
                for c in s.get("contributions") or []:
                    if isinstance(c, dict) and c.get("description"):
                        buf.append(str(c["description"]))
            elif isinstance(s, dict) and "entries" in s:
                for entry in s["entries"]:
                    if isinstance(entry, dict):
                        for v in entry.values():
                            if v:
                                buf.append(str(v))
        except Exception as e:
            print(f"Error processing resume section: {e}")
            continue

    text = " ".join(buf)
    kws = _extract_resume_keywords(resume_path)
    enriched = False

    # Enrichment loop: fetch from web if too short and not enough keywords
    for attempt in range(3):
        if len(text) >= min_chars and (kws & _tokenize_words(text)):
            _log(logs, "enrich.skip", {"reason": "long_enough_or_keywords_found"})
            break
        if enriched:
            time.sleep(rate_limit_s)
        _log(logs, "enrich.fetch", {"attempt": attempt + 1})
        desc = _fetch_with_trafilatura(resume_path, logs, timeout, user_agent)
        if not desc:
            desc = _fetch_with_bs4(resume_path, logs, timeout, user_agent)
        if desc and len(desc) >= min_chars:
            text = desc
            enriched = True
            _log(logs, "enrich.ok", {"chars": len(desc)})
        else:
            _log(logs, "enrich.fail", {"got_chars": len(desc or '')})
    # Save any new keywords found
    new_kws = _extract_resume_keywords(resume_path)
    if kws != new_kws:
        try:
            with open(resume_path, "a", encoding="utf-8") as f:
                if kws:
                    f.write("\n# Existing keywords\n")
                    for kw in sorted(kws):
                        f.write(f"- {kw}\n")
                if new_kws - kws:
                    f.write("\n# New keywords\n")
                    for kw in sorted(new_kws - kws):
                        f.write(f"- {kw}\n")
        except Exception as e:
            print(f"Error saving keywords to resume YAML: {e}")

# -----------------------------
# Aggregators
# -----------------------------
def fetch_remoteok(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    base = "https://remoteok.com/api"
    results: List[dict] = []
    queries = cfg.get("queries") or [""]
    limit = int(cfg.get("limit") or 200)
    for q in queries:
        url = base
        if q:
            url = f"{base}?search={requests.utils.quote(q)}"
        r, _ = _http_get(url, logs, headers={"User-Agent": "Mozilla/5.0"})
        if r and r.status_code == 200:
            try:
                data = r.json()
                # RemoteOK returns a list, sometimes with a metadata dict at index 0
                jobs = data if isinstance(data, list) else []
                # Remove metadata if present
                if jobs and isinstance(jobs[0], dict) and jobs[0].get("id") is None:
                    jobs = jobs[1:]
                for it in jobs:
                    # Some jobs may be missing fields; be robust
                    final_url = it.get("url") or it.get("link") or it.get("apply_url") or ""
                    results.append({
                        "id": it.get("id") or _hash_id(final_url or it.get("position", "")),
                        "title": it.get("position") or it.get("title"),
                        "company": it.get("company"),
                        "location": it.get("location") or it.get("region") or "",
                        "remote": True,
                        "url": final_url,
                        "source": "remoteok",
                        "posted_at": normalize_datetime(it.get("date") or it.get("epoch")),
                        "pulled_at": normalize_datetime(utcnow()),
                        "description": it.get("description") or "",
                    })
            except Exception as e:
                print(f"Error parsing remoteok data: {e}")
                _log(logs, "remoteok.parse_error", {"error": str(e)})
    return results[:limit]

def fetch_remotive(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    base = "https://remotive.com/api/remote-jobs"
    results: List[dict] = []
    queries = cfg.get("queries") or [""]
    categories = cfg.get("categories") or []
    limit = int(cfg.get("limit") or 200)
    for q in queries:
        base_url = f"{base}?search={requests.utils.quote(q)}"
        if categories:
            for c in categories:
                u = f"{base_url}&category={requests.utils.quote(c)}"
                r, _ = _http_get(u, logs)
                if r and r.status_code == 200:
                    try:
                        data = r.json()
                        exclude_archived = cfg.get("exclude_archived", True)
                        for it in data.get("jobs", []):
                            # Optionally skip archived/closed/expired jobs. Remotive doesn't always use a single
                            # canonical flag, so check common keys and textual hints.
                            if exclude_archived:
                                archived = False
                                # common boolean flags
                                if isinstance(it.get("is_archived"), bool) and it.get("is_archived"):
                                    archived = True
                                if isinstance(it.get("archived"), bool) and it.get("archived"):
                                    archived = True
                                # active/status flags
                                st_active = it.get("is_active")
                                if st_active is not None and not bool(st_active):
                                    archived = True
                                status = it.get("status")
                                if status and str(status).strip().lower() in ("archived", "closed", "expired", "deleted"):
                                    archived = True
                                # textual heuristics
                                title_l = (it.get("title") or "").lower()
                                desc_l = (it.get("description") or "").lower()
                                if any(k in title_l for k in ("archived", "expired", "closed", "no longer")):
                                    archived = True
                                if any(k in desc_l for k in ("archived", "expired", "closed", "no longer")):
                                    archived = True
                                if archived:
                                    _log(logs, "remotive.archived_skipped", {"id": it.get("id"), "title": it.get("title")})
                                    continue

                            results.append({
                                "id": it.get("id") or _hash_id(it.get("url","")),
                                "title": it.get("title"),
                                "company": it.get("company_name"),
                                "location": it.get("candidate_required_location") or it.get("job_type") or "",
                                "remote": True,
                                "url": it.get("url"),
                                "source": "remotive",
                                "posted_at": normalize_datetime(it.get("publication_date")),
                                "pulled_at": normalize_datetime(utcnow()),
                                "description": it.get("description") or "",
                            })
                    except Exception as e:
                        print(f"Error parsing remotive data: {e}")
                        _log(logs, "remotive.parse_error", {"error": str(e)})
        else:
            r, _ = _http_get(base_url, logs)
            if r and r.status_code == 200:
                try:
                    data = r.json()
                    for it in data.get("jobs", []):
                        results.append({
                            "id": it.get("id") or _hash_id(it.get("url","")),
                            "title": it.get("title"),
                            "company": it.get("company_name"),
                            "location": it.get("candidate_required_location") or it.get("job_type") or "",
                            "remote": True,
                            "url": it.get("url"),
                            "source": "remotive",
                            "posted_at": it.get("publication_date"),
                            "pulled_at": utcnow(),
                            "description": it.get("description") or "",
                        })
                except Exception as e:
                    print(f"Error parsing remotive data: {e}")
                    _log(logs, "remotive.parse_error", {"error": str(e)})
    return results[:limit]

def fetch_weworkremotely(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    feeds = cfg.get("feeds") or [
        "https://weworkremotely.com/remote-jobs.rss",
        "https://weworkremotely.com/categories/remote-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-data-science-jobs.rss",
        "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
        "https://weworkremotely.com/categories/remote-product-jobs.rss",
    ]
    results: List[dict] = []

    for u in feeds:
        # fetch with headers, then parse bytes (more reliable)
        r, _ = _http_get(
            u,
            logs,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
            },
            timeout=25,
        )
        if not r or r.status_code != 200:
            _log(logs, "rss.http_fail", {"url": u, "source": "weworkremotely", "status": getattr(r, "status_code", None)})
            continue
        try:
            f = feedparser.parse(r.content)
        except Exception as e:
            print(f"Error parsing WWR RSS: {e}")
            _log(logs, "rss.parse_error", {"url": u, "source": "weworkremotely", "error": str(e)})
            continue

        entries = getattr(f, "entries", []) or []
        _log(logs, "rss.entries", {"source": "weworkremotely", "url": u, "count": len(entries)})

        for e in entries:
            url = e.get("link") or ""
            title_raw = (e.get("title") or "").strip()
            desc = e.get("summary") or e.get("description") or ""
            posted = e.get("published") or e.get("updated") or None

            company = ""
            title = title_raw

            # Primary split: "Company: Title"
            if ":" in title_raw:
                left, right = title_raw.split(":", 1)
                if left.strip() and right.strip():
                    company, title = left.strip(), right.strip()
            # Optional fallback if some feeds use a semicolon
            elif ";" in title_raw:
                left, right = title_raw.split(";", 1)
                if left.strip() and right.strip():
                    company, title = left.strip(), right.strip()
            else:
                # Fallback: "... at Company"
                m = re.search(r"\bat\s+([^-–—\|]+)$", title_raw, flags=re.I)
                if m:
                    company = m.group(1).strip()
                    title = re.sub(r"\s+\bat\s+[^\-–—\|]+$", "", title_raw, flags=re.I).strip()

            results.append({
                "id": _hash_id(url or title_raw),
                "title": title,
                "company": company,
                "location": "",
                "remote": True,
                "url": url,
                "source": "weworkremotely",
                "posted_at": normalize_datetime(posted),
                "pulled_at": normalize_datetime(utcnow()),
                "description": desc,
            })
    return results

def fetch_hnrss(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    feed = cfg.get("feed") or "https://hnrss.org/jobs"
    # Patch fetch_rss to normalize posted_at and pulled_at
    raw = fetch_rss(feed, "hnrss", logs)
    for r in raw:
        r["posted_at"] = normalize_datetime(r.get("posted_at"))
        r["pulled_at"] = normalize_datetime(r.get("pulled_at"))
    return raw

def fetch_jobicy(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    # default public endpoint; allow override
    api = cfg.get("api") or "https://jobicy.com/api/v2/remote-jobs"
    params = {"count": int(cfg.get("count") or 200)}
    r, _ = _http_get(api, logs, params=params, headers={"User-Agent": "Mozilla/5.0"})
    out: List[dict] = []
    if r and r.status_code == 200:
        try:
            data = r.json()
            for it in data.get("jobs", []):
                out.append({
                    "id": str(it.get("id") or _hash_id(it.get("url",""))),
                    "title": it.get("jobTitle") or it.get("title"),
                    "company": it.get("companyName") or it.get("company"),
                    "location": it.get("jobGeo") or it.get("jobType") or "",
                    "remote": True,
                    "url": it.get("url") or it.get("jobUrl"),
                    "source": "jobicy",
                    "posted_at": normalize_datetime(it.get("pubDate") or it.get("date")),
                    "pulled_at": normalize_datetime(utcnow()),
                    "description": it.get("jobDescription") or it.get("description") or "",
                })
        except Exception as e:
            print(f"Error parsing jobicy data: {e}")
            _log(logs, "jobicy.parse_error", {"error": str(e)})
    return out

def fetch_arbeitnow(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    base = "https://arbeitnow.com/api/job-board-api"
    page = 1
    limit_pages = int(cfg.get("pages") or 2)
    out: List[dict] = []
    while page <= limit_pages:
        url = f"{base}?page={page}"
        r, _ = _http_get(url, logs)
        if not r or r.status_code != 200:
            break
        try:
            data = r.json()
            for it in data.get("data", []):
                out.append({
                    "id": it.get("slug") or _hash_id(it.get("url","")),
                    "title": it.get("title"),
                    "company": it.get("company_name") or it.get("company"),
                    "location": it.get("location") or "",
                    "remote": True,
                    "url": it.get("url"),
                    "source": "arbeitnow",
                    "posted_at": normalize_datetime(it.get("created_at") or it.get("published_at")),
                    "pulled_at": normalize_datetime(utcnow()),
                    "description": it.get("description") or "",
                })
        except Exception as e:
            print(f"Error parsing arbeitnow data: {e}")
            _log(logs, "arbeitnow.parse_error", {"error": str(e)})
        page += 1
    return out

def fetch_usajobs(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    key = _getenv_or(cfg, "api_key", "USAJOBS_API_KEY")
    if not key:
        _log(logs, "usajobs.no_key", {})
        return []
    user_agent = cfg.get("user_agent") or os.getenv("USAJOBS_USER_AGENT") or "job-hunter"
    base = "https://data.usajobs.gov/api/search"
    params = {
        "Keyword": cfg.get("keyword") or "software",
        "ResultsPerPage": int(cfg.get("results_per_page") or 100),
        "WhoMayApply": cfg.get("who") or "",  # e.g. "all"
    }
    r, _ = _http_get(base, logs, headers={
        "Host": "data.usajobs.gov",
        "User-Agent": user_agent,
        "Authorization-Key": key,
    }, params=params)
    out: List[dict] = []
    if r and r.status_code == 200:
        try:
            data = r.json()
            for it in (data.get("SearchResult", {}).get("SearchResultItems") or []):
                pos = it.get("MatchedObjectDescriptor", {})
                url = pos.get("PositionURI")
                out.append({
                    "id": str(pos.get("PositionID") or _hash_id(url or "")),
                    "title": pos.get("PositionTitle"),
                    "company": pos.get("OrganizationName"),
                    "location": ", ".join([l.get("LocationName","") for l in pos.get("PositionLocation",[])]),
                    "remote": False,
                    "url": url,
                    "source": "usajobs",
                    "posted_at": normalize_datetime(pos.get("PublicationStartDate")),
                    "pulled_at": normalize_datetime(utcnow()),
                    "description": pos.get("UserArea", {}).get("Details", {}).get("JobSummary",""),
                })
        except Exception as e:
            print(f"Error parsing usajobs data: {e}")
            _log(logs, "usajobs.parse_error", {"error": str(e)})
    return out

def fetch_adzuna(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    app_id = _getenv_or(cfg, "app_id", "ADZUNA_APP_ID")
    app_key = _getenv_or(cfg, "app_key", "ADZUNA_APP_KEY")
    if not app_id or not app_key:
        _log(logs, "adzuna.no_keys", {})
        return []
    country = (cfg.get("country") or "us").lower()
    what = cfg.get("what") or "software"
    where = cfg.get("where") or ""
    page_count = int(cfg.get("pages") or 1)
    per_page = int(cfg.get("per_page") or 50)
    out: List[dict] = []
    for page in range(1, page_count + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"
        params = {
            "app_id": app_id, "app_key": app_key,
            "what": what, "where": where, "results_per_page": per_page, "content-type": "application/json"
        }
        r, _ = _http_get(url, logs, params=params)
        if not r or r.status_code != 200:
            break
        try:
            data = r.json()
            for it in data.get("results", []):
                out.append({
                    "id": str(it.get("id") or _hash_id(it.get("redirect_url",""))),
                    "title": it.get("title"),
                    "company": (it.get("company") or {}).get("display_name"),
                    "location": (it.get("location") or {}).get("display_name",""),
                    "remote": False,
                    "url": it.get("redirect_url"),
                    "source": "adzuna",
                    "posted_at": normalize_datetime(it.get("created")),
                    "pulled_at": normalize_datetime(utcnow()),
                    "description": it.get("description") or "",
                })
        except Exception as e:
            print(f"Error parsing adzuna data: {e}")
            _log(logs, "adzuna.parse_error", {"error": str(e)})
    return out

def fetch_jooble(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    key = _getenv_or(cfg, "api_key", "JOOBLE_API_KEY")
    if not key:
        _log(logs, "jooble.no_key", {})
        return []
    url = f"https://jooble.org/api/{key}"
    payload = {
        "keywords": cfg.get("keywords") or "software",
        "location": cfg.get("location") or "",
        "radius": int(cfg.get("radius") or 0),
        "page": 1,
        "size": int(cfg.get("size") or 50),
    }
    r, _ = _http_post_json(url, payload, logs)
    out: List[dict] = []
    if r and r.status_code == 200:
        try:
            data = r.json()
            for it in data.get("jobs", []):
                out.append({
                    "id": str(it.get("id") or _hash_id(it.get("link",""))),
                    "title": it.get("title"),
                    "company": it.get("company"),
                    "location": it.get("location"),
                    "remote": "remote" in (it.get("type","").lower()),
                    "url": it.get("link"),
                    "source": "jooble",
                    "posted_at": normalize_datetime(it.get("updated")),
                    "pulled_at": normalize_datetime(utcnow()),
                    "description": it.get("snippet") or "",
                })
        except Exception as e:
            print(f"Error parsing jooble data: {e}")
            _log(logs, "jooble.parse_error", {"error": str(e)})
    return out

def fetch_themuse(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    base = "https://www.themuse.com/api/public/jobs"
    category = [cfg.get("category") or "Software Engineering"]
    pages = int(cfg.get("pages") or 2)
    out: List[dict] = []
    for cat in category:
        for p in range(1, pages + 1):
            params = {"page": p, "category": cat}
            r, _ = _http_get(base, logs, params=params)
            if not r or r.status_code != 200:
                break
            try:
                data = r.json()
                for it in data.get("results", []):
                    locs = ", ".join([l.get("name","") for l in it.get("locations",[])])
                    out.append({
                        "id": str(it.get("id") or _hash_id(it.get("refs",{}).get("landing_page",""))),
                        "title": it.get("name"),
                        "company": (it.get("company") or {}).get("name"),
                        "location": locs,
                        "remote": "remote" in locs.lower(),
                        "url": (it.get("refs") or {}).get("landing_page"),
                        "source": "themuse",
                        "posted_at": normalize_datetime(it.get("publication_date")),
                        "pulled_at": normalize_datetime(utcnow()),
                        "description": it.get("contents") or "",
                    })
            except Exception as e:
                print(f"Error parsing themuse data: {e}")
                _log(logs, "themuse.parse_error", {"error": str(e)})
    return out

def fetch_findwork(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    key = _getenv_or(cfg, "api_key", "FINDWORK_API_KEY")
    if not key:
        _log(logs, "findwork.no_key", {})
        return []
    base = "https://findwork.dev/api/jobs/"
    params = {"search": cfg.get("search") or "software", "limit": int(cfg.get("limit") or 100)}
    r, _ = _http_get(base, logs, headers={"Authorization": f"Token {key}"}, params=params)
    out: List[dict] = []
    if r and r.status_code == 200:
        try:
            data = r.json()
            for it in data.get("results", []):
                out.append({
                    "id": str(it.get("id") or _hash_id(it.get("url",""))),
                    "title": it.get("role") or it.get("title"),
                    "company": it.get("company_name"),
                    "location": it.get("location") or "",
                    "remote": bool(it.get("remote")),
                    "url": it.get("url"),
                    "source": "findwork",
                    "posted_at": normalize_datetime(it.get("date_posted")),
                    "pulled_at": normalize_datetime(utcnow()),
                    "description": it.get("text") or "",
                })
        except Exception as e:
            print(f"Error parsing findwork data: {e}")
            _log(logs, "findwork.parse_error", {"error": str(e)})
    return out

# ATS families (per-company lists)
def fetch_greenhouse(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    companies = cfg.get("companies") or []
    out: List[dict] = []
    for comp in companies:
        url = f"https://boards-api.greenhouse.io/v1/boards/{comp}/jobs"
        r, _ = _http_get(url, logs)
        if r and r.status_code == 200:
            try:
                data = r.json()
                for it in data.get("jobs", []):
                    out.append({
                        "id": str(it.get("id")),
                        "title": it.get("title"),
                        "company": comp,
                        "location": (it.get("location") or {}).get("name",""),
                        "remote": "remote" in ((it.get("location") or {}).get("name","" ).lower()),
                        "url": it.get("absolute_url"),
                        "source": "greenhouse",
                        "posted_at": normalize_datetime(it.get("updated_at") or it.get("created_at")),
                        "pulled_at": normalize_datetime(utcnow()),
                        "description": "",  # can be enriched later
                    })
            except Exception as e:
                print(f"Error parsing greenhouse data for {comp}: {e}")
                _log(logs, "greenhouse.parse_error", {"company": comp, "error": str(e)})
    return out

def fetch_lever(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    companies = cfg.get("companies") or []
    out: List[dict] = []
    for comp in companies:
        url = f"https://api.lever.co/v0/postings/{comp}?mode=json"
        r, _ = _http_get(url, logs)
        if r and r.status_code == 200:
            try:
                data = r.json()
                for it in data:
                    out.append({
                        "id": it.get("id") or _hash_id(it.get("hostedUrl","")),
                        "title": it.get("text") or it.get("title"),
                        "company": comp,
                        "location": (it.get("categories") or {}).get("location",""),
                        "remote": "remote" in ((it.get("categories") or {}).get("location","" ).lower()),
                        "url": it.get("hostedUrl"),
                        "source": "lever",
                        "posted_at": normalize_datetime(it.get("createdAt")),
                        "pulled_at": normalize_datetime(utcnow()),
                        "description": (it.get("descriptionPlain") or ""),
                    })
            except Exception as e:
                print(f"Error parsing lever data for {comp}: {e}")
                _log(logs, "lever.parse_error", {"company": comp, "error": str(e)})
    return out

def fetch_workable(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    accounts = cfg.get("accounts") or [] # subdomains
    out: List[dict] = []
    for acc in accounts:
        url = f"https://apply.workable.com/api/v3/accounts/{acc}/jobs"
        params = {"state": "published"}
        r, _ = _http_get(url, logs, params=params)
        if r and r.status_code == 200:
            try:
                data = r.json()
                for it in data.get("results", []):
                    loc = it.get("location") or {}
                    loc_str = ", ".join(filter(None, [loc.get("city"), loc.get("region"), loc.get("country_code")]))
                    out.append({
                        "id": it.get("id") or it.get("shortcode"),
                        "title": it.get("title"),
                        "company": acc,
                        "location": loc_str,
                        "remote": bool(loc.get("remote")),
                        "url": it.get("url"),
                        "source": "workable",
                        "posted_at": normalize_datetime(it.get("published_on")),
                        "pulled_at": normalize_datetime(utcnow()),
                        "description": "",  # can enrich
                    })
            except Exception as e:
                print(f"Error parsing workable data for {acc}: {e}")
                _log(logs, "workable.parse_error", {"account": acc, "error": str(e)})
    return out

def fetch_smartrecruiters(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    companies = cfg.get("companies") or []
    out: List[dict] = []
    for comp in companies:
        url = f"https://api.smartrecruiters.com/v1/companies/{comp}/postings"
        params = {"limit": int(cfg.get("limit") or 100)}
        r, _ = _http_get(url, logs, params=params)
        if r and r.status_code == 200:
            try:
                data = r.json()
                for it in data.get("content", []):
                    out.append({
                        "id": it.get("id"),
                        "title": (it.get("name") or {}).get("value") or it.get("name"),
                        "company": comp,
                        "location": (it.get("location") or {}).get("city",""),
                        "remote": False,
                        "url": (it.get("ref") or {}).get("jobAd") or it.get("applyUrl"),
                        "source": "smartrecruiters",
                        "posted_at": normalize_datetime(it.get("releasedDate")),
                        "pulled_at": normalize_datetime(utcnow()),
                        "description": "",
                    })
            except Exception as e:
                print(f"Error parsing smartrecruiters data for {comp}: {e}")
                _log(logs, "smartrecruiters.parse_error", {"company": comp, "error": str(e)})
    return out

def fetch_recruitee(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    companies = cfg.get("companies") or []
    out: List[dict] = []
    for comp in companies:
        url = f"https://{comp}.recruitee.com/api/offers/"
        r, _ = _http_get(url, logs)
        if r and r.status_code == 200:
            try:
                data = r.json()
                for it in data.get("offers", []):
                    out.append({
                        "id": str(it.get("id")),
                        "title": it.get("title"),
                        "company": comp,
                        "location": (it.get("location") or {}).get("city",""),
                        "remote": bool((it.get("remote") or {}).get("status") == "remote"),
                        "url": f"https://{comp}.recruitee.com/o/{it.get('slug')}",
                        "source": "recruitee",
                        "posted_at": normalize_datetime(it.get("created_at")),
                        "pulled_at": normalize_datetime(utcnow()),
                        "description": it.get("description") or "",
                    })
            except Exception as e:
                print(f"Error parsing recruitee data for {comp}: {e}")
                _log(logs, "recruitee.parse_error", {"company": comp, "error": str(e)})
    return out

def fetch_personio(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    domains = cfg.get("domains") or []  # like "acme.jobs.personio.de"
    out: List[dict] = []
    for dom in domains:
        url = f"https://{dom}/search.json"
        r, _ = _http_get(url, logs, headers={"User-Agent": "Mozilla/5.0"})
        if r and r.status_code == 200:
            try:
                data = r.json()
                for it in data.get("jobs", []):
                    url_apply = it.get("url") or f"https://{dom}{it.get('url','')}"
                    loc = it.get("office") or {}
                    loc_str = ", ".join(filter(None, [loc.get("city"), loc.get("country"), loc.get("name")]))
                    out.append({
                        "id": str(it.get("id") or _hash_id(url_apply)),
                        "title": it.get("name") or it.get("title"),
                        "company": dom.split(".")[0],
                        "location": loc_str,
                        "remote": "remote" in (loc_str.lower()),
                        "url": url_apply,
                        "source": "personio",
                        "posted_at": normalize_datetime(it.get("created_at")),
                        "pulled_at": normalize_datetime(utcnow()),
                        "description": it.get("description") or "",
                    })
            except Exception as e:
                print(f"Error parsing personio data for {dom}: {e}")
                _log(logs, "personio.parse_error", {"domain": dom, "error": str(e)})
    return out

def fetch_generic_rss(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    feeds = cfg.get("feeds") or []
    out: List[dict] = []
    for u in feeds:
        out.extend(fetch_rss(u, "rss", logs))
    return out

def fetch_rss(url: str, tag: str, logs: list) -> List[dict]:
    """Generic RSS fetcher that returns a list of normalized job dicts."""
    out: List[dict] = []
    try:
        r, err = _http_get(url, logs, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        if not r or r.status_code != 200:
            _log(logs, "rss.http_fail", {"url": url, "tag": tag, "status": getattr(r, "status_code", None)})
            return out
        f = feedparser.parse(r.content)
        entries = getattr(f, "entries", []) or []
        for e in entries:
            url_entry = e.get("link") or ""
            title = (e.get("title") or "").strip()
            desc = e.get("summary") or e.get("description") or ""
            posted = e.get("published") or e.get("updated") or None
            out.append({
                "id": _hash_id(url_entry or title),
                "title": title,
                "company": "",
                "location": "",
                "remote": "remote" in (desc or "").lower(),
                "url": url_entry,
                "source": tag,
                "posted_at": normalize_datetime(posted),
                "pulled_at": normalize_datetime(utcnow()),
                "description": desc,
            })
    except Exception as e:
        _log(logs, "rss.parse_error", {"url": url, "error": str(e)})
    return out

# -----------------------------
# Description enrichment (optional)
# -----------------------------
def _strip_tags_to_text(html: str) -> str:
    if not html:
        return ""
    if BeautifulSoup is None:
        return re.sub(r"<[^>]+>", " ", html)
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def _fetch_with_trafilatura(url: str, logs: list, timeout: int, user_agent: Optional[str]) -> Optional[str]:
    if trafilatura is None:
        return None
    try:
        _log(logs, "enrich.fetch.trafilatura", {"url": url})
        raw = trafilatura.fetch_url(url, timeout=timeout, user_agent=user_agent)
        if not raw:
            _log(logs, "enrich.trafilatura.empty", {"url": url})
            return None
        extracted = trafilatura.extract(raw, include_comments=False, include_tables=False, favor_recall=True)
        if extracted:
            return extracted.strip()
    except Exception as e:
        print(f"Error fetching with trafilatura: {e}")
        _log(logs, "enrich.trafilatura.error", {"url": url, "error": str(e)})
    return None

def _fetch_with_bs4(url: str, logs: list, timeout: int, user_agent: Optional[str]) -> Optional[str]:
    try:
        headers = {"User-Agent": user_agent or "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=timeout)
        _log(logs, "enrich.fetch.request", {"url": url, "status": r.status_code, "len": len(r.content)})
        if r.status_code != 200 or not r.text:
            return None
        return _strip_tags_to_text(r.text)
    except Exception as e:
        print(f"Error fetching with bs4: {e}")
        _log(logs, "enrich.request.error", {"url": url, "error": str(e)})
        return None

def _need_enrichment(item: dict, only_if_shorter_than: int, sources_pref: Optional[List[str]]) -> bool:
    if sources_pref:
        if item.get("source") not in sources_pref:
            return False
    desc = item.get("description") or ""
    plain = _strip_tags_to_text(desc)
    return len(plain) < max(0, int(only_if_shorter_than))

def _enrich_items_descriptions(items: List[dict], defaults: dict, logs: list) -> None:
    enrich_opts = defaults.get("enrichment") or {}
    max_fetch = int(enrich_opts.get("max_fetch", 20))
    only_if_shorter_than = int(enrich_opts.get("only_if_shorter_than", 400))
    min_chars = int(enrich_opts.get("min_chars", 600))
    rate_limit_s = float(enrich_opts.get("rate_limit_s", 1.0))
    timeout = int(enrich_opts.get("timeout_s", 20))
    user_agent = enrich_opts.get("user_agent", "Mozilla/5.0 (job-hunter)")

    sources_pref = enrich_opts.get("sources")
    if sources_pref is None:
        sources_pref = ["remoteok", "weworkremotely", "hnrss", "greenhouse", "lever", "workable", "smartrecruiters", "recruitee", "personio"]

    fetched = 0
    for it in items:
        if fetched >= max_fetch:
            _log(logs, "enrich.limit_reached", {"max_fetch": max_fetch})
            break
        url = it.get("url")
        if not url:
            continue
        if not _need_enrichment(it, only_if_shorter_than, sources_pref):
            _log(logs, "enrich.skip", {"url": url, "reason": "long_enough_or_source_filtered"})
            continue

        time.sleep(rate_limit_s)
        text = _fetch_with_trafilatura(url, logs, timeout, user_agent)
        if not text:
            text = _fetch_with_bs4(url, logs, timeout, user_agent)

        if text and len(text) >= min_chars:
            it["description"] = text
            it["description_source"] = "enriched"
            _log(logs, "enrich.ok", {"url": url, "chars": len(text)})
            fetched += 1
        else:
            _log(logs, "enrich.fail", {"url": url, "got_chars": len(text or '')})

# -----------------------------
# Main orchestration
# -----------------------------
def hunt_jobs(
    resume_path: str,
    sources_yaml: str,
    use_embeddings: bool = False,
    embedding_model: str = "text-embedding-3-small",
    debug: bool = False,
    logger=None,
) -> Dict[str, Any]:
    logs: List[dict] = []
    _log(logs, "yaml.read", {"path": sources_yaml})
    cfg, errs = load_sources_yaml(sources_yaml, debug=debug)
    if errs:
        for e in errs:
            print(f"Error validating YAML: {e}")
            _log(logs, "yaml.validation_error", {"error": e})
        return {
            "generated_at": utcnow(),
            "count": 0,
            "items": [],
            "stats": {"errors": errs},
            "debug": logs,
        }

    aggs = cfg.get("aggregators") or {}
    filters = cfg.get("filters") or {}
    defaults = cfg.get("defaults") or {}

    # Build resume text early so scoring functions can use it
    resume_text = _build_resume_text(resume_path)

    include = [s.lower() for s in (filters.get("include") or [])]
    exclude = [s.lower() for s in (filters.get("exclude") or [])]
    remote_only = bool(defaults.get("remote_only", True))
    max_results = int(defaults.get("max_results", 500))
    min_score = float(defaults.get("min_score", 0.0))

    # Fetch from aggregators
    items: List[dict] = []
    for name, acfg in aggs.items():
        if not isinstance(acfg, dict) or not acfg.get("enabled"):
            continue
        try:
            before = len(items)
            if name == "remotive":
                items.extend(fetch_remotive(acfg, logs))
            elif name == "remoteok":
                items.extend(fetch_remoteok(acfg, logs))
            elif name == "weworkremotely":
                items.extend(fetch_weworkremotely(acfg, logs))
            elif name == "hnrss":
                items.extend(fetch_hnrss(acfg, logs))
            elif name == "jobicy":
                items.extend(fetch_jobicy(acfg, logs))
            elif name == "arbeitnow":
                items.extend(fetch_arbeitnow(acfg, logs))
            elif name == "usajobs":
                items.extend(fetch_usajobs(acfg, logs))
            elif name == "adzuna":
                items.extend(fetch_adzuna(acfg, logs))
            elif name == "jooble":
                items.extend(fetch_jooble(acfg, logs))
            elif name == "themuse":
                items.extend(fetch_themuse(acfg, logs))
            elif name == "findwork":
                items.extend(fetch_findwork(acfg, logs))
            elif name == "greenhouse":
                items.extend(fetch_greenhouse(acfg, logs))
            elif name == "lever":
                items.extend(fetch_lever(acfg, logs))
            elif name == "workable":
                items.extend(fetch_workable(acfg, logs))
            elif name == "smartrecruiters":
                items.extend(fetch_smartrecruiters(acfg, logs))
            elif name == "recruitee":
                items.extend(fetch_recruitee(acfg, logs))
            elif name == "personio":
                items.extend(fetch_personio(acfg, logs))
            elif name == "rss":
                items.extend(fetch_generic_rss(acfg, logs))
            else:
                _log(logs, "aggregator.unknown", {"name": name})
            added = len(items) - before
            _log(logs, "agg.results", {"source": name, "count": added})
        except Exception as e:
            print(f"Error fetching jobs from aggregator {name}: {e}")
            _log(logs, "aggregator.error", {"name": name, "error": str(e)})

    # ------ STAGE 0: collect raw + per-source raw counts
    all_items = items[:]  # keep original
    raw_total = len(all_items)
    raw_counts = Counter((it.get("source") or "unknown") for it in all_items)

    # ------ STAGE 1: remote-only filter (robust)
    if remote_only:
        s1_items = [it for it in all_items if _is_remote(it)]
    else:
        s1_items = all_items[:]
    after_remote = len(s1_items)
    s1_counts = Counter((it.get("source") or "unknown") for it in s1_items)

    # ------ STAGE 2: include/exclude keyword filters
    def _s(x: str):
        return str(x or "").strip().lower() if x else None
    include_set = set(include)
    exclude_set = set(exclude)
    def _filter_keywords(items: List[dict], incl: set, excl: set) -> List[dict]:
        out = []
        for it in items:
            text = " ".join([_s(it.get(k)) for k in ("title", "description", "company", "location")])
            if incl and not incl.intersection(_tokenize_words(text)):
                continue
            if excl and excl.intersection(_tokenize_words(text)):
                continue
            out.append(it)
        return out

    items = _filter_keywords(s1_items, include_set, exclude_set)
    after_filter = len(items)

    # ------ STAGE 3: deduplication (strong)
    seen = {}
    def _dedup_key(it):
        return (it.get("title"), it.get("company"), it.get("location"), it.get("source"))
    for it in items:
        k = _dedup_key(it)
        if k in seen:
            _log(logs, "dedup.skip", {"id": it.get("id"), "key": k})
            continue
        seen[k] = it
    items = list(seen.values())
    after_dedupe = len(items)

    # ------ STAGE 4: scoring (embeddings or TF-IDF)
    if use_embeddings:
        # 4a: Embeddings
        all_texts = [resume_text] + [it.get("title") + " " + (it.get("description") or "") for it in items]
        try:
            scores = _score_embeddings(resume_text, all_texts, embedding_model, logs)
            for i, it in enumerate(items):
                it["score"] = scores[i + 1]  # offset by 1 for resume_text
        except Exception as e:
            print(f"Error scoring with embeddings: {e}")
            _log(logs, "score.embeddings.error", {"error": str(e)})
            # Fallback to TF-IDF
            scores = _score_tfidf(resume_text, [it.get("title") + " " + (it.get("description") or "") for it in items])
            for i, it in enumerate(items):
                it["score"] = scores[i]
    else:
        # 4b: TF-IDF
        scores = _score_tfidf(resume_text, [it.get("title") + " " + (it.get("description") or "") for it in items])
        for i, it in enumerate(items):
            it["score"] = scores[i]

    # ------ STAGE 5: apply global default filters
    def _filter_defaults(items: List[dict], cfg: dict) -> List[dict]:
        out = []
        min_salary = float(cfg.get("minimum_salary") or 0)
        min_hourly = float(cfg.get("minimum_hourly") or 0)
        exclude_archived = cfg.get("exclude_archived", True)
        exclude_hybrid = cfg.get("exclude_hybrid", False)
        max_age_days = int(cfg.get("max_posting_age") or 0)
        now = datetime.now(timezone.utc)
        for it in items:
            # Salary filters: if pay info exists, compare the maximum of the range against threshold
            if min_salary > 0 or min_hourly > 0:
                pay_info = _parse_pay_info(it)
                max_pay = pay_info.get("max") or 0
                period = pay_info.get("period")
                # If we couldn't detect pay info, don't filter by pay
                if period == "hourly" and min_hourly > 0 and max_pay and max_pay < min_hourly:
                    continue
                if period == "annual" and min_salary > 0 and max_pay and max_pay < min_salary:
                    continue
            # Archived/closed filters
            if exclude_archived and _is_archived_item(it):
                continue
            # Hybrid filter
            if exclude_hybrid and _is_hybrid(it):
                continue
            # Age filter
            if max_age_days > 0:
                posted_at = it.get("posted_at")
                if isinstance(posted_at, str):
                    try:
                        posted_dt = dateutil.parser.isoparse(posted_at)
                        age_days = (now - posted_dt).days
                        if age_days > max_age_days:
                            continue
                    except Exception:
                        pass
            out.append(it)
        return out

    # capture the items before applying defaults so we can detect what was filtered
    pre_defaults = items[:]

    items = _filter_defaults(items, defaults)
    after_defaults = len(items)

    # Automatically mark filtered-out jobs as not_suitable in the DB (if they exist and aren't already marked)
    try:
        import job_store
        try:
            conn = job_store.connect(None)
        except Exception:
            conn = None
        # compute the same default thresholds that _filter_defaults used
        min_salary = float(defaults.get("minimum_salary") or 0)
        min_hourly = float(defaults.get("minimum_hourly") or 0)
        exclude_archived = defaults.get("exclude_archived", True)
        exclude_hybrid = defaults.get("exclude_hybrid", True)

        kept_ids = set([it.get("id") for it in items if it.get("id")])
        filtered_items = [it for it in pre_defaults if it.get("id") and it.get("id") not in kept_ids]
        marked_count = 0
        if conn is not None and filtered_items:
            for it in filtered_items:
                try:
                    cur = conn.execute("SELECT id, not_suitable FROM jobs WHERE id = ? OR hash_key = ? OR url = ? LIMIT 1", (it.get("id"), job_store._hash_key(it), it.get("url")))
                    row = cur.fetchone()
                    if row and not bool(row[1]):
                        # mark as not suitable with the filter reason (best-effort)
                        reason = None
                        try:
                            # try to re-evaluate why it was filtered
                            if exclude_archived and _is_archived_item(it):
                                reason = "archived"
                            elif exclude_hybrid and _is_hybrid(it):
                                reason = "hybrid"
                            else:
                                # salary/age/other
                                p = _parse_pay_info(it)
                                if (p.get("period") == "hourly" and min_hourly > 0 and p.get("max") and p.get("max") < min_hourly):
                                    reason = f"salary_below_hourly_{min_hourly}"
                                elif (p.get("period") == "annual" and min_salary > 0 and p.get("max") and p.get("max") < min_salary):
                                    reason = f"salary_below_annual_{min_salary}"
                                else:
                                    reason = "filtered"
                        except Exception:
                            reason = "filtered"
                        try:
                            ok = job_store.mark_not_suitable(job_store.DB_PATH, it.get("id"), reasons=[reason], note="auto-filtered")
                            if ok:
                                marked_count += 1
                        except Exception:
                            # best-effort; don't fail the hunt
                            pass
                except Exception:
                    continue
        _log(logs, "auto_mark.filtered", {"count": marked_count})
    except Exception as e:
        _log(logs, "auto_mark.error", {"error": str(e)})

    # ------ FINAL: sort by score (highest first)
    items.sort(key=lambda it: it.get("score", 0), reverse=True)
    final_count = len(items)
    if max_results > 0 and final_count > max_results:
        items = items[:max_results]

    # Log summary
    _log(logs, "hunt.summary", {
        "raw_total": raw_total,
        "after_remote": after_remote,
        "after_filter": after_filter,
        "after_dedupe": after_dedupe,
        "final_count": final_count,
    })
    return {
        "generated_at": utcnow(),
        "count": final_count,
        "items": items,
        "stats": {
            "raw_total": raw_total,
            "after_remote": after_remote,
            "after_filter": after_filter,
            "after_dedupe": after_dedupe,
            "final_count": final_count,
            "errors": [],
        },
        "debug": logs,
    }

# -----------------------------
# Probe sources (UI helper)
# -----------------------------
def probe_sources(sources_yaml: str, debug: bool = False) -> List[dict]:
    """Check configured aggregator endpoints and return status rows for UI display."""
    rows: List[dict] = []
    cfg, errs = load_sources_yaml(sources_yaml, debug=debug)
    if errs:
        for e in errs:
            rows.append({"when": utcnow(), "tag": "yaml.error", "status": None, "reason": e, "url": sources_yaml})
        return rows

    aggs = cfg.get("aggregators") or {}
    test_endpoints = {
        "remotive": "https://remotive.com/api/remote-jobs",
        "remoteok": "https://remoteok.com/api",
        "weworkremotely": "https://weworkremotely.com/remote-jobs.rss",
        "hnrss": "https://hnrss.org/jobs",
        "jobicy": "https://jobicy.com/api/v2/remote-jobs",
        "arbeitnow": "https://arbeitnow.com/api/job-board-api",
        "adzuna": "https://api.adzuna.com/v1/api/jobs/us/search/1",
        "usajobs": "https://data.usajobs.gov/api/search",
        "jooble": "https://jooble.org/api/",
        "themuse": "https://www.themuse.com/api/public/jobs",
        "findwork": "https://findwork.dev/api/jobs/",
    }

    for name, acfg in aggs.items():
        if not isinstance(acfg, dict) or not acfg.get("enabled"):
            rows.append({"when": utcnow(), "tag": f"probe.{name}", "status": "skipped", "reason": "disabled"})
            continue
        url = test_endpoints.get(name)
        if not url:
            rows.append({"when": utcnow(), "tag": f"probe.{name}", "status": "n/a", "reason": "company-scope or rss-only"})
            continue
        try:
            r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            rows.append({"when": utcnow(), "tag": f"probe.{name}", "status": r.status_code, "reason": getattr(r, "reason", None), "url": url})
        except Exception as e:
            rows.append({"when": utcnow(), "tag": f"probe.{name}", "status": None, "reason": str(e)})
    return rows
