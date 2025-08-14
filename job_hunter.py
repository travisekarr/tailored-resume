# job_hunter.py
import os
import re
_re = re
import json
import time
import math
import hashlib
import requests
import feedparser
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional OpenAI embeddings
try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception as e:
    print(f"Error importing OpenAI: {e}")
    _openai_client = None

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
    return datetime.now(timezone.utc).isoformat()

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
            # ANSI escape code for bright red
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

# -----------------------------
# RSS helper
# -----------------------------
def fetch_rss(feed_url: str, source: str, logs: list) -> List[dict]:
    """Fetch RSS with browser-like headers, then parse; logs entry counts; handles WWR 'Company: Title'."""
    _log(logs, "rss.fetch", {"url": feed_url, "source": source})
    r, err = _http_get(
        feed_url,
        logs,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
        },
        timeout=25,
    )
    if not r or r.status_code != 200:
        _log(logs, "rss.http_fail", {
            "url": feed_url, "source": source, "status": getattr(r, "status_code", None), "error": err
        })
        return []

    try:
        f = feedparser.parse(r.content)  # parse the bytes we fetched
    except Exception as e:
        _log(logs, "rss.parse_error", {"url": feed_url, "source": source, "error": str(e)})
        return []

    entries = getattr(f, "entries", []) or []
    _log(logs, "rss.entries", {"source": source, "url": feed_url, "count": len(entries)})

    out: List[dict] = []
    for e in entries:
        url = e.get("link") or ""
        desc = e.get("summary") or e.get("description") or ""
        title_raw = e.get("title") or ""
        posted = e.get("published") or e.get("updated") or None

        title = title_raw
        company = ""

        if source == "weworkremotely":
            # WWR pattern is commonly "Company: Role"
            parts = [p.strip() for p in title_raw.split(": ", 1)]
            if len(parts) == 2:
                company, title = parts[0], parts[1]
        else:
            m = re.search(r" at ([^-–—]+)", title_raw)
            if m:
                company = m.group(1).strip()

        out.append({
            "id": _hash_id(url or title_raw),
            "title": title,
            "company": company,
            "location": "",
            "remote": True,
            "url": url,
            "source": source,
            "posted_at": posted,
            "pulled_at": utcnow(),
            "description": desc,
        })
    return out

def _log_openai_usage_resp(resp, *, endpoint: str, model: str, context: str = "cli", db_path: str | None = None):
    try:
        from job_store import log_api_usage
        # Chat/Responses usage fields differ; Embeddings has .usage.total_tokens
        usage = getattr(resp, "usage", None)
        if usage:
            # embeddings: input tokens are counted as total_tokens
            itok = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None) or getattr(usage, "total_tokens", None)
            otok = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None) or 0
            ttok = getattr(usage, "total_tokens", None) or ((itok or 0) + (otok or 0))
        else:
            itok = getattr(resp, "usage", {}).get("total_tokens", 0) if isinstance(resp.__dict__.get("usage"), dict) else 0
            otok, ttok = 0, itok
        req_id = getattr(resp, "id", "") or getattr(resp, "request_id", "") or ""
        log_api_usage(
            db_path=db_path, endpoint=endpoint, model=model,
            input_tokens=itok or 0, output_tokens=otok or 0, total_tokens=ttok or 0,
            request_id=req_id, context=context, meta={}
        )
    except Exception as e:
        print(f"Error logging OpenAI usage: {e}")
        pass

# ---------- Embedding batching helpers ----------
# Safe budgets; tweak down if you still see "max_tokens_per_request"
_EMBED_REQ_TOKEN_BUDGET = 160_000     # effective budget per request (<< 300k)
_MAX_INPUTS_PER_BATCH   = 96          # hard cap on items per batch
_MAX_CHARS_PER_INPUT    = 4_000       # reduce per-input length to keep requests small
_SAFETY_FACTOR          = 1.35        # multiply estimates to be conservative

def _estimate_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Conservative token estimate. Uses tiktoken if available; else ~3 chars/token."""
    s = text or ""
    if _tiktoken:
        try:
            enc = _tiktoken.encoding_for_model(model)
        except Exception:
            enc = _tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(s))
    # Conservative heuristic (smaller divisor => larger estimate)
    return max(1, (len(s) // 3) + 1)

def _clean_for_embedding(s) -> str:
    s = _stringify(s)
    s = _html.unescape(s)
    s = _re.sub(r"<[^>]+>", " ", s)
    s = _re.sub(r"`{3,}.*?`{3,}", " ", s, flags=_re.S)
    s = _re.sub(r"`([^`]+)`", r"\1", s)
    s = _re.sub(r"[ \t\r\f\v]+", " ", s)
    s = _re.sub(r"\s*\n+\s*", "\n", s)
    s = s.strip().lower()
    if len(s) > _MAX_CHARS_PER_INPUT:
        s = s[:_MAX_CHARS_PER_INPUT]
    return s

def _chunk_by_token_budget(texts: list[str], model: str, budget: int, max_items: int) -> list[list[str]]:
    """Split texts into batches under (budget / SAFETY_FACTOR) and <= max_items."""
    batches, current, current_est = [], [], 0
    eff_budget = int(budget / _SAFETY_FACTOR)
    for t in texts:
        tt = _estimate_tokens(t, model)
        # cut the batch if adding this item would exceed budget or max_items
        if current and (current_est + tt > eff_budget or len(current) >= max_items):
            batches.append(current)
            current, current_est = [], 0
        current.append(t)
        current_est += tt
    if current:
        batches.append(current)
    return batches

def _cosine_sim(vec_a, vec_b) -> float:
    num = sum(a*b for a, b in zip(vec_a, vec_b))
    da = _math.sqrt(sum(a*a for a in vec_a))
    db = _math.sqrt(sum(b*b for b in vec_b))
    return (num / (da * db)) if da and db else 0.0

def _score_embeddings(
    resume_text: str,
    job_texts: list[str],
    embedding_model: str = "text-embedding-3-small",
    logs: list | None = None,
) -> list[float]:
    """Embedding-based scoring with strict batching and recursive split on 300k-cap errors."""
    if _openai_client is None:
        return [0.0] * len(job_texts)

    # 1) Clean inputs
    resume_clean = _clean_for_embedding(resume_text or "")
    jobs_clean   = [_clean_for_embedding(t or "") for t in job_texts]

    if not embedding_model or not isinstance(embedding_model, str):
        raise ValueError(f"Embedding model missing/invalid: {embedding_model!r}")
    if not embedding_model.startswith("text-embedding-"):
        raise ValueError(f"Not an embeddings model: {embedding_model}")

    # 2) Embed resume once (retry on model mismatch with -3-small)
    try:
        res = _openai_client.embeddings.create(model=embedding_model, input=resume_clean)
    except Exception:
        res = _openai_client.embeddings.create(model="text-embedding-3-small", input=resume_clean)
        embedding_model = "text-embedding-3-small"
    resume_vec = res.data[0].embedding

    # 3) Prepare batches
    batches = _chunk_by_token_budget(
        jobs_clean, model=embedding_model,
        budget=_EMBED_REQ_TOKEN_BUDGET, max_items=_MAX_INPUTS_PER_BATCH
    )
    if logs is not None:
        _log(logs, "embeddings.start", {"model": embedding_model, "jobs": len(jobs_clean), "batches": len(batches)})

    job_vecs: list[list[float]] = []

    def _embed_batch(texts: list[str]) -> list[list[float]]:
        # try whole batch
        est = sum(_estimate_tokens(t, embedding_model) for t in texts)
        if logs is not None:
            _log(logs, "embeddings.batch", {
                "model": embedding_model, "batch_size": len(texts), "est_tokens": int(est * _SAFETY_FACTOR)
            })
        try:
            resp = _openai_client.embeddings.create(model=embedding_model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception as e:
            msg = str(e)
            # If we hit the token cap, split and recurse
            if ("max_tokens_per_request" in msg) or ("max 300000 tokens" in msg):
                if logs is not None:
                    _log(logs, "embeddings.batch.split", {
                        "reason": "max_tokens_per_request", "size": len(texts), "est_tokens": int(est * _SAFETY_FACTOR)
                    })
                if len(texts) == 1:
                    # last-ditch: hard-trim further and try once more
                    t0 = texts[0][: max(1000, _MAX_CHARS_PER_INPUT // 2)]
                    try:
                        resp2 = _openai_client.embeddings.create(model=embedding_model, input=t0)
                        return [resp2.data[0].embedding]
                    except Exception:
                        # fallback tiny vector of zeros to keep alignment
                        return [[0.0] * len(resume_vec)]
                mid = len(texts) // 2
                left  = _embed_batch(texts[:mid])
                right = _embed_batch(texts[mid:])
                return left + right
            # Other errors: rethrow (handled by caller to fall back TF-IDF if desired)
            raise

    # 4) Embed all batches in order
    try:
        for b in batches:
            job_vecs.extend(_embed_batch(b))
        if logs is not None:
            _log(logs, "embeddings.ok", {"model": embedding_model, "vectors": len(job_vecs)})
    except Exception as e:
        if logs is not None:
            _log(logs, "embeddings.error", {"model": embedding_model, "error": str(e)})
        # Fallback to TF-IDF on any unexpected failure
        return _score_tfidf(resume_text, job_texts)

    # 5) Cosine similarities
    def _cos(a, b):
        num = sum(x*y for x, y in zip(a, b))
        da = sum(x*x for x in a) ** 0.5
        db = sum(y*y for y in b) ** 0.5
        return (num / (da * db)) if da and db else 0.0

    scores = [_cos(resume_vec, v) for v in job_vecs]

    # 6) Align lengths
    if len(scores) != len(job_texts):
        if len(scores) < len(job_texts):
            scores += [0.0] * (len(job_texts) - len(scores))
        else:
            scores = scores[:len(job_texts)]
    return scores

# -----------------------------
# Aggregators
# -----------------------------
def fetch_remoteok(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    url = "https://remoteok.com/api"
    r, _ = _http_get(url, logs, headers={"User-Agent": "Mozilla/5.0"})
    results: List[dict] = []
    if r and r.status_code == 200:
        try:
            data = r.json()
            for it in data:
                if not isinstance(it, dict) or not (it.get("slug") or it.get("id")):
                    continue
                results.append({
                    "id": str(it.get("id") or it.get("slug")),
                    "title": it.get("position") or it.get("title"),
                    "company": it.get("company"),
                    "location": it.get("location") or it.get("region") or "",
                    "remote": True,
                    "url": "https://remoteok.com" + (it.get("url") or ""),
                    "source": "remoteok",
                    "posted_at": it.get("date") or it.get("epoch"),
                    "pulled_at": utcnow(),
                    "description": it.get("description") or "",
                })
        except Exception as e:
            print(f"Error parsing remoteok data: {e}")
            _log(logs, "remoteok.parse_error", {"error": str(e)})
    return results

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
                "posted_at": posted,
                "pulled_at": utcnow(),
                "description": desc,
            })
    return results

def fetch_hnrss(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    feed = cfg.get("feed") or "https://hnrss.org/jobs"
    return fetch_rss(feed, "hnrss", logs)

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
                    "posted_at": it.get("pubDate") or it.get("date"),
                    "pulled_at": utcnow(),
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
                    "posted_at": it.get("created_at") or it.get("published_at"),
                    "pulled_at": utcnow(),
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
                    "posted_at": pos.get("PublicationStartDate"),
                    "pulled_at": utcnow(),
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
                    "posted_at": it.get("created"),
                    "pulled_at": utcnow(),
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
                    "posted_at": it.get("updated"),
                    "pulled_at": utcnow(),
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
    category = cfg.get("category") or "Software Engineering"
    pages = int(cfg.get("pages") or 2)
    out: List[dict] = []
    for p in range(1, pages + 1):
        params = {"page": p, "category": category}
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
                    "posted_at": it.get("publication_date"),
                    "pulled_at": utcnow(),
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
                    "posted_at": it.get("date_posted"),
                    "pulled_at": utcnow(),
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
                        "remote": "remote" in ((it.get("location") or {}).get("name","").lower()),
                        "url": it.get("absolute_url"),
                        "source": "greenhouse",
                        "posted_at": it.get("updated_at") or it.get("created_at"),
                        "pulled_at": utcnow(),
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
                        "remote": "remote" in ((it.get("categories") or {}).get("location","").lower()),
                        "url": it.get("hostedUrl"),
                        "source": "lever",
                        "posted_at": it.get("createdAt"),
                        "pulled_at": utcnow(),
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
                        "posted_at": it.get("published_on"),
                        "pulled_at": utcnow(),
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
                        "posted_at": it.get("releasedDate"),
                        "pulled_at": utcnow(),
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
                        "posted_at": it.get("created_at"),
                        "pulled_at": utcnow(),
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
                        "posted_at": it.get("created_at"),
                        "pulled_at": utcnow(),
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

def _tokenize_words(s: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[a-z0-9+\-/#\.]{3,}", s or "")}

def _extract_resume_keywords(resume_yaml_path: str) -> set[str]:
    """
    Pull skills/tags/tech from the YAML so we can reward jobs that mention them,
    even when the phrasing differs from the summary prose.
    """
    import yaml as _y
    kws = set()
    try:
        with open(resume_yaml_path, "r", encoding="utf-8") as f:
            sections = _y.safe_load(f) or []
    except Exception as e:
        print(f"Error extracting resume keywords: {e}")
        sections = []
    for sec in sections if isinstance(sections, list) else []:
        if isinstance(sec, dict):
            for field in ("tags",):
                for t in sec.get(field) or []:
                    if isinstance(t, str):
                        kws.add(t.lower())
            if sec.get("type") == "experience":
                for c in sec.get("contributions") or []:
                    for f in ("skills_used", "impact_tags"):
                        for t in (c or {}).get(f) or []:
                            if isinstance(t, str):
                                kws.add(t.lower())
    # Optional: expand a few common synonyms
    synonyms = {
        "aws": {"amazon web services"},
        "gcp": {"google cloud"},
        "ci/cd": {"cicd", "continuous integration", "continuous delivery"},
        "k8s": {"kubernetes"},
        "sre": {"site reliability"},
        "saas": {"software as a service"},
    }
    for k, alts in synonyms.items():
        if k in kws:
            kws.update(alts)
    return kws

def _overlap_terms(text: str, terms: set[str], k: int = 15) -> list[str]:
    low = (text or "").lower()
    hits = [t for t in terms if t in low]
    # longer terms first (more specific), then alpha
    hits.sort(key=lambda x: (-len(x), x))
    return hits[:k]

# -----------------------------
# Probe sources (UI helper)
# -----------------------------
def probe_sources(sources_yaml: str, debug: bool = False) -> List[dict]:
    logs: List[dict] = []
    cfg, errs = load_sources_yaml(sources_yaml, debug=debug)
    rows: List[dict] = []
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
            print(f"Error probing source {name}: {e}")
            rows.append({"when": utcnow(), "tag": f"probe.{name}", "status": None, "reason": str(e)})
    return rows

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
    def _s(x):  # (kept from your code)
        return x if isinstance(x, str) else ("" if x is None else str(x))

    def ok_by_filters(it: dict) -> bool:  # (kept from your code)
        t = " ".join([_s(it.get("title")), _s(it.get("company")), _s(it.get("description"))]).lower()
        if include and not any(s in t for s in include):
            return False
        if exclude and any(s in t for s in exclude):
            return False
        return True

    s2_items = [it for it in s1_items if ok_by_filters(it)]
    after_keywords = len(s2_items)
    s2_counts = Counter((it.get("source") or "unknown") for it in s2_items)

    # ------ STAGE 3: de-dupe
    seen = set()
    deduped = []
    per_source_after_dedupe = defaultdict(int)
    for it in s2_items:
        key = (it.get("url") or "").lower() or ((it.get("company","") + "|" + it.get("title","")).lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
        per_source_after_dedupe[it.get("source") or "unknown"] += 1
    items = deduped
    after_dedupe = len(items)

    # ------ (optional) enrichment
    if bool(defaults.get("fetch_full_descriptions", False)):
        _log(logs, "enrich.start", defaults.get("enrichment") or {})
        _enrich_items_descriptions(items, defaults, logs)

    # ------ STAGE 4: scoring
    resume_text = _build_resume_text(resume_path)
    resume_terms = _extract_resume_keywords(resume_path)

    # Build job texts (boost titles implicitly by putting them first & twice)
    job_texts = [f"{it.get('title',' ')} {it.get('title',' ')} — {it.get('description','')}" for it in items]

    if use_embeddings:
        plain_job_texts = [f"{it.get('title','')} — {it.get('description','')}" for it in items]
        _log(logs, "embeddings.start", {
            "model": embedding_model,
            "jobs": len(plain_job_texts)
        })
        try:
            scores = _score_embeddings(
                resume_text=resume_text,
                job_texts=plain_job_texts,
                embedding_model=embedding_model,
                logs=logs,  # <- so you see per-batch logs
            )
            tfidf_scores = scores
            _log(logs, "embeddings.ok", {"model": embedding_model, "scores": len(tfidf_scores)})
        except Exception as e:
            _log(logs, "embeddings.error", {"model": embedding_model, "error": str(e)})
            # graceful fallback
            tfidf_scores = _score_tfidf(resume_text, job_texts)
    else:
        tfidf_scores = _score_tfidf(resume_text, job_texts)

    source_prior = {
        "greenhouse": 0.02, "lever": 0.02, "workable": 0.01,
        "remoteok": 0.02, "remotive": 0.02, "weworkremotely": 0.02,
        "hnrss": 0.01, "adzuna": 0.0, "usajobs": 0.0
    }

    scored = []
    for it, base in zip(items, tfidf_scores):
        title = it.get("title", "")
        desc = it.get("description", "")
        title_terms = _overlap_terms(title, resume_terms, k=10)
        desc_terms  = _overlap_terms(desc,  resume_terms, k=10)
        title_boost = min(0.30, 0.05 * len(title_terms))
        body_boost  = min(0.15, 0.02 * len(desc_terms))
        prior       = source_prior.get((it.get("source") or "").lower(), 0.0)
        final = 0.70 * float(base) + 0.25 * title_boost + 0.05 * (body_boost + prior)
        it = dict(it)
        it["score"] = float(final)
        it["match_terms_title"] = title_terms
        it["match_terms_body"]  = desc_terms
        it["id"] = it.get("id") or _hash_id((it.get("url") or it.get("title","")))
        scored.append(it)

    # ------ STAGE 5: threshold + cap
    items = [it for it in scored if it.get("score", 0.0) >= min_score]
    items.sort(key=lambda x: (-x.get("score", 0.0), (x.get("posted_at") or "")))
    if max_results and len(items) > max_results:
        items = items[:max_results]
    after_threshold = len(items)

    # per-source counts after threshold
    s5_counts = Counter((it.get("source") or "unknown") for it in items)

    # ------ LOG PER-SOURCE STAGE COUNTS
    all_sources = set(raw_counts) | set(s1_counts) | set(s2_counts) | set(per_source_after_dedupe) | set(s5_counts)
    for src in sorted(all_sources):
        _log(logs, "stage.counts.source", {
            "source": src,
            "raw_total": int(raw_counts.get(src, 0)),
            "after_remote_filter": int(s1_counts.get(src, 0)),
            "after_keyword_filters": int(s2_counts.get(src, 0)),
            "after_dedupe": int(per_source_after_dedupe.get(src, 0)),
            "after_threshold": int(s5_counts.get(src, 0)),
        })

    # ------ LOG GLOBAL STAGE COUNTS
    _log(logs, "stage.counts", {
        "raw_total": raw_total,
        "after_remote_filter": after_remote,
        "after_keyword_filters": after_keywords,
        "after_dedupe": after_dedupe,
        "after_threshold": after_threshold,
        "min_score": min_score,
        "remote_only": remote_only,
    })

    # ------ STATS + RESULT
    stats = {
        "raw_total": raw_total,
        "after_remote_filter": after_remote,
        "after_keyword_filters": after_keywords,
        "after_dedupe": after_dedupe,
        "after_threshold": after_threshold,
        "enriched": len([it for it in items if it.get("description_source") == "enriched"]),
    }
    _log(logs, "done", {"stats": stats})

    res = {
        "generated_at": utcnow(),
        "count": len(items),
        "items": items,
        "stats": stats,
        "debug": logs,
        "use_embeddings": use_embeddings,
        "embedding_model": embedding_model,
        "resume_path": resume_path,
        "sources_yaml": sources_yaml,
    }
    if logger:
        try: logger("hunt_jobs.complete", res)
        except Exception: pass
    return res
