# job_hunter.py
import os
import re
import json
import time
import math
import hashlib
import requests
import feedparser
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional OpenAI embeddings
try:
    from openai import OpenAI
    _openai_client = OpenAI()
except Exception:
    _openai_client = None

# Optional content extractors (we handle absence gracefully)
try:
    import trafilatura  # best-effort main-content extractor
except Exception:
    trafilatura = None

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


# -----------------------------
# Utilities
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
    logs.append({"ts": utcnow(), "msg": msg, "data": data or {}})


# -----------------------------
# YAML loader (with clear errors)
# -----------------------------
def load_sources_yaml(path: str, debug: bool = False) -> Tuple[Optional[dict], List[str]]:
    """
    Loads and normalizes the aggregator config YAML.
    Returns (config, errors). Errors is a list of string codes/messages.
    Accepts both:
      - canonical: {'aggregators': {...}, 'filters': {...}, 'defaults': {...}}
      - legacy:    {'sources': {...},    'filters': {...}, 'defaults': {...}}
    """
    errs: List[str] = []
    if not path or not os.path.exists(path):
        errs.append(f"file_not_found: {path}")
        return None, errs

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        errs.append(f"read_error: {e}")
        return None, errs

    try:
        data = yaml.safe_load(raw) or {}
    except Exception as e:
        errs.append(f"yaml_parse_error: {e}")
        return None, errs

    # Normalize legacy root key
    if "aggregators" not in data and "sources" in data and isinstance(data["sources"], dict):
        data["aggregators"] = data["sources"]

    aggs = data.get("aggregators")
    if not isinstance(aggs, dict) or not aggs:
        errs.append("validation_error: top-level 'aggregators' missing or not a mapping")

    enabled_count = 0
    if isinstance(aggs, dict):
        for name, cfg in aggs.items():
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
    except Exception:
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
        except Exception:
            continue
    return " ".join(buf)


# -----------------------------
# Scoring helpers
# -----------------------------
def _score_tfidf(resume_text: str, job_texts: List[str]) -> List[float]:
    corpus = [resume_text] + job_texts
    tfidf = TfidfVectorizer().fit_transform(corpus)
    sims = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
    return [float(x) for x in sims]

def _score_embeddings(job_texts: List[str], embedding_model: str, resume_text: str) -> List[float]:
    if _openai_client is None:
        return [0.0] * len(job_texts)
    inputs = [resume_text] + job_texts
    resp = _openai_client.embeddings.create(model=embedding_model, input=inputs)
    resume_vec = resp.data[0].embedding
    scores: List[float] = []
    for row in resp.data[1:]:
        scores.append(_cosine(resume_vec, row.embedding))
    return scores


# -----------------------------
# HTTP helper (logs every call)
# -----------------------------
def _http_get(url: str, logs: list, headers: Optional[dict] = None, timeout: int = 20) -> Tuple[Optional[requests.Response], Optional[str]]:
    try:
        r = requests.get(url, headers=headers or {}, timeout=timeout)
        _log(logs, "http.get", {"url": url, "status": r.status_code, "reason": getattr(r, "reason", None), "len": len(r.content)})
        return r, None
    except Exception as e:
        _log(logs, "http.error", {"url": url, "error": str(e)})
        return None, str(e)


# -----------------------------
# Aggregator fetchers
# -----------------------------
def fetch_remotive(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    base = "https://remotive.com/api/remote-jobs"
    results: List[dict] = []
    queries = cfg.get("queries") or [""]
    categories = cfg.get("categories") or []
    limit = int(cfg.get("limit") or 200)
    for q in queries:
        url = f"{base}?search={requests.utils.quote(q)}"
        if categories:
            for c in categories:
                u = f"{url}&category={requests.utils.quote(c)}"
                r, err = _http_get(u, logs)
                if r and r.status_code == 200:
                    try:
                        data = r.json()
                        for it in data.get("jobs", []):
                            results.append({
                                "id": it.get("id") or _hash_id(it.get("url","")),
                                "title": it.get("title"),
                                "company": it.get("company_name"),
                                "location": it.get("candidate_required_location") or it.get("job_type"),
                                "remote": True,
                                "url": it.get("url"),
                                "source": "remotive",
                                "posted_at": it.get("publication_date"),
                                "pulled_at": utcnow(),
                                "description": it.get("description") or "",
                            })
                    except Exception as e:
                        _log(logs, "remotive.parse_error", {"error": str(e)})
        else:
            r, err = _http_get(url, logs)
            if r and r.status_code == 200:
                try:
                    data = r.json()
                    for it in data.get("jobs", []):
                        results.append({
                            "id": it.get("id") or _hash_id(it.get("url","")),
                            "title": it.get("title"),
                            "company": it.get("company_name"),
                            "location": it.get("candidate_required_location") or it.get("job_type"),
                            "remote": True,
                            "url": it.get("url"),
                            "source": "remotive",
                            "posted_at": it.get("publication_date"),
                            "pulled_at": utcnow(),
                            "description": it.get("description") or "",
                        })
                except Exception as e:
                    _log(logs, "remotive.parse_error", {"error": str(e)})
    return results[:limit]

def fetch_remoteok(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    url = "https://remoteok.com/api"
    r, err = _http_get(url, logs, headers={"User-Agent": "Mozilla/5.0"})
    results: List[dict] = []
    if r and r.status_code == 200:
        try:
            data = r.json()
            for it in data:
                if not isinstance(it, dict):
                    continue
                if not it.get("slug") and not it.get("id"):
                    continue
                tags = [t.lower() for t in (it.get("tags") or []) if isinstance(t, str)]
                include = cfg.get("tags_include") or []
                exclude = cfg.get("tags_exclude") or []
                if include:
                    if not any(t in tags for t in [x.lower() for x in include]):
                        continue
                if exclude:
                    if any(t in tags for t in [x.lower() for x in exclude]):
                        continue
                results.append({
                    "id": str(it.get("id") or it.get("slug")),
                    "title": it.get("position") or it.get("title"),
                    "company": it.get("company"),
                    "location": it.get("location") or it.get("region"),
                    "remote": True,
                    "url": "https://remoteok.com" + it.get("url",""),
                    "source": "remoteok",
                    "posted_at": it.get("date") or it.get("epoch"),
                    "pulled_at": utcnow(),
                    "description": it.get("description") or "",
                })
        except Exception as e:
            _log(logs, "remoteok.parse_error", {"error": str(e)})
    return results

def fetch_rss(feed_url: str, source: str, logs: list) -> List[dict]:
    _log(logs, "rss.fetch", {"url": feed_url})
    try:
        f = feedparser.parse(feed_url)
    except Exception as e:
        _log(logs, "rss.parse_error", {"url": feed_url, "error": str(e)})
        return []
    out: List[dict] = []
    for e in f.entries:
        url = e.get("link") or ""
        desc = e.get("summary") or e.get("description") or ""
        title = e.get("title") or ""
        company = ""
        m = re.search(r" at ([^-–—]+)", title)
        if m:
            company = m.group(1).strip()
        posted = None
        if e.get("published"):
            posted = e.get("published")
        out.append({
            "id": _hash_id(url or title),
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

def fetch_weworkremotely(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    feeds = cfg.get("feeds") or []
    results: List[dict] = []
    for u in feeds:
        results.extend(fetch_rss(u, "weworkremotely", logs))
    return results

def fetch_hnrss(cfg: dict, logs: list) -> List[dict]:
    if not cfg.get("enabled"):
        return []
    feed = cfg.get("feed") or "https://hnrss.org/jobs"
    return fetch_rss(feed, "hnrss", logs)


# -----------------------------
# Description enrichment
# -----------------------------
def _strip_tags_to_text(html: str) -> str:
    if not html:
        return ""
    if BeautifulSoup is None:
        # crude fallback
        return re.sub(r"<[^>]+>", " ", html)
    soup = BeautifulSoup(html, "lxml") if "lxml" in (globals().get("__loader__", None) or {}).__class__.__name__ else BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # collapse whitespace
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
    """
    Modifies `items` in place when enrichment is enabled.
    """
    enrich_opts = defaults.get("enrichment") or {}
    max_fetch = int(enrich_opts.get("max_fetch", 20))
    only_if_shorter_than = int(enrich_opts.get("only_if_shorter_than", 400))
    min_chars = int(enrich_opts.get("min_chars", 600))
    rate_limit_s = float(enrich_opts.get("rate_limit_s", 1.0))
    timeout = int(enrich_opts.get("timeout_s", 20))
    user_agent = enrich_opts.get("user_agent", "Mozilla/5.0 (job-hunter)")

    # If provided, only enrich these sources; otherwise heuristic defaults
    sources_pref = enrich_opts.get("sources")
    if sources_pref is None:
        sources_pref = ["remoteok", "weworkremotely", "hnrss"]

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

        time.sleep(rate_limit_s)  # be polite
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
# Probe sources (UI button)
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
    for name, acfg in aggs.items():
        if not isinstance(acfg, dict) or not acfg.get("enabled"):
            rows.append({"when": utcnow(), "tag": f"probe.{name}", "status": "skipped", "reason": "disabled"})
            continue
        try:
            if name == "remotive":
                test_url = "https://remotive.com/api/remote-jobs"
            elif name == "remoteok":
                test_url = "https://remoteok.com/api"
            elif name == "weworkremotely":
                test_url = "https://weworkremotely.com/remote-jobs.rss"
            elif name == "hnrss":
                test_url = "https://hnrss.org/jobs"
            else:
                rows.append({"when": utcnow(), "tag": f"probe.{name}", "status": "unknown", "reason": "unrecognized aggregator"})
                continue

            r = requests.get(test_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            rows.append({"when": utcnow(), "tag": f"probe.{name}", "status": r.status_code, "reason": getattr(r, "reason", None), "url": test_url})
        except Exception as e:
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
        if name == "remotive":
            items.extend(fetch_remotive(acfg, logs))
        elif name == "remoteok":
            items.extend(fetch_remoteok(acfg, logs))
        elif name == "weworkremotely":
            items.extend(fetch_weworkremotely(acfg, logs))
        elif name == "hnrss":
            items.extend(fetch_hnrss(acfg, logs))
        else:
            _log(logs, "aggregator.unknown", {"name": name})

    raw_total = len(items)

    # Remote filter
    if remote_only:
        items = [it for it in items if it.get("remote") in (True, "true", 1)]
    after_remote = len(items)

    # Include/exclude substring filter
    def ok_by_filters(it: dict) -> bool:
        t = " ".join([it.get("title",""), it.get("company",""), it.get("description","")]).lower()
        if include and not any(s in t for s in include):
            return False
        if exclude and any(s in t for s in exclude):
            return False
        return True

    items = [it for it in items if ok_by_filters(it)]
    after_keywords = len(items)

    # De-dupe by URL/company+title
    seen = set()
    deduped: List[dict] = []
    for it in items:
        key = (it.get("url") or "").lower() or (it.get("company","").lower() + "|" + it.get("title","").lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    items = deduped
    after_dedupe = len(items)

    # (NEW) Enrich descriptions BEFORE scoring if enabled via YAML defaults
    if bool(defaults.get("fetch_full_descriptions", False)):
        _log(logs, "enrich.start", defaults.get("enrichment") or {})
        _enrich_items_descriptions(items, defaults, logs)

    # Score
    resume_text = _build_resume_text(resume_path)
    job_texts = [f"{it.get('title','')} — {it.get('description','')}" for it in items]
    if use_embeddings:
        scores = _score_embeddings(job_texts, embedding_model, resume_text)
    else:
        scores = _score_tfidf(resume_text, job_texts)

    for it, sc in zip(items, scores):
        it["score"] = float(sc)
        it["id"] = it.get("id") or _hash_id((it.get("url") or it.get("title","")))

    # Threshold + max
    items = [it for it in items if it.get("score", 0.0) >= min_score]
    items.sort(key=lambda x: (-x.get("score", 0.0), (x.get("posted_at") or "")), reverse=False)
    if max_results and len(items) > max_results:
        items = items[:max_results]
    after_threshold = len(items)

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
        try:
            logger("hunt_jobs.complete", res)
        except Exception:
            pass
    return res
