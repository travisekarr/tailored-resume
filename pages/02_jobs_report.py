# pages/02_Job_Reports.py
import os
import csv
import io
import yaml
import json
import re
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from job_store import (
    query_new_since,
    query_changed_since,
    query_top_matches,
    load_latest,
    set_job_status,
)

# ====== OPTIONAL: load resume to build profile terms for score debugging ======
# Uses same path your main app uses
RESUME_PATH = "modular_resume_full.yaml"

# ---------- Defaults ----------
DEFAULTS_PATH = "report_defaults.yaml"

DEFAULTS_FALLBACK = {
    "db": "jobs.db",
    "since": "24h",
    "min_score": 0.0,
    "hide_stale_days": None,    # e.g. 14
    "search": "",
    "top_count": 15,
    "new_limit": 100,
    "changed_limit": 200,
    "filters": {
        "title_contains": "",
        "company_contains": "",
        "location_contains": "",
        "description_contains": "",
        "sources": [],          # e.g. ["greenhouse","lever","rss"]
        "remote_only": False,
        "posted_after": "",     # "YYYY-MM-DD"
        "changed_fields": [],   # for changed view
        "statuses": [],         # e.g. ["saved","interested","applied"]
        "starred_only": False,

        # NEW: Location allowlist
        "use_location_allowlist": True,
        "allowed_locations": ["USA", "VA", "North America", "Remote", "Worldwide"],
        "allow_empty_location": True,
    },
}

# -------------------- small stopword set to avoid noisy highlights --------------------
STOPWORDS = {
    "and","the","with","for","your","you","our","their","this","that","from","have","will","are","was",
    "skills","experience","years","team","work","ability","in","to","of","on","as","by","or","an","be",
    "a","at","we","us","it","its","about","who","what","when","where","how"
}

# ================== Load/save defaults ==================
def load_defaults(path=DEFAULTS_PATH):
    if not os.path.exists(path):
        return DEFAULTS_FALLBACK.copy()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = {**DEFAULTS_FALLBACK, **data}
    merged["filters"] = {**DEFAULTS_FALLBACK["filters"], **(merged.get("filters") or {})}
    return merged

def save_defaults(cfg, path=DEFAULTS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

# ================== Helpers ==================
def parse_since(s: str) -> str:
    s = (s or "").strip().lower()
    now = datetime.now(timezone.utc)
    if s.endswith("h"):
        hrs = float(s[:-1])
        return (now - timedelta(hours=hrs)).isoformat()
    if s.endswith("d"):
        days = float(s[:-1])
        return (now - timedelta(days=days)).isoformat()
    try:
        dt = datetime.fromisoformat(s.replace("z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()
    except Exception:
        return (now - timedelta(hours=24)).isoformat()

def to_csv(rows, field_order=None) -> bytes:
    if not rows:
        return b""
    fieldnames = field_order or list(rows[0].keys())
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue().encode("utf-8")

def search_filter(rows, q: str):
    if not q:
        return rows
    q = q.lower().strip()
    out = []
    for r in rows:
        blob = " ".join(
            str(r.get(k, "")) for k in ("title", "company", "location", "description", "url")
        ).lower()
        if all(part in blob for part in q.split()):
            out.append(r)
    return out

def filter_jobs_advanced(
    rows,
    title_q,
    company_q,
    loc_q,
    desc_q,
    source_sel,
    remote_only,
    posted_after_str,
    statuses=None,
    starred_only=False,
    *,
    use_location_allowlist=False,
    allowed_locations=None,
    allow_empty_location=True,
):
    def contains(val, needle):
        return (needle.lower() in (val or "").lower()) if needle else True

    allowed_locations = allowed_locations or []

    posted_after = None
    if posted_after_str.strip():
        try:
            posted_after = datetime.fromisoformat(posted_after_str.strip())
        except Exception:
            try:
                posted_after = dtparse.parse(posted_after_str.strip())
            except Exception:
                posted_after = None

    out = []
    for r in rows:
        if not contains(r.get("title",""), title_q):      continue
        if not contains(r.get("company",""), company_q):  continue
        if not contains(r.get("location",""), loc_q):     continue
        if not contains(r.get("description",""), desc_q): continue
        if source_sel and (r.get("source") not in source_sel): continue
        if remote_only and not bool(r.get("remote")):     continue
        if statuses and (r.get("status") or "new") not in statuses: continue
        if starred_only and int(r.get("starred") or 0) != 1: continue

        # NEW: allowlist check
        if use_location_allowlist:
            if not location_allowed(r.get("location"), allowed_locations, allow_empty_location):
                continue

        if posted_after:
            pa = r.get("posted_at")
            try:
                if not pa or dtparse.parse(pa) < posted_after:
                    continue
            except Exception:
                continue

        out.append(r)
    return out

def filter_changed(events, fields):
    if not fields:
        return events
    return [e for e in events if e.get("field") in fields]

def fmt_est(dt_val):
    """Return YYYY-MM-DDTHH:MM:SS in America/New_York; drop microseconds."""
    if not dt_val:
        return ""
    try:
        # numeric epoch (RemoteOK occasionally)
        if isinstance(dt_val, (int, float)) or (isinstance(dt_val, str) and dt_val.strip().isdigit()):
            ts = int(float(dt_val))
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        else:
            s = str(dt_val).strip().replace("Z", "+00:00")
            dt = dtparse.parse(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)

        est = dt.astimezone(ZoneInfo("America/New_York")).replace(microsecond=0)
        return est.strftime("%Y-%m-%dT%H:%M:%S")
    except Exception:
        # if parsing fails, just return original string
        return str(dt_val)

def location_allowed(loc_value: str | None, allowed: list[str], allow_empty: bool) -> bool:
    """
    Return True if:
      - location is empty/None and allow_empty is True, or
      - any token in `allowed` is a substring of location (case-insensitive).
    """
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

def make_arrow_friendly(df):
    """
    Ensure DF is Arrow-compatible:
    - id / job_id => strings
    - date-like cols already formatted via fmt_est(), keep as strings
    - score numeric, starred as string/icon
    """
    for col in df.columns:
        low = col.lower()
        if low in ("id", "job_id"):
            df[col] = df[col].astype("string")
        elif low in ("posted_at", "first_seen", "last_seen", "pulled_at", "created_at", "updated_at", "changed_at"):
            df[col] = df[col].astype("string")
        elif low == "score":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif low == "starred":
            df[col] = df[col].astype("string")
        else:
            if df[col].dtype == "object":
                df[col] = df[col].astype("string")
    return df

# ================== Profile term builder (for score debugging) ==================
def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9+\-/#\.]{3,}", text.lower())

def _load_resume_terms(path=RESUME_PATH) -> dict:
    """
    Build a simple profile from resume YAML:
      - all 'tags' across sections
      - tokens from summaries/experience contributions
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            sections = yaml.safe_load(f) or []
    except Exception:
        return {"tags": set(), "tokens": set()}

    tags = set()
    tokens = set()

    for sec in sections:
        if isinstance(sec, dict):
            # tags
            for t in (sec.get("tags") or []):
                t2 = str(t).strip().lower()
                if t2 and t2 not in STOPWORDS:
                    tags.add(t2)
            # summaries/entries
            if "summary" in sec and sec["summary"]:
                for tok in _tokenize(sec["summary"]):
                    if tok not in STOPWORDS:
                        tokens.add(tok)
            if sec.get("type") == "experience":
                for contrib in sec.get("contributions") or []:
                    d = (contrib or {}).get("description", "")
                    for tok in _tokenize(d):
                        if tok not in STOPWORDS:
                            tokens.add(tok)
            elif "entries" in sec and sec.get("entries"):
                for entry in sec.get("entries") or []:
                    if isinstance(entry, dict):
                        for v in entry.values():
                            for tok in _tokenize(str(v)):
                                if tok not in STOPWORDS:
                                    tokens.add(tok)
    return {"tags": tags, "tokens": tokens}

PROFILE = _load_resume_terms()

def _job_blob(job: dict) -> str:
    return " ".join([str(job.get("title","")), str(job.get("description",""))])

def _overlap(a: set[str], b: set[str]) -> list[str]:
    return sorted(a & b, key=lambda x: (len(x), x), reverse=True)

def _interesting_tokens(tokens: list[str]) -> list[str]:
    # Surface tokens that often matter in tech JDs
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if any(ch in t for ch in ".#+-/"):
            out.append(t)
        elif t.isupper() and len(t) <= 6:
            out.append(t)   # e.g., SRE, PCI, SOC2
    return out

def _highlight(text: str, hits: set[str]) -> str:
    if not text or not hits:
        return text or ""
    safe = re.sub(r"&", "&amp;", text)
    safe = re.sub(r"<", "&lt;", safe)
    # longest-first replacement to avoid nesting
    for w in sorted(hits, key=len, reverse=True):
        pattern = re.compile(rf"(?<![\w])({re.escape(w)})s?(?![\w])", flags=re.IGNORECASE)
        safe = pattern.sub(r"<mark>\g<1></mark>", safe)
    # Light/dark friendly mark color
    style = """
    <style>
      mark { background: #ffeb99; color: #000; padding: 0 .1em; border-radius: 2px; }
      @media (prefers-color-scheme: dark) {
        mark { background: #775500; color: #fff; }
      }
      .jdbox { white-space: pre-wrap; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial; }
    </style>
    """
    return style + f"<div class='jdbox'>{safe}</div>"

# ================== UI ==================
st.set_page_config(page_title="Job Reports", layout="wide")
st.title("📊 Job Reports")

defaults = load_defaults()

with st.sidebar:
    st.header("Filters")

    db_path = st.text_input("Database file", value=defaults["db"])

    presets = ["24h", "48h", "3d", "7d", "14d", "30d"]
    since_default = defaults["since"]
    preset_index = presets.index(since_default) if since_default in presets else len(presets) - 1
    since_choice = st.selectbox("Window for 'New' & 'Changed'", presets + ["Custom (ISO)"],
                                index=preset_index if since_default in presets else len(presets))
    if since_choice == "Custom (ISO)":
        since_value = st.text_input("Custom ISO (e.g., 2025-08-01T12:00:00Z)", value=since_default if since_default not in presets else "")
    else:
        since_value = since_choice

    min_score = st.slider("Min score", 0.0, 1.0, float(defaults["min_score"]), 0.01)

    hide_stale = st.checkbox("Hide roles not seen within N days", value=defaults["hide_stale_days"] is not None)
    stale_days = st.slider("Stale days", 1, 60, int(defaults["hide_stale_days"] or 14), disabled=not hide_stale)

    search_query = st.text_input("Global search (title/company/location/description/url)", value=defaults["search"])

    top_count = st.slider("Top matches to show", 5, 100, int(defaults["top_count"]))
    new_limit = st.slider("Max 'New since' items", 10, 500, int(defaults["new_limit"]))
    changed_limit = st.slider("Max 'Changed since' events", 10, 1000, int(defaults["changed_limit"]))

    st.markdown("---")
    st.subheader("Field-specific filters")
    fdefs = defaults["filters"]
    title_q   = st.text_input("Title contains", fdefs.get("title_contains",""))
    company_q = st.text_input("Company contains", fdefs.get("company_contains",""))
    loc_q     = st.text_input("Location contains", fdefs.get("location_contains",""))
    desc_q    = st.text_input("Description contains", fdefs.get("description_contains",""))
    source_sel = st.multiselect("Source", ["greenhouse","lever","rss"], fdefs.get("sources") or [])
    remote_only = st.checkbox("Remote only", value=bool(fdefs.get("remote_only")))
    posted_after_str = st.text_input("Posted after (YYYY-MM-DD)", fdefs.get("posted_after",""))

    st.markdown("**Location allowlist**")
    use_location_allowlist = st.checkbox(
        "Apply location allowlist",
        value=bool(fdefs.get("use_location_allowlist", True))
    )
    allowed_locations_csv = st.text_input(
        "Allowed location keywords (comma-separated)",
        ", ".join(fdefs.get("allowed_locations") or ["USA", "VA", "North America", "Remote", "Worldwide"])
    )
    allow_empty_location = st.checkbox(
        "Include jobs with no location listed",
        value=bool(fdefs.get("allow_empty_location", True))
    )
    allowed_locations_list = [s.strip() for s in allowed_locations_csv.split(",") if s.strip()]

    statuses_sel = st.multiselect(
        "Status",
        ["new","saved","interested","applied","archived"],
        fdefs.get("statuses") or []
    )
    starred_only = st.checkbox("Starred only", value=bool(fdefs.get("starred_only")))

    st.markdown("---")
    changed_fields = st.multiselect(
        "Changed fields (for 'Changed since' view)",
        ["title","location","url","posted_at","description_sha","score","status","user_notes","starred"],
        fdefs.get("changed_fields") or []
    )

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        refresh = st.button("🔄 Refresh")
    with right:
        if st.button("💾 Save current filters as defaults"):
            new_cfg = {
                "db": db_path,
                "since": since_value,
                "min_score": float(min_score),
                "hide_stale_days": int(stale_days) if hide_stale else None,
                "search": search_query,
                "top_count": int(top_count),
                "new_limit": int(new_limit),
                "changed_limit": int(changed_limit),
                "filters": {
                    "title_contains": title_q,
                    "company_contains": company_q,
                    "location_contains": loc_q,
                    "description_contains": desc_q,
                    "sources": source_sel,
                    "remote_only": bool(remote_only),
                    "posted_after": posted_after_str,
                    "statuses": statuses_sel,
                    "starred_only": bool(starred_only),
                    "changed_fields": changed_fields,
                    "use_location_allowlist": bool(use_location_allowlist),
                    "allowed_locations": allowed_locations_list,
                    "allow_empty_location": bool(allow_empty_location),
                }
            }
            save_defaults(new_cfg)
            st.success(f"Saved defaults → {DEFAULTS_PATH}")

since_iso = parse_since(since_value)
hide_days = int(stale_days) if hide_stale else None

# ---------- Query ----------
NEW = query_new_since(db_path, since_iso, min_score=float(min_score), hide_stale_days=hide_days, limit=int(new_limit))
CHANGED = query_changed_since(db_path, since_iso, limit=int(changed_limit))
TOP = query_top_matches(db_path, limit=int(top_count), min_score=float(min_score), hide_stale_days=hide_days)
LATEST = load_latest(db_path, limit=20)

# Global search
NEW_f = search_filter(NEW, search_query)
CHANGED_f = search_filter(CHANGED, search_query)
TOP_f = search_filter(TOP, search_query)
LATEST_f = search_filter(LATEST, search_query)

# Field-specific + status/star filters
NEW_f = filter_jobs_advanced(
    NEW_f, title_q, company_q, loc_q, desc_q, source_sel, remote_only, posted_after_str,
    statuses_sel, starred_only,
    use_location_allowlist=use_location_allowlist,
    allowed_locations=allowed_locations_list,
    allow_empty_location=allow_empty_location,
)
TOP_f = filter_jobs_advanced(
    TOP_f, title_q, company_q, loc_q, desc_q, source_sel, remote_only, posted_after_str,
    statuses_sel, starred_only,
    use_location_allowlist=use_location_allowlist,
    allowed_locations=allowed_locations_list,
    allow_empty_location=allow_empty_location,
)
LATEST_f = filter_jobs_advanced(
    LATEST_f, title_q, company_q, loc_q, desc_q, source_sel, remote_only, posted_after_str,
    statuses_sel, starred_only,
    use_location_allowlist=use_location_allowlist,
    allowed_locations=allowed_locations_list,
    allow_empty_location=allow_empty_location,
)
CHANGED_f = filter_changed(CHANGED_f, changed_fields)

# ---------- KPIs ----------
k1, k2, k3, k4 = st.columns(4)
k1.metric("New since window", len(NEW_f))
k2.metric("Changed since window", len(CHANGED_f))
k3.metric("Top matches (shown)", len(TOP_f))
k4.metric("Latest seen (shown)", len(LATEST_f))

st.markdown("---")

def section_update_controls(label, rows, db_path):
    if not rows:
        return

    # Stable, section-specific prefix for widget keys
    prefix = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "sec"

    with st.expander(f"Update status / notes — {label}"):

        # Pick a job (key is unique per section)
        options = {f"{r.get('title','')} — {r.get('company','')}": r for r in rows[:300]}
        sel = st.selectbox(
            "Pick a job",
            list(options.keys()),
            key=f"{prefix}_pick_job",
        )
        job = options[sel]
        job_id = job.get("job_id") or job.get("id")

        statuses = ["new", "saved", "interested", "applied", "archived"]
        current_status = (job.get("status") or "new")
        try:
            idx = statuses.index(current_status)
        except ValueError:
            idx = 0

        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            new_status = st.selectbox(
                "Status",
                statuses,
                index=idx,
                key=f"{prefix}_status_{job_id}",
            )
        with c2:
            new_star = st.checkbox(
                "⭐ Star",
                value=bool(int(job.get("starred") or 0)),
                key=f"{prefix}_star_{job_id}",
            )
        with c3:
            new_notes = st.text_input(
                "Notes",
                value=job.get("user_notes") or "",
                placeholder="Optional…",
                key=f"{prefix}_notes_{job_id}",
            )

        if st.button("Save status/notes", key=f"{prefix}_save_{job_id}"):
            ok = set_job_status(
                db_path,
                job_id,
                status=new_status,
                user_notes=new_notes,
                starred=new_star,
            )
            if ok:
                st.success("Updated. Refresh to see the latest values in the tables.")
            else:
                st.error("Update failed (job not found).")

def table_rows(rows, cols):
    date_fields = {"posted_at","first_seen","last_seen","pulled_at","created_at","updated_at","changed_at"}

    def norm_row(r):
        # normalize core fields
        out = {
            "score": round(float(r.get("score", 0.0) or 0.0), 3),
            "status": r.get("status","new"),
            "starred": "⭐" if int(r.get("starred") or 0) == 1 else "",
        }
        # if source has 'id' but not 'job_id', expose as job_id for display/export
        if "job_id" in cols or "job_id" in r:
            out["job_id"] = r.get("job_id") or r.get("id", "")
        for k in cols:
            if k == "job_id":
                continue  # already handled
            v = r.get(k, "")
            if k in date_fields:
                v = fmt_est(v)
            out[k] = v
        return out

    return [norm_row(r) for r in rows]

# ---------- New since ----------
st.subheader("🆕 New since window")
if NEW_f:
    df_new = pd.DataFrame(table_rows(
        NEW_f, ["title","company","location","source","posted_at","first_seen","last_seen","url"]
    ))
    df_new = make_arrow_friendly(df_new)
    st.dataframe(
        df_new,
        use_container_width=True,
        height=360,
        column_config={"url": st.column_config.LinkColumn("URL", display_text="Open")},
    )
    with st.expander("Quick links"):
        for r in NEW_f[:30]:
            st.markdown(f"- **{r.get('title','')}** — {r.get('company','')}  \n  {r.get('url','')}")
    st.download_button(
        "⬇️ Export 'New since' CSV",
        data=to_csv(NEW_f, field_order=[
            "job_id","score","status","starred","title","company","location","remote","source","url",
            "posted_at","first_seen","last_seen","description","user_notes"
        ]),
        file_name="jobs_new_since.csv",
        mime="text/csv",
    )
    section_update_controls("New since", NEW_f, db_path)
else:
    st.info("No new jobs in the selected window (after filters).")

st.markdown("---")

# ---------- Changed since ----------
st.subheader("🔁 Changed since window")
if CHANGED_f:
    # Normalize changed_at as well
    df_changed = pd.DataFrame([
        {
            "changed_at": fmt_est(r.get("changed_at", "")),
            "field": r.get("field", ""),
            "old_value": r.get("old_value", ""),
            "new_value": r.get("new_value", ""),
            "title": r.get("title", ""),
            "company": r.get("company", ""),
            "score": round(float(r.get("score", 0.0) or 0.0), 3),
            "status": r.get("status", ""),
            "source": r.get("source", ""),
            "url": r.get("url", ""),
            "posted_at": fmt_est(r.get("posted_at", "")),
            "last_seen": fmt_est(r.get("last_seen", "")),
            "job_id": r.get("job_id", "") or r.get("id", ""),
        }
        for r in CHANGED_f
    ])
    df_changed = make_arrow_friendly(df_changed)
    st.dataframe(
        df_changed,
        use_container_width=True,
        height=360,
        column_config={"url": st.column_config.LinkColumn("URL", display_text="Open")},
    )
    st.download_button(
        "⬇️ Export 'Changed since' CSV",
        data=to_csv(CHANGED_f, field_order=[
            "changed_at", "job_id", "field", "old_value", "new_value",
            "title", "company", "score", "status", "source", "url",
            "posted_at", "last_seen"
        ]),
        file_name="jobs_changed_since.csv",
        mime="text/csv",
    )
else:
    st.info("No changes in the selected window (after filters).")


st.markdown("---")

# ---------- Top matches ----------
st.subheader("🏆 Top matches")
if TOP_f:
    df_top = pd.DataFrame(table_rows(
        TOP_f, ["title","company","location","source","posted_at","last_seen","url"]
    ))
    df_top = make_arrow_friendly(df_top)
    st.dataframe(
        df_top,
        use_container_width=True,
        height=360,
        column_config={"url": st.column_config.LinkColumn("URL", display_text="Open")},
    )
    st.download_button(
        "⬇️ Export 'Top matches' CSV",
        data=to_csv(TOP_f, field_order=[
            "job_id","score","status","starred","title","company","location","remote","source","url",
            "posted_at","first_seen","last_seen","description","user_notes"
        ]),
        file_name="jobs_top_matches.csv",
        mime="text/csv",
    )
    section_update_controls("Top matches", TOP_f, db_path)
else:
    st.info("No top matches (consider lowering min score).")


st.markdown("---")

# ---------- Latest seen ----------
st.subheader("🗂️ Latest seen")
if LATEST_f:
    df_latest = pd.DataFrame(table_rows(
        LATEST_f, ["title","company","source","posted_at","last_seen","url"]
    ))
    df_latest = make_arrow_friendly(df_latest)
    st.dataframe(
        df_latest,
        use_container_width=True,
        height=320,
        column_config={"url": st.column_config.LinkColumn("URL", display_text="Open")},
    )
    section_update_controls("Latest seen", LATEST_f, db_path)
else:
    st.info("No jobs stored yet. Run your hunter first.")

# ---------- Score distribution ----------
with st.expander("Score distribution (Top + New + Latest)"):
    scores = [r.get("score",0.0) for r in (TOP_f + NEW_f + LATEST_f)]
    if scores:
        hist, bins = np.histogram(scores, bins=20, range=(0.0, 1.0))
        st.bar_chart(hist)
        st.caption(f"Min: {min(scores):.3f} | Median: {np.median(scores):.3f} | Max: {max(scores):.3f}")
    else:
        st.write("No scores to show.")

# ================== NEW: Score Debugger ==================
st.markdown("---")
st.subheader("🔎 Score Debugger (Why is this score high/low?)")

# Pool candidates for inspection (dedupe by job_id/url)
def _dedupe(rows):
    seen = set()
    out = []
    for r in rows:
        key = r.get("job_id") or r.get("url") or r.get("id")
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out

ALL_ROWS = _dedupe(TOP_f + NEW_f + LATEST_f)

if not ALL_ROWS:
    st.info("No jobs available to debug. Run a hunt or relax filters.")
else:
    options = {
        f"{r.get('title','(no title)')} — {r.get('company','(no company)')} [{r.get('source','src')}]": r
        for r in ALL_ROWS[:500]
    }
    pick = st.selectbox("Select a job to inspect", list(options.keys()), key="dbg_pick")
    job = options[pick]

    # Basic info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{float(job.get('score',0.0) or 0.0):.3f}")
    c2.metric("Source", job.get("source",""))
    c3.metric("Posted", fmt_est(job.get("posted_at")))
    c4.metric("Last seen", fmt_est(job.get("last_seen")))

    # If your pipeline ever stores a JSON score breakdown, show it
    score_info = job.get("score_info") or job.get("score_components")
    if score_info:
        try:
            parsed = score_info if isinstance(score_info, dict) else json.loads(score_info)
            with st.expander("Raw score components (from store)"):
                st.json(parsed)
        except Exception:
            pass

    # Build overlaps against your profile terms
    title = (job.get("title") or "").strip()
    desc = (job.get("description") or "").strip()
    blob = _job_blob(job).lower()

    job_tokens = set(_tokenize(blob)) - STOPWORDS
    profile_tags = set(PROFILE.get("tags") or set())
    profile_tokens = set(PROFILE.get("tokens") or set())

    tag_hits = _overlap(profile_tags, job_tokens)
    token_hits = _overlap(profile_tokens, job_tokens)

    # Densities
    n_chars = max(len(desc), 1)
    density = (len(token_hits) / n_chars) * 1000  # hits per 1000 chars

    # Title matching quick signal
    title_tokens = set(_tokenize(title)) - STOPWORDS
    title_overlap = len(title_tokens & (profile_tags | profile_tokens))
    title_strength = "weak"
    if title_overlap >= 3:
        title_strength = "strong"
    elif title_overlap == 2:
        title_strength = "medium"

    # Missing-but-interesting tokens (appear in job, not in your profile)
    missing = job_tokens - (profile_tags | profile_tokens)
    missing_interesting = _interesting_tokens(list(missing))
    # Show top 15 by length (rough proxy for specificity)
    missing_surfaced = sorted(set(missing_interesting), key=len, reverse=True)[:15]

    with st.expander("Overlap metrics"):
        st.write(f"- **Title strength:** {title_strength} (overlap {title_overlap})")
        st.write(f"- **Profile tag hits:** {len(tag_hits)}")
        st.write(f"- **Profile token hits:** {len(token_hits)}")
        st.write(f"- **Hit density:** {density:.3f} matches / 1,000 chars of description")

        if tag_hits:
            st.write("**Matched tags:** " + ", ".join(tag_hits[:50]))
        if token_hits:
            st.write("**Matched tokens:** " + ", ".join(token_hits[:50]))

    with st.expander("Job description with highlights (profile matches)"):
        hits = set(tag_hits) | set(token_hits)
        st.markdown(_highlight(desc or "(no description provided)", hits), unsafe_allow_html=True)

    with st.expander("Potential keywords to add to your resume/tags"):
        if missing_surfaced:
            st.write(", ".join(missing_surfaced))
            st.caption("These appear in the job text but not in your profile/tags. Consider adding if relevant.")
        else:
            st.write("Nothing obvious — your profile already covers most of this posting.")
