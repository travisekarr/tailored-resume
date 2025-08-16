# Model config helpers
from models_config import load_models_cfg, ui_choices, ui_default
_model_cfg = load_models_cfg()

def _model_selectbox(label: str, group: str, *, key: str, disabled: bool = False):
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
# pages/03_job_history.py
import csv
from io import StringIO
from datetime import datetime, timedelta, timezone
import pytz
import os
from dotenv import load_dotenv

import pandas as pd
import streamlit as st

from job_store import query_jobs, prune_duplicates

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
    except Exception:
        return str(val)
st.set_page_config(page_title="Job History", layout="wide")
st.title("🗂️ Job History — Saved Results")

# ---------- Helpers ----------
def _boolish(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "y"):
        return True
    if s in ("0", "false", "no", "n", ""):
        return False
    try:
        return bool(int(s))
    except Exception:
        return bool(x)

def _arrow(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    for c in df.columns:
        lc = c.lower()
        if lc in ("id", "job_id"):
            df[c] = df[c].astype("string")
        elif lc in ("posted_at", "first_seen", "last_seen", "pulled_at", "created_at", "updated_at",
                    "submitted_at", "not_suitable_at"):
            df[c] = df[c].astype("string")
        elif lc in ("score", "resume_score"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif df[c].dtype == "object":
            df[c] = df[c].astype("string")
    return df

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Filters")
    sources = st.multiselect(
        "Sources",
        ["remotive", "remoteok", "weworkremotely", "hnrss"],
        default=["remotive", "remoteok", "weworkremotely", "hnrss"]
    )
    min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.01)
    days = st.selectbox("Posted within (days)", [0, 1, 3, 7, 14, 30, 60, 90, 180], index=7)
    query = st.text_input("Keyword (title/company/desc/url)")


    st.markdown("---")
    st.subheader("Model Selection")
    embedding_model_sel = _model_selectbox(
        "Embeddings model",
        group="embeddings",
        key="jh_embed_model",
        disabled=False,
    )
    order = st.selectbox("Sort by", ["score_desc", "date_desc", "updated_desc"], index=0)
    limit = st.slider("Max rows", 50, 2000, 500, 50)

    st.markdown("---")
    if st.button("🧹 Prune duplicates (keep best by score)", use_container_width=True):
        removed = prune_duplicates()
        st.success(f"Pruned {removed} duplicate rows.")

# ---------- Date window ----------
posted_start = None
posted_end = None
if days and days > 0:
    posted_end = datetime.now(timezone.utc).isoformat()
    posted_start = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

# ---------- Query ----------
rows = query_jobs(
    sources=sources,
    min_score=min_score,
    posted_start=posted_start,
    posted_end=posted_end,
    q=query if query.strip() else None,
    limit=limit,
    order=order
)

# Defensive normalize and flag filtering
if not rows:
    st.info("No rows match the current filters.")
    st.stop()

# Normalize basic fields + flags
norm = []
for r in rows:
    d = dict(r)
    d["job_id"] = d.get("job_id") or d.get("id")
    d["submitted"] = _boolish(d.get("submitted"))
    d["not_suitable"] = _boolish(d.get("not_suitable"))
    d["interviewed"] = _boolish(d.get("interviewed"))
    d["rejected"] = _boolish(d.get("rejected"))
    for k in [
        "posted_at", "first_seen", "last_seen", "pulled_at", "created_at", "updated_at",
        "submitted_at", "not_suitable_at", "interviewed_at", "rejected_at"
    ]:
        if k in d:
            d[k] = format_dt(d.get(k))
    norm.append(d)

# Apply flag filters (mutually-aware, but simple rules)

# ---------- Counts ----------
total = len(norm)
submitted_ct = sum(1 for r in norm if r["submitted"])
ns_ct = sum(1 for r in norm if r["not_suitable"])
neutral_ct = total - submitted_ct - ns_ct

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total (after filters)", total)
k2.metric("Submitted", submitted_ct)
k3.metric("Not suitable", ns_ct)
k4.metric("Neither", neutral_ct)

st.write(f"Showing {total} row(s).")

# ---------- Table ----------

# ---------- Tabs ----------
tab_labels = [
    "All", "Resume", "Submitted", "Un suitable", "Interviewed", "Rejected"
]
tab_keys = [
    "all", "resume", "submitted", "not_suitable", "interviewed", "rejected"
]
tabs = st.tabs(tab_labels)

tab_filters = {
    "all": lambda r: True,
    "resume": lambda r: bool(r.get("resume_path")),
    "submitted": lambda r: r.get("submitted", False),
    "not_suitable": lambda r: r.get("not_suitable", False),
    "interviewed": lambda r: r.get("interviewed", False),
    "rejected": lambda r: r.get("rejected", False),
}

for i, tab in enumerate(tabs):
    with tab:
        filtered = [r for r in norm if tab_filters[tab_keys[i]](r)]
        df = pd.DataFrame(filtered)
        preferred_cols = [
            "score", "title", "company", "location", "source",
            "posted_at", "submitted", "not_suitable", "interviewed", "rejected", "url"
        ]
        show_cols = [c for c in preferred_cols if c in df.columns]
        if "job_id" in df.columns and "job_id" not in show_cols:
            show_cols = ["job_id"] + show_cols
        df = _arrow(df)
        st.dataframe(
            df[show_cols],
            use_container_width=True,
            height=520,
            column_config={
                "url": st.column_config.LinkColumn("URL", display_text="Open"),
                "submitted": st.column_config.CheckboxColumn("Submitted"),
                "not_suitable": st.column_config.CheckboxColumn("Not suitable"),
                "interviewed": st.column_config.CheckboxColumn("Interviewed"),
                "rejected": st.column_config.CheckboxColumn("Rejected"),
            },
        )
        with st.expander("Show advanced columns (timestamps, resume path/score)"):
            adv_cols = [
                "job_id", "status", "user_notes", "first_seen", "last_seen", "updated_at",
                "submitted", "submitted_at", "not_suitable", "not_suitable_at",
                "interviewed", "interviewed_at", "rejected", "rejected_at",
                "not_suitable_reasons", "unsuitable_reason_note",
                "resume_path", "resume_score"
            ]
            adv_cols = [c for c in adv_cols if c in df.columns]
            if adv_cols:
                st.dataframe(df[adv_cols], use_container_width=True, height=300)
            else:
                st.caption("No advanced columns available in this dataset.")

# ---------- Export CSV ----------

# Collect all possible keys from all rows for CSV export
all_keys = set()
for r in norm:
    all_keys.update(r.keys())
csv_buf = StringIO()
writer = csv.DictWriter(csv_buf, fieldnames=sorted(all_keys))
writer.writeheader()
for r in norm:
    writer.writerow(r)

st.download_button(
    "⬇️ Download CSV",
    data=csv_buf.getvalue().encode("utf-8"),
    file_name="job_history.csv",
    mime="text/csv",
    use_container_width=True,
)
