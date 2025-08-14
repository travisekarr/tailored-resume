# pages/03_job_history.py
import csv
from io import StringIO
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from job_store import query_jobs, prune_duplicates

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
    days = st.selectbox("Posted within (days)", [0, 1, 3, 7, 14, 30, 60, 90], index=4)
    query = st.text_input("Keyword (title/company/desc/url)")

    # New: flag filters
    st.markdown("---")
    st.subheader("Flags")
    only_submitted = st.checkbox("Show submitted only", value=False)
    only_not_suitable = st.checkbox("Show not suitable only", value=False)
    hide_flagged = st.checkbox("Hide submitted & not suitable", value=False)

    st.markdown("---")
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
    # Add unified job_id for display/export if only 'id' exists
    d["job_id"] = d.get("job_id") or d.get("id")

    # Flags may be missing in older rows — coerce safely
    d["submitted"] = _boolish(d.get("submitted"))
    d["not_suitable"] = _boolish(d.get("not_suitable"))

    norm.append(d)

# Apply flag filters (mutually-aware, but simple rules)
if hide_flagged:
    norm = [r for r in norm if not r["submitted"] and not r["not_suitable"]]
elif only_submitted and only_not_suitable:
    norm = [r for r in norm if r["submitted"] or r["not_suitable"]]
elif only_submitted:
    norm = [r for r in norm if r["submitted"]]
elif only_not_suitable:
    norm = [r for r in norm if r["not_suitable"]]
# If none of the toggles are on, show all jobs (unless hide_flagged is checked above)

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
df = pd.DataFrame(norm)

# Choose a concise but useful default column set
preferred_cols = [
    "score", "title", "company", "location", "source",
    "posted_at", "submitted", "not_suitable", "url"
]
show_cols = [c for c in preferred_cols if c in df.columns]
# Always add job_id at front if present
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
    },
)

# Optional: expanded view with more columns
with st.expander("Show advanced columns (timestamps, resume path/score)"):
    adv_cols = [
        "job_id", "status", "user_notes", "first_seen", "last_seen", "updated_at",
        "submitted", "submitted_at", "not_suitable", "not_suitable_at",
        "not_suitable_reasons", "unsuitable_reason_note",
        "resume_path", "resume_score"
    ]
    adv_cols = [c for c in adv_cols if c in df.columns]
    if adv_cols:
        st.dataframe(df[adv_cols], use_container_width=True, height=300)
    else:
        st.caption("No advanced columns available in this dataset.")

# ---------- Export CSV ----------
csv_buf = StringIO()
writer = csv.DictWriter(csv_buf, fieldnames=list(df.columns))
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
