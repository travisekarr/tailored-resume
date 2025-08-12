# pages/03_job_history.py
import csv
from io import StringIO
from datetime import datetime, timedelta, timezone

import streamlit as st
from job_store import query_jobs, prune_duplicates

st.set_page_config(page_title="Job History", layout="wide")
st.title("🗂️ Job History — Saved Results")

with st.sidebar:
    st.header("Filters")
    sources = st.multiselect("Sources", ["remotive", "remoteok", "weworkremotely", "hnrss"], default=["remotive","remoteok","weworkremotely","hnrss"])
    min_score = st.slider("Min score", 0.0, 1.0, 0.0, 0.01)
    days = st.selectbox("Posted within (days)", [0, 1, 3, 7, 14, 30, 60, 90], index=4)
    query = st.text_input("Keyword")

    order = st.selectbox("Sort by", ["score_desc", "date_desc", "updated_desc"], index=0)
    limit = st.slider("Max rows", 50, 2000, 500, 50)

    st.markdown("---")
    if st.button("🧹 Prune duplicates (keep best by score)", use_container_width=True):
        removed = prune_duplicates()
        st.success(f"Pruned {removed} duplicate rows.")

# Date range
posted_start = None
posted_end = None
if days and days > 0:
    posted_end = datetime.now(timezone.utc).isoformat()
    posted_start = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

rows = query_jobs(
    sources=sources,
    min_score=min_score,
    posted_start=posted_start,
    posted_end=posted_end,
    q=query if query.strip() else None,
    limit=limit,
    order=order
)

st.write(f"Showing {len(rows)} rows.")
if rows:
    # Short table view
    tbl = [
        {
            "score": round(r["score"], 3),
            "title": r["title"],
            "company": r["company"],
            "source": r["source"],
            "posted_at": r["posted_at"],
            "url": r["url"],
        } for r in rows
    ]
    st.dataframe(tbl, use_container_width=True, height=520)

    # Export CSV
    if st.button("⬇️ Export CSV", use_container_width=True):
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        st.download_button("Download jobs.csv", data=output.getvalue().encode("utf-8"), file_name="jobs.csv", mime="text/csv")
else:
    st.info("No rows match the current filters.")
