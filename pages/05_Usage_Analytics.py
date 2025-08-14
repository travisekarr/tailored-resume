import dateutil.parser
def format_dt(val):
    if not val:
        return ""
    try:
        # Handle integer/float timestamps
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.strip().isdigit()):
            dt = datetime.fromtimestamp(float(val), tz=pytz.UTC)
        else:
            # Try parsing RFC2822, ISO, etc.
            dt = dateutil.parser.parse(str(val))
            if not dt.tzinfo:
                dt = dt.replace(tzinfo=pytz.UTC)
        tz = pytz.timezone(TZ)
        dt = dt.astimezone(tz)
        return dt.strftime(DT_FMT)
    except Exception:
        return str(val)
# pages/05_Usage_Analytics.py
import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, timezone
import pytz
from dotenv import load_dotenv

from job_store import fetch_usage

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
st.set_page_config(page_title="Usage Analytics", layout="wide")
st.title("📊 OpenAI Usage & Cost (Est.)")

def _now():
    return datetime.now(timezone.utc)

def _iso(d: datetime) -> str:
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.isoformat()

with st.sidebar:
    st.header("Filters")
    db_path = st.text_input("Database file", value="jobs.db")
    days = st.slider("Lookback (days)", 1, 90, 30)
    group = st.selectbox("Group by", ["day", "week", "month"], index=0)
    model_filter = st.text_input("Model contains", value="")
    endpoint_filter = st.selectbox("Endpoint", ["(any)", "chat.completions", "responses", "embeddings"], index=0)
    context_filter = st.text_input("Context equals (optional)", value="")  # e.g. ui:04_Generate_From_Jobs

since = _iso(_now() - timedelta(days=days))
until = _iso(_now())

usage = fetch_usage(
    db_path=db_path,
    since=since,
    until=until,
    model=None,  # substring filter below
    endpoint=None if endpoint_filter == "(any)" else endpoint_filter,
    context=context_filter or None,
    limit=50000,
)

if model_filter.strip():
    usage = [u for u in usage if model_filter.lower() in (u.get("model","").lower())]

if not usage:
    st.info("No usage records in the selected window.")
    st.stop()

df = pd.DataFrame(usage)
df["called_at"] = df["called_at"].apply(format_dt)

# Top summary
total_calls = len(df)
sum_in  = int(df["input_tokens"].fillna(0).sum())
sum_out = int(df["output_tokens"].fillna(0).sum())
sum_tot = int(df["total_tokens"].fillna(0).sum())
sum_cost = float(df["cost_usd"].fillna(0.0).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Calls", f"{total_calls:,}")
c2.metric("Input tokens", f"{sum_in:,}")
c3.metric("Output tokens", f"{sum_out:,}")
c4.metric("Total tokens", f"{sum_tot:,}")
c5.metric("Estimated cost", f"${sum_cost:,.4f}")

st.markdown("---")

# Breakdown by model + endpoint
st.subheader("By Model × Endpoint")
g_me = df.groupby(["model","endpoint"], dropna=False, as_index=False).agg(
    calls=("id","count"),
    input_tokens=("input_tokens","sum"),
    output_tokens=("output_tokens","sum"),
    total_tokens=("total_tokens","sum"),
    est_cost=("cost_usd","sum"),
)
st.dataframe(g_me.sort_values(["est_cost","total_tokens","calls"], ascending=False), use_container_width=True)

# Time breakdown
st.subheader(f"Time breakdown (per {group})")
key = {"day":"date","week":"week","month":"month"}[group]
g_time = df.groupby(key, as_index=False).agg(
    calls=("id","count"),
    input_tokens=("input_tokens","sum"),
    output_tokens=("output_tokens","sum"),
    total_tokens=("total_tokens","sum"),
    est_cost=("cost_usd","sum"),
)
st.dataframe(g_time.sort_values(key), use_container_width=True)

st.markdown("---")
st.subheader("Raw usage (latest first)")
st.dataframe(
    df[["called_at","context","endpoint","model","input_tokens","output_tokens","total_tokens","cost_usd","request_id"]]
      .sort_values("called_at", ascending=False),
    use_container_width=True, height=400
)

# Export
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv, file_name="api_usage.csv", mime="text/csv")
