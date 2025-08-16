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
# pages/01_job_finder.py
import os
import json
import streamlit as st
import yaml
import pytz
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime

from job_hunter import hunt_jobs, probe_sources, load_sources_yaml
from job_store import store_jobs


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

st.set_page_config(page_title="Job Finder (Aggregators)", layout="wide")
st.title("🔎 Job Finder — Aggregator Search")

st.markdown(
    "Searches broadly across Remotive, RemoteOK, WeWorkRemotely RSS, and HN RSS using your `job_sources.yaml`."
)

# ---------- Helpers ----------
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
def _safe_str(x):
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""

def _http_events_from_logs(logs):
    rows = []
    for ev in logs or []:
        tag = ev.get("msg", "")
        d = ev.get("data", {}) or {}
        if (
            tag in (
                "http.get", "http.error",
                "rss.fetch", "rss.http_fail", "rss.entries",
                "agg.results",
                "stage.counts", "stage.counts.source"
            )
            or tag.startswith("probe.")
            or tag.startswith("yaml.")
            or tag.startswith("embeddings.")
        ):
            rows.append({
                "when": ev.get("ts", ""),
                "tag": tag,
                "source": d.get("source"),
                "status": d.get("status"),
                "reason": d.get("reason") or d.get("error"),
                "len": d.get("len"),
                "url": d.get("url"),
                "count": d.get("count"),
                "raw_total": d.get("raw_total"),
                "after_remote": d.get("after_remote_filter"),
                "after_keywords": d.get("after_keyword_filters"),
                "after_dedupe": d.get("after_dedupe"),
                "after_threshold": d.get("after_threshold"),
                "min_score": d.get("min_score"),
                "remote_only": d.get("remote_only"),
                "model": d.get("model"),   # <-- NEW: see which model was used
                "jobs": d.get("jobs"),     # <-- NEW: how many inputs we sent
                "scores": d.get("scores"), # <-- NEW: embeddings.ok count
                "batch_index": d.get("batch_index"),
                "batch_size": d.get("batch_size"),
                "est_tokens": d.get("est_tokens"),
                "budget": d.get("budget"),
            })
    return rows

# Only allow **known embedding models** to reach the API.
# We'll intersect with account-visible models if we can list; otherwise we show this safe list.
_KNOWN_EMBEDDINGS = [
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",  # legacy, optional
]

def _embedding_model_options():
    if _openai_client is None:
        return list(_KNOWN_EMBEDDINGS)
    try:
        have = {m.id for m in getattr(_openai_client.models.list(), "data", [])}
        # Keep only known embeddings that your org actually has
        opts = [m for m in _KNOWN_EMBEDDINGS if m in have]
        return opts or list(_KNOWN_EMBEDDINGS)
    except Exception:
        return list(_KNOWN_EMBEDDINGS)

def _effective_embedding_model(selected: str, options: list[str]) -> str:
    """Guarantee a valid model name; fallback to text-embedding-3-small."""
    if selected in options:
        return selected
    # last resort
    return "text-embedding-3-small"

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Config")
    resume_path = st.text_input("Resume YAML", value="modular_resume_full.yaml")
    sources_yaml = st.text_input("Sources YAML (aggregators)", value="job_sources.yaml")

    st.markdown("### YAML Inspector")
    # Show/Hide toggle for YAML inspector
    try:
        show_yaml = st.toggle("Show", value=False, key="jf_show_yaml")
    except Exception:
        show_yaml = st.checkbox("Show", value=False, key="jf_show_yaml")

    if show_yaml:
        exists = os.path.exists(sources_yaml)
        st.write(f"File exists: **{exists}**")
        if exists:
            try:
                txt = open(sources_yaml, "r", encoding="utf-8").read()
            except Exception as e:
                txt = ""
                st.error(f"Read error: {e}")

            # Tabs: Parsed / Validation / Raw / Probe
            tabs = st.tabs(["Parsed", "Validation", "Raw", "Probe"])

            with tabs[0]:
                try:
                    parsed_cfg, parse_errs = load_sources_yaml(sources_yaml, debug=True)
                except Exception as e:
                    parsed_cfg, parse_errs = None, [f"read_error: {e}"]
                st.write("**Parsed config:**")
                if isinstance(parsed_cfg, dict):
                    st.json(parsed_cfg)
                else:
                    st.info("No parsed config available.")

            with tabs[1]:
                st.write("**Validation:**")
                if parse_errs:
                    for err in parse_errs:
                        st.error(err)
                else:
                    st.success("No validation errors.")

            with tabs[2]:
                st.write("**Raw YAML:**")
                st.code(txt or "# (empty)", language="yaml")

            with tabs[3]:
                st.write("**Endpoint probe:**")
                if st.button("Probe enabled sources", key="jf_probe_sources"):
                    try:
                        probe_rows = probe_sources(sources_yaml, debug=False)
                        if probe_rows:
                            # Ensure string types for Arrow friendliness
                            for r in probe_rows:
                                for k, v in list(r.items()):
                                    r[k] = _safe_str(v)
                            st.dataframe(probe_rows, use_container_width=True, height=300)
                        else:
                            st.info("No probe results.")
                    except Exception as e:
                        st.error(f"Probe failed: {e}")
        else:
            st.warning("Path not found. Check the filename and working directory.")

    st.markdown("---")
    use_embeddings = st.checkbox("Use OpenAI embeddings", value=False)
    embedding_model_sel = _model_selectbox(
        "Embeddings model",
        group="embeddings",
        key="jf_embed_model",
        disabled=not use_embeddings,
    )

    st.markdown("---")
    save_to_db = st.checkbox("Save results to jobs.db", value=True)
    show_debug = st.checkbox("Show debug logs", value=True)
    test_sources = st.button("🧪 Test Sources (HTTP)")

# ---------- Action Buttons ----------
col1, col2 = st.columns([1,1])
with col1:
    run = st.button("🚀 Fetch & Score Jobs", use_container_width=True)
with col2:
    download = st.button("⬇️ Download Last JSON", use_container_width=True)

# ---------- Placeholders ----------
results_container = st.empty()
stats_container = st.empty()
quicklinks_container = st.empty()
table_container = st.empty()
http_container = st.empty()
debug_container = st.empty()
download_container = st.empty()
store_container = st.empty()

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------- Probe sources ----------
if test_sources:
    rows = probe_sources(sources_yaml, debug=True)
    if rows:
        # normalize rows to strings for Arrow compatibility
        for r in rows:
            for k, v in list(r.items()):
                r[k] = _safe_str(v)
        st.markdown("### 🧪 Source Probes")
        st.dataframe(rows, use_container_width=True, height=260)
    else:
        st.warning("No aggregators configured or YAML invalid (open the YAML Inspector and check).")

# ---------- Run fetch ----------
if run:
    eff_embed_model = _effective_embedding_model(embedding_model_sel, emb_options)

    if use_embeddings and embedding_model_sel != eff_embed_model:
        st.warning(
            f"Selected embedding model `{embedding_model_sel}` isn’t available; "
            f"falling back to `{eff_embed_model}`."
        )

    with st.spinner("Fetching and scoring…"):
        res = hunt_jobs(
            resume_path=resume_path,
            sources_yaml=sources_yaml,
            use_embeddings=use_embeddings,
            embedding_model=eff_embed_model,
            debug=show_debug,
        )

    st.session_state.last_result = res

    if res.get("stats", {}).get("errors"):
        results_container.error("YAML errors detected — see debug log below and YAML Inspector.")
    else:
        results_container.success(
            f"Found {res['count']} jobs (generated at {res['generated_at']}). Sorted by score, then date."
        )

    stats = res.get("stats", {})
    stats_container.info(
        f"Stage counts → raw: {stats.get('raw_total',0)}, "
        f"after remote-only: {stats.get('after_remote_filter',0)}, "
        f"after keyword filters: {stats.get('after_keyword_filters',0)}, "
        f"after de-dupe: {stats.get('after_dedupe',0)}, "
        f"after threshold/max: {stats.get('after_threshold',0)}"
    )

    if save_to_db and res["count"]:
        info = store_jobs(res["items"])
        store_container.success(f"Saved to DB — inserted/updated: {info.get('inserted',0)}/{info.get('updated',0)}")

    # Quick links
    if res["count"]:
        top = res["items"][:10]
        quicklinks_container.markdown("**Top 10 (quick links):**")
        for it in top:
            quicklinks_container.markdown(
                f"- **{_safe_str(it.get('title',''))}** — {_safe_str(it.get('company',''))}  "
                f"[score={float(it.get('score',0)):.3f}]  \n  {_safe_str(it.get('url',''))}"
            )

    # Table
    if res["count"]:
        rows = []
        for it in res["items"]:
                rows.append({
                    "score": float(it.get("score", 0.0) or 0.0),
                    "title": _safe_str(it.get("title","")),
                    "company": _safe_str(it.get("company","")),
                    "location": _safe_str(it.get("location","")),
                    "remote": bool(it.get("remote")),
                    "source": _safe_str(it.get("source","")),
                    "posted_at": format_dt(it.get("posted_at","")),
                    "url": _safe_str(it.get("url","")),
                    "id": _safe_str(it.get("id","")),
                    "description": (
                        (_safe_str(it.get("description",""))[:300] + "…")
                        if len(_safe_str(it.get("description",""))) > 300
                        else _safe_str(it.get("description",""))
                    ),
                })
        table_container.dataframe(rows, use_container_width=True, height=480)

    # HTTP activity (every request + YAML events)
    logs = res.get("debug", []) if res else []
    http_rows = _http_events_from_logs(logs)
    if http_rows:
        st.markdown("### 🌐 HTTP & YAML Activity")
        st.dataframe(http_rows, use_container_width=True, height=260)

    # Full debug
    if show_debug:
        if logs:
            with debug_container.expander("🔍 Debug log (full trace)", expanded=False):
                for ev in logs:
                    st.markdown(f"**{_safe_str(ev.get('ts',''))} — {_safe_str(ev.get('msg',''))}**")
                    if "data" in ev and ev["data"] is not None:
                        st.json(ev["data"])
        else:
            debug_container.info("No debug events captured. Enable 'Show debug logs' and run again.")

# ---------- Download last JSON ----------
if download:
    if not st.session_state.last_result:
        download_container.warning("Run a fetch first.")
    else:
        jl = json.dumps(st.session_state.last_result, ensure_ascii=False, indent=2).encode("utf-8")
        download_container.download_button(
            "⬇️ Download JSON", data=jl, file_name="jobs_ranked.json", mime="application/json"
        )
