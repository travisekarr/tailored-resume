# pages/01_job_finder.py
import os
import json
import streamlit as st
import yaml

from job_hunter import hunt_jobs, probe_sources, load_sources_yaml
from job_store import store_jobs

st.set_page_config(page_title="Job Finder (Aggregators)", layout="wide")
st.title("🔎 Job Finder — Aggregator Search")

st.markdown("Searches broadly across Remotive, RemoteOK, WeWorkRemotely RSS, and HN RSS using your `job_sources.yaml`.")

# Sidebar
with st.sidebar:
    st.header("Config")
    resume_path = st.text_input("Resume YAML", value="modular_resume_full.yaml")
    sources_yaml = st.text_input("Sources YAML (aggregators)", value="job_sources.yaml")

    st.markdown("### YAML Inspector")
    exists = os.path.exists(sources_yaml)
    st.write(f"File exists: **{exists}**")
    if exists:
        try:
            txt = open(sources_yaml, "r", encoding="utf-8").read()
            st.code(txt, language="yaml")
        except Exception as e:
            st.error(f"Read error: {e}")

        cfg, errs = load_sources_yaml(sources_yaml, debug=True)
        if errs:
            st.error("Validation errors:")
            for e in errs:
                st.write(f"- {e}")
        else:
            st.success("YAML parsed ✔")
            with st.expander("Parsed structure", expanded=False):
                st.json(cfg)
    else:
        st.warning("Path not found. Check the filename and working directory.")

    st.markdown("---")
    use_embeddings = st.checkbox("Use OpenAI embeddings", value=False)
    embedding_model = st.selectbox(
        "Embeddings model",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=0
    )

    st.markdown("---")
    save_to_db = st.checkbox("Save results to jobs.db", value=True)
    show_debug = st.checkbox("Show debug logs", value=True)
    test_sources = st.button("🧪 Test Sources (HTTP)")

col1, col2 = st.columns([1,1])
with col1:
    run = st.button("🚀 Fetch & Score Jobs", use_container_width=True)
with col2:
    download = st.button("⬇️ Download Last JSON", use_container_width=True)

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

def _http_events_from_logs(logs):
    rows = []
    for ev in logs or []:
        tag = ev.get("msg", "")
        d = ev.get("data", {}) or {}
        if tag in ("http.get", "http.error") or tag.startswith("probe.") or tag.startswith("yaml."):
            rows.append({
                "when": ev.get("ts", ""),
                "tag": tag,
                "status": d.get("status"),
                "reason": d.get("reason") or d.get("error"),
                "len": d.get("len"),
                "url": d.get("url"),
            })
    return rows

# Probe sources
if test_sources:
    rows = probe_sources(sources_yaml, debug=True)
    if rows:
        st.markdown("### 🧪 Source Probes")
        st.dataframe(rows, use_container_width=True, height=260)
    else:
        st.warning("No aggregators configured or YAML invalid (see YAML Inspector at left).")

# Run fetch
if run:
    with st.spinner("Fetching and scoring…"):
        res = hunt_jobs(
            resume_path=resume_path,
            sources_yaml=sources_yaml,
            use_embeddings=use_embeddings,
            embedding_model=embedding_model,
            debug=show_debug,
        )

    st.session_state.last_result = res

    if res.get("stats", {}).get("errors"):
        results_container.error("YAML errors detected — see debug log below and YAML Inspector.")
    else:
        results_container.success(f"Found {res['count']} jobs (generated at {res['generated_at']}). Sorted by score, then date.")

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
                f"- **{it.get('title','')}** — {it.get('company','')}  "
                f"[score={it.get('score',0):.3f}]  \n  {it.get('url','')}"
            )

    # Table
    if res["count"]:
        rows = [
            {
                "score": round(it.get("score", 0.0), 3),
                "title": it.get("title",""),
                "company": it.get("company",""),
                "location": it.get("location",""),
                "remote": bool(it.get("remote")),
                "source": it.get("source",""),
                "posted_at": it.get("posted_at",""),
                "url": it.get("url",""),
                "id": it.get("id",""),
                "description": (it.get("description","")[:300] + "…") if len(it.get("description","")) > 300 else it.get("description",""),
            }
            for it in res["items"]
        ]
        table_container.dataframe(rows, use_container_width=True, height=480)

    # HTTP activity (every request + YAML events)
    logs = res.get("debug", []) if res else []
    http_rows = _http_events_from_logs(logs)
    if http_rows:
        st.markdown("### 🌐 HTTP & YAML Activity")
        http_container.dataframe(http_rows, use_container_width=True, height=260)

    # Full debug
    if show_debug:
        if logs:
            with debug_container.expander("🔍 Debug log (full trace)", expanded=False):
                for ev in logs:
                    st.markdown(f"**{ev.get('ts','')} — {ev.get('msg','')}**")
                    if "data" in ev and ev["data"] is not None:
                        st.json(ev["data"])
        else:
            debug_container.info("No debug events captured. Enable 'Show debug logs' and run again.")

# Download last JSON
if download:
    if not st.session_state.last_result:
        download_container.warning("Run a fetch first.")
    else:
        jl = json.dumps(st.session_state.last_result, ensure_ascii=False, indent=2).encode("utf-8")
        download_container.download_button("⬇️ Download JSON", data=jl, file_name="jobs_ranked.json", mime="application/json")
