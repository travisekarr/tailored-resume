# scheduled_fetch.py
"""
Run this on a schedule to fetch & store new jobs.

Examples:
  # Use defaults
  python scheduled_fetch.py

  # Custom paths
  python scheduled_fetch.py --resume modular_resume_full.yaml --sources job_sources.yaml
"""
import argparse
from job_hunter import hunt_jobs
from job_store import store_jobs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default="modular_resume_full.yaml")
    ap.add_argument("--sources", default="job_sources.yaml")
    ap.add_argument("--use-embeddings", action="store_true")
    ap.add_argument("--embedding-model", default="text-embedding-3-small")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    try:
        res = hunt_jobs(
            resume_path=args.resume,
            sources_yaml=args.sources,
            use_embeddings=args.use_embeddings,
            embedding_model=args.embedding_model,
            debug=args.debug,
        )
        if res["count"]:
            info = store_jobs(res["items"])
            print(f"[scheduled_fetch] Stored {res['count']} jobs (inserted/updated approx: {info['inserted']}/{info['updated']})")
        else:
            print("[scheduled_fetch] 0 jobs this run.")
    except Exception as e:
        print(f"Error in scheduled fetch: {e}")

if __name__ == "__main__":
    main()
