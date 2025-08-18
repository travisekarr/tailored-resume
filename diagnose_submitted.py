import job_store
from datetime import datetime, timezone, timedelta

sources = ["remotive","remoteok","weworkremotely","hnrss"]

submitted = job_store.query_submitted()
print(f"DB submitted count: {len(submitted)}")
submitted_keys = []
for r in submitted:
    k = r.get('job_id') or r.get('id')
    submitted_keys.append(str(k))
    print(f"SUBMITTED: id={r.get('id')} job_id={r.get('job_id')} source={r.get('source')} posted_at={r.get('posted_at')} submitted_at={r.get('submitted_at')}")

posted_start = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
posted_end = datetime.now(timezone.utc).isoformat()
rows = job_store.query_jobs(sources=sources, min_score=0.0, posted_start=posted_start, posted_end=posted_end, q=None, limit=2000, order='score_desc')
print(f"\nUI query (sources={sources}, days=90) count: {len(rows)}")
rows_keys = []
for r in rows:
    k = r.get('job_id') or r.get('id')
    rows_keys.append(str(k))
    print(f"ROW: id={r.get('id')} job_id={r.get('job_id')} source={r.get('source')} posted_at={r.get('posted_at')} submitted={r.get('submitted')}")

missing = set(submitted_keys) - set(rows_keys)
print(f"\nMissing keys (in DB submitted but not in UI query): {missing}")

if missing:
    print("\nDetails for missing items:")
    with job_store.connect(None) as conn:
        for k in missing:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ? OR id = ? OR url = ? OR hash_key = ? LIMIT 1", (k,k,k,k)).fetchone()
            print(row)

print('\nDone')
