import sys
import os
import json
# ensure repo root is on sys.path so local imports work when running the script
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import job_store

if len(sys.argv) < 2:
    print("Usage: inspect_job.py <job_key>")
    sys.exit(1)

key = sys.argv[1]
conn = job_store.connect()
cur = conn.execute("""
    SELECT * FROM jobs
    WHERE job_id = ? OR id = ? OR url = ? OR hash_key = ?
    LIMIT 1
""", (key, key, key, key))
row = cur.fetchone()
if not row:
    print(f"No row found for key={key}")
    sys.exit(0)

try:
    d = job_store._row_to_dict(row)
except Exception as e:
    print("_row_to_dict error:", e)
    d = {"raw": list(row)}

# Also fetch full columns
cols = job_store.REPORT_COLS
out = {cols[i]: row[i] if i < len(row) else None for i in range(len(cols))}
# Normalize job_id alias
out['job_id'] = out.get('id')
print(json.dumps({'raw_row': out, 'normalized': d}, indent=2, ensure_ascii=False))
