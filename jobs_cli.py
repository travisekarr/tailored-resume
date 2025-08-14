import dateutil.parser
from datetime import datetime, timezone
# --- Date/time migration utility ---
def normalize_datetime_cli(val):
    if not val:
        return None
    try:
        if isinstance(val, (int, float)) or (isinstance(val, str) and val.strip().isdigit()):
            return datetime.fromtimestamp(float(val), tz=timezone.utc).isoformat()
        dt = dateutil.parser.parse(str(val))
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return None

def cmd_migrate_datetime_fields(db):
    con = sqlite3.connect(db)
    cur = con.execute("PRAGMA table_info(jobs)")
    columns = [r[1] for r in cur.fetchall()]
    dt_fields = [c for c in columns if c.endswith('_at') or c in ('posted_at', 'pulled_at', 'created_at', 'updated_at')]
    cur = con.execute(f"SELECT id, {', '.join(dt_fields)} FROM jobs")
    jobs = cur.fetchall()
    updated = 0
    for job in jobs:
        id = job[0]
        updates = {}
        for idx, field in enumerate(dt_fields, start=1):
            orig = job[idx]
            norm = normalize_datetime_cli(orig)
            if norm and norm != orig:
                updates[field] = norm
        if updates:
            set_clause = ', '.join([f"{k}=?" for k in updates.keys()])
            vals = list(updates.values()) + [id]
            con.execute(f"UPDATE jobs SET {set_clause} WHERE id=?", vals)
            updated += 1
    con.commit()
    con.close()
    print(json.dumps({"db": db, "updated_rows": updated, "fields": dt_fields}))
def cmd_fix_remoteok_urls(db):
    con = sqlite3.connect(db)
    cur = con.execute("""
        UPDATE jobs
        SET url = REPLACE(url, 'https://remoteok.comhttps://remoteOK.com/', 'https://remoteok.com/')
        WHERE source = 'remoteok' AND url LIKE 'https://remoteok.comhttps://remoteOK.com/%';
    """)
    affected = cur.rowcount
    con.commit()
    con.close()
    print(json.dumps({"db": db, "fixed_rows": affected}))
# jobs_cli.py
import os, sys, sqlite3, argparse, json

def abs_db(path): 
    return os.path.abspath(path or "jobs.db")

def q(db, sql, args=()):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    cur = con.execute(sql, args)
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def cmd_counts(db):
    rows = q(db, """
        SELECT
          SUM(COALESCE(submitted,0))      AS submitted_cnt,
          SUM(COALESCE(not_suitable,0))   AS not_suitable_cnt,
          COUNT(*)                         AS total_rows
        FROM jobs
    """)
    print(json.dumps({"db": db, **(rows[0] if rows else {})}, indent=2))

def cmd_flags(db, limit):
    rows = q(db, """
        SELECT id AS job_id, title, company,
               submitted, submitted_at,
               not_suitable, not_suitable_at,
               not_suitable_reasons, unsuitable_reason_note,
               updated_at, pulled_at, created_at
        FROM jobs
        WHERE COALESCE(submitted,0)=1 OR COALESCE(not_suitable,0)=1
        ORDER BY COALESCE(submitted_at, not_suitable_at, updated_at, pulled_at, created_at) DESC
        LIMIT ?
    """, (int(limit),))
    print(json.dumps({"db": db, "count": len(rows), "rows": rows}, indent=2))

def cmd_schema(db):
    rows = q(db, "PRAGMA table_info(jobs)")
    print(json.dumps({"db": db, "columns": rows}, indent=2))

def _make_like_pattern(query: str, exact: bool) -> str:
    # If user already provided wildcards, honor them; otherwise wrap with %
    if exact: 
        return query
    if "%" in query or "_" in query:
        return query
    return f"%{query}%"

def cmd_search_company(db, query, limit, exact=False, show_desc=False):
    pattern = _make_like_pattern(query, exact)
    rows = q(db, """
        SELECT id AS job_id,
               title, company, location, remote, source, url,
               score, starred,
               submitted, submitted_at,
               not_suitable, not_suitable_at,
               COALESCE(updated_at, pulled_at, created_at) AS last_seen,
               description
        FROM jobs
        WHERE COALESCE(company,'') LIKE ? COLLATE NOCASE
        ORDER BY last_seen DESC
        LIMIT ?
    """, (pattern, int(limit)))
    # Trim description unless requested
    if not show_desc:
        for r in rows:
            r.pop("description", None)
    else:
        for r in rows:
            d = (r.get("description") or "")
            r["description"] = (d[:240] + "…") if len(d) > 240 else d
    print(json.dumps({
        "db": db,
        "query": query,
        "pattern_used": pattern,
        "count": len(rows),
        "rows": rows
    }, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Jobs DB CLI (SQLite)")
    ap.add_argument("--db", default="jobs.db", help="Path to SQLite DB (default: jobs.db)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("counts")
    p_flags = sub.add_parser("flags"); p_flags.add_argument("--limit", type=int, default=50)
    sub.add_parser("schema")

    p_search = sub.add_parser("search-company", help="Search by full or partial company name (case-insensitive)")
    p_search.add_argument("query", help="Company name or fragment (e.g., 'Acme' or '%Acme%')")
    p_search.add_argument("--limit", type=int, default=100, help="Max rows to return")
    p_search.add_argument("--exact", action="store_true", help="Match exactly (no wildcards added)")
    p_search.add_argument("--show-desc", action="store_true", help="Include a short description snippet")

    sub.add_parser("fix-remoteok-urls", help="Fix malformed RemoteOK URLs in jobs table")

    sub.add_parser("migrate-datetime-fields", help="Normalize all job date/time fields to ISO UTC format")

    args = ap.parse_args()
    db = abs_db(args.db)

    if args.cmd == "counts":         cmd_counts(db)
    elif args.cmd == "flags":        cmd_flags(db, args.limit)
    elif args.cmd == "schema":       cmd_schema(db)
    elif args.cmd == "search-company":
        cmd_search_company(db, args.query, args.limit, exact=args.exact, show_desc=args.show_desc)
    elif args.cmd == "fix-remoteok-urls":
        cmd_fix_remoteok_urls(db)
    elif args.cmd == "migrate-datetime-fields":
        cmd_migrate_datetime_fields(db)
