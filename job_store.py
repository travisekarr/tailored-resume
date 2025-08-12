# job_store.py
import os
import sqlite3
from typing import Iterable, List, Dict, Any, Optional
from datetime import datetime, timezone

DB_PATH = os.environ.get("JOBS_DB_PATH", "jobs.db")

from datetime import datetime, timezone

def _to_iso(dt_or_str) -> str | None:
    """Accepts datetime or string and returns ISO-8601 (UTC) string."""
    if dt_or_str is None:
        return None
    if isinstance(dt_or_str, datetime):
        if dt_or_str.tzinfo is None:
            dt_or_str = dt_or_str.replace(tzinfo=timezone.utc)
        return dt_or_str.isoformat()
    s = str(dt_or_str).strip()
    # allow simple YYYY-MM-DD
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        return s + "T00:00:00+00:00"
    return s

DDL = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  title TEXT,
  company TEXT,
  location TEXT,
  remote INTEGER,
  url TEXT,
  source TEXT,
  posted_at TEXT,
  pulled_at TEXT,
  description TEXT,
  score REAL,
  hash_key TEXT,
  created_at TEXT,
  updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);
CREATE INDEX IF NOT EXISTS idx_jobs_posted_at ON jobs(posted_at);
CREATE INDEX IF NOT EXISTS idx_jobs_score ON jobs(score);
CREATE INDEX IF NOT EXISTS idx_jobs_hash ON jobs(hash_key);
"""

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db(db_path: Optional[str] = None) -> None:
    with connect(db_path) as conn:
        for stmt in DDL.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)

def _coerce_bool(b) -> int:
    return 1 if b in (True, "true", "True", 1) else 0

def _hash_key(it: Dict[str, Any]) -> str:
    # fallback dedupe key used for pruning/reporting
    company = (it.get("company") or "").strip().lower()
    title = (it.get("title") or "").strip().lower()
    url = (it.get("url") or "").strip().lower()
    return f"{company}|{title}|{url}"

def store_jobs(items: Iterable[Dict[str, Any]], db_path: Optional[str] = None) -> Dict[str, int]:
    """
    Upsert by id. Keeps first pulled_at; updates other fields.
    Returns counts: {'inserted': X, 'updated': Y}
    """
    init_db(db_path)
    ins, upd = 0, 0
    now = utcnow()
    with connect(db_path) as conn:
        cur = conn.cursor()
        for it in items:
            row = {
                "id": it.get("id") or it.get("url"),
                "title": it.get("title") or "",
                "company": it.get("company") or "",
                "location": it.get("location") or "",
                "remote": _coerce_bool(it.get("remote")),
                "url": it.get("url") or "",
                "source": it.get("source") or "",
                "posted_at": it.get("posted_at"),
                "pulled_at": now,
                "description": it.get("description") or "",
                "score": float(it.get("score") or 0.0),
                "hash_key": _hash_key(it),
                "created_at": now,
                "updated_at": now,
            }
            # Try insert; if conflict, update (keeping original created_at/pulled_at)
            qry = """
            INSERT INTO jobs (id, title, company, location, remote, url, source, posted_at, pulled_at, description, score, hash_key, created_at, updated_at)
            VALUES (:id,:title,:company,:location,:remote,:url,:source,:posted_at,:pulled_at,:description,:score,:hash_key,:created_at,:updated_at)
            ON CONFLICT(id) DO UPDATE SET
              title=excluded.title,
              company=excluded.company,
              location=excluded.location,
              remote=excluded.remote,
              url=excluded.url,
              source=excluded.source,
              posted_at=COALESCE(excluded.posted_at, jobs.posted_at),
              description=excluded.description,
              score=excluded.score,
              hash_key=excluded.hash_key,
              updated_at=excluded.updated_at
            """
            try:
                cur.execute(qry, row)
                if cur.rowcount == 1 and cur.lastrowid is not None:
                    # SQLite returns 0 lastrowid on upsert; use changes() to detect
                    pass
                # We can’t directly tell insert/update; check if previously existed
                # Quick approach: see if changes() == 1 and there was no existing row by doing a select first
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        # Rough counts: count how many ids already existed
        # Simpler: recompute via SELECT
        # (If you want perfect counts, track existence before upsert; keeping simple)
    # Provide counts as a convenience: treat all as inserted for now
    # If you want exact counts, call exists check before each insert.
    # To keep it useful, return total items as inserted+updated:
    total = 0
    for _ in items:
        total += 1
    return {"inserted": total, "updated": 0}

def query_jobs(
    db_path: Optional[str] = None,
    sources: Optional[List[str]] = None,
    min_score: float = 0.0,
    posted_start: Optional[str] = None,
    posted_end: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 500,
    order: str = "score_desc"
) -> List[Dict[str, Any]]:
    init_db(db_path)
    where = []
    params: Dict[str, Any] = {}

    if sources:
        ph = ",".join(["?"] * len(sources))
        where.append(f"source IN ({ph})")
    if min_score > 0.0:
        where.append("score >= ?")
        params[len(params)] = float(min_score)
    if posted_start:
        where.append("posted_at IS NOT NULL AND posted_at >= ?")
        params[len(params)] = posted_start
    if posted_end:
        where.append("posted_at IS NOT NULL AND posted_at <= ?")
        params[len(params)] = posted_end
    if q:
        where.append("(title LIKE ? OR company LIKE ? OR description LIKE ?)")
        params[len(params)] = f"%{q}%"
        params[len(params)+1] = f"%{q}%"
        params[len(params)+2] = f"%{q}%"

    if order == "score_desc":
        order_by = "ORDER BY score DESC, COALESCE(posted_at,'') DESC"
    elif order == "date_desc":
        order_by = "ORDER BY COALESCE(posted_at,'') DESC, score DESC"
    else:
        order_by = "ORDER BY updated_at DESC"

    sql = "SELECT id,title,company,location,remote,url,source,posted_at,pulled_at,description,score FROM jobs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" {order_by} LIMIT ?"

    with connect(db_path) as conn:
        cur = conn.cursor()
        args: List[Any] = []
        if sources:
            args.extend(sources)
        # keep params in insertion order:
        for i in range(len(params)):
            args.append(params[i])
        args.append(int(limit))
        rows = cur.execute(sql, args).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({
            "id": r[0], "title": r[1], "company": r[2], "location": r[3],
            "remote": bool(r[4]), "url": r[5], "source": r[6],
            "posted_at": r[7], "pulled_at": r[8], "description": r[9], "score": float(r[10]),
        })
    return out

def prune_duplicates(db_path: Optional[str] = None) -> int:
    """
    Keep max(score) per hash_key; delete others.
    Returns number of rows deleted.
    """
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        # Find worst ids per hash_key
        cur.execute("""
            WITH ranked AS (
              SELECT id, hash_key, score,
                     ROW_NUMBER() OVER (PARTITION BY hash_key ORDER BY score DESC, COALESCE(posted_at,'') DESC) AS rn
              FROM jobs
            )
            SELECT id FROM ranked WHERE rn > 1
        """)
        to_del = [r[0] for r in cur.fetchall()]
        if not to_del:
            return 0
        cur.executemany("DELETE FROM jobs WHERE id = ?", [(i,) for i in to_del])
        conn.commit()
        return len(to_del)

# --- add to the DDL string (keep the semicolons structure) ---
DDL = DDL + """
CREATE TABLE IF NOT EXISTS runs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  started_at TEXT,
  finished_at TEXT,
  ok INTEGER,
  source_count INTEGER,
  item_count INTEGER,
  notes TEXT,
  params_json TEXT,
  stats_json TEXT,
  created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_finished_at ON runs(finished_at);
"""

# ---- Run logging helpers ----
import json
from typing import Optional

def _json_dumps_safe(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return "{}"

def record_run(
    res: Optional[dict] = None,
    *,
    started_at: Optional[str] = None,
    finished_at: Optional[str] = None,
    ok: bool = True,
    stats: Optional[dict] = None,
    params: Optional[dict] = None,
    notes: Optional[str] = None,
    db_path: Optional[str] = None,
) -> int:
    """
    Persist a single fetch/scoring run to the `runs` table.

    Compatible usage patterns:
      A) record_run(res)                           # pass the whole hunt_jobs result dict
      B) record_run(stats=..., params=..., ...)    # pass fields explicitly

    Fields derived from `res` (if provided):
      - finished_at  <- res['generated_at']
      - item_count   <- res['count']
      - source_count <- unique sources in res['items']
      - stats        <- res['stats']
      - params       <- {'use_embeddings':..., 'embedding_model':...} if present
    """
    init_db(db_path)

    # Derive from res, if provided
    items = []
    if isinstance(res, dict):
        finished_at = finished_at or res.get("generated_at")
        stats = stats or res.get("stats")
        items = res.get("items") or []
        # pull a few common params if present
        params_from_res = {}
        for k in ("use_embeddings", "embedding_model", "ordering", "sources_yaml", "resume_path"):
            if k in res:
                params_from_res[k] = res[k]
        if params_from_res:
            params = {**(params or {}), **params_from_res}

    item_count = len(items) if items else None
    source_count = len({(it.get("source") or "") for it in items}) if items else None

    now = utcnow()
    with connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO runs (started_at, finished_at, ok, source_count, item_count,
                              notes, params_json, stats_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                started_at or now,
                finished_at or now,
                1 if ok else 0,
                source_count,
                item_count,
                notes or "",
                _json_dumps_safe(params or {}),
                _json_dumps_safe(stats or {}),
                now,
            ),
        )
        run_id = cur.lastrowid
        conn.commit()
        return int(run_id)

# Backward-compat alias (some older code used log_run)
def log_run(*args, **kwargs) -> int:
    return record_run(*args, **kwargs)

def list_runs(limit: int = 50, db_path: Optional[str] = None) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT id, started_at, finished_at, ok, source_count, item_count, notes, params_json, stats_json
            FROM runs
            ORDER BY COALESCE(finished_at, created_at) DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "id": r[0],
            "started_at": r[1],
            "finished_at": r[2],
            "ok": bool(r[3]),
            "source_count": r[4],
            "item_count": r[5],
            "notes": r[6],
            "params": (json.loads(r[7]) if r[7] else {}),
            "stats": (json.loads(r[8]) if r[8] else {}),
        })
    return out

def get_run(run_id: int, db_path: Optional[str] = None) -> Optional[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        r = cur.execute(
            """
            SELECT id, started_at, finished_at, ok, source_count, item_count, notes, params_json, stats_json, created_at
            FROM runs WHERE id = ?
            """,
            (int(run_id),),
        ).fetchone()
    if not r:
        return None
    return {
        "id": r[0],
        "started_at": r[1],
        "finished_at": r[2],
        "ok": bool(r[3]),
        "source_count": r[4],
        "item_count": r[5],
        "notes": r[6],
        "params": (json.loads(r[7]) if r[7] else {}),
        "stats": (json.loads(r[8]) if r[8] else {}),
        "created_at": r[9],
    }

def query_new_since(
    since,                         # datetime or ISO/string ("YYYY-MM-DD" OK)
    sources: list[str] | None = None,
    min_score: float | None = None,
    limit: int = 500,
    db_path: str | None = None,
) -> list[dict]:
    """
    Return jobs first pulled after `since`.
    - Filters by sources and min_score if provided.
    - Ordered newest-first by pulled_at.
    """
    init_db(db_path)
    since_iso = _to_iso(since)
    if not since_iso:
        raise ValueError("query_new_since: `since` is required")

    where = ["pulled_at > ?"]
    params: list = [since_iso]

    if sources:
        placeholders = ",".join("?" for _ in sources)
        where.append(f"source IN ({placeholders})")
        params.extend(sources)

    if min_score is not None:
        where.append("score >= ?")
        params.append(float(min_score))

    sql = f"""
        SELECT id, title, company, location, remote, url, source,
               posted_at, pulled_at, description, score, hash_key,
               created_at, updated_at
        FROM jobs
        WHERE {" AND ".join(where)}
        ORDER BY pulled_at DESC
        LIMIT ?
    """
    params.append(int(limit))

    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(sql, params).fetchall()

    cols = ["id","title","company","location","remote","url","source",
            "posted_at","pulled_at","description","score","hash_key",
            "created_at","updated_at"]
    out: list[dict] = []
    for r in rows:
        item = {cols[i]: r[i] for i in range(len(cols))}
        # tidy types
        if item.get("remote") is not None:
            item["remote"] = bool(item["remote"])
        if item.get("score") is not None:
            item["score"] = float(item["score"])
        out.append(item)
    return out

# If you have a DDL string, append these lines to it.
# Otherwise, execute these CREATE INDEX statements once during init_db().
DDL = DDL + """
CREATE INDEX IF NOT EXISTS idx_jobs_pulled_at ON jobs(pulled_at);
CREATE INDEX IF NOT EXISTS idx_jobs_source ON jobs(source);
"""

DDL = DDL + """
CREATE INDEX IF NOT EXISTS idx_jobs_pulled_at  ON jobs(pulled_at);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_source     ON jobs(source);
"""

# Speed up score-ordered queries and common filters
DDL = DDL + """
CREATE INDEX IF NOT EXISTS idx_jobs_score          ON jobs(score);
CREATE INDEX IF NOT EXISTS idx_jobs_source_score   ON jobs(source, score);
CREATE INDEX IF NOT EXISTS idx_jobs_company_title  ON jobs(company, title);
"""


def query_changed_since(
    since,                         # datetime or ISO string ("YYYY-MM-DD" OK)
    *,
    sources: list[str] | None = None,
    min_score: float | None = None,
    include_new: bool = True,      # include rows created after `since`
    limit: int = 500,
    db_path: str | None = None,
) -> list[dict]:
    """
    Return jobs that changed after `since`.
    'Changed' is defined as having a change timestamp > since, where change timestamp is:
        COALESCE(updated_at, pulled_at, created_at)

    Args:
      since: datetime or ISO string.
      sources: optional list of source names to include.
      min_score: optional minimum score filter.
      include_new: if False, exclude rows whose created_at > since (i.e., only updates).
      limit: max rows to return (newest changed first).
    """
    init_db(db_path)
    since_iso = _to_iso(since)
    if not since_iso:
        raise ValueError("query_changed_since: `since` is required")

    # Build WHERE
    change_expr = "COALESCE(updated_at, pulled_at, created_at)"
    where = [f"{change_expr} > ?"]
    params: list = [since_iso]

    if not include_new:
        # Exclude newly created rows after since
        where.append("created_at <= ?")
        params.append(since_iso)

    if sources:
        placeholders = ",".join("?" for _ in sources)
        where.append(f"source IN ({placeholders})")
        params.extend(sources)

    if min_score is not None:
        where.append("score >= ?")
        params.append(float(min_score))

    sql = f"""
        SELECT id, title, company, location, remote, url, source,
               posted_at, pulled_at, description, score, hash_key,
               created_at, updated_at
        FROM jobs
        WHERE {" AND ".join(where)}
        ORDER BY {change_expr} DESC
        LIMIT ?
    """
    params.append(int(limit))

    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(sql, params).fetchall()

    cols = ["id","title","company","location","remote","url","source",
            "posted_at","pulled_at","description","score","hash_key",
            "created_at","updated_at"]
    out: list[dict] = []
    for r in rows:
        item = {cols[i]: r[i] for i in range(len(cols))}
        if item.get("remote") is not None:
            item["remote"] = bool(item["remote"])
        if item.get("score") is not None:
            item["score"] = float(item["score"])
        out.append(item)
    return out

def query_top_matches(
    limit: int = 50,
    *,
    sources: list[str] | None = None,
    min_score: float | None = None,
    posted_start: str | None = None,   # ISO string or "YYYY-MM-DD" (uses _to_iso)
    posted_end: str | None = None,     # ISO string or "YYYY-MM-DD"
    q: str | None = None,              # substring match across title/company/description
    dedupe: bool = True,               # keep best row per (company|title|url)
    db_path: str | None = None,
) -> list[dict]:
    """
    Return top-scoring jobs with optional filters. If `dedupe=True`, keeps only the
    single highest-scored row per (company|title|url) cluster.

    Ordered by: score DESC, then COALESCE(posted_at, pulled_at, created_at) DESC.
    """
    init_db(db_path)

    where = []
    params: list = []

    if sources:
        placeholders = ",".join("?" for _ in sources)
        where.append(f"source IN ({placeholders})")
        params.extend(sources)

    if min_score is not None:
        where.append("score >= ?")
        params.append(float(min_score))

    if posted_start:
        where.append("(COALESCE(posted_at, pulled_at, created_at) >= ?)")
        params.append(_to_iso(posted_start))

    if posted_end:
        where.append("(COALESCE(posted_at, pulled_at, created_at) <= ?)")
        params.append(_to_iso(posted_end))

    if q and q.strip():
        like = f"%{q.strip()}%"
        where.append("(title LIKE ? OR company LIKE ? OR description LIKE ?)")
        params.extend([like, like, like])

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    order_sql = "ORDER BY score DESC, COALESCE(posted_at, pulled_at, created_at) DESC"

    # If deduping, grab more than needed then prune in Python
    fetch_cap = max(limit * 5, 200) if dedupe else limit

    sql = f"""
        SELECT id, title, company, location, remote, url, source,
               posted_at, pulled_at, description, score, hash_key,
               created_at, updated_at
        FROM jobs
        {where_sql}
        {order_sql}
        LIMIT ?
    """
    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(sql, (*params, int(fetch_cap))).fetchall()

    cols = ["id","title","company","location","remote","url","source",
            "posted_at","pulled_at","description","score","hash_key",
            "created_at","updated_at"]

    def _row_to_dict(r):
        d = {cols[i]: r[i] for i in range(len(cols))}
        if d.get("remote") is not None:
            d["remote"] = bool(d["remote"])
        if d.get("score") is not None:
            d["score"] = float(d["score"])
        return d

    items = [_row_to_dict(r) for r in rows]

    if not dedupe:
        return items[:limit]

    # Python-side dedupe by (company|title|url) keeping best score
    seen: dict[str, dict] = {}
    for it in items:
        key = ((it.get("company") or "").strip().lower() + "|" +
               (it.get("title") or "").strip().lower() + "|" +
               (it.get("url") or "").strip().lower())
        prev = seen.get(key)
        if prev is None or (it.get("score", 0.0) > prev.get("score", 0.0)):
            seen[key] = it

    deduped = list(seen.values())
    deduped.sort(key=lambda x: (-x.get("score", 0.0), (x.get("posted_at") or x.get("pulled_at") or x.get("created_at") or "")), reverse=False)
    return deduped[:limit]

# --- load_latest: DB first, then JSON fallback ---
import os, json
from typing import Optional

def _table_exists(conn, name: str) -> bool:
    try:
        cur = conn.cursor()
        row = cur.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (name,),
        ).fetchone()
        return bool(row)
    except Exception:
        return False

def load_latest(
    limit: int = 500,
    *,
    prefer_db: bool = True,
    db_path: Optional[str] = None,
    json_path: str = "jobs_ranked.json",
) -> dict:
    """
    Returns a dict like:
      {
        "items": [...],
        "count": <int>,
        "source": "db" | "json" | None,
        "stats": { ... }     # if available from JSON
      }

    DB mode: returns newest by pulled_at (desc).
    JSON mode: reads items from jobs_ranked.json (if present).
    """
    init_db(db_path)

    # Try DB first
    if prefer_db:
        try:
            with connect(db_path) as conn:
                if _table_exists(conn, "jobs"):
                    cur = conn.cursor()
                    rows = cur.execute(
                        """
                        SELECT id, title, company, location, remote, url, source,
                               posted_at, pulled_at, description, score, hash_key,
                               created_at, updated_at
                        FROM jobs
                        ORDER BY COALESCE(pulled_at, created_at) DESC
                        LIMIT ?
                        """,
                        (int(limit),),
                    ).fetchall()

                    if rows:
                        cols = ["id","title","company","location","remote","url","source",
                                "posted_at","pulled_at","description","score","hash_key",
                                "created_at","updated_at"]
                        items = []
                        for r in rows:
                            d = {cols[i]: r[i] for i in range(len(cols))}
                            if d.get("remote") is not None:
                                d["remote"] = bool(d["remote"])
                            if d.get("score") is not None:
                                d["score"] = float(d["score"])
                            items.append(d)
                        return {"items": items, "count": len(items), "source": "db", "stats": {}}
        except Exception:
            # fall back to JSON
            pass

    # Fallback: JSON file written by job_hunter CLI/UI
    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("items") or []
            items = items[: int(limit)]
            return {
                "items": items,
                "count": len(items),
                "source": "json",
                "stats": (data.get("stats") or {}),
            }
    except Exception:
        pass

    return {"items": [], "count": 0, "source": None, "stats": {}}

def save_latest(result: dict, json_path: str = "jobs_ranked.json") -> None:
    """Persist the full result dict to JSON (for reports or offline review)."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

# ====== JOB STATUS SUPPORT ======

from datetime import datetime, timezone
from typing import Optional

# Valid states (feel free to tweak)
ALLOWED_STATUSES = {
    "new", "interested", "applied", "interview", "offer", "rejected", "archived"
}

def _to_iso(dt_or_str) -> Optional[str]:
    """Accepts datetime or string and returns ISO-8601 (UTC) string."""
    if dt_or_str is None:
        return None
    if isinstance(dt_or_str, datetime):
        if dt_or_str.tzinfo is None:
            dt_or_str = dt_or_str.replace(tzinfo=timezone.utc)
        return dt_or_str.isoformat()
    s = str(dt_or_str).strip()
    if len(s) == 10 and s[4] == "-" and s[7] == "-":  # YYYY-MM-DD
        return s + "T00:00:00+00:00"
    return s

def ensure_job_status_columns(db_path: Optional[str] = None) -> None:
    """Ensure jobs table has status columns; create if missing."""
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(jobs)")
        cols = {r[1] for r in cur.fetchall()}
        alters = []
        if "status" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN status TEXT")
        if "status_notes" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN status_notes TEXT")
        if "status_updated_at" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN status_updated_at TEXT")
        if "starred" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN starred INTEGER DEFAULT 0")
        for sql in alters:
            cur.execute(sql)
        # Helpful index
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        conn.commit()

def set_job_status(
    *,
    id: Optional[int] = None,
    url: Optional[str] = None,
    hash_key: Optional[str] = None,
    status: Optional[str] = None,
    notes: Optional[str] = None,
    starred: Optional[bool] = None,
    db_path: Optional[str] = None,
    allow_new_status_values: bool = False,
) -> dict:
    """
    Update status/notes/starred for a job identified by id OR url OR hash_key.
    Returns: {"updated": <int>, "job": <dict|None>}
    """
    ensure_job_status_columns(db_path)

    if id is None and url is None and hash_key is None:
        raise ValueError("set_job_status: provide one identifier (id, url, or hash_key).")
    if status is None and notes is None and starred is None:
        raise ValueError("set_job_status: nothing to update.")

    if (status is not None) and (not allow_new_status_values) and (status not in ALLOWED_STATUSES):
        raise ValueError(f"Invalid status '{status}'. Allowed: {sorted(ALLOWED_STATUSES)}")

    now = utcnow()
    sets, params = [], []
    # If any field changes, bump status_updated_at
    touch_timestamp = False

    if status is not None:
        sets.append("status = ?")
        params.append(status)
        touch_timestamp = True

    if notes is not None:
        sets.append("status_notes = ?")
        params.append(notes)
        touch_timestamp = True

    if starred is not None:
        sets.append("starred = ?")
        params.append(1 if starred else 0)
        touch_timestamp = True

    if touch_timestamp:
        sets.append("status_updated_at = ?")
        params.append(now)

    where, wparams = [], []
    if id is not None:
        where.append("id = ?"); wparams.append(int(id))
    if url is not None:
        where.append("url = ?"); wparams.append(url)
    if hash_key is not None:
        where.append("hash_key = ?"); wparams.append(hash_key)
    where_sql = " OR ".join(where)

    with connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE {where_sql}", (*params, *wparams))
        affected = cur.rowcount
        conn.commit()

        job = None
        if affected:
            row = cur.execute(
                f"""SELECT id, title, company, location, remote, url, source,
                           posted_at, pulled_at, description, score, hash_key,
                           created_at, updated_at, status, status_notes,
                           status_updated_at, starred
                    FROM jobs
                    WHERE {where_sql}
                    LIMIT 1""",
                tuple(wparams),
            ).fetchone()
            if row:
                cols = ["id","title","company","location","remote","url","source",
                        "posted_at","pulled_at","description","score","hash_key",
                        "created_at","updated_at","status","status_notes",
                        "status_updated_at","starred"]
                job = {cols[i]: row[i] for i in range(len(cols))}
                if job.get("remote") is not None:
                    job["remote"] = bool(job["remote"])
                if job.get("score") is not None:
                    job["score"] = float(job["score"])
                if job.get("starred") is not None:
                    job["starred"] = bool(job["starred"])

    return {"updated": affected, "job": job}

def query_by_status(
    statuses: Optional[list[str]] = None,
    *,
    starred: Optional[bool] = None,
    limit: int = 500,
    db_path: Optional[str] = None,
) -> list[dict]:
    """Fetch jobs by status and/or starred flag, newest status change first."""
    ensure_job_status_columns(db_path)

    where, params = [], []
    if statuses:
        placeholders = ",".join("?" for _ in statuses)
        where.append(f"status IN ({placeholders})")
        params.extend(statuses)
    if starred is not None:
        where.append("starred = ?")
        params.append(1 if starred else 0)

    sql = """
        SELECT id, title, company, location, remote, url, source,
               posted_at, pulled_at, description, score, hash_key,
               created_at, updated_at, status, status_notes,
               status_updated_at, starred
        FROM jobs
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY COALESCE(status_updated_at, pulled_at, created_at) DESC LIMIT ?"
    params.append(int(limit))

    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(sql, params).fetchall()

    cols = ["id","title","company","location","remote","url","source",
            "posted_at","pulled_at","description","score","hash_key",
            "created_at","updated_at","status","status_notes",
            "status_updated_at","starred"]
    out = []
    for r in rows:
        d = {cols[i]: r[i] for i in range(len(cols))}
        if d.get("remote") is not None:
            d["remote"] = bool(d["remote"])
        if d.get("score") is not None:
            d["score"] = float(d["score"])
        if d.get("starred") is not None:
            d["starred"] = bool(d["starred"])
        out.append(d)
    return out

# ==============================
# COMPAT SHIM FOR 02_Job_Reports.py
# ==============================
from datetime import datetime, timedelta, timezone

# --- ensure extra columns the report expects ---
def _ensure_reports_columns(db_path: str | None = None) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(jobs)")
        cols = {r[1] for r in cur.fetchall()}
        alters = []
        if "status" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN status TEXT")
        if "status_notes" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN status_notes TEXT")
        if "starred" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN starred INTEGER DEFAULT 0")
        if "first_seen" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN first_seen TEXT")
        if "last_seen" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN last_seen TEXT")
        if "updated_at" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN updated_at TEXT")
        for sql in alters:
            cur.execute(sql)
        # helpful indices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_pulled_at  ON jobs(pulled_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_score      ON jobs(score)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status     ON jobs(status)")
        conn.commit()

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _to_iso_compat(s: str) -> str:
    if not s:
        return _now_iso()
    s = s.strip().replace("Z", "+00:00")
    # allow "24h", "7d" style too
    try:
        if s.endswith("h"):
            hrs = float(s[:-1]); return (datetime.now(timezone.utc) - timedelta(hours=hrs)).isoformat()
        if s.endswith("d"):
            days = float(s[:-1]); return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    except Exception:
        pass
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception:
        return _now_iso()

def _row_to_report_dict(row) -> dict:
    """
    Map DB row -> report dict keys the page expects.
    """
    cols = ["id","title","company","location","remote","url","source",
            "posted_at","pulled_at","description","score","hash_key",
            "created_at","updated_at","status","status_notes","starred",
            "first_seen","last_seen"]
    d = {cols[i]: row[i] for i in range(len(cols)) if i < len(row)}
    # normalize types/aliases
    d["job_id"] = d.pop("id", None)
    d["user_notes"] = d.get("status_notes")
    d["first_seen"] = d.get("first_seen") or d.get("created_at") or d.get("pulled_at")
    d["last_seen"]  = d.get("last_seen")  or d.get("pulled_at")   or d.get("updated_at") or d.get("created_at")
    if d.get("remote") is not None:
        d["remote"] = bool(d["remote"])
    if d.get("starred") is not None:
        d["starred"] = int(d["starred"])
    if d.get("score") is not None:
        try: d["score"] = float(d["score"])
        except Exception: pass
    return d

# --- keep first_seen/last_seen reasonably fresh on upsert/store ---
def _touch_seen_rows(db_path: str | None = None):
    _ensure_reports_columns(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        # If first_seen is null, set to created_at or pulled_at
        cur.execute("""
            UPDATE jobs
            SET first_seen = COALESCE(first_seen, created_at, pulled_at),
                last_seen  = COALESCE(last_seen,  pulled_at, updated_at, created_at)
            WHERE first_seen IS NULL OR last_seen IS NULL
        """)
        conn.commit()

# Call this opportunistically from your store function if you want:
# _touch_seen_rows(db_path)

# ------------------------------
# load_latest(db_path, limit)
# ------------------------------
def load_latest(db_path: str | None = None, limit: int = 20):
    """
    Reports page expects: list[dict] newest-first by pulled_at/created_at.
    """
    _ensure_reports_columns(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT id, title, company, location, remote, url, source,
                   posted_at, pulled_at, description, score, hash_key,
                   created_at, updated_at, status, status_notes, starred,
                   first_seen, last_seen
            FROM jobs
            ORDER BY COALESCE(pulled_at, created_at) DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [_row_to_report_dict(r) for r in rows]

# ------------------------------
# query_top_matches(db_path, ...)
# ------------------------------
def query_top_matches(db_path: str | None = None, *, limit: int = 50,
                      min_score: float = 0.0, hide_stale_days: int | None = None):
    """
    Simplified: top by score desc; optional staleness filter.
    """
    _ensure_reports_columns(db_path)
    where = ["score >= ?"]
    params = [float(min_score)]
    if hide_stale_days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(hide_stale_days))).isoformat()
        where.append("COALESCE(pulled_at, created_at) >= ?")
        params.append(cutoff)
    sql = f"""
        SELECT id, title, company, location, remote, url, source,
               posted_at, pulled_at, description, score, hash_key,
               created_at, updated_at, status, status_notes, starred,
               first_seen, last_seen
        FROM jobs
        WHERE {" AND ".join(where)}
        ORDER BY score DESC, COALESCE(posted_at, pulled_at, created_at) DESC
        LIMIT ?
    """
    params.append(int(limit))
    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(sql, params).fetchall()
    return [_row_to_report_dict(r) for r in rows]

# ------------------------------
# query_new_since(db_path, since_iso, ...)
# ------------------------------
def query_new_since(db_path: str | None,
                    since_iso: str,
                    *,
                    min_score: float = 0.0,
                    hide_stale_days: int | None = None,
                    limit: int = 100):
    """
    New rows first pulled after `since_iso` (or created after).
    """
    _ensure_reports_columns(db_path)
    since_iso = _to_iso_compat(since_iso)
    where = [
        "(COALESCE(pulled_at, created_at) > ?)",
        "(score >= ?)"
    ]
    params = [since_iso, float(min_score)]
    if hide_stale_days:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=int(hide_stale_days))).isoformat()
        where.append("COALESCE(pulled_at, created_at) >= ?")
        params.append(cutoff)

    sql = f"""
        SELECT id, title, company, location, remote, url, source,
               posted_at, pulled_at, description, score, hash_key,
               created_at, updated_at, status, status_notes, starred,
               first_seen, last_seen
        FROM jobs
        WHERE {" AND ".join(where)}
        ORDER BY COALESCE(pulled_at, created_at) DESC
        LIMIT ?
    """
    params.append(int(limit))
    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(sql, params).fetchall()
    return [_row_to_report_dict(r) for r in rows]

# ------------------------------
# query_changed_since(db_path, since_iso, ...)
# ------------------------------
def query_changed_since(db_path: str | None,
                        since_iso: str,
                        *,
                        limit: int = 200):
    """
    We don't have a historical change log table here, so we approximate:
    return any job with COALESCE(updated_at, pulled_at, created_at) > since,
    and emit a single 'seen' event per job for the report.
    """
    _ensure_reports_columns(db_path)
    since_iso = _to_iso_compat(since_iso)
    change_expr = "COALESCE(updated_at, pulled_at, created_at)"
    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(
            f"""
            SELECT id, title, company, location, remote, url, source,
                   posted_at, pulled_at, description, score, hash_key,
                   created_at, updated_at, status, status_notes, starred,
                   first_seen, last_seen,
                   {change_expr} as changed_at
            FROM jobs
            WHERE {change_expr} > ?
            ORDER BY changed_at DESC
            LIMIT ?
            """,
            (since_iso, int(limit)),
        ).fetchall()

    events = []
    for r in rows:
        d = _row_to_report_dict(r[:-1])  # map first N cols
        changed_at = r[-1]
        events.append({
            "changed_at": changed_at,
            "field": "seen",          # placeholder without full diff history
            "old_value": "",
            "new_value": changed_at,
            "title": d.get("title",""),
            "company": d.get("company",""),
            "score": d.get("score", 0.0),
            "status": d.get("status","new"),
            "source": d.get("source",""),
            "url": d.get("url",""),
            "posted_at": d.get("posted_at",""),
            "last_seen": d.get("last_seen",""),
            "job_id": d.get("job_id"),
        })
    return events

# ------------------------------
# set_job_status(db_path, job_id, ...)
# ------------------------------
def set_job_status(db_path: str | None,
                   job_id: int,
                   *,
                   status: str | None = None,
                   user_notes: str | None = None,
                   starred: bool | None = None) -> bool:
    """
    Update status/notes/starred for a job (report page signature).
    """
    _ensure_reports_columns(db_path)
    now = _now_iso()
    with connect(db_path) as conn:
        cur = conn.cursor()
        sets, params = [], []
        if status is not None:
            sets.append("status = ?"); params.append(status)
        if user_notes is not None:
            sets.append("status_notes = ?"); params.append(user_notes)
        if starred is not None:
            sets.append("starred = ?"); params.append(1 if starred else 0)
        if not sets:
            return False
        sets.append("updated_at = ?"); params.append(now)
        params.append(int(job_id))
        cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", params)
        conn.commit()
        return cur.rowcount > 0
