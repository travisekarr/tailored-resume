from typing import Optional, List, Dict, Any
# job_store.py (refactored)
import os
import json
import sqlite3
import traceback
from typing import Iterable, List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone

DB_PATH = os.environ.get("JOBS_DB_PATH", "jobs.db")
OPENAI_PRICING_PATH = os.environ.get("OPENAI_PRICING_PATH", "openai_pricing.yaml")

# =========================
# Time helpers
# =========================
def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def _to_iso(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    try:
        s = s.strip().replace("Z", "+00:00")
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return s + "T00:00:00+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception as e:
        print(f"[job_store] _to_iso error: {e}")
        return s

def _to_iso_compat(s: str) -> str:
    if not s:
        return utcnow()
    s = s.strip().replace("Z", "+00:00")
    try:
        if s.endswith("h"):
            hrs = float(s[:-1]); return (datetime.now(timezone.utc) - timedelta(hours=hrs)).isoformat()
        if s.endswith("d"):
            days = float(s[:-1]); return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    except Exception as e:
        print(f"[job_store] _to_iso_compat reltime parse error: {e}")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception as e:
        print(f"[job_store] _to_iso_compat ISO parse error: {e}")
        return utcnow()

# =========================
# Connection / schema
# =========================
DDL_BASE = """
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
    interviewed INTEGER DEFAULT 0,
    interviewed_at TEXT,
    rejected INTEGER DEFAULT 0,
    rejected_at TEXT,
  created_at TEXT,
  updated_at TEXT
);
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
CREATE INDEX IF NOT EXISTS idx_jobs_source      ON jobs(source);
CREATE INDEX IF NOT EXISTS idx_jobs_posted_at   ON jobs(posted_at);
CREATE INDEX IF NOT EXISTS idx_jobs_score       ON jobs(score);
CREATE INDEX IF NOT EXISTS idx_jobs_hash        ON jobs(hash_key);
CREATE INDEX IF NOT EXISTS idx_jobs_pulled_at   ON jobs(pulled_at);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at  ON jobs(updated_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at  ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_runs_finished_at ON runs(finished_at);
"""

# Columns we may add over time (kept in one place)
MIGRATION_COLS = [
    ("status",                     "ALTER TABLE jobs ADD COLUMN status TEXT"),
    ("status_notes",               "ALTER TABLE jobs ADD COLUMN status_notes TEXT"),
    ("status_updated_at",          "ALTER TABLE jobs ADD COLUMN status_updated_at TEXT"),
    ("starred",                    "ALTER TABLE jobs ADD COLUMN starred INTEGER DEFAULT 0"),
    ("first_seen",                 "ALTER TABLE jobs ADD COLUMN first_seen TEXT"),
    ("last_seen",                  "ALTER TABLE jobs ADD COLUMN last_seen TEXT"),
    ("resume_path",                "ALTER TABLE jobs ADD COLUMN resume_path TEXT"),
    ("resume_score",               "ALTER TABLE jobs ADD COLUMN resume_score REAL"),
    ("resume_template",            "ALTER TABLE jobs ADD COLUMN resume_template TEXT"),
    ("not_suitable",               "ALTER TABLE jobs ADD COLUMN not_suitable INTEGER DEFAULT 0"),
    ("not_suitable_at",            "ALTER TABLE jobs ADD COLUMN not_suitable_at TEXT"),
    ("not_suitable_reasons",       "ALTER TABLE jobs ADD COLUMN not_suitable_reasons TEXT"),
    # canonical name expected by pages: 'unsuitable_reason_note'
    ("unsuitable_reason_note",     "ALTER TABLE jobs ADD COLUMN unsuitable_reason_note TEXT"),
    ("submitted",                  "ALTER TABLE jobs ADD COLUMN submitted INTEGER DEFAULT 0"),
    ("submitted_at",               "ALTER TABLE jobs ADD COLUMN submitted_at TEXT"),
    ("job_id",                     "ALTER TABLE jobs ADD COLUMN job_id TEXT"),
    ("interviewed",                "ALTER TABLE jobs ADD COLUMN interviewed INTEGER DEFAULT 0"),
    ("interviewed_at",             "ALTER TABLE jobs ADD COLUMN interviewed_at TEXT"),
    ("rejected",                   "ALTER TABLE jobs ADD COLUMN rejected INTEGER DEFAULT 0"),
    ("rejected_at",                "ALTER TABLE jobs ADD COLUMN rejected_at TEXT"),
    ("resume_model",               "ALTER TABLE jobs ADD COLUMN resume_model TEXT"),
]

def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    _ensure_schema(conn)
    return conn

def init_db(db_path: Optional[str] = None) -> None:
    try:
        with sqlite3.connect(db_path or DB_PATH) as conn:
            for stmt in DDL_BASE.strip().split(";"):
                s = stmt.strip()
                if s:
                    conn.execute(s)
            _ensure_schema(conn)
    except Exception as e:
        print(f"[job_store] init_db error: {e}\n{traceback.format_exc()}")

def _get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        return {r[1] for r in cur.fetchall()}
    except Exception as e:
        print(f"[job_store] _get_columns error: {e}")
        return set()

def _ensure_schema(conn: sqlite3.Connection):
    try:
        # base tables / indexes
        for stmt in DDL_BASE.strip().split(";"):
            s = stmt.strip()
            if s:
                conn.execute(s)
        # add missing columns
        cols = _get_columns(conn, "jobs")
        for col, ddl in MIGRATION_COLS:
            if col not in cols:
                try:
                    conn.execute(ddl)
                except Exception as e:
                    print(f"[job_store] migration '{ddl}' failed: {e}")
        # old column migration: copy not_suitable_note -> unsuitable_reason_note (if present)
        cols = _get_columns(conn, "jobs")
        if "not_suitable_note" in cols and "unsuitable_reason_note" in cols:
            try:
                conn.execute("""
                    UPDATE jobs
                       SET unsuitable_reason_note = COALESCE(unsuitable_reason_note, not_suitable_note)
                     WHERE not_suitable_note IS NOT NULL
                """)
            except Exception as e:
                print(f"[job_store] note column migrate error: {e}")
        # helpful indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status       ON jobs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_submitted    ON jobs(submitted)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_not_suitable ON jobs(not_suitable)")
        conn.commit()
    except Exception as e:
        print(f"[job_store] _ensure_schema error: {e}\n{traceback.format_exc()}")

# =========================
# Ingest / upsert
# =========================
def _coerce_bool(b) -> int:
    return 1 if b in (True, "true", "True", 1) else 0

def _hash_key(it: Dict[str, Any]) -> str:
    company = (it.get("company") or "").strip().lower()
    title = (it.get("title") or "").strip().lower()
    url = (it.get("url") or "").strip().lower()
    return f"{company}|{title}|{url}"

def store_jobs(items: Iterable[Dict[str, Any]], db_path: Optional[str] = None) -> Dict[str, int]:
    """
    Upsert by id. Keeps original posted_at if the new one is missing.
    """
    init_db(db_path)
    now = utcnow()
    inserted = updated = 0
    try:
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
                cur.execute("""
                    INSERT INTO jobs (id, title, company, location, remote, url, source,
                                      posted_at, pulled_at, description, score, hash_key,
                                      created_at, updated_at)
                    VALUES (:id,:title,:company,:location,:remote,:url,:source,
                            :posted_at,:pulled_at,:description,:score,:hash_key,
                            :created_at,:updated_at)
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
                """, row)
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    updated += 1
            conn.commit()
    except Exception as e:
        print(f"[job_store] store_jobs error: {e}\n{traceback.format_exc()}")
    return {"inserted": inserted, "updated": updated}

def prune_duplicates(db_path: Optional[str] = None) -> int:
    """
    Keep max(score) per hash_key; delete others.
    """
    init_db(db_path)
    try:
        with connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute("""
                WITH ranked AS (
                  SELECT id, hash_key, score,
                         ROW_NUMBER() OVER (PARTITION BY hash_key
                                            ORDER BY score DESC, COALESCE(posted_at,'') DESC) AS rn
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
    except Exception as e:
        print(f"[job_store] prune_duplicates error: {e}\n{traceback.format_exc()}")
        return 0

# =========================
# OpenAI usage logging (optional)
# =========================
def _load_pricing_yaml(path: str) -> dict:
    try:
        import yaml as _y
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = _y.safe_load(f) or {}
                return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[job_store] load_pricing error: {e}")
    return {}

def _estimate_cost_usd(model: str, endpoint: str, in_tok: int, out_tok: int, pricing: dict) -> tuple[Optional[float], Optional[float], Optional[float]]:
    d_ep = (pricing.get(endpoint) or {})
    d_mod = (d_ep.get(model) or {})
    unit_in  = d_mod.get("in_per_mtok")
    unit_out = d_mod.get("out_per_mtok")
    if unit_in is None and unit_out is None:
        return (None, None, None)
    in_cost  = (in_tok or 0)  / 1_000_000 * float(unit_in  or 0.0)
    out_cost = (out_tok or 0) / 1_000_000 * float(unit_out or 0.0)
    return (round(in_cost + out_cost, 6), unit_in, unit_out)

def log_api_usage(
    *,
    db_path: str | None = None,
    endpoint: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int | None = None,
    request_id: str | None = None,
    duration_ms: int | None = None,
    context: str | None = None,
    meta: dict | None = None,
    called_at: str | None = None,
) -> int:
    """
    Insert a single usage row. Cost is estimated if pricing YAML is present.
    """
    init_db(db_path)
    try:
        pricing = _load_pricing_yaml(OPENAI_PRICING_PATH)
        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        cost_usd, unit_in, unit_out = _estimate_cost_usd(model, endpoint, input_tokens or 0, output_tokens or 0, pricing)
        with connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  called_at TEXT,
                  endpoint TEXT,
                  model TEXT,
                  input_tokens INTEGER,
                  output_tokens INTEGER,
                  total_tokens INTEGER,
                  request_id TEXT,
                  duration_ms INTEGER,
                  context TEXT,
                  meta_json TEXT,
                  unit_in_per_mtok REAL,
                  unit_out_per_mtok REAL,
                  cost_usd REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_called_at ON api_usage(called_at)")
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO api_usage (called_at, endpoint, model, input_tokens, output_tokens, total_tokens,
                                       request_id, duration_ms, context, meta_json, unit_in_per_mtok, unit_out_per_mtok, cost_usd)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                called_at or utcnow(), endpoint, model,
                int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0),
                request_id or "", int(duration_ms or 0) if duration_ms is not None else None,
                context or "", json.dumps(meta or {}, ensure_ascii=False),
                unit_in, unit_out, cost_usd
            ))
            rid = int(cur.lastrowid)
            conn.commit()
            return rid
    except Exception as e:
        print(f"[job_store] log_api_usage error: {e}\n{traceback.format_exc()}")
        return 0

# =========================
# Query helpers / reporting
# =========================
REPORT_COLS = [
    "id","title","company","location","remote","url","source",
    "posted_at","pulled_at","description","score","hash_key",
    "interviewed","interviewed_at","rejected","rejected_at",
    "created_at","updated_at","status","status_notes","starred",
    "first_seen","last_seen","resume_path","resume_score",
    "not_suitable","not_suitable_at","not_suitable_reasons","unsuitable_reason_note",
    "submitted","submitted_at"
]

def _row_to_dict(row) -> dict:
    cols = REPORT_COLS
    d = {cols[i]: row[i] for i in range(min(len(cols), len(row)))}
    # normalize
    d["job_id"] = d.pop("id", None)
    if d.get("remote") is not None:
        d["remote"] = bool(d["remote"])
    if d.get("starred") is not None:
        try: d["starred"] = int(d["starred"])
        except Exception: pass
    if d.get("score") is not None:
        try: d["score"] = float(d["score"])
        except Exception: pass
    if d.get("interviewed") is not None:
        try: d["interviewed"] = int(d["interviewed"])
        except Exception: pass
    if d.get("rejected") is not None:
        try: d["rejected"] = int(d["rejected"])
        except Exception: pass
    return d

def _touch_seen_rows(db_path: Optional[str] = None):
    try:
        with connect(db_path) as conn:
            conn.execute("""
                UPDATE jobs
                   SET first_seen = COALESCE(first_seen, created_at, pulled_at),
                       last_seen  = COALESCE(last_seen,  pulled_at, updated_at, created_at)
                 WHERE first_seen IS NULL OR last_seen IS NULL
            """)
            conn.commit()
    except Exception as e:
        print(f"[job_store] _touch_seen_rows error: {e}\n{traceback.format_exc()}")

def load_latest(db_path: Optional[str] = None, limit: int = 20):
    _touch_seen_rows(db_path)
    try:
        with connect(db_path) as conn:
            rows = conn.execute(
                f"SELECT {', '.join(REPORT_COLS)} FROM jobs "
                "ORDER BY COALESCE(pulled_at, created_at) DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        print(f"[job_store] load_latest error: {e}\n{traceback.format_exc()}")
        return []

def query_top_matches(db_path: str | None = None, *, limit: int = 50,
                      min_score: float = 0.0, hide_stale_days: int | None = None):
    _touch_seen_rows(db_path)
    try:
        where = ["score >= ?"]
        params = [float(min_score)]
        if hide_stale_days:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=int(hide_stale_days))).isoformat()
            where.append("COALESCE(pulled_at, created_at) >= ?")
            params.append(cutoff)
        sql = f"""
            SELECT {", ".join(REPORT_COLS)}
            FROM jobs
            WHERE {" AND ".join(where)}
            ORDER BY score DESC, COALESCE(posted_at, pulled_at, created_at) DESC
            LIMIT ?
        """
        params.append(int(limit))
        with connect(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        print(f"[job_store] query_top_matches error: {e}\n{traceback.format_exc()}")
        return []

def query_submitted(db_path: str | None = None, *, limit: int = 500) -> list[dict]:
    _touch_seen_rows(db_path)
    try:
        with connect(db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT {", ".join(REPORT_COLS)}
                FROM jobs
                WHERE COALESCE(submitted,0)=1
                ORDER BY COALESCE(submitted_at, updated_at, pulled_at, created_at) DESC
                LIMIT ?
                """, (int(limit),)
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        print(f"[job_store] query_submitted error: {e}\n{traceback.format_exc()}")
        return []

def query_not_suitable(db_path: str | None = None, *, limit: int = 500) -> list[dict]:
    _touch_seen_rows(db_path)
    try:
        with connect(db_path) as conn:
            rows = conn.execute(
                f"""
                SELECT {", ".join(REPORT_COLS)}
                FROM jobs
                WHERE COALESCE(not_suitable,0)=1
                ORDER BY COALESCE(submitted_at, updated_at, pulled_at, created_at) DESC
                LIMIT ?
                """, (int(limit),)
            ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as e:
        print(f"[job_store] query_not_suitable error: {e}\n{traceback.format_exc()}")
        return []

def query_sections_for_report(db_path: str | None = None, *, limit_per: int = 200) -> dict[str, list[dict]]:
    """
    Returns sectioned lists for jobs report:
      - actionable: not submitted and not marked not_suitable
      - saved: has a resume_path (or status='saved')
      - submitted: submitted == 1
      - not_suitable: not_suitable == 1
    """
    _touch_seen_rows(db_path)
    try:
        with connect(db_path) as conn:
            # Actionable
            rows_actionable = conn.execute(
                f"""
                SELECT {", ".join(REPORT_COLS)}
                FROM jobs
                WHERE COALESCE(submitted,0)=0
                  AND COALESCE(not_suitable,0)=0
                ORDER BY COALESCE(score,0) DESC,
                         COALESCE(posted_at, updated_at, pulled_at, created_at) DESC
                LIMIT ?
                """, (int(limit_per),)
            ).fetchall()
            # Saved
            rows_saved = conn.execute(
                f"""
                SELECT {", ".join(REPORT_COLS)}
                FROM jobs
                WHERE (resume_path IS NOT NULL AND TRIM(resume_path) <> '')
                   OR LOWER(COALESCE(status,'')) = 'saved'
                ORDER BY COALESCE(updated_at, pulled_at, created_at) DESC
                LIMIT ?
                """, (int(limit_per),)
            ).fetchall()
            # Submitted
            rows_submitted = conn.execute(
                f"""
                SELECT {", ".join(REPORT_COLS)}
                FROM jobs
                WHERE COALESCE(submitted,0)=1
                ORDER BY COALESCE(submitted_at, updated_at, pulled_at, created_at) DESC
                LIMIT ?
                """, (int(limit_per),)
            ).fetchall()
            # Not suitable
            rows_ns = conn.execute(
                f"""
                SELECT {", ".join(REPORT_COLS)}
                FROM jobs
                WHERE COALESCE(not_suitable,0)=1
                ORDER BY COALESCE(submitted_at, updated_at, pulled_at, created_at) DESC
                LIMIT ?
                """, (int(limit_per),)
            ).fetchall()

        norm = lambda rows: [_row_to_dict(r) for r in rows]
        return {
            "actionable":   norm(rows_actionable),
            "saved":        norm(rows_saved),
            "submitted":    norm(rows_submitted),
            "not_suitable": norm(rows_ns),
        }
    except Exception as e:
        print(f"[job_store] query_sections_for_report error: {e}\n{traceback.format_exc()}")
        return {"actionable": [], "saved": [], "submitted": [], "not_suitable": []}

# Wide, filterable list (used by history / CLI)
def query_jobs(
    db_path: Optional[str] = None,
    sources: Optional[List[str]] = None,
    min_score: float = 0.0,
    posted_start: Optional[str] = None,
    posted_end: Optional[str] = None,
    q: Optional[str] = None,
    limit: int = 500,
    order: str = "score_desc",
) -> List[Dict[str, Any]]:
    init_db(db_path)
    try:
        where_sql, params = [], []
        if sources:
            ph = ",".join(["?"] * len(sources))
            where_sql.append(f"source IN ({ph})")
            params.extend(sources)
        if min_score is not None and float(min_score) > 0.0:
            where_sql.append("score >= ?"); params.append(float(min_score))
        if posted_start:
            where_sql.append("COALESCE(posted_at, pulled_at, created_at) >= ?")
            params.append(_to_iso(posted_start))
        if posted_end:
            where_sql.append("COALESCE(posted_at, pulled_at, created_at) <= ?")
            params.append(_to_iso(posted_end))
        if q and q.strip():
            like = f"%{q.strip()}%"
            where_sql.append("(title LIKE ? OR company LIKE ? OR description LIKE ?)")
            params.extend([like, like, like])
        where_clause = (" WHERE " + " AND ".join(where_sql)) if where_sql else ""
        if order == "date_desc":
            order_by = "ORDER BY COALESCE(posted_at, pulled_at, created_at) DESC, score DESC"
        elif order == "updated_desc":
            order_by = "ORDER BY COALESCE(updated_at, pulled_at, created_at) DESC"
        else:
            order_by = "ORDER BY score DESC, COALESCE(posted_at, pulled_at, created_at) DESC"

        sql = f"""
            SELECT {', '.join(REPORT_COLS)}
            FROM jobs
            {where_clause}
            {order_by}
            LIMIT ?
        """
        params.append(int(limit))
        with connect(db_path) as conn:
            rows = conn.execute(sql, params).fetchall()

        out = []
        for r in rows:
            d = _row_to_dict(r)
            # Add user_notes alias for UI compatibility
            d["user_notes"] = d.get("status_notes")
            d["first_seen"] = d.get("first_seen") or d.get("created_at") or d.get("pulled_at")
            d["last_seen"]  = d.get("last_seen")  or d.get("pulled_at")   or d.get("updated_at") or d.get("created_at")
            out.append(d)
        return out
    except Exception as e:
        print(f"[job_store] query_jobs error: {e}\n{traceback.format_exc()}")
        return []

# =========================
# Updates / actions
# =========================
ALLOWED_STATUSES = {"new","interested","applied","interview","offer","rejected","archived","saved"}

def _where_by_any_key(conn: sqlite3.Connection, job_key: str) -> tuple[str, list]:
    cols = _get_columns(conn, "jobs")
    wh, params = [], []
    if "job_id" in cols:
        wh.append("job_id = ?"); params.append(job_key)
    wh.append("id = ?");  params.append(job_key)
    wh.append("url = ?"); params.append(job_key)
    wh.append("hash_key = ?"); params.append(job_key)
    return " OR ".join(wh), params

def set_job_status(db_path: Optional[str],
                   job_id: str,
                   *,
                   status: Optional[str] = None,
                   status_notes: Optional[str] = None,
                   starred: Optional[bool] = None) -> bool:
    """
    Robust status update; matches by id OR job_id OR url OR hash_key.
    """
    init_db(db_path)
    try:
        now = utcnow()
        sets, params = [], []
        if status is not None:
            if status not in ALLOWED_STATUSES:
                raise ValueError(f"Invalid status '{status}'. Allowed: {sorted(ALLOWED_STATUSES)}")
            sets += ["status = ?", "status_updated_at = ?"]
            params += [status, now]
        if status_notes is not None:
            sets += ["status_notes = ?", "status_updated_at = ?"]
            params += [status_notes, now]
        if starred is not None:
            sets += ["starred = ?", "status_updated_at = ?"]
            params += [1 if starred else 0, now]
        if not sets:
            return False
        sets.append("updated_at = ?"); params.append(now)

        with connect(db_path) as conn:
            where_sql, wparams = _where_by_any_key(conn, str(job_id))
            sql = f"UPDATE jobs SET {', '.join(sets)} WHERE {where_sql}"
            cur = conn.execute(sql, (*params, *wparams))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        print(f"[job_store] set_job_status error: {e}\n{traceback.format_exc()}")
        return False

def set_job_resume(
    db_path: str,
    *,
    job_id: str | None = None,
    job_key: str | None = None,
    resume_path: str | None = None,
    resume_score: float | None = None,
    resume_model: str | None = None,
    resume_template: str | None = None,
) -> bool:
    """
    Persist resume_path/score; robust match by id OR job_id OR url OR hash_key.
    """
    init_db(db_path)
    try:
        key = job_key or job_id
        if not key:
            return False
        with connect(db_path) as conn:
            cols = _get_columns(conn, "jobs")
            # ensure present
            if "resume_path" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN resume_path TEXT")
            if "resume_score" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN resume_score REAL")
            if "resume_model" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN resume_model TEXT")
            if "resume_template" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN resume_template TEXT")
            now = utcnow()
            sets = "resume_path = ?, resume_score = ?, resume_model = ?, resume_template = ?, updated_at = ?"
            params = (
                resume_path,
                resume_score,
                resume_model if resume_model is not None else "No model",
                resume_template if resume_template is not None else "unknown",
                now,
            )
            where_sql, wparams = _where_by_any_key(conn, str(key))
            cur = conn.execute(f"UPDATE jobs SET {sets} WHERE {where_sql}", (*params, *wparams))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        print(f"[job_store] set_job_resume error: {e}\n{traceback.format_exc()}")
        return False

def mark_not_suitable(db_path: str, job_id: str, *, reasons=None, note: str | None = None) -> bool:
    """
    Mark a job as not suitable; stores reasons JSON and an optional note.
    """
    init_db(db_path)
    try:
        ts = utcnow()
        reasons_json = json.dumps(reasons or [])
        with connect(db_path) as conn:
            where_sql, wparams = _where_by_any_key(conn, str(job_id))
            sql = f"""
                UPDATE jobs
                   SET not_suitable = 1,
                       not_suitable_at = ?,
                       not_suitable_reasons = ?,
                       unsuitable_reason_note = ?,
                       updated_at = ?
                 WHERE {where_sql}
            """
            cur = conn.execute(sql, (ts, reasons_json, (note or ""), ts, *wparams))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        print(f"[job_store] mark_not_suitable error: {e}\n{traceback.format_exc()}")
        return False

def mark_submitted(db_path: str, job_id: str, *, submitted: bool = True, submitted_at: str | None = None) -> bool:
    """
    Mark/unmark a job as submitted; records submitted_at.
    """
    init_db(db_path)
    try:
        ts = submitted_at or utcnow()
        with connect(db_path) as conn:
            where_sql, wparams = _where_by_any_key(conn, str(job_id))
            if submitted:
                set_clause = "submitted = 1, submitted_at = ?, updated_at = ?"
                params = (ts, ts)
            else:
                set_clause = "submitted = 0, submitted_at = NULL, updated_at = ?"
                params = (ts,)
            sql = f"UPDATE jobs SET {set_clause} WHERE {where_sql}"
            cur = conn.execute(sql, (*params, *wparams))
            conn.commit()
            return cur.rowcount > 0
    except Exception as e:
        print(f"[job_store] mark_submitted error: {e}\n{traceback.format_exc()}")
        return False

# Back-compat aliases
set_job_not_suitable = mark_not_suitable
set_job_submitted    = mark_submitted

# =========================
# Runs logging
# =========================
def _json_dumps_safe(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        print(f"[job_store] json dump error: {e}")
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
    init_db(db_path)
    try:
        items = []
        if isinstance(res, dict):
            finished_at = finished_at or res.get("generated_at")
            stats = stats or res.get("stats")
            items = res.get("items") or []
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
    except Exception as e:
        print(f"[job_store] record_run error: {e}\n{traceback.format_exc()}")
        return 0

def list_runs(limit: int = 50, db_path: Optional[str] = None) -> list[dict]:
    init_db(db_path)
    try:
        with connect(db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, started_at, finished_at, ok, source_count, item_count, notes, params_json, stats_json
                  FROM runs
                 ORDER BY COALESCE(finished_at, created_at) DESC
                 LIMIT ?
                """, (int(limit),)
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
    except Exception as e:
        print(f"[job_store] list_runs error: {e}\n{traceback.format_exc()}")
        return []

def get_run(run_id: int, db_path: Optional[str] = None) -> Optional[dict]:
    init_db(db_path)
    try:
        with connect(db_path) as conn:
            r = conn.execute(
                """
                SELECT id, started_at, finished_at, ok, source_count, item_count, notes, params_json, stats_json, created_at
                  FROM runs
                 WHERE id = ?
                """, (int(run_id),)
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
    except Exception as e:
        print(f"[job_store] get_run error: {e}\n{traceback.format_exc()}")
        return None

# =========================
# Dangerous utilities
# =========================
def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (name,)).fetchone()
    return bool(row)

def clear_jobs_data(db_path: Optional[str] = None, *, include_runs: bool = True, vacuum: bool = True) -> dict:
    init_db(db_path)
    try:
        with connect(db_path) as conn:
            jobs_deleted = runs_deleted = 0
            if _table_exists(conn, "jobs"):
                (jobs_deleted,) = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
                conn.execute("DELETE FROM jobs")
            if include_runs and _table_exists(conn, "runs"):
                (runs_deleted,) = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
                conn.execute("DELETE FROM runs")
            conn.commit()
            if vacuum:
                try: conn.execute("VACUUM")
                except Exception as e: print(f"[job_store] VACUUM error: {e}")
        return {"jobs_deleted": int(jobs_deleted), "runs_deleted": int(runs_deleted)}
    except Exception as e:
        print(f"[job_store] clear_jobs_data error: {e}\n{traceback.format_exc()}")
        return {"jobs_deleted": 0, "runs_deleted": 0}

def nuke_database(db_path: Optional[str] = None) -> bool:
    path = os.path.abspath(db_path or DB_PATH)
    try:
        # try to close handles
        try:
            with sqlite3.connect(path):
                pass
        except Exception as e:
            print(f"[job_store] nuke pre-close error: {e}")
        for p in (path, path + "-wal", path + "-shm"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                print(f"[job_store] nuke remove error for {p}: {e}")
                return False
        init_db(path)
        return True
    except Exception as e:
        print(f"[job_store] nuke_database error: {e}\n{traceback.format_exc()}")
        return False

# ====== OpenAI usage table helpers & fetcher (restored) ======

def _ensure_api_usage_table(conn: sqlite3.Connection):
    """Create api_usage table + indexes if missing."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS api_usage (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          called_at TEXT,
          endpoint TEXT,           -- 'chat.completions' | 'responses' | 'embeddings' | ...
          model TEXT,
          input_tokens INTEGER,
          output_tokens INTEGER,
          total_tokens INTEGER,
          request_id TEXT,
          duration_ms INTEGER,
          context TEXT,            -- 'ui:04_Generate_From_Jobs' | 'ui:job_finder' | 'cli' ...
          meta_json TEXT,          -- extras per call
          unit_in_per_mtok REAL,   -- pricing snapshot
          unit_out_per_mtok REAL,
          cost_usd REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_called_at ON api_usage(called_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_model ON api_usage(model)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage(endpoint)")

def fetch_usage(
    *,
    db_path: str | None = None,
    since: str | None = None,   # ISO or 'YYYY-MM-DD'
    until: str | None = None,   # ISO or 'YYYY-MM-DD'
    model: str | None = None,
    endpoint: str | None = None,
    context: str | None = None,
    limit: int = 5000,
) -> list[dict]:
    """
    Return raw API-usage rows for analytics dashboards.
    """
    init_db(db_path)
    try:
        with connect(db_path) as conn:
            _ensure_api_usage_table(conn)
            where, args = [], []
            if since:
                where.append("called_at >= ?"); args.append(_to_iso(since))
            if until:
                where.append("called_at <= ?"); args.append(_to_iso(until))
            if model:
                where.append("model = ?"); args.append(model)
            if endpoint:
                where.append("endpoint = ?"); args.append(endpoint)
            if context:
                where.append("context = ?"); args.append(context)

            sql = (
                "SELECT id, called_at, endpoint, model, input_tokens, output_tokens, total_tokens, "
                "request_id, duration_ms, context, meta_json, unit_in_per_mtok, unit_out_per_mtok, cost_usd "
                "FROM api_usage"
            )
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY called_at DESC LIMIT ?"
            args.append(int(limit))

            rows = conn.execute(sql, args).fetchall()

        cols = [
            "id","called_at","endpoint","model","input_tokens","output_tokens","total_tokens",
            "request_id","duration_ms","context","meta_json","unit_in_per_mtok","unit_out_per_mtok","cost_usd"
        ]
        out: list[dict] = []
        for r in rows:
            d = {cols[i]: r[i] for i in range(len(cols))}
            try:
                d["meta"] = json.loads(d.pop("meta_json") or "{}")
            except Exception:
                d["meta"] = {}
            out.append(d)
        return out
    except Exception as e:
        print(f"[job_store] fetch_usage error: {e}")
        import traceback; print(traceback.format_exc())
        return []

def query_new_since(db_path: Optional[str],
                    since_iso: str,
                    *,
                    min_score: float = 0.0,
                    hide_stale_days: Optional[int] = None,
                    limit: int = 100):
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
        SELECT {', '.join(REPORT_COLS)}
        FROM jobs
        WHERE {' AND '.join(where)}
        ORDER BY COALESCE(pulled_at, created_at) DESC
        LIMIT ?
    """
    params.append(int(limit))
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]

def query_changed_since(db_path: Optional[str],
                        since_iso: str,
                        *,
                        limit: int = 200):
    since_iso = _to_iso_compat(since_iso)
    change_expr = "COALESCE(updated_at, pulled_at, created_at)"
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT {', '.join(REPORT_COLS)}, {change_expr} as changed_at
            FROM jobs
            WHERE {change_expr} > ?
            ORDER BY changed_at DESC
            LIMIT ?
            """,
            (since_iso, int(limit)),
        ).fetchall()
    events = []
    for r in rows:
        core = r[:-1]  # everything except changed_at
        d = _row_to_dict(core)
        changed_at = r[-1]
        events.append({
            "changed_at": changed_at,
            "field": "seen",
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