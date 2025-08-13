# job_store.py
import os
import json
import json as _json
import sqlite3
from typing import Iterable, List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone

DB_PATH = os.environ.get("JOBS_DB_PATH", "jobs.db")

# ---------------------------
# Time / ISO helpers
# ---------------------------
def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()

def _to_iso(dt_or_str) -> Optional[str]:
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

def _to_iso_compat(s: str) -> str:
    """Accepts ISO, 'YYYY-MM-DD', or relative like '24h', '7d'."""
    if not s:
        return utcnow()
    s = s.strip().replace("Z", "+00:00")
    try:
        if s.endswith("h"):
            hrs = float(s[:-1]); return (datetime.now(timezone.utc) - timedelta(hours=hrs)).isoformat()
        if s.endswith("d"):
            days = float(s[:-1]); return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    except Exception as e:
        print(f"Error parsing relative time: {e}")
        pass
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()
    except Exception as e:
        print(f"Error parsing ISO date: {e}")
        return utcnow()

# ---------------------------
# DB bootstrap / connect
# ---------------------------
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
  updated_at TEXT,
  -- optional columns that may be added later:
  status TEXT,
  status_notes TEXT,
  status_updated_at TEXT,
  starred INTEGER DEFAULT 0,
  first_seen TEXT,
  last_seen TEXT,
  resume_path TEXT,
  resume_score REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_source      ON jobs(source);
CREATE INDEX IF NOT EXISTS idx_jobs_posted_at   ON jobs(posted_at);
CREATE INDEX IF NOT EXISTS idx_jobs_score       ON jobs(score);
CREATE INDEX IF NOT EXISTS idx_jobs_hash        ON jobs(hash_key);
CREATE INDEX IF NOT EXISTS idx_jobs_pulled_at   ON jobs(pulled_at);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at  ON jobs(updated_at);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at  ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status      ON jobs(status);
"""

# Minimal runs table (used by history/reporting elsewhere)
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

# ====== OPENAI USAGE LOGGING ======

OPENAI_PRICING_PATH = os.environ.get("OPENAI_PRICING_PATH", "openai_pricing.yaml")

def _ensure_api_usage_table(conn: sqlite3.Connection):
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
          unit_in_per_mtok REAL,   -- snapshot of pricing used
          unit_out_per_mtok REAL,
          cost_usd REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_called_at ON api_usage(called_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_model ON api_usage(model)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage(endpoint)")

def _load_pricing_yaml(path: str) -> dict:
    try:
        import yaml as _y
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = _y.safe_load(f) or {}
                return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Error loading pricing YAML: {e}")
        pass
    return {}

def _estimate_cost_usd(model: str, endpoint: str, in_tok: int, out_tok: int, pricing: dict) -> tuple[float|None, float|None, float|None]:
    """
    Pricing YAML shape:
      chat.completions:
        gpt-4o:          { in_per_mtok: 5.0,  out_per_mtok: 15.0 }
        gpt-4o-mini:     { in_per_mtok: 0.15, out_per_mtok: 0.60 }
      embeddings:
        text-embedding-3-small: { in_per_mtok: 0.02 }   # embeddings typically 'input' only

    Returns (cost_usd, unit_in, unit_out). If no pricing found -> (None, None, None).
    """
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
    Insert a single usage row. Cost is estimated using openai_pricing.yaml if present.
    Returns inserted row id.
    """
    init_db(db_path)
    pricing = _load_pricing_yaml(OPENAI_PRICING_PATH)

    # compute total if missing
    if total_tokens is None:
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    cost_usd, unit_in, unit_out = _estimate_cost_usd(model, endpoint, input_tokens or 0, output_tokens or 0, pricing)

    with connect(db_path) as conn:
        _ensure_api_usage_table(conn)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO api_usage (called_at, endpoint, model, input_tokens, output_tokens, total_tokens,
                                   request_id, duration_ms, context, meta_json, unit_in_per_mtok, unit_out_per_mtok, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            called_at or utcnow(),
            endpoint, model,
            int(input_tokens or 0), int(output_tokens or 0), int(total_tokens or 0),
            request_id or "",
            int(duration_ms or 0) if duration_ms is not None else None,
            context or "",
            _json.dumps(meta or {}, ensure_ascii=False),
            unit_in, unit_out,
            cost_usd
        ))
        rid = int(cur.lastrowid)
        conn.commit()
        return rid

# Simple readers for the analytics page
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
    init_db(db_path)
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
        sql = "SELECT id, called_at, endpoint, model, input_tokens, output_tokens, total_tokens, request_id, duration_ms, context, meta_json, unit_in_per_mtok, unit_out_per_mtok, cost_usd FROM api_usage"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY called_at DESC LIMIT ?"
        args.append(int(limit))
        rows = conn.execute(sql, args).fetchall()

    cols = ["id","called_at","endpoint","model","input_tokens","output_tokens","total_tokens","request_id","duration_ms","context","meta_json","unit_in_per_mtok","unit_out_per_mtok","cost_usd"]
    out = []
    for r in rows:
        d = {cols[i]: r[i] for i in range(len(cols))}
        try:
            d["meta"] = _json.loads(d.pop("meta_json") or "{}")
        except Exception:
            d["meta"] = {}
        out.append(d)
    return out

# ---------------------------
# Column helpers (for safe ALTERs)
# ---------------------------
def _table_has_column(conn: sqlite3.Connection, table: str, col: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(r[1] == col for r in cur.fetchall())

def _ensure_columns(conn: sqlite3.Connection, table: str, cols: list[tuple[str, str]]) -> None:
    for name, decl in cols:
        if not _table_has_column(conn, table, name):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    conn.commit()

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
        # NEW: actionable/submission flags
        if "not_suitable" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN not_suitable INTEGER DEFAULT 0")
        if "submitted" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN submitted INTEGER DEFAULT 0")
        if "submitted_at" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN submitted_at TEXT")
        # NEW: resume columns (no-ops if already added elsewhere)
        if "resume_path" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN resume_path TEXT")
        if "resume_score" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN resume_score REAL")

        for sql in alters:
            cur.execute(sql)

        # helpful indices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_pulled_at   ON jobs(pulled_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created_at  ON jobs(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_updated_at  ON jobs(updated_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_score       ON jobs(score)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status      ON jobs(status)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_submitted   ON jobs(submitted)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_not_suitable ON jobs(not_suitable)")
        conn.commit()

def _pick_id_column(conn: sqlite3.Connection, table: str = "jobs") -> str:
    # Prefer 'job_id' if it exists; otherwise 'id'
    return "job_id" if _table_has_column(conn, table, "job_id") else "id"

# ---------------------------
# Basic utils
# ---------------------------
def _coerce_bool(b) -> int:
    return 1 if b in (True, "true", "True", 1) else 0

def _hash_key(it: Dict[str, Any]) -> str:
    company = (it.get("company") or "").strip().lower()
    title = (it.get("title") or "").strip().lower()
    url = (it.get("url") or "").strip().lower()
    return f"{company}|{title}|{url}"

# ---------- Danger-zone helpers ----------

def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return bool(row)

def clear_jobs_data(db_path: Optional[str] = None, *, include_runs: bool = True, vacuum: bool = True) -> dict:
    """
    Delete all rows from jobs (and runs if include_runs=True), keeping the schema.
    Returns counts: {"jobs_deleted": int, "runs_deleted": int}
    """
    init_db(db_path)
    with connect(db_path) as conn:
        jobs_deleted = 0
        runs_deleted = 0

        if _table_exists(conn, "jobs"):
            (jobs_deleted,) = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()
            conn.execute("DELETE FROM jobs")

        if include_runs and _table_exists(conn, "runs"):
            (runs_deleted,) = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
            conn.execute("DELETE FROM runs")

        conn.commit()
        if vacuum:
            # Reclaim disk space after large deletes
            try:
                conn.execute("VACUUM")
            except Exception as e:
                print(f"Error running VACUUM: {e}")
                pass

    return {"jobs_deleted": int(jobs_deleted), "runs_deleted": int(runs_deleted)}

def nuke_database(db_path: Optional[str] = None) -> bool:
    """
    Hard reset: delete the SQLite file (and -wal/-shm siblings) and recreate schema.
    Returns True on success.
    """
    path = db_path or DB_PATH
    try:
        # Ensure any on-disk file path
        path = os.path.abspath(path)
        # Close any stray connections by using a short-lived one first
        try:
            with connect(path):
                pass
        except Exception as e:
            print(f"Error closing stray connections: {e}")
            pass

        # Remove main + WAL/SHM if present
        for p in (path, path + "-wal", path + "-shm"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                print(f"Error removing DB file {p}: {e}")
                # On Windows a locked file can raise; bubble up as False
                return False

        # Recreate schema
        init_db(path)
        return True
    except Exception:
        return False

# ---------------------------
# Ingest / Upsert
# ---------------------------
def store_jobs(items: Iterable[Dict[str, Any]], db_path: Optional[str] = None) -> Dict[str, int]:
    """
    Upsert by id. Keeps first pulled_at; updates other fields.
    Returns counts: {'inserted': X, 'updated': Y} (approximate).
    """
    init_db(db_path)
    now = utcnow()
    total = 0
    with connect(db_path) as conn:
        cur = conn.cursor()
        for it in items:
            total += 1
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
            """, row)
        conn.commit()
    return {"inserted": total, "updated": 0}

def prune_duplicates(db_path: Optional[str] = None) -> int:
    """
    Keep max(score) per hash_key; delete others. Returns number of rows deleted.
    """
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
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

# ---------------------------
# Runs logging (used by history/reporting)
# ---------------------------
def _json_dumps_safe(obj) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception as e:
        print(f"Error dumping JSON: {e}")
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

def log_run(*args, **kwargs) -> int:
    return record_run(*args, **kwargs)

def list_runs(limit: int = 50, db_path: Optional[str] = None) -> list[dict]:
    init_db(db_path)
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

def get_run(run_id: int, db_path: Optional[str] = None) -> Optional[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        r = conn.execute(
            """
            SELECT id, started_at, finished_at, ok, source_count, item_count, notes, params_json, stats_json, created_at
            FROM runs WHERE id = ?
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

# --- Action flags columns (not_suitable/submitted) ---
def ensure_action_columns(db_path: str | None = None) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute("PRAGMA table_info(jobs)")
        cols = {r[1] for r in cur.fetchall()}

        alters = []
        if "not_suitable" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN not_suitable INTEGER DEFAULT 0")
        if "not_suitable_at" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN not_suitable_at TEXT")
        if "not_suitable_reasons" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN not_suitable_reasons TEXT")
        if "not_suitable_note" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN not_suitable_note TEXT")

        if "submitted" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN submitted INTEGER DEFAULT 0")
        if "submitted_at" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN submitted_at TEXT")

        # keep resume columns handy for the Generate page
        if "resume_path" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN resume_path TEXT")
        if "resume_score" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN resume_score REAL")

        for sql in alters:
            conn.execute(sql)

        # If an older column exists, migrate its values into the new one once.
        try:
            if "not_suitable_reasons" in cols:
                conn.execute("""
                    UPDATE jobs
                    SET not_suitable_reasons = COALESCE(not_suitable_reasons, not_suitable_reasons)
                    WHERE not_suitable_reasons IS NOT NULL
                      AND (not_suitable_reasons IS NULL OR TRIM(not_suitable_reasons) = '')
                """)
        except Exception:
            # ignore if the old column doesn't exist in this DB
            pass

        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_not_suitable ON jobs(not_suitable)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_submitted     ON jobs(submitted)")
        conn.commit()

# ---------------------------
# Report query helpers
# ---------------------------
# Standard column list all report queries will SELECT (order matters)
REPORT_COLS = [
    "id","title","company","location","remote","url","source",
    "posted_at","pulled_at","description","score","hash_key",
    "created_at","updated_at","status","status_notes","starred",
    "first_seen","last_seen","resume_path","resume_score"
]

def _row_to_report_dict(row) -> dict:
    """
    Map DB row -> report dict keys the page expects.
    Safe with SELECTs that don't include all columns (we guard with len(row)).
    """
    cols = [
        "id","title","company","location","remote","url","source",
        "posted_at","pulled_at","description","score","hash_key",
        "created_at","updated_at","status","status_notes","starred",
        "first_seen","last_seen",
        # New (will be present if SELECT includes them)
        "resume_path","resume_score",
        "not_suitable","not_suitable_at",
        "not_suitable_reasons","unsuitable_reason_note",
        "submitted","submitted_at",
    ]
    d = {cols[i]: row[i] for i in range(min(len(cols), len(row)))}
    # normalize types/aliases
    d["job_id"] = d.pop("id", None)
    d["user_notes"] = d.get("status_notes")
    d["first_seen"] = d.get("first_seen") or d.get("created_at") or d.get("pulled_at")
    d["last_seen"]  = d.get("last_seen")  or d.get("pulled_at")   or d.get("updated_at") or d.get("created_at")
    if d.get("remote") is not None:
        d["remote"] = bool(d["remote"])
    if d.get("starred") is not None:
        try: d["starred"] = int(d["starred"])
        except Exception: pass
    if d.get("score") is not None:
        try: d["score"] = float(d["score"])
        except Exception: pass
    return d

def _touch_seen_rows(db_path: Optional[str] = None):
    _ensure_reports_columns(db_path)
    with connect(db_path) as conn:
        conn.execute("""
            UPDATE jobs
            SET first_seen = COALESCE(first_seen, created_at, pulled_at),
                last_seen  = COALESCE(last_seen,  pulled_at, updated_at, created_at)
            WHERE first_seen IS NULL OR last_seen IS NULL
        """)
        conn.commit()

def load_latest(db_path: Optional[str] = None, limit: int = 20):
    _ensure_reports_columns(db_path)
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT {", ".join(REPORT_COLS)}
            FROM jobs
            ORDER BY COALESCE(pulled_at, created_at) DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    return [_row_to_report_dict(r) for r in rows]

def query_top_matches(db_path: str | None = None, *, limit: int = 50,
                      min_score: float = 0.0, hide_stale_days: int | None = None):
    _ensure_reports_columns(db_path)
    ensure_action_columns(db_path) 
    with connect(db_path) as conn:
        cur = conn.cursor()
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
                first_seen, last_seen,
                resume_path, resume_score,
                not_suitable, not_suitable_at,
                not_suitable_reasons,
                submitted, submitted_at
            FROM jobs
            WHERE {" AND ".join(where)}
            ORDER BY score DESC, COALESCE(posted_at, pulled_at, created_at) DESC
            LIMIT ?
        """
        params.append(int(limit))
        rows = cur.execute(sql, params).fetchall()

    cols = ["id","title","company","location","remote","url","source",
            "posted_at","pulled_at","description","score","hash_key",
            "created_at","updated_at","status","status_notes","starred",
            "first_seen","last_seen","resume_path","resume_score",
            "not_suitable","not_suitable_at",
            "not_suitable_reasons",
            "submitted","submitted_at"]

    out = []
    for r in rows:
        d = {cols[i]: r[i] for i in range(len(cols))}
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
        if d.get("resume_score") is not None:
            try: d["resume_score"] = float(d["resume_score"])
            except Exception: pass
        if d.get("not_suitable") is not None:
            d["not_suitable"] = int(d["not_suitable"])
        if d.get("not_suitable_at") is not None:
            d["not_suitable_at"] = d["not_suitable_at"]
        if d.get("not_suitable_reasons") is not None:
            d["not_suitable_reasons"] = d["not_suitable_reasons"]
        if d.get("unsuitable_reason_note") is not None:
            d["unsuitable_reason_note"] = d["unsuitable_reason_note"]
        if d.get("submitted") is not None:
            d["submitted"] = int(d["submitted"])
        if d.get("submitted_at") is not None:
            d["submitted_at"] = d["submitted_at"]
        out.append(d)

    return out

def query_new_since(db_path: Optional[str],
                    since_iso: str,
                    *,
                    min_score: float = 0.0,
                    hide_stale_days: Optional[int] = None,
                    limit: int = 100):
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
        SELECT {", ".join(REPORT_COLS)}
        FROM jobs
        WHERE {" AND ".join(where)}
        ORDER BY COALESCE(pulled_at, created_at) DESC
        LIMIT ?
    """
    params.append(int(limit))
    with connect(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row_to_report_dict(r) for r in rows]

def query_changed_since(db_path: Optional[str],
                        since_iso: str,
                        *,
                        limit: int = 200):
    _ensure_reports_columns(db_path)
    since_iso = _to_iso_compat(since_iso)
    change_expr = "COALESCE(updated_at, pulled_at, created_at)"
    with connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT {", ".join(REPORT_COLS)}, {change_expr} as changed_at
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
        d = _row_to_report_dict(core)
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

# ---------------------------
# Status API used by reports page
# ---------------------------
ALLOWED_STATUSES = {"new","interested","applied","interview","offer","rejected","archived"}

def set_job_status(db_path: Optional[str],
                   job_id: int,
                   *,
                   status: Optional[str] = None,
                   user_notes: Optional[str] = None,
                   starred: Optional[bool] = None) -> bool:
    _ensure_reports_columns(db_path)
    now = utcnow()
    sets, params = [], []
    if status is not None:
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Allowed: {sorted(ALLOWED_STATUSES)}")
        sets.append("status = ?"); params.append(status)
        sets.append("status_updated_at = ?"); params.append(now)
    if user_notes is not None:
        sets.append("status_notes = ?"); params.append(user_notes)
        # bump timestamp if only notes changed
        if "status_updated_at = ?" not in sets:
            sets.append("status_updated_at = ?"); params.append(now)
    if starred is not None:
        sets.append("starred = ?"); params.append(1 if starred else 0)
        if "status_updated_at = ?" not in sets:
            sets.append("status_updated_at = ?"); params.append(now)
    if not sets:
        return False
    sets.append("updated_at = ?"); params.append(now)

    with connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", (*params, int(job_id)))
        conn.commit()
        return cur.rowcount > 0

# ---------------------------
# Resume path / score persistence (used by 04_Generate_From_Jobs)
# ---------------------------
import sqlite3

def _get_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {r[1] for r in cur.fetchall()}

def _ensure_resume_columns(conn: sqlite3.Connection):
    cols = _get_columns(conn, "jobs")
    if "resume_path" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN resume_path TEXT")
    if "resume_score" not in cols:
        conn.execute("ALTER TABLE jobs ADD COLUMN resume_score REAL")
    conn.commit()

def _update_by_any_key(conn: sqlite3.Connection, job_key: str, sets_sql: str, params: tuple) -> int:
    cols = _get_columns(conn, "jobs")
    id_col = "job_id" if "job_id" in cols else "id"
    cur = conn.execute(f"UPDATE jobs SET {sets_sql} WHERE {id_col} = ?", (*params, job_key))
    if cur.rowcount == 0 and isinstance(job_key, str) and job_key.startswith(("http://", "https://")):
        cur = conn.execute(f"UPDATE jobs SET {sets_sql} WHERE url = ?", (*params, job_key))
    conn.commit()
    return cur.rowcount

def set_job_resume(
    db_path: str,
    *,
    job_key: str | None = None,
    resume_path: str | None = None,
    resume_score: float | None = None,
    **kwargs
) -> bool:
    # Back-compat: allow callers to pass job_id=
    if not job_key:
        job_key = kwargs.get("job_id")
    if not db_path or not job_key:
        return False

    with sqlite3.connect(db_path) as conn:
        _ensure_resume_columns(conn)
        changed = _update_by_any_key(conn, job_key, "resume_path = ?, resume_score = ?", (resume_path, resume_score))
        return changed > 0

def _ensure_action_columns(conn: sqlite3.Connection):
    cols = _get_columns(conn, "jobs")
    q = []
    if "not_suitable" not in cols:
        q.append("ALTER TABLE jobs ADD COLUMN not_suitable INTEGER DEFAULT 0")
    if "submitted" not in cols:
        q.append("ALTER TABLE jobs ADD COLUMN submitted INTEGER DEFAULT 0")
    if "submitted_at" not in cols:
        q.append("ALTER TABLE jobs ADD COLUMN submitted_at TEXT")
    for sql in q:
        conn.execute(sql)
    conn.commit()

def set_job_not_suitable(
    db_path: str,
    *,
    job_key: str | None = None,
    **kwargs
) -> bool:
    if not job_key:
        job_key = kwargs.get("job_id")  # back-compat
    if not db_path or not job_key:
        return False
    with sqlite3.connect(db_path) as conn:
        _ensure_action_columns(conn)
        changed = _update_by_any_key(conn, job_key, "not_suitable = 1", tuple())
        return changed > 0

def set_job_submitted(
    db_path: str,
    *,
    job_key: str | None = None,
    submitted: bool = True,
    submitted_at: str | None = None,
    **kwargs
) -> bool:
    if not job_key:
        job_key = kwargs.get("job_id")  # back-compat
    if not db_path or not job_key:
        return False
    if submitted and not submitted_at:
        submitted_at = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        _ensure_action_columns(conn)
        if submitted:
            sets = "submitted = 1, submitted_at = ?"
            params = (submitted_at,)
        else:
            sets = "submitted = 0, submitted_at = NULL"
            params = tuple()
        changed = _update_by_any_key(conn, job_key, sets, params)
        return changed > 0

def query_submitted(db_path: str | None = None, *, limit: int = 500) -> list[dict]:
    _ensure_reports_columns(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT id, title, company, location, remote, url, source,
                   posted_at, pulled_at, description, score, hash_key,
                   created_at, updated_at, status, status_notes, starred,
                   first_seen, last_seen,
                   resume_path, resume_score,
                   not_suitable, not_suitable_at,
                   not_suitable_reasons, unsuitable_reason_note,
                   submitted, submitted_at
            FROM jobs
            WHERE submitted = 1
            ORDER BY COALESCE(submitted_at, updated_at, pulled_at, created_at) DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()

    cols = ["id","title","company","location","remote","url","source",
            "posted_at","pulled_at","description","score","hash_key",
            "created_at","updated_at","status","status_notes","starred",
            "first_seen","last_seen","resume_path","resume_score",
            "not_suitable","not_suitable_at","not_suitable_reasons",
            "unsuitable_reason_note","submitted","submitted_at"]

    out = []
    for r in rows:
        d = {cols[i]: r[i] for i in range(len(cols))}
        d["job_id"] = d.pop("id", None)
        if d.get("remote") is not None:
            d["remote"] = bool(d["remote"])
        if d.get("starred") is not None:
            d["starred"] = int(d["starred"])
        if d.get("score") is not None:
            try: d["score"] = float(d["score"])
            except Exception: pass
        if d.get("resume_score") is not None:
            try: d["resume_score"] = float(d["resume_score"])
            except Exception: pass
        if d.get("not_suitable") is not None:
            d["not_suitable"] = bool(d["not_suitable"])
        if d.get("not_suitable_at") is not None:
            d["not_suitable_at"] = d["not_suitable_at"]
        if d.get("not_suitable_reasons") is not None:
            d["not_suitable_reasons"] = d["not_suitable_reasons"]
        if d.get("unsuitable_reason_note") is not None:
            d["unsuitable_reason_note"] = d["unsuitable_reason_note"]
        if d.get("submitted") is not None:
            d["submitted"] = bool(d["submitted"])
        if d.get("submitted_at") is not None:
            d["submitted_at"] = d["submitted_at"]
        out.append(d)
    return out

# ===== Preference / Training columns (flags + reasons) =====

def ensure_preference_columns(db_path: Optional[str] = None) -> None:
    """Ensure jobs table has columns needed for suitability & submission tracking."""
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("PRAGMA table_info(jobs)")
        cols = {r[1] for r in cur.fetchall()}
        alters = []
        if "not_suitable" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN not_suitable INTEGER DEFAULT 0")
        if "not_suitable_at" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN not_suitable_at TEXT")
        if "not_suitable_reasons" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN not_suitable_reasons TEXT")
        if "unsuitable_reason_note" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN unsuitable_reason_note TEXT")
        if "submitted" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN submitted INTEGER DEFAULT 0")
        if "submitted_at" not in cols:
            alters.append("ALTER TABLE jobs ADD COLUMN submitted_at TEXT")
        for sql in alters:
            cur.execute(sql)
        # Helpful indices
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_not_suitable ON jobs(not_suitable)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_jobs_submitted    ON jobs(submitted)")
        conn.commit()

def _where_by_job_key(conn: sqlite3.Connection, job_key: str) -> tuple[str, list]:
    cur = conn.execute("PRAGMA table_info(jobs)")
    cols = {r[1] for r in cur.fetchall()}
    wh, params = [], []
    if "job_id" in cols:
        wh.append("job_id = ?"); params.append(job_key)
    wh.append("id = ?"); params.append(job_key)
    wh.append("url = ?"); params.append(job_key)
    return " OR ".join(wh), params

def mark_not_suitable(db_path: str, job_id: str, *, reasons=None, note: str | None = None) -> bool:
    ensure_action_columns(db_path)
    ts = utcnow()
    reasons_json = json.dumps(reasons or [])
    with connect(db_path) as conn:
        where_sql, wparams = _where_by_job_key(conn, job_id)
        sql = f"""
            UPDATE jobs
            SET not_suitable = 1,
                not_suitable_at = ?,
                not_suitable_reasons = ?,
                not_suitable_note = ?,
                updated_at = COALESCE(updated_at, ?)
            WHERE {where_sql}
        """
        cur = conn.execute(sql, (ts, reasons_json, (note or ""), ts, *wparams))
        conn.commit()
        return cur.rowcount > 0

def mark_submitted(db_path: str, job_id: str, *, submitted: bool = True, submitted_at: str | None = None) -> bool:
    ensure_action_columns(db_path)
    ts = submitted_at or utcnow()
    with connect(db_path) as conn:
        where_sql, wparams = _where_by_job_key(conn, job_id)
        if submitted:
            set_clause = "submitted = 1, submitted_at = ?, updated_at = COALESCE(updated_at, ?)"
            params = (ts, ts, *wparams)
        else:
            set_clause = "submitted = 0, submitted_at = NULL, updated_at = COALESCE(updated_at, ?)"
            params = (ts, *wparams)
        sql = f"UPDATE jobs SET {set_clause} WHERE {where_sql}"
        cur = conn.execute(sql, params)
        conn.commit()
        return cur.rowcount > 0

# optional aliases to keep older imports working
set_job_not_suitable = mark_not_suitable
set_job_submitted    = mark_submitted

def query_submitted(db_path: str, *, limit: int = 500) -> list[dict]:
    ensure_action_columns(db_path)
    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(
            """
            SELECT id, title, company, location, remote, url, source,
                   posted_at, pulled_at, description, score, hash_key,
                   created_at, updated_at,
                   submitted, submitted_at,
                   resume_path, resume_score,
                   not_suitable, not_suitable_at
            FROM jobs
            WHERE submitted = 1
            ORDER BY COALESCE(submitted_at, updated_at, pulled_at, created_at) DESC
            LIMIT ?
            """, (int(limit),)
        ).fetchall()
    cols = ["id","title","company","location","remote","url","source",
            "posted_at","pulled_at","description","score","hash_key",
            "created_at","updated_at",
            "submitted","submitted_at",
            "resume_path","resume_score",
            "not_suitable","not_suitable_at"]
    out = []
    for r in rows:
        d = {cols[i]: r[i] for i in range(len(cols))}
        if d.get("remote") is not None: d["remote"] = bool(d["remote"])
        if d.get("submitted") is not None: d["submitted"] = bool(d["submitted"])
        if d.get("not_suitable") is not None: d["not_suitable"] = bool(d["not_suitable"])
        if d.get("score") is not None:
            try: d["score"] = float(d["score"])
            except Exception: pass
        return out + [d]

set_job_not_suitable = mark_not_suitable
set_job_submitted    = mark_submitted

# ---- Legacy/utility query: wide, filterable job list ----
from typing import List, Dict, Any, Optional

def query_jobs(
    db_path: Optional[str] = None,
    sources: Optional[List[str]] = None,
    min_score: float = 0.0,
    posted_start: Optional[str] = None,   # ISO or "YYYY-MM-DD"
    posted_end: Optional[str] = None,     # ISO or "YYYY-MM-DD"
    q: Optional[str] = None,              # substring across title/company/description
    limit: int = 500,
    order: str = "score_desc",            # "score_desc" | "date_desc" | "updated_desc"
) -> List[Dict[str, Any]]:
    """
    Return jobs with simple server-side filters. This is a compatibility wrapper
    used by pages like 03_job_history.py.

    Ordering:
      - score_desc:   score DESC, posted_at DESC
      - date_desc:    posted_at DESC, score DESC
      - updated_desc: updated_at DESC
    """
    init_db(db_path)

    where_sql = []
    params: list[Any] = []

    if sources:
        ph = ",".join(["?"] * len(sources))
        where_sql.append(f"source IN ({ph})")
        params.extend(sources)

    if min_score is not None and float(min_score) > 0.0:
        where_sql.append("score >= ?")
        params.append(float(min_score))

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
        SELECT id, title, company, location, remote, url, source,
               posted_at, pulled_at, description, score, hash_key,
               created_at, updated_at,
               status, status_notes, starred,
               first_seen, last_seen
        FROM jobs
        {where_clause}
        {order_by}
        LIMIT ?
    """
    params.append(int(limit))

    with connect(db_path) as conn:
        cur = conn.cursor()
        rows = cur.execute(sql, params).fetchall()

    cols = ["id","title","company","location","remote","url","source",
            "posted_at","pulled_at","description","score","hash_key",
            "created_at","updated_at","status","status_notes","starred",
            "first_seen","last_seen"]

    out: List[Dict[str, Any]] = []
    for r in rows:
        d = {cols[i]: r[i] for i in range(len(cols))}
        # normalize types
        d["remote"] = bool(d.get("remote")) if d.get("remote") is not None else None
        if d.get("starred") is not None:
            d["starred"] = int(d["starred"])
        if d.get("score") is not None:
            try: d["score"] = float(d["score"])
            except Exception: pass
        # compat aliases used by some pages
        d["job_id"] = d.get("id")
        d["user_notes"] = d.get("status_notes")
        d["first_seen"] = d.get("first_seen") or d.get("created_at") or d.get("pulled_at")
        d["last_seen"]  = d.get("last_seen")  or d.get("pulled_at")   or d.get("updated_at") or d.get("created_at")
        out.append(d)

    return out
