import asyncio
import csv
import io
import json
import html
import os
import secrets
import sqlite3
import string
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ── Config ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/sckeptic/icrm-hitek-full-db-mixed/resolve/main",
).rstrip("/")
INDEX_SOURCE = os.environ.get("ICMR_INDEX_SOURCE", "remote").lower()
PARALLELISM = int(os.environ.get("ICMR_PARALLEL", "2"))
THREADS_PER_CONN = int(os.environ.get("ICMR_THREADS_PER_CONN", "2"))
DUPLICATE_CAP = 2

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]

REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── IST Timezone ──────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

def now_ist() -> datetime:
    return datetime.now(IST)

def _fmt_ist(dt: Optional[datetime], fallback: str = "Never") -> str:
    if dt is None:
        return fallback
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=IST)
    return dt.astimezone(IST).strftime("%d %b %Y %I:%M %p IST")

def _from_iso(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=IST)

# ── Admin Config ───────────────────────────────────────────────────────────────
ADMIN_PASSWORD = "PRIYANSHU295"
DEVELOPER_CREDIT = "@VORTEX_PRIYANSHU"
_sess_lock = threading.Lock()

def _new_session() -> str:
    tok = secrets.token_hex(32)
    exp = now_ist() + timedelta(hours=24)
    with _db_lock:
        with _db() as con:
            con.execute(
                "INSERT OR REPLACE INTO admin_sessions(token, expires_at) VALUES (?, ?)",
                (tok, exp.isoformat()),
            )
            con.commit()
    return tok

def _check_session(tok: str) -> bool:
    if not tok:
        return False
    now = now_ist()
    with _db_lock:
        with _db() as con:
            row = con.execute(
                "SELECT expires_at FROM admin_sessions WHERE token=?",
                (tok,),
            ).fetchone()
            if not row:
                return False
            exp = _from_iso(row["expires_at"])
            if not exp or now > exp:
                con.execute("DELETE FROM admin_sessions WHERE token=?", (tok,))
                con.commit()
                return False
    return True

def _require_admin(request: Request):
    tok = request.headers.get("X-Admin-Token", "")
    if not _check_session(tok):
        raise HTTPException(401, {"error": "Admin authentication required"})

# ── SQLite Keys DB ─────────────────────────────────────────────────────────────
# Use /data (Render persistent disk) if available, else local folder
_DATA_DIR = "/data" if os.path.isdir("/data") else BASE
DB_PATH = os.path.join(_DATA_DIR, "keys.db")
_db_lock = threading.Lock()

def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def _init_db():
    with _db_lock:
        with _db() as con:
            con.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                key            TEXT PRIMARY KEY,
                owner          TEXT NOT NULL,
                plan           TEXT NOT NULL,
                rate_limit     INTEGER NOT NULL DEFAULT 100,
                created_at     TEXT NOT NULL,
                expires_at     TEXT,
                requests_today INTEGER NOT NULL DEFAULT 0,
                total_requests INTEGER NOT NULL DEFAULT 0,
                last_reset     TEXT NOT NULL,
                last_used      TEXT,
                status         TEXT NOT NULL DEFAULT 'active',
                note           TEXT DEFAULT ''
            )""")
            con.execute("""
            CREATE TABLE IF NOT EXISTS admin_sessions (
                token       TEXT PRIMARY KEY,
                expires_at  TEXT NOT NULL
            )""")
            con.execute("DELETE FROM admin_sessions WHERE expires_at <= ?", (now_ist().isoformat(),))
            con.commit()

# ── Key Helpers ────────────────────────────────────────────────────────────────
def _rand(n: int) -> str:
    return "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(n))

def _make_key(plan: str, rate_limit: int, username: str = "", note: str = "") -> dict:
    key = "vx_" + _rand(32)
    suffix = _rand(7)
    uname = username.lstrip("@").strip()
    owner = f"@{uname}_{suffix}" if uname else f"@vortex_priyanshu{suffix}"
    now = now_ist()
    if plan == "day":
        expires = (now + timedelta(days=1)).isoformat()
    elif plan == "month":
        expires = (now + timedelta(days=30)).isoformat()
    else:
        expires = None
    with _db_lock:
        with _db() as con:
            con.execute(
                "INSERT INTO api_keys VALUES (?,?,?,?,?,?,0,0,?,NULL,'active',?)",
                (key, owner, plan, rate_limit, now.isoformat(), expires, now.isoformat(), note),
            )
            con.commit()
    return {
        "key": key, "owner": owner, "plan": plan,
        "rate_limit": rate_limit, "expires_at": expires,
        "note": note, "created_at": now.isoformat(),
    }

def _validate_key(api_key: str) -> dict:
    with _db_lock:
        with _db() as con:
            row = con.execute("SELECT * FROM api_keys WHERE key=?", (api_key,)).fetchone()
    if not row:
        raise HTTPException(401, {"error": "Invalid API key", "api_developer": DEVELOPER_CREDIT})
    row = dict(row)
    if row["status"] == "revoked":
        raise HTTPException(401, {"error": "API key has been revoked", "api_developer": DEVELOPER_CREDIT})
    now = now_ist()
    if row["expires_at"]:
        exp = _from_iso(row["expires_at"])
        if now > exp:
            raise HTTPException(401, {
                "error": "API key expired",
                "expired_at": _fmt_ist(exp),
                "api_developer": DEVELOPER_CREDIT,
            })
    # Daily reset
    last_reset = _from_iso(row["last_reset"])
    if now.date() > last_reset.date():
        with _db_lock:
            with _db() as con:
                con.execute(
                    "UPDATE api_keys SET requests_today=0, last_reset=? WHERE key=?",
                    (now.isoformat(), api_key),
                )
                con.commit()
        row["requests_today"] = 0
    # Rate limit
    rl = row["rate_limit"]
    if rl > 0 and row["requests_today"] >= rl:
        raise HTTPException(429, {
            "error": "Rate limit exceeded",
            "limit_per_day": rl,
            "used_today": row["requests_today"],
            "resets_at": "Midnight IST",
            "api_developer": DEVELOPER_CREDIT,
        })
    # Increment
    with _db_lock:
        with _db() as con:
            con.execute(
                "UPDATE api_keys SET requests_today=requests_today+1, total_requests=total_requests+1, last_used=? WHERE key=?",
                (now.isoformat(), api_key),
            )
            con.commit()
    row["requests_today"] += 1
    return row

def _key_meta(row: dict) -> dict:
    exp_dt = _from_iso(row.get("expires_at"))
    valid_till = _fmt_ist(exp_dt, "Lifetime — Never expires") if exp_dt else "Lifetime — Never expires"
    rl = row["rate_limit"]
    return {
        "api_key_plan": row["plan"],
        "api_key_valid_till": valid_till,
        "api_key_requests_today": f"{row['requests_today']} / {'Unlimited' if rl == 0 else rl}",
        "api_developer": DEVELOPER_CREDIT,
    }

def _row_to_dict(row: dict) -> dict:
    now = now_ist()
    exp_dt = _from_iso(row.get("expires_at"))
    if row["status"] == "revoked":
        eff = "revoked"
    elif exp_dt and now > exp_dt:
        eff = "expired"
    else:
        eff = "active"
    expiring_soon = False
    if eff == "active" and exp_dt:
        expiring_soon = exp_dt <= now + timedelta(days=3)
    return {
        **row,
        "effective_status": eff,
        "expires_at_ist": _fmt_ist(exp_dt, "Never expires"),
        "last_used_ist": _fmt_ist(_from_iso(row.get("last_used")), "Never used"),
        "expiring_soon": expiring_soon,
    }

# ── DuckDB Connection Pool ─────────────────────────────────────────────────────
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")

def _idx_ready(kind: str) -> bool:
    return kind in REMOTE_INDEXES

def _new_conn() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET home_directory='/tmp'")
    con.execute("SET extension_directory='/tmp/duckdb_extensions'")
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    for kind, urls in REMOTE_INDEXES.items():
        view = f"people_{kind}"
        lst = ", ".join(f"'{u}'" for u in urls)
        con.execute(f"CREATE OR REPLACE VIEW {view} AS SELECT * FROM read_parquet([{lst}])")
    con.execute(f"SET threads = {THREADS_PER_CONN}")
    return con

def _get_conn() -> duckdb.DuckDBPyConnection:
    con = getattr(_thread_local, "conn", None)
    if con is None:
        con = _new_conn()
        _thread_local.conn = con
    return con

# ── Dedup & Connected Records ──────────────────────────────────────────────────
def _person_key(row: dict) -> tuple:
    ph = (row.get("phoneNumber") or "").strip()
    ad = (row.get("aadharNumber") or "").strip()
    if ph or ad:
        return (ph, ad)
    return (row.get("name") or "").strip(), (row.get("fathersName") or "").strip()

def _connected_numbers(row: dict) -> list[dict]:
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None:
            continue
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        connected.append({"field": field, "value": value})
    return connected

def _cap_duplicates(rows: list[dict]) -> list[dict]:
    seen: dict[tuple, int] = {}
    out = []
    for r in rows:
        k = _person_key(r)
        n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1
            record = dict(r)
            record["connected_numbers"] = _connected_numbers(record)
            out.append(record)
    return out

# ── Search Logic ───────────────────────────────────────────────────────────────
def _run_field_search(field: str, value: str, mode: str, limit: int) -> dict:
    if field not in SEARCH_FIELDS:
        raise ValueError(f"Unknown field: {field}")
    if mode not in ("exact", "contains"):
        raise ValueError(f"Unknown mode: {mode}")
    v = value.replace("'", "''")
    v2 = v.replace("%", r"\%").replace("_", r"\_")
    predicate = f"{field} = '{v}'" if mode == "exact" else f"{field} ILIKE '%{v2}%' ESCAPE '\\'"

    views = []
    if _idx_ready("phone"):
        views.append("people_phone")
    if _idx_ready("aadhar"):
        views.append("people_aadhar")

    if not views:
        return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}

    con = _get_conn()
    per_view = limit * DUPLICATE_CAP + 20
    parts = [f"SELECT * FROM {view} WHERE {predicate} LIMIT {per_view}" for view in views]
    sql = " UNION ALL ".join(parts)
    rows = con.execute(sql).fetchall()
    cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}

def _unified_search(q: str, limit: int = 10) -> dict:
    q = q.strip()
    if not q:
        return {"query": q, "searched_fields": [], "count": 0, "results": []}

    if q.isdigit() and len(q) >= 8:
        all_rows, searched = [], []
        if _idx_ready("phone"):
            r = _run_field_search("phoneNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("phoneNumber")
        if _idx_ready("aadhar"):
            r = _run_field_search("aadharNumber", q, "exact", limit)
            all_rows.extend(r["results"])
            searched.append("aadharNumber")
        all_rows = _cap_duplicates(all_rows)[:limit]
        return {"query": q, "searched_fields": searched, "count": len(all_rows), "results": all_rows}

    # Text searches: name first, then address/district/town/state as fallback.
    searched = []
    all_rows = []
    for field in ("name", "fathersName", "address", "district", "town", "state"):
        r = _run_field_search(field, q, "contains", limit)
        searched.append(field)
        all_rows.extend(r["results"])
        if len(_cap_duplicates(all_rows)) >= limit:
            break

    all_rows = _cap_duplicates(all_rows)[:limit]
    return {"query": q, "searched_fields": searched, "count": len(all_rows), "results": all_rows}

# ── Admin HTML ─────────────────────────────────────────────────────────────────
ADMIN_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>VORTEX API — Admin</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;font-family:'Segoe UI',Arial,sans-serif}
body{background:#1a0000;min-height:100vh;display:flex;align-items:flex-start;justify-content:center;padding:20px}
#login-wrap{width:100%;max-width:380px;margin:80px auto}
.login-card{background:linear-gradient(160deg,#fff,#f5f5f5);border-radius:20px;padding:40px 32px;box-shadow:0 24px 64px rgba(0,0,0,0.5),0 8px 20px rgba(220,38,38,0.3)}
.login-logo{text-align:center;margin-bottom:28px}
.login-logo h1{font-size:22px;font-weight:600;color:#b91c1c;letter-spacing:1px}
.login-logo p{font-size:12px;color:#bbb;margin-top:4px}
.login-input{width:100%;padding:12px 16px;border:1.5px solid #e5e5e5;border-radius:10px;font-size:14px;color:#333;background:#fff;margin-bottom:16px;transition:border .2s}
.login-input:focus{outline:none;border-color:#dc2626}
.login-btn{width:100%;padding:13px;background:linear-gradient(90deg,#b91c1c,#dc2626);border:none;border-radius:10px;color:#fff;font-size:14px;font-weight:600;cursor:pointer;letter-spacing:.5px;box-shadow:0 6px 20px rgba(185,28,28,.4)}
.login-btn:hover{background:linear-gradient(90deg,#991b1b,#b91c1c)}
.err{font-size:12px;color:#dc2626;margin-bottom:12px;display:none}
#app{width:100%;max-width:900px;display:none}
.topbar{background:linear-gradient(90deg,#b91c1c,#dc2626);padding:12px 20px;border-radius:16px 16px 0 0;display:flex;align-items:center;justify-content:space-between;box-shadow:0 4px 12px rgba(185,28,28,.5)}
.logo{font-size:14px;font-weight:600;color:#fff;letter-spacing:1px}
.live{display:flex;align-items:center;gap:6px;font-size:11px;color:rgba(255,255,255,.85)}
.live-dot{width:7px;height:7px;border-radius:50%;background:#fff;box-shadow:0 0 6px #fff;animation:pulse 1.5s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.card{background:linear-gradient(160deg,#fff,#fafafa);border-radius:0 0 16px 16px;box-shadow:0 20px 60px rgba(0,0,0,.4),0 8px 20px rgba(220,38,38,.2)}
.tabs{display:flex;border-bottom:1.5px solid #f0f0f0;padding:0 20px;background:#fff}
.tab{font-size:12px;padding:12px 16px;color:#999;cursor:pointer;border-bottom:2.5px solid transparent;font-weight:500;transition:color .2s}
.tab.active{color:#b91c1c;border-color:#dc2626}
.tab-content{display:none;padding:20px}
.tab-content.active{display:block}
.sec-lbl{font-size:10px;font-weight:600;color:#b91c1c;letter-spacing:1px;text-transform:uppercase;margin-bottom:12px}
.stat-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px}
@media(max-width:600px){.stat-grid{grid-template-columns:repeat(2,1fr)}}
.stat-card{background:#fff;border-radius:12px;padding:14px;border:1px solid #f0f0f0;box-shadow:0 4px 12px rgba(0,0,0,.06)}
.stat-num{font-size:26px;font-weight:600;color:#111}
.stat-num.red{color:#dc2626}
.stat-lbl{font-size:11px;color:#bbb;margin-top:2px;font-weight:500}
.warn-banner{background:linear-gradient(90deg,#fff5f5,#fff);border:1px solid #fecaca;border-left:3px solid #dc2626;border-radius:8px;padding:10px 14px;display:flex;align-items:center;gap:10px;margin-bottom:16px;font-size:12px;color:#991b1b;font-weight:500}
.search-bar{display:flex;gap:10px;margin-bottom:12px}
.search-bar input{flex:1;padding:9px 14px;border:1px solid #e8e8e8;border-radius:8px;font-size:12px;background:#fff}
.search-bar input:focus{outline:none;border-color:#dc2626}
.filter-row{display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap}
.chip{font-size:11px;padding:5px 14px;border-radius:20px;border:1px solid #e8e8e8;background:#fff;color:#999;cursor:pointer;transition:all .2s}
.chip.active{background:linear-gradient(90deg,#b91c1c,#dc2626);color:#fff;border-color:#dc2626;box-shadow:0 2px 8px rgba(220,38,38,.25)}
.key-card{background:#fff;border-radius:12px;padding:12px 14px;margin-bottom:8px;border:1px solid #f0f0f0;display:flex;align-items:flex-start;gap:12px;box-shadow:0 2px 8px rgba(0,0,0,.04);transition:box-shadow .2s}
.key-card:hover{box-shadow:0 4px 16px rgba(0,0,0,.08)}
.key-card.warn{border-color:#fecaca;background:#fff5f5}
.key-card.revoked{opacity:.45}
.avatar{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:600;flex-shrink:0}
.av-r{background:linear-gradient(135deg,#b91c1c,#dc2626);color:#fff}
.av-g{background:#f0f0f0;color:#999}
.av-w{background:#fff5f5;color:#b91c1c;border:1px solid #fecaca}
.key-info{flex:1;min-width:0}
.k-owner{font-size:12px;font-weight:600;color:#222}
.k-val{font-size:10px;color:#ccc;font-family:monospace;margin-top:2px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.k-note{font-size:10px;color:#dc2626;margin-top:2px;font-style:italic}
.bar-wrap{height:3px;background:#f0f0f0;border-radius:2px;margin-top:6px}
.bar-fill{height:3px;border-radius:2px}
.bar-lo{background:linear-gradient(90deg,#16a34a,#22c55e)}
.bar-mi{background:linear-gradient(90deg,#ea580c,#f97316)}
.bar-hi{background:linear-gradient(90deg,#b91c1c,#dc2626)}
.bar-txt{font-size:9px;color:#bbb;margin-top:2px}
.key-right{text-align:right;flex-shrink:0;min-width:100px}
.plan-pill{font-size:9px;padding:2px 8px;border-radius:20px;display:inline-block;margin-bottom:4px;font-weight:600}
.p-l{background:#fff5f5;color:#b91c1c;border:1px solid #fecaca}
.p-m{background:#fef3f3;color:#dc2626;border:1px solid #fca5a5}
.p-d{background:#f5f5f5;color:#666;border:1px solid #e5e5e5}
.p-rv{background:#f5f5f5;color:#ccc;border:1px solid #e5e5e5}
.exp-dt{font-size:10px;color:#bbb}
.exp-soon{color:#dc2626;font-weight:600}
.lu{font-size:9px;color:#ddd;margin-top:2px}
.btn-row{display:flex;gap:4px;margin-top:6px;justify-content:flex-end}
.btn-s{font-size:9px;padding:3px 10px;border-radius:5px;border:1px solid;cursor:pointer;background:transparent;font-weight:600;transition:all .2s}
.btn-cp{color:#dc2626;border-color:#fca5a5}
.btn-cp:hover{background:#fff5f5}
.btn-rv{color:#fff;border-color:#b91c1c;background:linear-gradient(90deg,#b91c1c,#dc2626);box-shadow:0 2px 6px rgba(220,38,38,.3)}
.form-group{margin-bottom:14px}
.f-lbl{font-size:10px;color:#999;font-weight:600;letter-spacing:.5px;text-transform:uppercase;margin-bottom:5px}
.f-input{width:100%;padding:10px 12px;border:1px solid #e8e8e8;border-radius:8px;font-size:12px;color:#333;background:#fff}
.f-input:focus{outline:none;border-color:#dc2626}
.plan-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px}
@media(max-width:500px){.plan-grid{grid-template-columns:repeat(2,1fr)}}
.plan-box{background:#fff;border:1px solid #e8e8e8;border-radius:10px;padding:10px;text-align:center;cursor:pointer;transition:all .2s}
.plan-box.sel{background:#fff5f5;border-color:#fca5a5;box-shadow:0 4px 12px rgba(220,38,38,.12)}
.plan-nm{font-size:12px;font-weight:600;color:#333}
.plan-box.sel .plan-nm{color:#b91c1c}
.plan-sub{font-size:10px;color:#bbb;margin-top:2px}
.rl-row{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.rl-lbl{font-size:12px;color:#777;flex:1}
.rl-inp{width:80px;padding:8px;border:1px solid #e8e8e8;border-radius:8px;font-size:12px;color:#333;text-align:center}
.rl-inp:focus{outline:none;border-color:#dc2626}
.rl-hint{font-size:11px;color:#dc2626;background:#fff5f5;padding:6px 10px;border-radius:6px;border:1px solid #fecaca;margin-bottom:14px}
.gen-btn{width:100%;padding:12px;background:linear-gradient(90deg,#b91c1c,#dc2626);border:none;border-radius:10px;color:#fff;font-size:13px;font-weight:600;cursor:pointer;letter-spacing:.5px;box-shadow:0 6px 20px rgba(185,28,28,.4)}
.gen-btn:hover{background:linear-gradient(90deg,#991b1b,#b91c1c)}
.result-box{margin-top:16px;background:#f8f8f8;border-radius:10px;padding:14px;border:1px solid #e8e8e8;display:none}
.result-box.show{display:block}
.result-key{font-family:monospace;font-size:13px;color:#b91c1c;word-break:break-all;font-weight:600}
.result-row{display:flex;justify-content:space-between;font-size:11px;color:#666;margin-top:6px}
.copy-full-btn{margin-top:10px;width:100%;padding:8px;background:#fff;border:1px solid #fca5a5;border-radius:7px;color:#dc2626;font-size:11px;font-weight:600;cursor:pointer}
.io-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:16px}
.io-btn{background:#fff;border:1px solid #e8e8e8;border-radius:12px;padding:16px;text-align:center;cursor:pointer;transition:all .2s;box-shadow:0 2px 8px rgba(0,0,0,.04)}
.io-btn:hover{border-color:#fca5a5;box-shadow:0 4px 12px rgba(220,38,38,.1)}
.io-icon{font-size:24px;margin-bottom:6px}
.io-lbl{font-size:12px;color:#555;font-weight:600}
.mode-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:10px}
.mode-box{border-radius:9px;padding:10px;text-align:center;cursor:pointer;border:1px solid #e8e8e8;background:#fff;transition:all .2s}
.mode-box.sel{background:#fff5f5;border-color:#fca5a5;box-shadow:0 3px 10px rgba(220,38,38,.1)}
.mode-nm{font-size:12px;font-weight:600;color:#333}
.mode-box.sel .mode-nm{color:#b91c1c}
.mode-sub{font-size:10px;color:#bbb;margin-top:2px}
.imp-area{background:#fafafa;border:1.5px dashed #e5e5e5;border-radius:10px;padding:20px;text-align:center;cursor:pointer;margin-top:12px;transition:border .2s}
.imp-area:hover,.imp-area.drag{border-color:#dc2626;background:#fff5f5}
.imp-area p{font-size:12px;color:#bbb}
.imp-btn{margin-top:12px;width:100%;padding:10px;background:linear-gradient(90deg,#b91c1c,#dc2626);border:none;border-radius:8px;color:#fff;font-size:12px;font-weight:600;cursor:pointer}
.toast{position:fixed;bottom:20px;right:20px;background:#222;color:#fff;padding:10px 18px;border-radius:8px;font-size:12px;display:none;z-index:999;animation:slideup .3s}
@keyframes slideup{from{transform:translateY(20px);opacity:0}to{transform:translateY(0);opacity:1}}
.empty{text-align:center;padding:30px;color:#ccc;font-size:13px}
#file-input{display:none}
</style>
</head>
<body>

<div id="login-wrap">
  <div class="login-card">
    <div class="login-logo">
      <h1>⚡ VORTEX API</h1>
      <p>Admin Panel — Secure Access</p>
    </div>
    <div class="err" id="err">Wrong password. Try again.</div>
    <input class="login-input" type="password" id="pw" placeholder="Enter admin password" onkeydown="if(event.key==='Enter')doLogin()">
    <button class="login-btn" onclick="doLogin()">Login</button>
  </div>
</div>

<div id="app">
  <div class="topbar">
    <div class="logo">⚡ VORTEX API — ADMIN</div>
    <div class="live"><div class="live-dot"></div>LIVE</div>
  </div>
  <div class="card">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('dash',this)">Dashboard</div>
      <div class="tab" onclick="switchTab('keys',this)">Keys</div>
      <div class="tab" onclick="switchTab('gen',this)">Generate</div>
      <div class="tab" onclick="switchTab('csv',this)">Import/Export</div>
    </div>

    <!-- DASHBOARD -->
    <div class="tab-content active" id="tab-dash">
      <div class="sec-lbl">Overview</div>
      <div class="stat-grid">
        <div class="stat-card"><div class="stat-num red" id="s-active">—</div><div class="stat-lbl">Active keys</div></div>
        <div class="stat-card"><div class="stat-num" id="s-exp">—</div><div class="stat-lbl">Expired</div></div>
        <div class="stat-card"><div class="stat-num" id="s-rev">—</div><div class="stat-lbl">Revoked</div></div>
        <div class="stat-card"><div class="stat-num red" id="s-today">—</div><div class="stat-lbl">Requests today</div></div>
        <div class="stat-card"><div class="stat-num" id="s-soon">—</div><div class="stat-lbl">Expiring soon</div></div>
        <div class="stat-card"><div class="stat-num" id="s-total" style="font-size:16px">—</div><div class="stat-lbl">Total requests</div></div>
      </div>
      <div class="warn-banner" id="warn-banner" style="display:none">
        ⚠️ <span id="warn-txt"></span>
      </div>
    </div>

    <!-- KEYS -->
    <div class="tab-content" id="tab-keys">
      <div class="search-bar">
        <input type="text" id="key-search" placeholder="Search by owner name..." oninput="loadKeys()">
      </div>
      <div class="filter-row">
        <div class="chip active" onclick="setFilter('all',this)">All</div>
        <div class="chip" onclick="setFilter('active',this)">Active</div>
        <div class="chip" onclick="setFilter('expired',this)">Expired</div>
        <div class="chip" onclick="setFilter('revoked',this)">Revoked</div>
      </div>
      <div id="keys-list"><div class="empty">Loading keys...</div></div>
    </div>

    <!-- GENERATE -->
    <div class="tab-content" id="tab-gen">
      <div class="sec-lbl">Generate new API key</div>
      <div class="form-group">
        <div class="f-lbl">Telegram username (optional)</div>
        <input class="f-input" id="g-user" placeholder="@username  →  auto: @vortex_priyanshuXXXXXXX">
      </div>
      <div class="form-group">
        <div class="f-lbl">Plan</div>
        <div class="plan-grid">
          <div class="plan-box sel" onclick="setPlan('day',this)"><div class="plan-nm">Day</div><div class="plan-sub">24 hours</div></div>
          <div class="plan-box" onclick="setPlan('month',this)"><div class="plan-nm">Month</div><div class="plan-sub">30 days</div></div>
          <div class="plan-box" onclick="setPlan('lifetime',this)"><div class="plan-nm">Lifetime</div><div class="plan-sub">Never expires</div></div>
          <div class="plan-box" onclick="setPlan('custom',this)" style="border-style:dashed;opacity:.6"><div class="plan-nm">Custom</div><div class="plan-sub">Manual days</div></div>
        </div>
        <div id="custom-days-wrap" style="display:none;margin-bottom:10px">
          <div class="f-lbl">Custom days</div>
          <input class="f-input" id="g-days" type="number" min="1" placeholder="e.g. 7">
        </div>
      </div>
      <div class="rl-row">
        <div class="rl-lbl">Rate limit (requests / day) — 0 = unlimited</div>
        <input class="rl-inp" id="g-rl" type="number" value="100" min="0">
      </div>
      <div class="rl-hint" id="rl-hint">Day plan → suggested: 100 req/day</div>
      <div class="form-group">
        <div class="f-lbl">Note (optional)</div>
        <input class="f-input" id="g-note" placeholder="e.g. Client XYZ, Testing...">
      </div>
      <button class="gen-btn" onclick="generateKey()">⚡ Generate API Key</button>
      <div class="result-box" id="gen-result">
        <div class="f-lbl">Generated key</div>
        <div class="result-key" id="r-key"></div>
        <div class="result-row"><span id="r-owner"></span><span id="r-plan"></span></div>
        <div class="result-row"><span id="r-exp"></span><span id="r-rl"></span></div>
        <button class="copy-full-btn" onclick="copyKey()">📋 Copy Key</button>
      </div>
    </div>

    <!-- CSV -->
    <div class="tab-content" id="tab-csv">
      <div class="sec-lbl">Export</div>
      <div class="io-grid">
        <div class="io-btn" onclick="doExport()">
          <div class="io-icon">📥</div>
          <div class="io-lbl">Export CSV</div>
        </div>
        <div class="io-btn" onclick="document.getElementById('file-input').click()">
          <div class="io-icon">📤</div>
          <div class="io-lbl">Select CSV file</div>
        </div>
      </div>
      <input type="file" id="file-input" accept=".csv" onchange="onFileSelect(this)">

      <div id="import-section" style="display:none">
        <div class="sec-lbl" style="margin-top:4px">Import mode</div>
        <div class="mode-grid">
          <div class="mode-box sel" id="mode-merge" onclick="setMode('merge')">
            <div class="mode-nm">Merge</div>
            <div class="mode-sub">Skip existing keys</div>
          </div>
          <div class="mode-box" id="mode-replace" onclick="setMode('replace')">
            <div class="mode-nm">Replace all</div>
            <div class="mode-sub">Full restore</div>
          </div>
        </div>
        <div style="font-size:11px;color:#bbb;margin-top:8px" id="file-name"></div>
        <button class="imp-btn" id="do-import" onclick="doImport()">📤 Import CSV</button>
      </div>
    </div>

  </div>
</div>

<div class="toast" id="toast"></div>

<script>
let TOKEN = '', CUR_FILTER = 'all', CUR_PLAN = 'day', CUR_MODE = 'merge', CSV_DATA = '';

async function doLogin() {
  const pw = document.getElementById('pw').value;
  const err = document.getElementById('err');
  err.style.display = 'none';
  try {
    const r = await fetch('/admin/login', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({password:pw})
    });
    if (!r.ok) {
      err.textContent = r.status === 401 ? 'Wrong password. Try again.' : 'Login service error (' + r.status + ').';
      err.style.display = 'block';
      return;
    }
    const d = await r.json();
    if (!d.token) throw new Error('No session token returned');
    TOKEN = d.token;
    document.getElementById('login-wrap').style.display = 'none';
    document.getElementById('app').style.display = 'block';
    await Promise.all([loadStats(), loadKeys()]);
  } catch (e) {
    err.textContent = 'Cannot connect to admin service. Check server logs.';
    err.style.display = 'block';
  }
}

function switchTab(id, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-' + id).classList.add('active');
  if (id === 'dash') loadStats();
  if (id === 'keys') loadKeys();
}

async function loadStats() {
  const r = await fetch('/admin/stats', {headers:{'X-Admin-Token':TOKEN}});
  if (!r.ok) return;
  const d = await r.json();
  document.getElementById('s-active').textContent = d.active;
  document.getElementById('s-exp').textContent = d.expired;
  document.getElementById('s-rev').textContent = d.revoked;
  document.getElementById('s-today').textContent = d.requests_today.toLocaleString();
  document.getElementById('s-soon').textContent = d.expiring_soon;
  document.getElementById('s-total').textContent = (d.total_requests || 0).toLocaleString();
  if (d.expiring_soon > 0) {
    document.getElementById('warn-banner').style.display = 'flex';
    document.getElementById('warn-txt').textContent = d.expiring_soon + ' key' + (d.expiring_soon > 1 ? 's' : '') + ' expiring within 3 days';
  }
}

function setFilter(f, el) {
  CUR_FILTER = f;
  document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  loadKeys();
}

async function loadKeys() {
  const s = document.getElementById('key-search').value;
  const r = await fetch('/admin/keys?status=' + CUR_FILTER + '&search=' + encodeURIComponent(s), {headers:{'X-Admin-Token':TOKEN}});
  if (!r.ok) return;
  const keys = await r.json();
  const el = document.getElementById('keys-list');
  if (!keys.length) { el.innerHTML = '<div class="empty">No keys found</div>'; return; }
  el.innerHTML = keys.map(k => {
    const pct = k.rate_limit > 0 ? Math.min(100, Math.round(k.requests_today / k.rate_limit * 100)) : 0;
    const barCls = pct >= 85 ? 'bar-hi' : pct >= 50 ? 'bar-mi' : 'bar-lo';
    const avCls = k.effective_status === 'revoked' ? 'av-g' : k.expiring_soon ? 'av-w' : 'av-r';
    const initials = k.owner.replace('@','').substring(0,2).toUpperCase();
    const planCls = k.plan === 'lifetime' ? 'p-l' : k.plan === 'month' ? 'p-m' : 'p-d';
    const cardCls = k.effective_status === 'revoked' ? 'key-card revoked' : k.expiring_soon ? 'key-card warn' : 'key-card';
    const usageStr = k.rate_limit === 0 ? '∞ unlimited' : k.requests_today + ' / ' + k.rate_limit + ' today';
    const esc = s => String(s ?? '').replace(/[&<>'\"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','\"':'&quot;'}[c]));
    const safeOwner = esc(k.owner), safeKey = esc(k.key), safeNote = esc(k.note), safeLast = esc(k.last_used_ist), safeExp = esc(k.expires_at_ist);
    const noteHtml = safeNote ? '<div class="k-note">' + safeNote + '</div>' : '';
    const revokeBtn = k.effective_status !== 'revoked' ? '<button class="btn-s btn-rv" onclick="revokeKey(\''+encodeURIComponent(k.key)+'\')">Revoke</button>' : '';
    return '<div class="' + cardCls + '"><div class="avatar ' + avCls + '">' + esc(initials) + '</div><div class="key-info"><div class="k-owner">' + safeOwner + '</div><div class="k-val">' + safeKey + '</div>' + noteHtml + '<div class="bar-wrap"><div class="bar-fill ' + barCls + '" style="width:' + pct + '%"></div></div><div class="bar-txt">' + usageStr + '</div><div class="bar-txt" style="color:#ddd">Last: ' + safeLast + '</div></div><div class="key-right"><span class="plan-pill ' + planCls + '">' + esc(k.plan) + '</span><div class="exp-dt' + (k.expiring_soon ? ' exp-soon' : '') + '">' + (k.expiring_soon ? '⚠ ' : '') + safeExp + '</div><div class="btn-row"><button class="btn-s btn-cp" onclick="copyText(\''+encodeURIComponent(k.key)+'\')">Copy</button>' + revokeBtn + '</div></div></div>';
  }).join('');
}

async function revokeKey(key) {
  key = decodeURIComponent(key);
  if (!confirm('Revoke this key?')) return;
  const r = await fetch('/admin/keys/' + encodeURIComponent(key) + '/revoke', {method:'POST',headers:{'X-Admin-Token':TOKEN}});
  if (r.ok) { toast('Key revoked'); loadKeys(); loadStats(); }
}

function setPlan(p, el) {
  CUR_PLAN = p;
  document.querySelectorAll('.plan-box').forEach(b => b.classList.remove('sel'));
  el.classList.add('sel');
  const hints = {day:'Day plan → suggested: 100 req/day',month:'Month plan → suggested: 3,000 req/day',lifetime:'Lifetime → suggested: 0 (unlimited)',custom:'Custom plan → set your own limit'};
  const vals = {day:100,month:3000,lifetime:0,custom:100};
  document.getElementById('rl-hint').textContent = hints[p] || '';
  document.getElementById('g-rl').value = vals[p] !== undefined ? vals[p] : 100;
  document.getElementById('custom-days-wrap').style.display = p === 'custom' ? 'block' : 'none';
}

async function generateKey() {
  const username = document.getElementById('g-user').value;
  const rl = parseInt(document.getElementById('g-rl').value) || 0;
  const note = document.getElementById('g-note').value;
  let plan = CUR_PLAN;
  let body = {plan, rate_limit: rl, username, note};
  if (plan === 'custom') {
    const days = parseInt(document.getElementById('g-days').value);
    if (!days || days < 1) { toast('Enter valid custom days'); return; }
    body.custom_days = days;
    plan = 'custom';
  }
  const r = await fetch('/admin/keys/generate', {method:'POST',headers:{'Content-Type':'application/json','X-Admin-Token':TOKEN},body:JSON.stringify(body)});
  if (!r.ok) { toast('Generation failed'); return; }
  const d = await r.json();
  document.getElementById('r-key').textContent = d.key;
  document.getElementById('r-owner').textContent = d.owner;
  document.getElementById('r-plan').textContent = d.plan.toUpperCase();
  document.getElementById('r-exp').textContent = d.expires_at ? 'Expires: ' + d.expires_at.substring(0,10) : 'Never expires';
  document.getElementById('r-rl').textContent = d.rate_limit === 0 ? 'Unlimited' : d.rate_limit + ' req/day';
  document.getElementById('gen-result').classList.add('show');
  toast('Key generated!');
}

function copyKey() { copyText(document.getElementById('r-key').textContent); }
function decodeURIComponentSafe(v) { try { return decodeURIComponent(v); } catch (_) { return v; } }

function copyText(t) {
  t = decodeURIComponentSafe(t);
  navigator.clipboard.writeText(t).then(() => toast('Copied!')).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = t; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); document.body.removeChild(ta); toast('Copied!');
  });
}

async function doExport() {
  const r = await fetch('/admin/export', {headers:{'X-Admin-Token':TOKEN}});
  if (!r.ok) { toast('Export failed'); return; }
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'vortex_keys.csv'; a.click();
  URL.revokeObjectURL(url);
  toast('Exported!');
}

function onFileSelect(input) {
  const file = input.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    CSV_DATA = e.target.result;
    document.getElementById('file-name').textContent = '📄 ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    document.getElementById('import-section').style.display = 'block';
  };
  reader.readAsText(file);
}

function setMode(m) {
  CUR_MODE = m;
  document.getElementById('mode-merge').classList.toggle('sel', m === 'merge');
  document.getElementById('mode-replace').classList.toggle('sel', m === 'replace');
}

async function doImport() {
  if (!CSV_DATA) { toast('No file selected'); return; }
  if (CUR_MODE === 'replace' && !confirm('This will DELETE all existing keys. Continue?')) return;
  const r = await fetch('/admin/import', {method:'POST',headers:{'Content-Type':'application/json','X-Admin-Token':TOKEN},body:JSON.stringify({csv:CSV_DATA,mode:CUR_MODE})});
  if (!r.ok) { toast('Import failed'); return; }
  const d = await r.json();
  toast('Imported: ' + d.imported + ', Skipped: ' + d.skipped);
  document.getElementById('import-section').style.display = 'none';
  CSV_DATA = '';
  loadStats();
}

function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg; t.style.display = 'block';
  setTimeout(() => t.style.display = 'none', 2500);
}
</script>
</body>
</html>"""

# ── FastAPI App ────────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="ICMR + HITEK Search API")

class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10

# ── Admin Routes ───────────────────────────────────────────────────────────────
@fastapi_app.get("/admin-panel", response_class=HTMLResponse)
async def admin_panel():
    return HTMLResponse(content=ADMIN_HTML)

@fastapi_app.post("/admin/login")
async def admin_login(body: dict):
    password = body.get("password") if isinstance(body, dict) else None
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail={"error": "Wrong password"})
    return {"token": _new_session()}

@fastapi_app.get("/admin/stats")
async def admin_stats(request: Request):
    _require_admin(request)
    now_str = now_ist().isoformat()
    soon_str = (now_ist() + timedelta(days=3)).isoformat()
    with _db_lock:
        with _db() as con:
            active = con.execute(
                "SELECT COUNT(*) FROM api_keys WHERE status='active' AND (expires_at IS NULL OR expires_at > ?)",
                (now_str,)
            ).fetchone()[0]
            expired = con.execute(
                "SELECT COUNT(*) FROM api_keys WHERE status='active' AND expires_at IS NOT NULL AND expires_at <= ?",
                (now_str,)
            ).fetchone()[0]
            revoked = con.execute(
                "SELECT COUNT(*) FROM api_keys WHERE status='revoked'"
            ).fetchone()[0]
            req_today = con.execute("SELECT COALESCE(SUM(requests_today),0) FROM api_keys").fetchone()[0]
            req_total = con.execute("SELECT COALESCE(SUM(total_requests),0) FROM api_keys").fetchone()[0]
            soon = con.execute(
                "SELECT COUNT(*) FROM api_keys WHERE status='active' AND expires_at IS NOT NULL AND expires_at > ? AND expires_at <= ?",
                (now_str, soon_str)
            ).fetchone()[0]
    return {
        "active": active, "expired": expired, "revoked": revoked,
        "requests_today": int(req_today), "total_requests": int(req_total),
        "expiring_soon": soon,
    }

@fastapi_app.get("/admin/keys")
async def admin_keys(request: Request, status: str = Query("all"), search: str = Query("")):
    _require_admin(request)
    now_str = now_ist().isoformat()
    like = f"%{search}%"
    with _db_lock:
        with _db() as con:
            if status == "active":
                rows = con.execute(
                    "SELECT * FROM api_keys WHERE status='active' AND (expires_at IS NULL OR expires_at > ?) AND owner LIKE ? ORDER BY created_at DESC",
                    (now_str, like)
                ).fetchall()
            elif status == "expired":
                rows = con.execute(
                    "SELECT * FROM api_keys WHERE status='active' AND expires_at IS NOT NULL AND expires_at <= ? AND owner LIKE ? ORDER BY created_at DESC",
                    (now_str, like)
                ).fetchall()
            elif status == "revoked":
                rows = con.execute(
                    "SELECT * FROM api_keys WHERE status='revoked' AND owner LIKE ? ORDER BY created_at DESC",
                    (like,)
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT * FROM api_keys WHERE owner LIKE ? ORDER BY created_at DESC",
                    (like,)
                ).fetchall()
    return [_row_to_dict(dict(r)) for r in rows]

@fastapi_app.post("/admin/keys/generate")
async def admin_generate(request: Request, body: dict):
    _require_admin(request)
    plan = body.get("plan", "day")
    if plan not in ("day", "month", "lifetime", "custom"):
        raise HTTPException(400, {"error": "Invalid plan"})
    try:
        rl = int(body.get("rate_limit", 100))
    except (TypeError, ValueError):
        raise HTTPException(400, {"error": "rate_limit must be an integer"})
    if rl < 0:
        raise HTTPException(400, {"error": "rate_limit must be >= 0"})
    username = str(body.get("username", "")).strip()[:100]
    note = str(body.get("note", "")).strip()[:500]
    key_data = _make_key(plan if plan != "custom" else "day", rl, username, note)
    if plan == "custom":
        try:
            days = int(body.get("custom_days", 1))
        except (TypeError, ValueError):
            raise HTTPException(400, {"error": "custom_days must be an integer"})
        if days < 1 or days > 3650:
            raise HTTPException(400, {"error": "custom_days must be between 1 and 3650"})
        new_exp = (now_ist() + timedelta(days=days)).isoformat()
        with _db_lock:
            with _db() as con:
                con.execute("UPDATE api_keys SET expires_at=?, plan='custom' WHERE key=?", (new_exp, key_data["key"]))
                con.commit()
        key_data["expires_at"] = new_exp
        key_data["plan"] = "custom"
    return key_data

@fastapi_app.post("/admin/keys/{key}/revoke")
async def admin_revoke(key: str, request: Request):
    _require_admin(request)
    with _db_lock:
        with _db() as con:
            r = con.execute("UPDATE api_keys SET status='revoked' WHERE key=?", (key,))
            con.commit()
            if r.rowcount == 0:
                raise HTTPException(404, {"error": "Key not found"})
    return {"success": True}

@fastapi_app.get("/admin/export")
async def admin_export(request: Request):
    _require_admin(request)
    with _db_lock:
        with _db() as con:
            rows = con.execute("SELECT * FROM api_keys ORDER BY created_at DESC").fetchall()
            cols = [d[0] for d in con.description]
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in rows:
        w.writerow(list(r))
    fname = f"vortex_keys_{now_ist().strftime('%Y%m%d_%H%M')}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={fname}"},
    )

@fastapi_app.post("/admin/import")
async def admin_import(request: Request, body: dict):
    _require_admin(request)
    mode = body.get("mode", "merge")
    if mode not in ("merge", "replace"):
        raise HTTPException(400, {"error": "mode must be merge or replace"})

    csv_raw = body.get("csv", "")
    if not isinstance(csv_raw, str) or not csv_raw.strip():
        raise HTTPException(400, {"error": "Empty CSV"})
    if len(csv_raw.encode("utf-8")) > 10 * 1024 * 1024:
        raise HTTPException(413, {"error": "CSV too large (max 10 MB)"})

    reader = csv.DictReader(io.StringIO(csv_raw))
    required = {
        "key", "owner", "plan", "rate_limit", "created_at", "expires_at",
        "requests_today", "total_requests", "last_reset", "last_used",
        "status", "note",
    }
    if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
        raise HTTPException(400, {"error": "CSV missing required columns"})

    rows = list(reader)
    if not rows:
        raise HTTPException(400, {"error": "Empty CSV"})

    valid_rows = []
    skipped = 0
    now_str = now_ist().isoformat()
    allowed_plans = {"day", "month", "lifetime", "custom"}
    allowed_status = {"active", "revoked"}

    for row in rows:
        try:
            key = str(row.get("key", "")).strip()
            owner = str(row.get("owner", "")).strip()
            plan = str(row.get("plan", "day")).strip()
            rate_limit = int(row.get("rate_limit", 100))
            created_at = str(row.get("created_at") or now_str).strip()
            expires_at = str(row.get("expires_at") or "").strip() or None
            requests_today = int(row.get("requests_today", 0))
            total_requests = int(row.get("total_requests", 0))
            last_reset = str(row.get("last_reset") or now_str).strip()
            last_used = str(row.get("last_used") or "").strip() or None
            status = str(row.get("status", "active")).strip()
            note = str(row.get("note", "")).strip()

            if not key or len(key) > 200 or not owner:
                raise ValueError("missing key/owner")
            if plan not in allowed_plans or status not in allowed_status:
                raise ValueError("invalid plan/status")
            if rate_limit < 0 or requests_today < 0 or total_requests < 0:
                raise ValueError("negative numeric value")
            # Validate timestamps before changing the database.
            for ts in (created_at, last_reset, expires_at, last_used):
                if ts:
                    _from_iso(ts)

            valid_rows.append((
                key, owner, plan, rate_limit, created_at, expires_at,
                requests_today, total_requests, last_reset, last_used,
                status, note,
            ))
        except (TypeError, ValueError, OverflowError):
            skipped += 1

    if mode == "replace" and not valid_rows:
        raise HTTPException(400, {"error": "No valid rows to import; existing keys were not changed"})

    imported = 0
    with _db_lock:
        with _db() as con:
            try:
                con.execute("BEGIN")
                if mode == "replace":
                    con.execute("DELETE FROM api_keys")
                for row in valid_rows:
                    key = row[0]
                    if mode == "merge" and con.execute(
                        "SELECT 1 FROM api_keys WHERE key=?", (key,)
                    ).fetchone():
                        skipped += 1
                        continue
                    con.execute(
                        """INSERT INTO api_keys
                        (key,owner,plan,rate_limit,created_at,expires_at,
                         requests_today,total_requests,last_reset,last_used,status,note)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                        row,
                    )
                    imported += 1
                con.commit()
            except Exception:
                con.rollback()
                raise HTTPException(400, {"error": "Import failed; no database changes were kept"})

    return {"imported": imported, "skipped": skipped, "mode": mode}

# ── Original Routes ────────────────────────────────────────────────────────────
@fastapi_app.get("/")
def root():
    return {
        "app": "ICMR + HITEK Search API",
        "records": 2_504_793_870,
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "index_source": INDEX_SOURCE,
        "columns": SEARCH_FIELDS,
        "docs": "/docs",
        "admin_panel": "/admin-panel",
        "developer": DEVELOPER_CREDIT,
    }

@fastapi_app.get("/health")
def health():
    return {
        "status": "ok",
        "indexes": {"phone": _idx_ready("phone"), "aadhar": _idx_ready("aadhar")},
        "index_source": INDEX_SOURCE,
    }

# ── Search Routes (API key required) ──────────────────────────────────────────
@fastapi_app.get("/search")
async def search(
    request: Request,
    q: Optional[str] = Query(None),
    mobile: Optional[str] = Query(None),
    api_key: Optional[str] = Query(None),
    field: Optional[str] = Query(None),
    mode: str = Query("exact"),
    limit: int = Query(10, ge=1, le=1000),
    pretty: bool = Query(True),
):
    key_val = api_key or request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    if not key_val:
        raise HTTPException(401, {
            "error": "API key required",
            "usage": "Pass ?api_key=YOUR_KEY or header X-API-Key: YOUR_KEY",
            "api_developer": DEVELOPER_CREDIT,
        })
    key_row = _validate_key(key_val)
    q_val = (q or mobile or "").strip()
    if not q_val:
        raise HTTPException(422, "Provide q or mobile parameter")
    loop = asyncio.get_running_loop()
    if field:
        data = await loop.run_in_executor(pool, _run_field_search, field, q_val, mode, limit)
    else:
        data = await loop.run_in_executor(pool, _unified_search, q_val, limit)
    result = {
        "success": bool(data["count"]),
        **data,
        "number": q_val,
        "total": data["count"],
        **_key_meta(key_row),
    }
    return Response(
        content=json.dumps(result, indent=2 if pretty else None, ensure_ascii=False),
        media_type="application/json",
    )

@fastapi_app.post("/search/parallel")
async def search_parallel(request: Request, req: BatchRequest, api_key: Optional[str] = Query(None)):
    key_val = api_key or request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    if not key_val:
        raise HTTPException(401, {
            "error": "API key required",
            "usage": "Pass ?api_key=YOUR_KEY or header X-API-Key: YOUR_KEY",
            "api_developer": DEVELOPER_CREDIT,
        })
    key_row = _validate_key(key_val)
    if not req.queries:
        raise HTTPException(400, "queries must not be empty")
    if len(req.queries) > 50:
        raise HTTPException(400, "max 50 queries per batch")
    loop = asyncio.get_running_loop()
    tasks = [
        loop.run_in_executor(
            pool, _run_field_search,
            item.get("field", "phoneNumber"),
            item.get("value", ""),
            item.get("mode", "exact"),
            int(item.get("limit", req.limit)),
        )
        for item in req.queries
    ]
    results = await asyncio.gather(*tasks)
    payload = {
        "searches": len(req.queries),
        "results": list(results),
        **_key_meta(key_row),
    }
    return Response(
        content=json.dumps(payload, indent=2, ensure_ascii=False),
        media_type="application/json",
    )

# ── Pinger ─────────────────────────────────────────────────────────────────────
async def pinger():
    port = os.getenv("PORT", "7860")
    url = f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try:
                resp = await client.get(url)
                print(f"[Pinger] {resp.status_code} at {asyncio.get_event_loop().time():.0f}")
            except Exception as e:
                print(f"[Pinger] Error: {e}")

@fastapi_app.on_event("startup")
async def startup_event():
    _init_db()
    asyncio.create_task(pinger())

# ── Gradio UI ──────────────────────────────────────────────────────────────────
def format_result(row: dict) -> str:
    lines = []
    for field in SEARCH_FIELDS:
        val = row.get(field, "")
        if val:
            lines.append(f"**{field}:** {val}")
    cn = row.get("connected_numbers", [])
    if cn:
        nums = ", ".join(f"{c['field']}={c['value']}" for c in cn)
        lines.append(f"**connected:** {nums}")
    return "\n\n".join(lines)

def search_ui(query: str, limit: int) -> str:
    if not query or not query.strip():
        return "⚠️ Kuch toh search karo — phone, aadhar, ya name daalo."
    try:
        data = _unified_search(query.strip(), int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    count = data["count"]
    results = data["results"]
    searched = ", ".join(data.get("searched_fields", []))
    if not results:
        return f"🔍 **Query:** `{query}`\n**Searched:** {searched}\n\n❌ **No data found** for this number."
    header = f"🔍 **Query:** `{query}`  |  **Found:** {count} results  |  **Searched:** {searched}\n\n---\n\n"
    parts = [f"### Result {i}\n{format_result(row)}" for i, row in enumerate(results, 1)]
    return header + "\n\n---\n\n".join(parts)

def build_ui():
    with gr.Blocks(
        title="ICMR Search API",
        theme=gr.themes.Soft(),
        css=".main-title{text-align:center;margin-bottom:0}.subtitle{text-align:center;color:#666;margin-top:0}.footer{text-align:center;color:#888;margin-top:20px}"
    ) as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API", elem_classes="main-title")
        gr.Markdown("Search **2.5 billion records** — phone, Aadhaar, name, address & more", elem_classes="subtitle")
        with gr.Row():
            with gr.Column(scale=3):
                query_input = gr.Textbox(label="Search Query", placeholder="Phone number, Aadhaar, ya name daalo...", lines=1)
            with gr.Column(scale=1):
                limit_slider = gr.Slider(minimum=1, maximum=50, value=10, step=1, label="Max Results")
        search_btn = gr.Button("🔍 Search", variant="primary", size="lg")
        output = gr.Markdown(label="Results")
        search_btn.click(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        query_input.submit(fn=search_ui, inputs=[query_input, limit_slider], outputs=output)
        gr.Markdown("---")
        with gr.Accordion("📡 API Info", open=False):
            gr.Markdown("""
**Endpoints** (via FastAPI):
- `GET /search?q=<number>&api_key=<key>` — Phone/Aadhaar search
- `POST /search/parallel?api_key=<key>` — Batch search
- `GET /health` — Health check
- `GET /admin-panel` — Admin panel
- `GET /docs` — Swagger UI
            """)
        gr.Markdown(
            "---\n<div class='footer'>👨‍💻 **Developer:** " + DEVELOPER_CREDIT + "</div>",
            elem_classes="footer",
        )
    return demo

# ── Mount ──────────────────────────────────────────────────────────────────────
demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
