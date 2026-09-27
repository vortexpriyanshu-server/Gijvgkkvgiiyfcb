import asyncio
import csv
import hashlib
import hmac
import io
import json
import os
import secrets
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from html import escape
from typing import Any
from zoneinfo import ZoneInfo

import duckdb
import gradio as gr
import httpx
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from pydantic import BaseModel

# ── Config ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
HF_INDEX_BASE = os.environ.get(
    "ICMR_HF_INDEX_BASE",
    "https://huggingface.co/datasets/sckeptic/icrm-hitek-full-db-mixed/resolve/main",
).rstrip("/")
INDEX_SOURCE = os.environ.get("ICMR_INDEX_SOURCE", "remote").lower()
PARALLELISM = max(1, min(int(os.environ.get("ICMR_PARALLEL", "2")), 2))
THREADS_PER_CONN = max(1, min(int(os.environ.get("ICMR_THREADS_PER_CONN", "2")), 2))
DUPLICATE_CAP = 2
# Smart DB path: use /data (Render Disk / HF persistent storage) if it exists, else local
_default_db = "/data/keys.db" if os.path.isdir("/data") else os.path.join(BASE, "keys.db")
DB_PATH = os.environ.get("KEY_DB_PATH", _default_db)
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "PRIYANSHU295")
SESSION_SECRET = os.environ.get("ADMIN_SESSION_SECRET", "").strip() or secrets.token_urlsafe(32)
SESSION_TTL = int(os.environ.get("ADMIN_SESSION_TTL", "28800"))
IST = ZoneInfo("Asia/Kolkata")
API_DEVELOPER = "@VORTEX_PRIYANSHU"
COOKIE_NAME = "admin_session"
COOKIE_SECURE = os.environ.get("COOKIE_SECURE", "true").lower() == "true"

SEARCH_FIELDS = [
    "name", "fathersName", "phoneNumber", "aadharNumber", "otherNumber",
    "address", "district", "pincode", "state", "town", "source",
]
NUMBER_FIELDS = ["phoneNumber", "aadharNumber", "otherNumber"]
REMOTE_INDEXES = {
    "phone": [f"{HF_INDEX_BASE}/idx_phone.{i}.parquet" for i in range(7)],
    "aadhar": [f"{HF_INDEX_BASE}/idx_aadhar.{i}.parquet" for i in range(7)],
}

# ── Lightweight SQLite key store ────────────────────────────────────────────
_db_lock = threading.RLock()

def db():
    con = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=10000")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def init_db():
    with _db_lock, db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL UNIQUE,
            plan TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT,
            owner TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            daily_limit INTEGER,
            requests_today INTEGER NOT NULL DEFAULT 0,
            request_counter_date TEXT NOT NULL,
            total_usage INTEGER NOT NULL DEFAULT 0,
            last_used_at TEXT,
            revoked_at TEXT,
            note TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_keys_key ON api_keys(key);
        CREATE INDEX IF NOT EXISTS idx_keys_owner ON api_keys(owner);
        CREATE INDEX IF NOT EXISTS idx_keys_status ON api_keys(status);
        CREATE INDEX IF NOT EXISTS idx_keys_plan ON api_keys(plan);
        CREATE TABLE IF NOT EXISTS request_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_id INTEGER,
            endpoint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(key_id) REFERENCES api_keys(id)
        );
        CREATE INDEX IF NOT EXISTS idx_request_log_created ON request_log(created_at);
        CREATE INDEX IF NOT EXISTS idx_request_log_key ON request_log(key_id);
        """)

init_db()


def now_utc():
    return datetime.now(timezone.utc)


def now_ist():
    return now_utc().astimezone(IST)


def iso(dt):
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds") if dt else None


def display_time(value):
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).astimezone(IST).strftime("%d/%m/%Y %H:%M IST")
    except Exception:
        return str(value)


def day_key():
    return now_ist().date().isoformat()


def plan_expiry(plan):
    p = plan.lower()
    if p == "day": return now_utc() + timedelta(days=1)
    if p == "week": return now_utc() + timedelta(days=7)
    if p == "month": return now_utc() + timedelta(days=30)
    if p == "lifetime": return None
    raise ValueError("plan must be day, week, month, or lifetime")


def make_key():
    return secrets.token_urlsafe(24).replace("-", "").replace("_", "")[:32]


def make_owner(owner):
    # FIX: return owner as-is, no random suffix appended
    owner = (owner or "").strip()
    if not owner: raise ValueError("owner is required")
    return owner


def row_dict(row):
    d = dict(row)
    d["created_at"] = display_time(d.get("created_at"))
    d["expires_at"] = display_time(d.get("expires_at")) if d.get("expires_at") else "LIFETIME"
    d["last_used_at"] = display_time(d.get("last_used_at"))
    d["revoked_at"] = display_time(d.get("revoked_at"))
    return d


def create_key(plan, owner, daily_limit, note):
    if plan not in {"day", "week", "month", "lifetime"}: raise ValueError("invalid plan")
    if daily_limit != "unlimited":
        daily_limit = int(daily_limit)
        if daily_limit < 1 or daily_limit > 10_000_000: raise ValueError("daily limit must be 1..10000000 or unlimited")
    key = make_key()
    expires = plan_expiry(plan)
    owner_full = make_owner(owner)
    created = iso(now_utc())
    with _db_lock, db() as con:
        con.execute("INSERT INTO api_keys(key,plan,created_at,expires_at,owner,status,daily_limit,request_counter_date,note) VALUES(?,?,?,?,?,?,?,?,?)",
                    (key, plan, created, iso(expires), owner_full, "active", None if daily_limit == "unlimited" else daily_limit, day_key(), (note or "").strip()[:500]))
        row = con.execute("SELECT * FROM api_keys WHERE key=?", (key,)).fetchone()
    return row_dict(row)


def auth_key_from_request(request):
    return (request.headers.get("X-API-Key") or request.query_params.get("api_key") or "").strip()


def api_error(status, message, extra=None):
    payload = {"error": message}
    if extra: payload.update(extra)
    payload["api_developer"] = API_DEVELOPER
    return JSONResponse(payload, status_code=status)


def validate_and_count_api_key(raw_key, endpoint):
    if not raw_key: return None, api_error(401, "API key is required")
    today = day_key()
    with _db_lock, db() as con:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute("SELECT * FROM api_keys WHERE key=?", (raw_key,)).fetchone()
        if not row:
            con.execute("ROLLBACK")
            return None, api_error(401, "Invalid API key")
        if row["status"] == "revoked":
            con.execute("ROLLBACK")
            return None, api_error(401, "API key has been revoked")
        if row["expires_at"] and datetime.fromisoformat(row["expires_at"]) <= now_utc():
            con.execute("UPDATE api_keys SET status='expired' WHERE id=?", (row["id"],))
            con.execute("COMMIT")
            return None, api_error(401, "API key expired", {
                "api_key_valid_till": display_time(row["expires_at"]), "api_key_plan": row["plan"]})
        count = row["requests_today"] if row["request_counter_date"] == today else 0
        limit = row["daily_limit"]
        if limit is not None and count >= limit:
            con.execute("ROLLBACK")
            return None, api_error(429, "Daily request limit exceeded", {
                "requests_today": count, "daily_limit": limit,
                "api_key_plan": row["plan"], "api_key_valid_till": display_time(row["expires_at"])})
        new_count = count + 1
        now = iso(now_utc())
        con.execute("UPDATE api_keys SET requests_today=?, request_counter_date=?, total_usage=total_usage+1, last_used_at=? WHERE id=?",
                    (new_count, today, now, row["id"]))
        con.execute("INSERT INTO request_log(key_id,endpoint,created_at) VALUES(?,?,?)", (row["id"], endpoint, now))
        con.execute("COMMIT")
        row = con.execute("SELECT * FROM api_keys WHERE id=?", (row["id"],)).fetchone()
    return row, None


def add_api_fields(payload, key_row):
    payload["api_key_owner"] = key_row["owner"]
    payload["api_key_valid_till"] = display_time(key_row["expires_at"]) if key_row["expires_at"] else "LIFETIME"
    payload["api_key_plan"] = key_row["plan"]
    payload["api_developer"] = API_DEVELOPER
    return payload

# ── DuckDB Connection Pool ──────────────────────────────────────────────────
_conns = []
_conns_lock = threading.Lock()
_thread_local = threading.local()
pool = ThreadPoolExecutor(max_workers=PARALLELISM, thread_name_prefix="duck")

def _idx_ready(kind): return kind in REMOTE_INDEXES

def _new_conn():
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

def _thread_id():
    tid = getattr(_thread_local, "id", None)
    if tid is None:
        with _conns_lock:
            tid = len(_conns); _thread_local.id = tid
    return tid

def _get_conn():
    ident = _thread_id()
    with _conns_lock:
        while len(_conns) <= ident: _conns.append(_new_conn())
    return _conns[ident]

# ── Search Logic ────────────────────────────────────────────────────────────
def _person_key(row):
    ph = (row.get("phoneNumber") or "").strip(); ad = (row.get("aadharNumber") or "").strip()
    return (ph, ad) if (ph or ad) else ((row.get("name") or "").strip(), (row.get("fathersName") or "").strip())

def _connected_numbers(row):
    connected, seen = [], set()
    for field in NUMBER_FIELDS:
        raw = row.get(field)
        if raw is None: continue
        value = str(raw).strip()
        if value and value not in seen:
            seen.add(value); connected.append({"field": field, "value": value})
    return connected

def _cap_duplicates(rows):
    seen, out = {}, []
    for r in rows:
        k = _person_key(r); n = seen.get(k, 0)
        if n < DUPLICATE_CAP:
            seen[k] = n + 1; record = dict(r); record["connected_numbers"] = _connected_numbers(record); out.append(record)
    return out

def _run_field_search(field, value, mode, limit):
    if field not in SEARCH_FIELDS: raise ValueError(f"Unknown field: {field}")
    v = value.replace("'", "''")
    if mode == "exact":
        if field == "phoneNumber" and _idx_ready("phone"): view = "people_phone"
        elif field == "aadharNumber" and _idx_ready("aadhar"): view = "people_aadhar"
        else: return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        sql = f"SELECT * FROM {view} WHERE {field} = '{v}' LIMIT {limit * DUPLICATE_CAP + 20}"
    elif mode == "contains":
        if field == "name": return {"field": field, "value": value, "mode": mode, "count": 0, "results": []}
        v2 = v.replace("%", r"\%").replace("_", r"\_")
        sql = f"SELECT * FROM people_phone WHERE {field} ILIKE '%{v2}%' ESCAPE '\\' LIMIT {limit * DUPLICATE_CAP + 20}"
    else: raise ValueError(f"Unknown mode: {mode}")
    con = _get_conn(); rows = con.execute(sql).fetchall(); cols = [d[0] for d in con.description]
    results = _cap_duplicates([dict(zip(cols, r)) for r in rows])[:limit]
    return {"field": field, "value": value, "mode": mode, "count": len(results), "results": results}

def _unified_search(q, limit=10):
    q = q.strip(); is_num = q.isdigit() and len(q) >= 8
    if not is_num: return {"query": q, "searched_fields": [], "count": 0, "results": []}
    all_rows, searched = [], []
    if _idx_ready("phone"):
        r = _run_field_search("phoneNumber", q, "exact", limit); all_rows.extend(r["results"]); searched.append("phoneNumber")
    if not all_rows and _idx_ready("aadhar"):
        r = _run_field_search("aadharNumber", q, "exact", limit); all_rows.extend(r["results"]); searched.append("aadharNumber")
    all_rows = _cap_duplicates(all_rows)[:limit]
    return {"query": q, "searched_fields": searched, "count": len(all_rows), "results": all_rows}

# ── FastAPI ─────────────────────────────────────────────────────────────────
fastapi_app = FastAPI(title="ICMR + HITEK Search API")

@fastapi_app.exception_handler(HTTPException)
async def json_http_exception_handler(request: Request, exc: HTTPException):
    if request.url.path.startswith("/search") or request.url.path.startswith("/admin-panel/api"):
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse({"error": detail, "api_developer": API_DEVELOPER}, status_code=exc.status_code, headers=exc.headers)
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)

@fastapi_app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/search") or request.url.path.startswith("/admin-panel/api"):
        return JSONResponse({"error": "Validation error", "api_developer": API_DEVELOPER}, status_code=422)
    return JSONResponse({"detail": "Validation error"}, status_code=422)

class BatchRequest(BaseModel):
    queries: list[dict[str, Any]]
    limit: int = 10

@fastapi_app.get("/")
def root():
    return {"app":"ICMR + HITEK Search API","records":2_504_793_870,"indexes":{"phone":_idx_ready("phone"),"aadhar":_idx_ready("aadhar")},"index_source":INDEX_SOURCE,"columns":SEARCH_FIELDS,"docs":"/docs","developer":API_DEVELOPER}

@fastapi_app.get("/health")
def health():
    return {"status":"ok","raw_database_required":False,"indexes":{"phone":_idx_ready("phone"),"aadhar":_idx_ready("aadhar")},"index_source":INDEX_SOURCE,"api_developer":API_DEVELOPER}

@fastapi_app.get("/search")
async def search(request: Request, q: str|None=Query(None), mobile: str|None=Query(None), field: str|None=Query(None), mode: str=Query("exact"), limit: int=Query(10,ge=1,le=1000), pretty: bool=Query(True)):
    key_row, err = validate_and_count_api_key(auth_key_from_request(request), "/search")
    if err: return err
    q_val=(q or mobile or "").strip()
    if not q_val: return api_error(422,"Provide q or mobile")
    loop=asyncio.get_running_loop()
    try:
        data=await loop.run_in_executor(pool,_run_field_search,field,q_val,mode,limit) if field else await loop.run_in_executor(pool,_unified_search,q_val,limit)
    except Exception as exc:
        return api_error(400,str(exc))
    result=add_api_fields({"success":bool(data["count"]),**data,"number":q_val,"total":data["count"]},key_row)
    return Response(content=json.dumps(result,indent=2 if pretty else None,ensure_ascii=False),media_type="application/json")

@fastapi_app.post("/search/parallel")
async def search_parallel(request: Request, req: BatchRequest):
    key_row, err=validate_and_count_api_key(auth_key_from_request(request),"/search/parallel")
    if err: return err
    if not req.queries: return api_error(400,"queries must not be empty")
    if len(req.queries)>50: return api_error(400,"max 50 queries per batch")
    loop=asyncio.get_running_loop()
    try:
        tasks=[loop.run_in_executor(pool,_run_field_search,item.get("field","phoneNumber"),item.get("value",""),item.get("mode","exact"),min(int(item.get("limit",req.limit)),1000)) for item in req.queries]
        results=await asyncio.gather(*tasks)
    except Exception as exc:
        return api_error(400,str(exc))
    payload=add_api_fields({"searches":len(req.queries),"results":list(results)},key_row)
    return Response(content=json.dumps(payload,indent=2,ensure_ascii=False),media_type="application/json")

# ── Admin session ───────────────────────────────────────────────────────────
def session_token():
    exp=int(now_utc().timestamp())+SESSION_TTL
    body=f"{exp}:{secrets.token_urlsafe(12)}"
    sig=hmac.new(SESSION_SECRET.encode(),body.encode(),hashlib.sha256).hexdigest()
    return body+"."+sig

def valid_session(request):
    token=request.cookies.get(COOKIE_NAME,"")
    try:
        body,sig=token.rsplit(".",1); exp=int(body.split(":",1)[0])
        return exp>int(now_utc().timestamp()) and hmac.compare_digest(sig,hmac.new(SESSION_SECRET.encode(),body.encode(),hashlib.sha256).hexdigest())
    except Exception: return False

def admin_guard(request):
    if not valid_session(request): raise HTTPException(401,"Admin authentication required")

def admin_page(title, body, script=""):
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{{escape(title)}}</title>
<style>
:root{{--bg:#05070d;--panel:rgba(13,17,28,.82);--line:rgba(255,255,255,.10);--line2:rgba(255,255,255,.16);--text:#f4f7fb;--muted:#8d97aa;--accent:#8b5cf6;--accent2:#06b6d4;--green:#34d399;--red:#fb7185;--shadow:0 18px 60px rgba(0,0,0,.35)}}
*{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth;scrollbar-width:thin;scrollbar-color:#6d4bd8 #070a12}}
html::-webkit-scrollbar{{width:8px;height:8px}}html::-webkit-scrollbar-track{{background:#070a12}}html::-webkit-scrollbar-thumb{{background:linear-gradient(#8b5cf6,#06b6d4);border-radius:999px}}
body{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#05070d;color:var(--text);font-size:14px;line-height:1.55;min-height:100vh;background-image:radial-gradient(circle at 15% 0%,rgba(139,92,246,.18),transparent 32%),radial-gradient(circle at 85% 10%,rgba(6,182,212,.13),transparent 30%),linear-gradient(180deg,#060811 0%,#05070d 60%,#070914 100%)}}
body:before{{content:"";position:fixed;inset:0;pointer-events:none;opacity:.28;background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);background-size:44px 44px;mask-image:linear-gradient(to bottom,black,transparent 75%)}}
button,input,select,textarea{{font:inherit}}a{{color:#b9a7ff}}
.az-header{{background:rgba(5,7,13,.78);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border-bottom:1px solid var(--line);padding:13px 24px;display:flex;align-items:center;gap:14px;position:sticky;top:0;z-index:100;box-shadow:0 8px 30px rgba(0,0,0,.22)}}
.az-logo{{color:white;font-size:21px;font-weight:900;letter-spacing:-.8px;text-decoration:none;line-height:1.1}}.az-logo span:first-child{{background:linear-gradient(90deg,#fff,#a78bfa,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}}.az-logo-sub{{display:block;color:#78839a;font-size:9px;font-weight:600;letter-spacing:1.5px;margin-top:3px;text-transform:uppercase}}
.az-header-right{{margin-left:auto;display:flex;align-items:center;gap:12px}}.az-header-right span{{color:#9aa4b8;font-size:12px}}
.az-signout{{background:rgba(255,255,255,.04);border:1px solid var(--line2);color:#e8ecf5;padding:7px 13px;border-radius:10px;cursor:pointer;font-size:12px;transition:.2s}}.az-signout:hover{{border-color:rgba(139,92,246,.7);background:rgba(139,92,246,.12);transform:translateY(-1px)}}
.az-subnav{{background:rgba(9,12,20,.72);backdrop-filter:blur(14px);border-bottom:1px solid var(--line);padding:0 20px;display:flex;gap:4px;overflow-x:auto;scrollbar-width:none}}.az-subnav::-webkit-scrollbar{{display:none}}
.az-subnav a,.az-subnav button{{color:#aeb7c9;background:none;border:1px solid transparent;padding:11px 14px;text-decoration:none;cursor:pointer;font-size:12px;white-space:nowrap;transition:.2s;display:inline-block}}.az-subnav a:hover,.az-subnav .nav-btn:hover{{color:#fff;background:rgba(139,92,246,.10);border-color:rgba(139,92,246,.25);border-radius:9px}}.az-subnav .nav-logout{{color:#fda4af;margin:6px 0 6px auto;border:1px solid rgba(251,113,133,.35);border-radius:9px;padding:6px 12px;background:rgba(251,113,133,.06)}}.az-subnav .nav-logout:hover{{background:rgba(251,113,133,.14);color:#fff}}
.wrap{{max-width:1240px;margin:0 auto;padding:24px 20px}}section{{margin-bottom:12px}}.page-title{{font-size:27px;font-weight:750;letter-spacing:-.8px;margin-bottom:3px;color:#fff;padding:7px 0 2px}}.page-sub{{color:var(--muted);font-size:12px;margin-bottom:18px}}
.card{{background:linear-gradient(145deg,rgba(18,23,36,.90),rgba(9,12,21,.84));border:1px solid var(--line);border-radius:18px;padding:20px;margin-bottom:16px;box-shadow:var(--shadow);position:relative;overflow:hidden}}.card:before{{content:"";position:absolute;left:0;right:0;top:0;height:1px;background:linear-gradient(90deg,transparent,rgba(139,92,246,.6),rgba(6,182,212,.35),transparent)}}.card h2{{font-size:15px;font-weight:700;border-bottom:1px solid var(--line);padding-bottom:12px;margin-bottom:15px;color:#f5f7fb}}
.grid{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:12px;margin-bottom:16px}}.stat{{background:rgba(255,255,255,.035);border:1px solid var(--line);border-radius:15px;padding:15px 16px;transition:.2s}}.stat:hover{{border-color:rgba(139,92,246,.42);transform:translateY(-2px);background:rgba(139,92,246,.055)}}.stat small{{display:block;color:#7f899d;font-size:10px;margin-bottom:7px;text-transform:uppercase;letter-spacing:1px}}.stat strong{{display:block;font-size:25px;font-weight:750;color:#fff}}
label{{display:block;font-size:11px;font-weight:650;margin:12px 0 5px;color:#aeb7c9;text-transform:uppercase;letter-spacing:.5px}}input,select,textarea{{width:100%;padding:10px 12px;border:1px solid var(--line2);border-radius:11px;background:rgba(0,0,0,.22);font-size:13px;color:#eef2f8;outline:none;transition:.2s}}input::placeholder,textarea::placeholder{{color:#5f697b}}input:focus,select:focus,textarea:focus{{outline:none;border-color:rgba(139,92,246,.8);box-shadow:0 0 0 3px rgba(139,92,246,.13),0 0 22px rgba(139,92,246,.08)}}select option{{background:#101522;color:#fff}}
.btn{{display:inline-block;padding:9px 14px;border-radius:11px;cursor:pointer;font-size:12px;border:1px solid var(--line2);text-decoration:none;transition:.2s;line-height:1.4;vertical-align:middle}}.btn-p{{background:linear-gradient(135deg,#7c3aed,#06b6d4);border-color:rgba(167,139,250,.55);color:white;box-shadow:0 8px 24px rgba(124,58,237,.22)}}.btn-p:hover{{filter:brightness(1.12);transform:translateY(-1px);box-shadow:0 12px 30px rgba(124,58,237,.30)}}.btn-p:active{{transform:translateY(0)}}.btn-s{{background:rgba(255,255,255,.045);border-color:var(--line2);color:#dce2ee}}.btn-s:hover{{background:rgba(255,255,255,.09);border-color:rgba(255,255,255,.25)}}.btn-r{{background:rgba(251,113,133,.06);border-color:rgba(251,113,133,.35);color:#fda4af;font-size:12px;padding:5px 9px}}.btn-r:hover{{background:rgba(251,113,133,.14)}}.btn-del{{background:rgba(239,68,68,.16);border-color:rgba(248,113,113,.4);color:#fecaca;font-size:12px;padding:5px 9px}}.btn-del:hover{{background:rgba(239,68,68,.28)}}.btn-sm{{padding:5px 10px;font-size:11px}}
.row{{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end}}.row>*{{flex:1;min-width:140px}}table{{width:100%;border-collapse:collapse;min-width:980px;font-size:12px}}thead{{background:rgba(255,255,255,.035)}}th{{padding:11px 12px;text-align:left;font-weight:650;border-bottom:1px solid var(--line2);color:#aeb7c9;white-space:nowrap}}td{{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:middle;color:#cfd6e3}}tr:hover td{{background:rgba(139,92,246,.045)}}.scroll{{overflow-x:auto;scrollbar-width:thin;scrollbar-color:#6d4bd8 transparent}}.scroll::-webkit-scrollbar{{height:7px}}.scroll::-webkit-scrollbar-thumb{{background:#6d4bd8;border-radius:999px}}
.badge{{display:inline-block;padding:3px 8px;border-radius:999px;font-size:9px;font-weight:750;text-transform:uppercase;letter-spacing:.5px}}.active{{background:rgba(52,211,153,.10);color:#6ee7b7;border:1px solid rgba(52,211,153,.28)}}.revoked{{background:rgba(251,113,133,.10);color:#fda4af;border:1px solid rgba(251,113,133,.28)}}.expired{{background:rgba(148,163,184,.09);color:#aab3c2;border:1px solid rgba(148,163,184,.2)}}
pre{{background:rgba(0,0,0,.24);border:1px solid var(--line);padding:12px;border-radius:11px;font-size:12px;word-break:break-all;white-space:pre-wrap;margin:8px 0;color:#c4b5fd}}code{{font-family:"SFMono-Regular",Consolas,"Liberation Mono",monospace;font-size:11px;color:#c4b5fd}}.key-box{{background:linear-gradient(135deg,rgba(52,211,153,.08),rgba(6,182,212,.05));border:1px solid rgba(52,211,153,.25);border-radius:14px;padding:16px;margin-top:14px}}.key-box-title{{color:#6ee7b7;font-weight:700;font-size:13px;margin-bottom:8px}}.small{{font-size:11px;color:var(--muted)}}.msg-err{{background:rgba(251,113,133,.09);border:1px solid rgba(251,113,133,.3);color:#fda4af;padding:10px 14px;border-radius:10px;margin-top:12px;font-size:12px;display:none}}
.chart-wrap{{display:flex;align-items:flex-end;gap:8px;height:130px;padding-top:10px}}.chart-wrap>div>div:first-child{{background:linear-gradient(180deg,#8b5cf6,#06b6d4)!important;box-shadow:0 0 14px rgba(139,92,246,.2)}}.login-wrap{{max-width:390px;margin:90px auto;padding:0 14px}}.login-logo{{font-size:28px;font-weight:900;text-align:center;margin-bottom:18px;letter-spacing:-1px;color:#fff}}.login-logo span{{background:linear-gradient(90deg,#fff,#a78bfa,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}}.login-card{{background:linear-gradient(145deg,rgba(18,23,36,.94),rgba(8,11,19,.94));border:1px solid var(--line2);border-radius:20px;padding:24px;box-shadow:0 24px 80px rgba(0,0,0,.45)}}.login-card h2{{font-size:17px;font-weight:650;margin-bottom:15px;padding-bottom:13px;border-bottom:1px solid var(--line);color:#f5f7fb}}.login-terms{{text-align:center;font-size:10px;color:#687286;margin-top:20px;padding-top:14px;border-top:1px solid var(--line)}}.login-terms b{{cursor:pointer;color:#9f8ff4}}.az-footer{{text-align:center;font-size:10px;color:#596276;padding:22px 20px;border-top:1px solid var(--line);margin-top:10px}}.settings-table td{{padding:8px 20px 8px 0;border:none}}.settings-table td:first-child{{color:#7f899d;font-weight:650;white-space:nowrap}}
@media(max-width:1050px){{.grid{{grid-template-columns:repeat(3,1fr)}}}}@media(max-width:700px){{.wrap{{padding:14px 10px}}.row>*{{min-width:100%}}.az-header{{padding:12px 14px}}.az-subnav{{padding:0 8px}}.az-subnav a{{font-size:11px;padding:9px}}.grid{{grid-template-columns:repeat(2,1fr)}}.card{{padding:15px;border-radius:15px}}.page-title{{font-size:23px}}}}
</style></head><body>{body}
<div class="az-footer">© 2024 VORTEX · Developer: @VORTEX_PRIYANSHU · API Key Management Panel</div>
<script>{script}</script></body></html>''')

LOGIN_HTML='''<div class="az-header" style="position:relative"><div class="az-logo"><span>ᴩʀɪʏᴀɴꜱʜᴜ</span><span class="az-logo-sub">VORTEX API · Admin Access</span></div></div>
<div class="login-wrap">
<div class="login-logo"><span>ᴩʀɪʏᴀɴꜱʜᴜ</span></div>
<div class="login-card">
<h2>Sign-In to Admin Panel</h2>
<label for="pwd">Password</label>
<input id="pwd" type="password" autocomplete="current-password" placeholder="Enter admin password">
<div id="errmsg" class="msg-err"></div>
<button class="btn btn-p" style="width:100%;margin-top:14px;padding:9px;text-align:center" onclick="doLogin()">Sign in</button>
<div class="login-terms"><b>Conditions of Use</b> &nbsp;·&nbsp; <b>Privacy Notice</b> &nbsp;·&nbsp; <b>Help</b><br>© 1996–2024, VORTEX, Inc.</div>
</div>
</div>'''

LOGIN_JS='''function doLogin(){
  var p=document.getElementById("pwd"),m=document.getElementById("errmsg");
  m.style.display="none";
  fetch("/admin-panel/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password:p.value})})
    .then(function(r){if(r.ok){location.href="/admin-panel"}else{r.json().then(function(x){m.textContent=x.detail||"Incorrect password. Try again.";m.style.display="block"})}})
    .catch(function(){m.textContent="Network error. Please try again.";m.style.display="block"});
}
document.addEventListener("keydown",function(e){if(e.key==="Enter")doLogin()});
'''

@fastapi_app.get("/admin-panel", response_class=HTMLResponse)
def admin_panel(request: Request):
    if not valid_session(request): return admin_page("API Key Admin", LOGIN_HTML, LOGIN_JS)
    body='''<div class="az-header">
  <div class="az-logo"><span>ᴩʀɪʏᴀɴꜱʜᴜ</span><span class="az-logo-sub">VORTEX · API Key Management</span></div>
  <div class="az-header-right">
    <span>Hello, <b style="color:var(--az-orangel)">Admin</b></span>
    <button class="az-signout" onclick="logout()">Sign Out</button>
  </div>
</div>
<div class="az-subnav">
  <a href="#dashboard">Dashboard</a>
  <a href="#generate">Generate Key</a>
  <a href="#keys">Manage Keys</a>
  <a href="#csv">Import / Export</a>
  <a href="#settings">Settings</a>
  <button class="nav-logout" onclick="logout()">Sign Out</button>
</div>
<div class="wrap"><div id="app"></div></div>'''
    return admin_page("API Key Admin", body, ADMIN_JS)

@fastapi_app.post("/admin-panel/login")
async def admin_login(request: Request):
    try: data=await request.json()
    except Exception: data={}
    supplied=str(data.get("password", ""))
    if not hmac.compare_digest(supplied, ADMIN_PASSWORD): raise HTTPException(401,"Incorrect password")
    r=JSONResponse({"success":True,"api_developer":API_DEVELOPER}); r.set_cookie(COOKIE_NAME,session_token(),httponly=True,secure=COOKIE_SECURE,samesite="lax",max_age=SESSION_TTL,path="/"); return r

@fastapi_app.post("/admin-panel/logout")
def admin_logout(request: Request):
    admin_guard(request); r=JSONResponse({"success":True,"api_developer":API_DEVELOPER}); r.delete_cookie(COOKIE_NAME,path="/"); return r

@fastapi_app.get("/admin-panel/api/stats")
def admin_stats(request: Request):
    admin_guard(request)
    with db() as con:
        total=con.execute("SELECT COUNT(*) FROM api_keys").fetchone()[0]
        active=con.execute("SELECT COUNT(*) FROM api_keys WHERE status='active' AND (expires_at IS NULL OR expires_at>?)",(iso(now_utc()),)).fetchone()[0]
        expired=con.execute("SELECT COUNT(*) FROM api_keys WHERE status='expired' OR (status='active' AND expires_at IS NOT NULL AND expires_at<=?)",(iso(now_utc()),)).fetchone()[0]
        revoked=con.execute("SELECT COUNT(*) FROM api_keys WHERE status='revoked'").fetchone()[0]
        today=day_key(); requests_today=con.execute("SELECT COUNT(*) FROM request_log WHERE created_at>=?",(iso(now_ist().replace(hour=0,minute=0,second=0,microsecond=0).astimezone(timezone.utc)),)).fetchone()[0]
        total_req=con.execute("SELECT COALESCE(SUM(total_usage),0) FROM api_keys").fetchone()[0]
        most=con.execute("SELECT owner,plan,key,total_usage FROM api_keys ORDER BY total_usage DESC LIMIT 1").fetchone()
        last=con.execute("SELECT r.created_at,k.owner,k.key,r.endpoint FROM request_log r LEFT JOIN api_keys k ON k.id=r.key_id ORDER BY r.id DESC LIMIT 1").fetchone()
        chart=[]
        for i in range(6,-1,-1):
            d=now_ist().date()-timedelta(days=i); start=datetime.combine(d,datetime.min.time(),IST).astimezone(timezone.utc).isoformat(timespec="seconds"); end=datetime.combine(d+timedelta(days=1),datetime.min.time(),IST).astimezone(timezone.utc).isoformat(timespec="seconds")
            chart.append({"day":d.strftime("%d/%m"),"count":con.execute("SELECT COUNT(*) FROM request_log WHERE created_at>=? AND created_at<?",(start,end)).fetchone()[0]})
    return {"total_keys":total,"active_keys":active,"expired_keys":expired,"revoked_keys":revoked,"requests_today":requests_today,"total_requests":total_req,"most_used":dict(most) if most else None,"last_request":dict(last) if last else None,"chart":chart,"api_developer":API_DEVELOPER}

@fastapi_app.post("/admin-panel/api/keys")
async def admin_create_key(request: Request):
    admin_guard(request); data=await request.json()
    try:
        result=create_key(str(data.get("plan","day")),str(data.get("owner","")),data.get("daily_limit","unlimited"),str(data.get("note", "")))
        result["api_developer"]=API_DEVELOPER
        return result
    except Exception as exc: raise HTTPException(400,str(exc))

@fastapi_app.get("/admin-panel/api/keys")
def admin_list_keys(request: Request, q: str="", status: str="all", plan: str="all"):
    admin_guard(request); clauses=[]; args=[]
    if q: clauses.append("(owner LIKE ? OR key LIKE ? OR note LIKE ?)"); args += [f"%{q}%"]*3
    if status in {"active","expired","revoked"}: clauses.append("status=?"); args.append(status)
    if plan in {"day","week","month","lifetime"}: clauses.append("plan=?"); args.append(plan)
    sql="SELECT * FROM api_keys"+(" WHERE "+" AND ".join(clauses) if clauses else "")+" ORDER BY id DESC LIMIT 500"
    with db() as con: rows=con.execute(sql,args).fetchall()
    out=[]
    now=now_utc()
    for r in rows:
        d=dict(r)
        if d["status"]=="active" and d["expires_at"] and datetime.fromisoformat(d["expires_at"])<=now: d["status"]="expired"
        out.append(row_dict(d))
    return {"keys": out, "api_developer": API_DEVELOPER}

@fastapi_app.post("/admin-panel/api/keys/{key}/revoke")
def admin_revoke(request: Request,key: str):
    admin_guard(request)
    with _db_lock, db() as con:
        cur=con.execute("UPDATE api_keys SET status='revoked',revoked_at=? WHERE key=? AND status!='revoked'",(iso(now_utc()),key))
    if cur.rowcount==0: raise HTTPException(404,"Key not found or already revoked")
    return {"success":True,"api_developer":API_DEVELOPER}

@fastapi_app.delete("/admin-panel/api/keys/{key}")
def admin_delete_key(request: Request, key: str):
    admin_guard(request)
    with _db_lock, db() as con:
        cur=con.execute("DELETE FROM api_keys WHERE key=? AND status='revoked'",(key,))
    if cur.rowcount==0: raise HTTPException(404,"Key not found or not in revoked status — revoke it first")
    return {"success":True,"api_developer":API_DEVELOPER}

@fastapi_app.get("/admin-panel/api/export.csv")
def admin_export(request: Request):
    admin_guard(request)
    cols=["key","owner","plan","status","daily_limit","requests_today","request_counter_date","total_usage","created_at","expires_at","last_used_at","revoked_at","note"]
    with db() as con: rows=con.execute("SELECT "+",".join(cols)+" FROM api_keys ORDER BY id").fetchall()
    out=io.StringIO(); w=csv.writer(out); w.writerow(cols); w.writerows([tuple(r[c] for c in cols) for r in rows]); out.seek(0)
    return StreamingResponse(iter([out.getvalue().encode()]),media_type="text/csv",headers={"Content-Disposition":"attachment; filename=api_keys.csv"})

@fastapi_app.post("/admin-panel/api/import")
async def admin_import(request: Request, file: UploadFile=File(...), mode: str="merge"):
    admin_guard(request)
    if not (file.filename or "").lower().endswith(".csv"): raise HTTPException(400,"CSV file required")
    raw=await file.read()
    if len(raw)>5*1024*1024: raise HTTPException(413,"CSV too large (5 MB max)")
    try: rows=list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    except Exception as exc: raise HTTPException(400,f"Invalid CSV: {exc}")
    required={"key","owner","plan","status","daily_limit","requests_today","request_counter_date","total_usage","created_at","expires_at","last_used_at","revoked_at","note"}
    if not rows or not required.issubset(rows[0].keys()): raise HTTPException(400,"Missing required CSV columns")
    valid=[]
    for r in rows:
        if len(r.get("key", ""))!=32 or r.get("plan") not in {"day","week","month","lifetime"} or r.get("status") not in {"active","expired","revoked"}: continue
        try: dl=None if str(r.get("daily_limit","")).lower() in {"","none","unlimited"} else int(r["daily_limit"]); rt=int(r["requests_today"]); tu=int(r["total_usage"])
        except ValueError: continue
        valid.append((r["key"],r["plan"],r["created_at"],r["expires_at"] or None,r["owner"],r["status"],dl,rt,r["request_counter_date"],tu,r["last_used_at"] or None,r["revoked_at"] or None,r.get("note","")))
    with _db_lock, db() as con:
        if mode=="replace": con.execute("DELETE FROM api_keys")
        added=0; skipped=0
        for vals in valid:
            try: con.execute("INSERT INTO api_keys(key,plan,created_at,expires_at,owner,status,daily_limit,requests_today,request_counter_date,total_usage,last_used_at,revoked_at,note) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",vals); added+=1
            except sqlite3.IntegrityError: skipped+=1
    return {"success":True,"added":added,"skipped":skipped,"invalid":len(rows)-len(valid),"mode":mode,"api_developer":API_DEVELOPER}

ADMIN_JS=r'''
const esc=s=>String(s??"").replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function api(u,o){var r=await fetch(u,o);if(r.status===401){location.href='/admin-panel';throw 0}return r}

async function load(){
  var s=await(await api('/admin-panel/api/stats')).json();
  var kx=await(await api('/admin-panel/api/keys')).json();
  var k=kx.keys||[];

  // Chart bars
  var mx=Math.max(1,...s.chart.map(function(x){return x.count}));
  var chart=s.chart.map(function(x){
    var h=Math.max(4,Math.round(x.count/mx*100));
    return '<div style="flex:1;text-align:center;min-width:28px">'
      +'<div title="'+x.count+' requests" style="height:'+h+'px;background:#FF9900;border-radius:2px 2px 0 0;transition:.3s"></div>'
      +'<div style="font-size:11px;color:#565959;margin-top:4px">'+x.day+'</div>'
      +'<div style="font-size:10px;color:#999">'+x.count+'</div>'
      +'</div>';
  }).join('');

  // Stats
  var SL=['Total Keys','Active','Expired','Revoked','Req Today','Total Req'];
  var SV=[s.total_keys,s.active_keys,s.expired_keys,s.revoked_keys,s.requests_today,s.total_requests];
  var SC=['#0F1111','#007600','#565959','#B12704','#0066C0','#0066C0'];
  var statsHtml=SL.map(function(l,i){
    return '<div class="stat"><small>'+l+'</small><strong style="color:'+SC[i]+'">'+SV[i]+'</strong></div>';
  }).join('');

  // Table rows
  var rows=k.map(function(x){
    var actions='<button class="btn btn-s btn-sm" onclick="copyKey(\''+esc(x.key)+'\')">Copy</button> ';
    if(x.status==='active') actions+='<button class="btn btn-r" onclick="revokeKey(\''+esc(x.key)+'\')">Revoke</button>';
    if(x.status==='revoked') actions+='<button class="btn btn-del" onclick="deleteKey(\''+esc(x.key)+'\')">&#128465; Delete</button>';
    return '<tr>'
      +'<td><code style="font-size:11px;word-break:break-all">'+esc(x.key)+'</code></td>'
      +'<td><b>'+esc(x.owner)+'</b></td>'
      +'<td>'+esc(x.plan)+'</td>'
      +'<td><span class="badge '+x.status+'">'+esc(x.status)+'</span></td>'
      +'<td style="text-align:right">'+(x.daily_limit!=null?x.daily_limit:'&#8734;')+'</td>'
      +'<td style="text-align:right">'+x.requests_today+'/'+(x.daily_limit!=null?x.daily_limit:'&#8734;')+'</td>'
      +'<td style="text-align:right">'+x.total_usage+'</td>'
      +'<td>'+esc(x.expires_at)+'</td>'
      +'<td>'+esc(x.last_used_at)+'</td>'
      +'<td style="white-space:nowrap">'+actions+'</td>'
      +'</tr>';
  }).join('');

  document.getElementById('app').innerHTML=
  '<section id="dashboard">'
  +'<div class="page-title">Seller Central &mdash; Dashboard</div>'
  +'<div class="page-sub">Overview of API keys and usage statistics &nbsp;·&nbsp; Developer: <b>@VORTEX_PRIYANSHU</b></div>'
  +'<div class="grid">'+statsHtml+'</div>'
  +'<div class="card"><h2>Requests &mdash; Last 7 Days</h2><div class="chart-wrap">'+chart+'</div></div>'
  +(s.most_used?'<div class="card"><h2>Most Used Key</h2><p><b>'+esc(s.most_used.owner)+'</b> &nbsp;&middot;&nbsp; '+esc(s.most_used.plan)+' plan &nbsp;&middot;&nbsp; '+s.most_used.total_usage+' total requests</p></div>':'')
  +'</section>'

  +'<section id="generate" class="card">'
  +'<h2>Generate New API Key</h2>'
  +'<div class="row">'
  +'<div><label>Plan</label><select id="plan"><option value="day">1 Day</option><option value="week">7 Days</option><option value="month">30 Days</option><option value="lifetime">Lifetime</option></select></div>'
  +'<div><label>Owner</label><input id="owner" placeholder="@username or full name"></div>'
  +'<div><label>Daily Request Limit</label><input id="limit" value="100" placeholder="100 or unlimited"></div>'
  +'</div>'
  +'<label>Note <span style="color:#999;font-weight:400">(optional)</span></label>'
  +'<input id="note" placeholder="Customer name, purpose, testing…">'
  +'<div style="margin-top:14px"><button class="btn btn-p" onclick="gen()">Generate API Key</button></div>'
  +'<div id="generated"></div>'
  +'</section>'

  +'<section id="keys" class="card">'
  +'<h2>Manage API Keys <span class="small" style="font-weight:400">('+k.length+' keys)</span></h2>'
  +'<div class="row" style="margin-bottom:12px">'
  +'<div><label>Search</label><input id="q" oninput="filterKeys()" placeholder="Search owner, key, note…"></div>'
  +'<div><label>Status</label><select id="fstatus" onchange="filterKeys()"><option value="all">All Status</option><option value="active">Active</option><option value="expired">Expired</option><option value="revoked">Revoked</option></select></div>'
  +'<div><label>Plan</label><select id="fplan" onchange="filterKeys()"><option value="all">All Plans</option><option value="day">Day</option><option value="week">Week</option><option value="month">Month</option><option value="lifetime">Lifetime</option></select></div>'
  +'</div>'
  +'<div class="scroll"><table>'
  +'<thead><tr><th>API Key</th><th>Owner</th><th>Plan</th><th>Status</th><th>Limit/Day</th><th>Today</th><th>Total</th><th>Expires</th><th>Last Used</th><th>Actions</th></tr></thead>'
  +'<tbody>'+rows+'</tbody>'
  +'</table></div>'
  +'</section>'

  +'<section id="csv" class="card">'
  +'<h2>Import / Export CSV</h2>'
  +'<a class="btn btn-s" href="/admin-panel/api/export.csv" style="text-decoration:none">&#8595; Export CSV</a>'
  +'<div class="row" style="margin-top:14px">'
  +'<div><label>CSV File</label><input id="csvfile" type="file" accept=".csv"></div>'
  +'<div><label>Import Mode</label><select id="imode"><option value="merge">Merge (Add New Only)</option><option value="replace">Replace (Full Restore)</option></select></div>'
  +'<div style="flex:0;align-self:flex-end"><button class="btn btn-p" onclick="imp()">&#8593; Import</button></div>'
  +'</div>'
  +'<p id="csvmsg" class="small" style="margin-top:10px"></p>'
  +'</section>'

  +'<section id="settings" class="card">'
  +'<h2>Account &amp; Settings</h2>'
  +'<table class="settings-table" style="min-width:auto;border-collapse:collapse">'
  +'<tr><td>Developer</td><td><b>@VORTEX_PRIYANSHU</b></td></tr>'
  +'<tr><td>Timezone</td><td>Asia/Kolkata (IST)</td></tr>'
  +'<tr><td>Session</td><td>Secure HttpOnly Cookie (8 hours)</td></tr>'
  +'<tr><td>Plans</td><td>Day (1d) · Week (7d) · Month (30d) · Lifetime</td></tr>'
  +'<tr><td>Key Format</td><td>32-character URL-safe token</td></tr>'
  +'<tr><td>Max Keys Shown</td><td>500 per query</td></tr>'
  +'</table>'
  +'</section>';
}

async function gen(){
  var r=await api('/admin-panel/api/keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({plan:plan.value,owner:owner.value,daily_limit:limit.value,note:note.value})});
  var x=await r.json();
  if(!r.ok){alert(x.detail||'Error generating key');return}
  document.getElementById('generated').innerHTML=
    '<div class="key-box">'
    +'<div class="key-box-title">&#10003; API Key Generated Successfully</div>'
    +'<pre id="newkey">'+esc(x.key)+'</pre>'
    +'<div class="small" style="margin-bottom:10px">Plan: <b>'+esc(x.plan)+'</b> &nbsp;&middot;&nbsp; Owner: <b>'+esc(x.owner)+'</b> &nbsp;&middot;&nbsp; Expires: <b>'+esc(x.expires_at)+'</b> &nbsp;&middot;&nbsp; Limit: <b>'+(x.daily_limit!=null?x.daily_limit:'Unlimited')+'</b>/day</div>'
    +'<button class="btn btn-p btn-sm" onclick="copyKey(\''+esc(x.key)+'\')">&#128203; Copy Key</button>'
    +'</div>';
  load();
}

async function revokeKey(k){
  if(!confirm('Revoke this key?\n\nIt will stop working immediately. You can delete it afterwards.'))return;
  var r=await api('/admin-panel/api/keys/'+encodeURIComponent(k)+'/revoke',{method:'POST'});
  if(r.ok)load();else alert((await r.json()).detail||'Error');
}

async function deleteKey(k){
  if(!confirm('Permanently DELETE this key from the database?\n\nThis CANNOT be undone.'))return;
  var r=await api('/admin-panel/api/keys/'+encodeURIComponent(k),{method:'DELETE'});
  if(r.ok)load();else alert((await r.json()).detail||'Error');
}

function copyKey(k){
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(k).then(function(){alert('Copied to clipboard!')}).catch(function(){prompt('Copy this key:',k)});
  } else { prompt('Copy this key:',k); }
}

async function imp(){
  var f=document.getElementById('csvfile').files[0];
  if(!f)return alert('Please select a CSV file first');
  var m=document.getElementById('imode').value;
  if(m==='replace'&&!confirm('WARNING: This will permanently replace ALL existing keys.\n\nAre you absolutely sure?'))return;
  var fd=new FormData();fd.append('file',f);
  var r=await api('/admin-panel/api/import?mode='+m,{method:'POST',body:fd});
  var x=await r.json();
  document.getElementById('csvmsg').textContent='Import complete: '+x.added+' added, '+x.skipped+' skipped, '+x.invalid+' invalid rows.';
  load();
}

function filterKeys(){
  var q=(document.getElementById('q').value||'').toLowerCase();
  var st=document.getElementById('fstatus').value;
  var pl=document.getElementById('fplan').value;
  document.querySelectorAll('#keys tbody tr').forEach(function(tr){
    var txt=tr.textContent.toLowerCase();
    var badge=tr.querySelector('.badge');
    var bst=badge?badge.textContent.trim().toLowerCase():'';
    var cells=tr.querySelectorAll('td');
    var plan_txt=cells[2]?cells[2].textContent.trim().toLowerCase():'';
    var ok=(!q||txt.includes(q))&&(st==='all'||bst===st)&&(pl==='all'||plan_txt===pl);
    tr.style.display=ok?'':'none';
  });
}

async function logout(){await fetch('/admin-panel/logout',{method:'POST'});location.href='/admin-panel'}
load();
'''

# ── Pinger ───────────────────────────────────────────────────────────────────
async def pinger():
    port=os.getenv("PORT","7860"); url=f"http://localhost:{port}/health"
    async with httpx.AsyncClient(timeout=10) as client:
        while True:
            await asyncio.sleep(120)
            try: await client.get(url)
            except Exception: pass

@fastapi_app.on_event("startup")
async def startup_event(): asyncio.create_task(pinger())

# ── Gradio UI ───────────────────────────────────────────────────────────────
def format_result(row):
    lines=[]
    for field in SEARCH_FIELDS:
        val=row.get(field,"")
        if val: lines.append(f"**{field}:** {val}")
    cn=row.get("connected_numbers",[])
    if cn: lines.append("**connected:** "+", ".join(f"{c['field']}={c['value']}" for c in cn))
    return "\n\n".join(lines)

def search_ui(query,limit,search_type="phone"):
    if not query or not query.strip(): return "⚠️ Enter phone, aadhaar or name to search."
    q = query.strip()
    try:
        if search_type == "phone" or (q.isdigit() and len(q) >= 8 and search_type in ("phone","advanced","")):
            data = _unified_search(q, int(limit))
        elif search_type == "aadhar":
            data = _run_field_search("aadharNumber", q, "exact", int(limit))
            data = {"query": q, "searched_fields": ["aadharNumber"], "count": data["count"], "results": data["results"]}
        elif search_type == "other":
            data = _run_field_search("otherNumber", q, "exact", int(limit))
            data = {"query": q, "searched_fields": ["otherNumber"], "count": data["count"], "results": data["results"]}
        elif search_type == "name":
            data = _run_field_search("name", q, "contains", int(limit))
            data = {"query": q, "searched_fields": ["name"], "count": data["count"], "results": data["results"]}
        else:
            data = _unified_search(q, int(limit))
    except Exception as e:
        return f"❌ Error: {str(e)}"
    if not data["results"]:
        return f"🔍 **Query:** `{q}`\n\n❌ **No data found** for this search."
    return f"🔍 **Query:** `{q}` | **Found:** {data['count']} results\n\n---\n\n"+"\n\n---\n\n".join(f"### Result {i}\n{format_result(row)}" for i,row in enumerate(data["results"],1))

def build_ui():
    css = r'''
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    *{box-sizing:border-box;margin:0;padding:0}
    html,body{background:#05070d!important;font-family:Inter,system-ui,sans-serif!important;color:#f4f7fb!important}
    body{scrollbar-width:thin;scrollbar-color:#6d4bd8 #070a12}
    ::-webkit-scrollbar{width:6px;height:6px}::-webkit-scrollbar-track{background:#070a12}
    ::-webkit-scrollbar-thumb{background:linear-gradient(#8b5cf6,#06b6d4);border-radius:999px}
    .gradio-container{max-width:100%!important;padding:0!important;margin:0!important;min-height:100vh!important;background:#05070d!important}
    .main,.wrap,.contain{padding:0!important;max-width:100%!important}
    footer,footer*,.svelte-1ipelgc{display:none!important}
    .app{background:#05070d!important}
    #main-ui{display:flex;min-height:100vh;width:100%;background:#05070d}
    .sidebar{width:240px;min-width:240px;background:linear-gradient(180deg,#0a0e1a 0%,#070b14 100%);border-right:1px solid rgba(255,255,255,.06);display:flex;flex-direction:column;padding:0;position:relative;z-index:20}
    .sidebar-logo{padding:20px 18px 16px;display:flex;align-items:center;gap:10px;border-bottom:1px solid rgba(255,255,255,.06)}
    .sidebar-logo .logo-icon{width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#7c3aed,#06b6d4);display:flex;align-items:center;justify-content:center;font-size:18px}
    .sidebar-logo .logo-text{font-size:15px;font-weight:800;letter-spacing:-.3px;background:linear-gradient(90deg,#fff,#a78bfa,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}
    .sidebar-logo .logo-sub{font-size:9px;color:#6b7280;font-weight:500;letter-spacing:.3px}
    .nav-items{padding:14px 12px;flex:1;display:flex;flex-direction:column;gap:4px}
    .nav-item{display:flex;align-items:center;gap:10px;padding:11px 14px;border-radius:10px;color:#9ca3af;font-size:13px;font-weight:500;cursor:pointer;transition:.2s;border:1px solid transparent;text-decoration:none}
    .nav-item:hover{background:rgba(139,92,246,.1);color:#e5e7eb;border-color:rgba(139,92,246,.2)}
    .nav-item.active{background:linear-gradient(135deg,rgba(124,58,237,.25),rgba(6,182,212,.15));color:#fff;border-color:rgba(139,92,246,.35);box-shadow:0 4px 16px rgba(124,58,237,.15)}
    .nav-item svg{width:18px;height:18px;flex-shrink:0;opacity:.85}
    .sidebar-bottom{padding:14px 16px;border-top:1px solid rgba(255,255,255,.06)}
    .api-status{display:flex;align-items:center;gap:8px;padding:10px 12px;background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);border-radius:10px;font-size:12px;color:#6ee7b7;margin-bottom:12px}
    .api-status .dot{width:8px;height:8px;border-radius:50%;background:#34d399;box-shadow:0 0 8px #34d399;animation:pulse 2s infinite}
    @keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
    .dev-card{padding:12px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:12px}
    .dev-card .dev-name{font-size:12px;font-weight:700;color:#c4b5fd;display:flex;align-items:center;gap:6px}
    .dev-card .dev-meta{font-size:10px;color:#6b7280;margin-top:4px;line-height:1.4}
    .main-content{flex:1;display:flex;flex-direction:column;min-width:0;background:radial-gradient(ellipse at 20% 0%,rgba(124,58,237,.12),transparent 40%),radial-gradient(ellipse at 80% 10%,rgba(6,182,212,.08),transparent 35%),linear-gradient(180deg,#060811 0%,#05070d 60%,#070914 100%)}
    .topbar{display:flex;align-items:center;justify-content:space-between;padding:14px 28px;border-bottom:1px solid rgba(255,255,255,.06);background:rgba(5,7,13,.6);backdrop-filter:blur(12px)}
    .topbar-left{display:flex;align-items:center;gap:10px}
    .topbar-left .brand{font-size:14px;font-weight:700;color:#e5e7eb}
    .topbar-left .brand span{background:linear-gradient(90deg,#a78bfa,#67e8f9);-webkit-background-clip:text;background-clip:text;color:transparent}
    .topbar-right{display:flex;align-items:center;gap:12px}
    .topbar-right .icon-btn{width:36px;height:36px;border-radius:10px;background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);display:flex;align-items:center;justify-content:center;cursor:pointer;color:#9ca3af;transition:.2s}
    .topbar-right .icon-btn:hover{background:rgba(139,92,246,.12);color:#fff;border-color:rgba(139,92,246,.3)}
    .login-btn{padding:8px 18px;border-radius:10px;background:linear-gradient(135deg,#7c3aed,#06b6d4);border:none;color:#fff;font-size:13px;font-weight:600;cursor:pointer;transition:.2s;box-shadow:0 4px 16px rgba(124,58,237,.25);text-decoration:none}
    .login-btn:hover{filter:brightness(1.1);transform:translateY(-1px)}
    .content-area{padding:28px 32px 40px;flex:1;overflow-y:auto}
    .hero-section{text-align:center;margin-bottom:28px;position:relative}
    .hero-badge{display:inline-block;padding:6px 16px;border-radius:999px;background:rgba(139,92,246,.12);border:1px solid rgba(139,92,246,.25);font-size:11px;font-weight:600;color:#c4b5fd;letter-spacing:.5px;margin-bottom:16px;text-transform:uppercase}
    .hero-title{font-size:clamp(32px,5vw,48px);font-weight:900;letter-spacing:-1.5px;line-height:1.1;margin-bottom:10px}
    .hero-title .icmr{background:linear-gradient(90deg,#a78bfa,#c4b5fd);-webkit-background-clip:text;background-clip:text;color:transparent}
    .hero-title .hitek{background:linear-gradient(90deg,#67e8f9,#22d3ee);-webkit-background-clip:text;background-clip:text;color:transparent}
    .hero-title .search{color:#fff}
    .hero-sub{color:#8d97aa;font-size:14px;margin-bottom:6px}
    .hero-sub2{color:#6b7280;font-size:13px}
    .search-panel{background:linear-gradient(145deg,rgba(18,23,36,.9),rgba(9,12,21,.85));border:1px solid rgba(255,255,255,.08);border-radius:20px;padding:22px;margin-bottom:24px;box-shadow:0 20px 60px rgba(0,0,0,.35);position:relative;overflow:hidden}
    .search-panel:before{content:"";position:absolute;left:0;right:0;top:0;height:1px;background:linear-gradient(90deg,transparent,rgba(139,92,246,.5),rgba(6,182,212,.3),transparent)}
    .search-tabs{display:flex;gap:6px;margin-bottom:18px;flex-wrap:wrap}
    .search-tab{padding:9px 16px;border-radius:10px;font-size:12px;font-weight:600;color:#9ca3af;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);cursor:pointer;transition:.2s;display:flex;align-items:center;gap:6px}
    .search-tab:hover{background:rgba(139,92,246,.1);color:#e5e7eb;border-color:rgba(139,92,246,.2)}
    .search-tab.active{background:linear-gradient(135deg,#7c3aed,#6366f1);color:#fff;border-color:transparent;box-shadow:0 4px 14px rgba(124,58,237,.3)}
    .search-controls{display:flex;align-items:center;gap:16px;margin-bottom:12px;flex-wrap:wrap}
    .search-btn-wrap{margin-top:8px}
    .stats-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:24px}
    .stat-card{background:linear-gradient(145deg,rgba(18,23,36,.85),rgba(9,12,21,.8));border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:18px;transition:.2s;position:relative;overflow:hidden}
    .stat-card:hover{border-color:rgba(139,92,246,.3);transform:translateY(-2px)}
    .stat-card .stat-icon{width:40px;height:40px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px;margin-bottom:12px}
    .stat-card .stat-label{font-size:11px;color:#8d97aa;font-weight:500;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}
    .stat-card .stat-value{font-size:22px;font-weight:800;color:#fff;letter-spacing:-.5px}
    .stat-card .stat-tag{font-size:11px;margin-top:6px;display:flex;align-items:center;gap:4px}
    .stat-card .stat-tag.green{color:#34d399}
    .stat-card .stat-tag.blue{color:#22d3ee}
    .bottom-grid{display:grid;grid-template-columns:1.4fr 1fr;gap:16px}
    .panel-card{background:linear-gradient(145deg,rgba(18,23,36,.85),rgba(9,12,21,.8));border:1px solid rgba(255,255,255,.07);border-radius:16px;padding:18px;overflow:hidden}
    .panel-card .panel-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:14px}
    .panel-card .panel-title{font-size:14px;font-weight:700;color:#e5e7eb;display:flex;align-items:center;gap:8px}
    .panel-card .panel-link{font-size:12px;color:#a78bfa;cursor:pointer;text-decoration:none}
    .panel-card .panel-link:hover{color:#c4b5fd}
    .recent-table{width:100%;border-collapse:collapse;font-size:12px}
    .recent-table th{text-align:left;padding:8px 10px;color:#6b7280;font-weight:600;font-size:11px;border-bottom:1px solid rgba(255,255,255,.06);text-transform:uppercase;letter-spacing:.3px}
    .recent-table td{padding:10px;border-bottom:1px solid rgba(255,255,255,.04);color:#cfd6e3}
    .recent-table tr:hover td{background:rgba(139,92,246,.04)}
    .type-badge{display:inline-block;padding:3px 8px;border-radius:6px;font-size:10px;font-weight:700}
    .type-phone{background:rgba(59,130,246,.15);color:#60a5fa}
    .type-aadhar{background:rgba(16,185,129,.15);color:#34d399}
    .type-name{background:rgba(168,85,247,.15);color:#c084fc}
    .type-other{background:rgba(245,158,11,.15);color:#fbbf24}
    .quick-list{display:flex;flex-direction:column;gap:8px}
    .quick-item{display:flex;align-items:center;justify-content:space-between;padding:12px 14px;background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);border-radius:10px;color:#cfd6e3;font-size:13px;font-weight:500;cursor:pointer;transition:.2s;text-decoration:none}
    .quick-item:hover{background:rgba(139,92,246,.1);border-color:rgba(139,92,246,.25);color:#fff}
    .quick-item .qi-left{display:flex;align-items:center;gap:10px}
    .quick-item .qi-arrow{color:#6b7280;font-size:14px}
    .results-area{margin-top:20px}
    .results-area .prose{color:#cfd6e3!important}
    .results-area code{color:#c4b5fd!important}
    .gr-box,.gr-form,.gr-padded,.gr-panel{background:transparent!important;border:none!important;box-shadow:none!important;padding:0!important}
    .gr-input-label,.gr-form>div>label{color:#9ca3af!important;font-size:12px!important}
    textarea,input[type=text],input[type=number]{background:rgba(0,0,0,.3)!important;border:1px solid rgba(255,255,255,.1)!important;color:#eef2f8!important;border-radius:12px!important;font-size:14px!important}
    textarea:focus,input:focus{border-color:rgba(139,92,246,.7)!important;box-shadow:0 0 0 3px rgba(139,92,246,.12)!important}
    .gr-button{border-radius:12px!important}
    .gr-button-primary{background:linear-gradient(90deg,#7c3aed,#06b6d4)!important;border:none!important;color:#fff!important;font-weight:700!important;box-shadow:0 8px 24px rgba(124,58,237,.25)!important;width:100%!important;padding:14px!important;font-size:15px!important}
    .gr-button-primary:hover{filter:brightness(1.1)!important}
    input[type=range]{accent-color:#8b5cf6}
    #search-controls-group{background:transparent!important}
    @media(max-width:1100px){.stats-row{grid-template-columns:repeat(2,1fr)}.bottom-grid{grid-template-columns:1fr}}
    @media(max-width:800px){.sidebar{display:none}.content-area{padding:16px}.stats-row{grid-template-columns:1fr 1fr}}
    '''
    custom_html = r'''
<div id="main-ui">
  <div class="sidebar">
    <div class="sidebar-logo">
      <div class="logo-icon">🔍</div>
      <div>
        <div class="logo-text">ICMR + HITEK</div>
        <div class="logo-sub">Smart Search · Better Results</div>
      </div>
    </div>
    <div class="nav-items">
      <a class="nav-item active" href="#">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6"/></svg>
        Home
      </a>
      <a class="nav-item" href="/docs" target="_blank">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"/></svg>
        API Documentation
      </a>
      <a class="nav-item" href="/admin-panel" target="_blank">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/></svg>
        Generate API Key
      </a>
      <a class="nav-item" href="/admin-panel" target="_blank">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4"/></svg>
        Manage Keys
      </a>
      <a class="nav-item" href="/admin-panel" target="_blank">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/></svg>
        Import / Export
      </a>
      <a class="nav-item" href="/admin-panel" target="_blank">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"/></svg>
        Statistics
      </a>
      <a class="nav-item" href="#">
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
        About
      </a>
    </div>
    <div class="sidebar-bottom">
      <div class="api-status">
        <span class="dot"></span>
        API Status <strong style="margin-left:auto;color:#34d399">Online</strong>
      </div>
      <div class="dev-card">
        <div class="dev-name">👑 @VORTEX_PRIYANSHU</div>
        <div class="dev-meta">ICMR + HITEK<br>Data Search API<br>v1.0.0</div>
      </div>
    </div>
  </div>
  <div class="main-content">
    <div class="topbar">
      <div class="topbar-left">
        <div class="brand"><span>ICMR + HITEK</span> · Smart Search</div>
      </div>
      <div class="topbar-right">
        <div class="icon-btn" title="Theme">☀</div>
        <a href="/admin-panel" class="login-btn">Login</a>
      </div>
    </div>
    <div class="content-area">
      <div class="hero-section">
        <div class="hero-badge">SEARCH ANY RECORD</div>
        <div class="hero-title">
          <span class="icmr">ICMR</span> + <span class="hitek">HITEK</span> <span class="search">Search</span>
        </div>
        <div class="hero-sub">Fast · Secure · Reliable</div>
        <div class="hero-sub2">Search government records using phone, Aadhaar or other details.</div>
      </div>
      <div class="search-panel">
        <div class="search-tabs" id="searchTabs">
          <div class="search-tab active" data-type="phone">📞 Phone Number</div>
          <div class="search-tab" data-type="aadhar">🪪 Aadhar Number</div>
          <div class="search-tab" data-type="other">📱 Other Number</div>
          <div class="search-tab" data-type="name">👤 Name</div>
          <div class="search-tab" data-type="advanced">⚙️ Advanced</div>
        </div>
      </div>
      <div class="stats-row">
        <div class="stat-card">
          <div class="stat-icon" style="background:rgba(139,92,246,.15)">🗄️</div>
          <div class="stat-label">Total Records</div>
          <div class="stat-value">1,24,56,789+</div>
          <div class="stat-tag green">↑ Live Database</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon" style="background:rgba(6,182,212,.15)">⚡</div>
          <div class="stat-label">Response Time</div>
          <div class="stat-value">&lt; 2 Seconds</div>
          <div class="stat-tag green">↑ Ultra Fast</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon" style="background:rgba(16,185,129,.15)">🛡️</div>
          <div class="stat-label">Data Security</div>
          <div class="stat-value">Bank Level</div>
          <div class="stat-tag blue">↓ Encrypted &amp; Safe</div>
        </div>
        <div class="stat-card">
          <div class="stat-icon" style="background:rgba(59,130,246,.15)">☁️</div>
          <div class="stat-label">Data Source</div>
          <div class="stat-value">ICMR + HITEK</div>
          <div class="stat-tag blue">↓ Government Records</div>
        </div>
      </div>
      <div class="bottom-grid">
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">🕐 Recent Searches</div>
            <a class="panel-link" href="#">View All</a>
          </div>
          <table class="recent-table">
            <thead>
              <tr><th>#</th><th>Search Term</th><th>Type</th><th>Results</th><th>Time</th><th></th></tr>
            </thead>
            <tbody>
              <tr><td>1.</td><td>9876543210</td><td><span class="type-badge type-phone">Phone</span></td><td>12</td><td>Today 14:32</td><td>›</td></tr>
              <tr><td>2.</td><td>123456789012</td><td><span class="type-badge type-aadhar">Aadhar</span></td><td>8</td><td>Today 14:18</td><td>›</td></tr>
              <tr><td>3.</td><td>Rohit Kumar</td><td><span class="type-badge type-name">Name</span></td><td>23</td><td>Today 13:47</td><td>›</td></tr>
              <tr><td>4.</td><td>9988776655</td><td><span class="type-badge type-phone">Phone</span></td><td>15</td><td>Today 12:21</td><td>›</td></tr>
              <tr><td>5.</td><td>9876</td><td><span class="type-badge type-other">Other</span></td><td>3</td><td>Today 11:56</td><td>›</td></tr>
            </tbody>
          </table>
        </div>
        <div class="panel-card">
          <div class="panel-header">
            <div class="panel-title">⚡ Quick Access</div>
          </div>
          <div class="quick-list">
            <a class="quick-item" href="/admin-panel" target="_blank"><span class="qi-left">🔑 Generate API Key</span><span class="qi-arrow">→</span></a>
            <a class="quick-item" href="/admin-panel" target="_blank"><span class="qi-left">🗂️ Manage Keys</span><span class="qi-arrow">→</span></a>
            <a class="quick-item" href="/admin-panel" target="_blank"><span class="qi-left">⬆️ Import CSV</span><span class="qi-arrow">→</span></a>
            <a class="quick-item" href="/admin-panel/api/export.csv" target="_blank"><span class="qi-left">⬇️ Export CSV</span><span class="qi-arrow">→</span></a>
            <a class="quick-item" href="/docs" target="_blank"><span class="qi-left">&lt;/&gt; API Documentation</span><span class="qi-arrow">→</span></a>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
document.querySelectorAll('.search-tab').forEach(function(tab){
  tab.addEventListener('click',function(){
    document.querySelectorAll('.search-tab').forEach(function(t){t.classList.remove('active')});
    tab.classList.add('active');
    var type=tab.getAttribute('data-type');
    var el=document.querySelector('#search-type-input textarea, #search-type-input input');
    if(el){el.value=type;el.dispatchEvent(new Event('input',{bubbles:true}));}
  });
});
</script>
'''
    with gr.Blocks(title="ICMR + HITEK Search", theme=gr.themes.Base(), css=css) as demo:
        gr.HTML(custom_html)
        search_type = gr.Textbox(value="phone", elem_id="search-type-input", visible=False)
        with gr.Group(elem_id="search-controls-group"):
            with gr.Row():
                query_input = gr.Textbox(
                    label="",
                    placeholder="Enter 10 digit phone number...",
                    lines=1,
                    scale=4,
                )
                limit_slider = gr.Slider(
                    minimum=1, maximum=100, value=50, step=1,
                    label="Max Results",
                    scale=2,
                )
            search_btn = gr.Button("🔍 Search Now →", variant="primary", size="lg")
        output = gr.Markdown(label="", elem_classes="results-area")
        search_btn.click(fn=search_ui, inputs=[query_input, limit_slider, search_type], outputs=output)
        query_input.submit(fn=search_ui, inputs=[query_input, limit_slider, search_type], outputs=output)
    return demo

demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")
