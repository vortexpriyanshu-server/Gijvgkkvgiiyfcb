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
    return HTMLResponse(f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>
:root{{--az-dark:#131921;--az-nav:#232F3E;--az-nav2:#37475A;--az-orange:#FF9900;--az-oranged:#E47911;--az-orangel:#FEBD69;--az-link:#007185;--az-bg:#EAEDED;--az-card:#fff;--az-border:#D5D9D9;--az-text:#0F1111;--az-muted:#565959;--az-green:#007600;--az-red:#B12704}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:Arial,'Helvetica Neue',sans-serif;background:var(--az-bg);color:var(--az-text);font-size:14px;line-height:1.5}}
button,input,select,textarea{{font:inherit}}
.az-header{{background:var(--az-dark);padding:10px 20px;display:flex;align-items:center;gap:12px;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px #0008}}
.az-logo{{color:white;font-size:21px;font-weight:900;letter-spacing:-1px;text-decoration:none;line-height:1.1}}
.az-logo span{{color:var(--az-orange)}}
.az-logo-sub{{display:block;color:#aaa;font-size:10px;font-weight:400;letter-spacing:.5px;margin-top:1px}}
.az-header-right{{margin-left:auto;display:flex;align-items:center;gap:12px}}
.az-header-right span{{color:#ccc;font-size:12px}}
.az-signout{{background:none;border:1px solid #666;color:white;padding:5px 13px;border-radius:3px;cursor:pointer;font-size:12px;transition:.15s}}
.az-signout:hover{{border-color:#aaa;background:#ffffff18}}
.az-subnav{{background:var(--az-nav2);padding:0 20px;display:flex;gap:0;overflow-x:auto}}
.az-subnav a,.az-subnav button{{color:white;background:none;border:2px solid transparent;padding:9px 15px;text-decoration:none;cursor:pointer;font-size:13px;white-space:nowrap;transition:.1s;display:inline-block;border-top:none;border-bottom:none}}
.az-subnav a:hover,.az-subnav .nav-btn:hover{{border-left-color:white;border-right-color:white;border-radius:2px}}
.az-subnav .nav-logout{{color:var(--az-orangel);margin-left:auto;border:1px solid var(--az-orangel);border-radius:3px;margin:6px 0;padding:5px 12px;font-size:12px;background:none}}
.az-subnav .nav-logout:hover{{background:var(--az-orangel);color:var(--az-dark)}}
.wrap{{max-width:1200px;margin:0 auto;padding:16px 20px}}
section{{margin-bottom:4px}}
.page-title{{font-size:22px;font-weight:400;margin-bottom:2px;color:var(--az-text);padding:14px 0 4px}}
.page-sub{{color:var(--az-muted);font-size:13px;margin-bottom:12px}}
.card{{background:var(--az-card);border:1px solid var(--az-border);border-radius:4px;padding:20px;margin-bottom:14px}}
.card h2{{font-size:16px;font-weight:700;border-bottom:1px solid var(--az-border);padding-bottom:10px;margin-bottom:14px;color:var(--az-dark)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:10px;margin-bottom:14px}}
.stat{{background:var(--az-card);border:1px solid var(--az-border);border-radius:4px;padding:14px 16px}}
.stat small{{display:block;color:var(--az-muted);font-size:11px;margin-bottom:6px;text-transform:uppercase;letter-spacing:.4px}}
.stat strong{{display:block;font-size:26px;font-weight:700;color:var(--az-dark)}}
label{{display:block;font-size:13px;font-weight:700;margin:12px 0 4px;color:var(--az-text)}}
input,select,textarea{{width:100%;padding:7px 10px;border:1px solid var(--az-border);border-radius:3px;background:white;font-size:13px;color:var(--az-text)}}
input:focus,select:focus{{outline:3px solid #F5A623;outline-offset:0;border-color:#E47911;box-shadow:none}}
.btn{{display:inline-block;padding:8px 14px;border-radius:3px;cursor:pointer;font-size:13px;border:1px solid;text-decoration:none;transition:.12s;line-height:1.4;vertical-align:middle}}
.btn-p{{background:linear-gradient(to bottom,#f5c142,#e9a812);border-color:#A88734;color:var(--az-text)}}
.btn-p:hover{{background:linear-gradient(to bottom,#f0b91a,#de9f0c)}}
.btn-p:active{{background:linear-gradient(to bottom,#de9f0c,#ca8e08)}}
.btn-s{{background:linear-gradient(to bottom,#f7f8f8,#e7e9e9);border-color:#adb1b8;color:var(--az-text)}}
.btn-s:hover{{background:linear-gradient(to bottom,#e7e9e9,#d8dadb)}}
.btn-r{{background:none;border-color:#B12704;color:#B12704;font-size:12px;padding:4px 9px}}
.btn-r:hover{{background:#FFF0EC}}
.btn-del{{background:linear-gradient(to bottom,#e05c4b,#c03020);border-color:#8A1E00;color:white;font-size:12px;padding:4px 9px}}
.btn-del:hover{{background:linear-gradient(to bottom,#c03020,#a01c00)}}
.btn-sm{{padding:4px 10px;font-size:12px}}
.row{{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end}}
.row>*{{flex:1;min-width:140px}}
table{{width:100%;border-collapse:collapse;min-width:980px;font-size:13px}}
thead{{background:#F0F2F2}}
th{{padding:10px 12px;text-align:left;font-weight:700;border-bottom:2px solid var(--az-border);color:var(--az-text);white-space:nowrap}}
td{{padding:9px 12px;border-bottom:1px solid #EAEDED;vertical-align:middle}}
tr:hover td{{background:#F7F8F8}}
.scroll{{overflow-x:auto}}
.badge{{display:inline-block;padding:3px 8px;border-radius:3px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.3px}}
.active{{background:#E7F5E7;color:var(--az-green);border:1px solid #9DC79D}}
.revoked{{background:#FDECEA;color:var(--az-red);border:1px solid #F9C6C0}}
.expired{{background:#F0F2F2;color:var(--az-muted);border:1px solid #CCC}}
pre{{background:#F0F2F2;border:1px solid var(--az-border);padding:12px;border-radius:3px;font-size:12px;word-break:break-all;white-space:pre-wrap;margin:8px 0}}
code{{font-family:'Courier New',monospace;font-size:12px;color:#333}}
.key-box{{background:#FFFBF0;border:1px solid #D5A430;border-radius:4px;padding:16px;margin-top:14px}}
.key-box-title{{color:var(--az-green);font-weight:700;font-size:14px;margin-bottom:8px}}
.small{{font-size:12px;color:var(--az-muted)}}
.msg-err{{background:#FFF0EC;border:1px solid #C45500;color:#C45500;padding:10px 14px;border-radius:4px;margin-top:12px;font-size:13px;display:none}}
.chart-wrap{{display:flex;align-items:flex-end;gap:8px;height:120px;padding-top:10px}}
.login-wrap{{max-width:360px;margin:80px auto;padding:0 12px}}
.login-logo{{font-size:26px;font-weight:900;text-align:center;margin-bottom:14px;letter-spacing:-1px;color:var(--az-text)}}
.login-logo span{{color:var(--az-orange)}}
.login-card{{background:white;border:1px solid var(--az-border);border-radius:4px;padding:22px}}
.login-card h2{{font-size:17px;font-weight:400;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid var(--az-border);color:var(--az-text)}}
.login-terms{{text-align:center;font-size:11px;color:var(--az-muted);margin-top:20px;padding-top:14px;border-top:1px solid var(--az-border)}}
.login-terms b{{cursor:pointer;color:var(--az-link)}}
.az-footer{{text-align:center;font-size:11px;color:var(--az-muted);padding:16px 20px;border-top:1px solid var(--az-border);margin-top:10px}}
.settings-table td{{padding:8px 20px 8px 0;border:none}}
.settings-table td:first-child{{color:var(--az-muted);font-weight:700;white-space:nowrap}}
@media(max-width:700px){{.wrap{{padding:10px}}.row>*{{min-width:100%}}.az-subnav a{{font-size:11px;padding:8px 9px}}.grid{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body>{body}
<div class="az-footer">© 2024 VORTEX · Developer: @VORTEX_PRIYANSHU · API Key Management Panel</div>
<script>{script}</script></body></html>''')

LOGIN_HTML='''<div class="az-header" style="position:relative"><div class="az-logo"><span>amazon</span><span class="az-logo-sub">VORTEX API · Admin Access</span></div></div>
<div class="login-wrap">
<div class="login-logo"><span>amazon</span></div>
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
  <div class="az-logo"><span>amazon</span><span class="az-logo-sub">VORTEX · API Key Management</span></div>
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

def search_ui(query,limit):
    if not query or not query.strip(): return "⚠️ Kuch toh search karo — phone, aadhar, ya name daalo."
    try: data=_unified_search(query.strip(),int(limit))
    except Exception as e: return f"❌ Error: {str(e)}"
    if not data["results"]: return f"🔍 **Query:** `{query.strip()}`\n\n❌ **No data found** for this number."
    return f"🔍 **Query:** `{query.strip()}` | **Found:** {data['count']} results\n\n---\n\n"+"\n\n---\n\n".join(f"### Result {i}\n{format_result(row)}" for i,row in enumerate(data["results"],1))

def build_ui():
    with gr.Blocks(title="ICMR Search API",theme=gr.themes.Soft(),css=".main-title{text-align:center}.subtitle{text-align:center;color:#666}.footer{text-align:center;color:#888;margin-top:20px}") as demo:
        gr.Markdown("# 🔍 ICMR + HITEK Search API",elem_classes="main-title")
        gr.Markdown("Search phone/Aadhaar data",elem_classes="subtitle")
        with gr.Row():
            with gr.Column(scale=3): query_input=gr.Textbox(label="Search Query",placeholder="Phone number ya Aadhaar daalo...",lines=1)
            with gr.Column(scale=1): limit_slider=gr.Slider(minimum=1,maximum=50,value=10,step=1,label="Max Results")
        search_btn=gr.Button("🔍 Search",variant="primary",size="lg"); output=gr.Markdown(label="Results")
        search_btn.click(fn=search_ui,inputs=[query_input,limit_slider],outputs=output); query_input.submit(fn=search_ui,inputs=[query_input,limit_slider],outputs=output)
        gr.Markdown("---")
        with gr.Accordion("📡 API Info",open=False): gr.Markdown("**Endpoints:** GET /search · POST /search/parallel · GET /health · GET /admin-panel")
    return demo

demo=build_ui()
app=gr.mount_gradio_app(fastapi_app,demo,path="/")
