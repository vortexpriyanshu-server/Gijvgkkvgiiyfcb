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
DB_PATH = os.environ.get("KEY_DB_PATH", os.path.join(BASE, "keys.db"))
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
    owner = (owner or "").strip()
    if not owner: raise ValueError("owner is required")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return owner + "".join(secrets.choice(alphabet) for _ in range(7))


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
    # Keep public API errors predictable and always append developer as final field.
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
    # One authenticated API call consumes one request. Individual queries are not separately counted.
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
    return HTMLResponse(f'''<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>{escape(title)}</title>
<style>
:root{{--red:#c9152b;--dark:#7d0918;--bg:#f7f7f8;--card:#fff;--muted:#73777d;--ok:#1f9d62;--warn:#d98b00}}
*{{box-sizing:border-box}}body{{margin:0;font-family:Inter,system-ui,Arial;background:var(--bg);color:#17191c}}button,input,select,textarea{{font:inherit}}.wrap{{max-width:1200px;margin:auto;padding:20px}}.nav{{background:linear-gradient(135deg,#a80d22,#e3213d);color:white;padding:18px 22px;border-radius:22px;box-shadow:0 12px 30px #b30f2430;display:flex;gap:14px;align-items:center;flex-wrap:wrap}}.nav b{{font-size:20px;margin-right:auto}}.nav a,.nav button{{color:white;background:#ffffff18;border:1px solid #ffffff33;padding:10px 13px;border-radius:12px;text-decoration:none;cursor:pointer}}.card{{background:var(--card);border-radius:20px;padding:20px;box-shadow:0 10px 30px #11111112;border:1px solid #eeeeef;margin-top:18px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px}}.stat{{padding:18px;border-radius:18px;background:linear-gradient(145deg,#fff,#f2f2f3);box-shadow:inset 1px 1px 0 white,8px 8px 20px #1111110d}}.stat strong{{display:block;font-size:28px;color:var(--red);margin-top:8px}}label{{font-size:13px;color:var(--muted);display:block;margin:12px 0 6px}}input,select,textarea{{width:100%;padding:12px;border:1px solid #ddd;border-radius:12px;background:#fff}}.btn{{border:0;border-radius:13px;padding:12px 16px;cursor:pointer;transition:.16s;box-shadow:0 5px 0 #87091b;background:var(--red);color:white;font-weight:700}}.btn:active{{transform:translateY(3px);box-shadow:0 2px 0 #87091b}}.btn.secondary{{background:white;color:var(--red);border:1px solid #f0bcc4;box-shadow:0 4px 0 #ead6d9}}.row{{display:flex;gap:10px;align-items:end;flex-wrap:wrap}}.row>*{{flex:1;min-width:150px}}table{{width:100%;border-collapse:collapse;min-width:980px}}th,td{{padding:10px;border-bottom:1px solid #eee;text-align:left;font-size:13px}}th{{color:var(--muted)}}.scroll{{overflow:auto}}.badge{{padding:5px 8px;border-radius:99px;font-size:11px;font-weight:700;background:#eee}}.active{{background:#ddf7e9;color:#087342}}.revoked{{background:#ffe1e4;color:#9c1020}}.expired{{background:#eee;color:#666}}.warn{{background:#fff0c7;color:#8a5900}}.login{{max-width:430px;margin:12vh auto}}.login .icon{{font-size:48px;text-align:center}}.msg{{margin-top:12px;padding:10px;border-radius:10px;background:#fff0f2;color:#a20e22;display:none}}.small{{font-size:12px;color:var(--muted)}}pre{{white-space:pre-wrap;word-break:break-all;background:#101114;color:#fff;padding:15px;border-radius:12px}}@media(max-width:700px){{.wrap{{padding:10px}}.nav{{border-radius:15px}}.nav a{{font-size:12px}}}}
</style></head><body><div class="wrap">{body}</div><script>{script}</script></body></html>''')

LOGIN_HTML='''<div class="card login"><div class="icon">🔐</div><h1 style="text-align:center">API Key Admin</h1><p style="text-align:center;color:#73777d">Secure Admin Access</p><form id="f"><label>Administrator password</label><input id="p" type="password" autocomplete="current-password" required><button class="btn" style="width:100%;margin-top:16px">Login</button><div id="m" class="msg"></div></form></div>'''
LOGIN_JS='''document.getElementById("f").onsubmit=async(e)=>{e.preventDefault();let m=document.getElementById("m");m.style.display="none";let r=await fetch("/admin-panel/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({password:document.getElementById("p").value})});if(r.ok)location.href="/admin-panel";else{let x=await r.json();m.textContent=x.detail||"Login failed";m.style.display="block"}}'''

@fastapi_app.get("/admin-panel", response_class=HTMLResponse)
def admin_panel(request: Request):
    if not valid_session(request): return admin_page("API Key Admin",LOGIN_HTML,LOGIN_JS)
    body='''<div class="nav"><b>🔑 API Key Admin</b><a href="#dashboard">Dashboard</a><a href="#generate">Generate</a><a href="#keys">Manage Keys</a><a href="#csv">CSV</a><a href="#settings">Settings</a><button onclick="logout()">Logout</button></div>
<div id="app"></div>'''
    return admin_page("API Key Admin",body,ADMIN_JS)

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

ADMIN_JS=r'''const esc=s=>String(s??"").replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
async function api(u,o){let r=await fetch(u,o);if(r.status===401){location.href='/admin-panel';throw 0}return r}
async function load(){let s=await (await api('/admin-panel/api/stats')).json();let kx=await (await api('/admin-panel/api/keys')).json();let k=kx.keys||[];
let chart=s.chart.map(x=>`<div style="flex:1;text-align:center"><div style="height:${Math.max(4,x.count/Math.max(1,Math.max(...s.chart.map(y=>y.count)))*130)}px;background:#c9152b;border-radius:8px 8px 2px 2px"></div><small>${x.day}</small></div>`).join('');
let rows=k.map(x=>`<tr><td><code>${esc(x.key)}</code></td><td>${esc(x.owner)}</td><td>${esc(x.plan)}</td><td><span class="badge ${x.status}">${esc(x.status)}</span></td><td>${x.daily_limit??'Unlimited'}</td><td>${x.requests_today}/${x.daily_limit??'∞'}</td><td>${x.total_usage}</td><td>${esc(x.expires_at)}</td><td>${esc(x.last_used_at)}</td><td><button class="btn secondary" onclick="copyKey('${esc(x.key)}')">Copy</button> ${x.status!=='revoked'?`<button class="btn" onclick="revokeKey('${esc(x.key)}')">Revoke</button>`:''}</td></tr>`).join('');
document.getElementById('app').innerHTML=`<section id="dashboard"><h1>Dashboard</h1><p class="small">Overview of your API keys and usage statistics</p><div class="grid">${[['Total Keys',s.total_keys],['Active Keys',s.active_keys],['Expired Keys',s.expired_keys],['Revoked Keys',s.revoked_keys],['Requests Today',s.requests_today],['Total Requests',s.total_requests]].map(x=>`<div class="stat">${x[0]}<strong>${x[1]}</strong></div>`).join('')}</div><div class="card"><h2>Requests — Last 7 Days</h2><div style="height:160px;display:flex;align-items:end;gap:10px">${chart}</div></div><div class="card"><h2>Most Used Key</h2><p>${s.most_used?`<b>${esc(s.most_used.owner)}</b> · ${esc(s.most_used.plan)} · ${s.most_used.total_usage} requests`:'No usage yet'}</p></div></section>
<section id="generate" class="card"><h2>Generate Key</h2><div class="row"><div><label>Plan</label><select id="plan"><option value="day">1 Day</option><option value="week">7 Days</option><option value="month">30 Days</option><option value="lifetime">Lifetime</option></select></div><div><label>Owner</label><input id="owner" placeholder="@vortex_priyanshu"></div><div><label>Daily Request Limit</label><input id="limit" value="100" placeholder="100 or unlimited"></div></div><label>Note</label><input id="note" placeholder="Testing key"><button class="btn" style="margin-top:14px" onclick="gen()">Generate API Key</button><div id="generated"></div></section>
<section id="keys" class="card"><h2>Manage Keys</h2><div class="row"><div><label>Search</label><input id="q" oninput="filterKeys()" placeholder="owner, key, note"></div><div><label>Status</label><select id="status" onchange="filterKeys()"><option>all</option><option>active</option><option>expired</option><option>revoked</option></select></div><div><label>Plan</label><select id="kp" onchange="filterKeys()"><option>all</option><option>day</option><option>week</option><option>month</option><option>lifetime</option></select></div></div><div class="scroll"><table><thead><tr><th>Key</th><th>Owner</th><th>Plan</th><th>Status</th><th>Limit</th><th>Today</th><th>Total</th><th>Expires</th><th>Last Used</th><th>Actions</th></tr></thead><tbody>${rows}</tbody></table></div></section>
<section id="csv" class="card"><h2>CSV Import / Export</h2><a class="btn" href="/admin-panel/api/export.csv">Export CSV</a><div class="row" style="margin-top:16px"><div><label>CSV file</label><input id="csvfile" type="file" accept=".csv"></div><div><label>Mode</label><select id="imode"><option value="merge">Merge / Add New Only</option><option value="replace">Replace / Full Restore</option></select></div><button class="btn" onclick="imp()">Import CSV</button></div><p id="csvmsg" class="small"></p></section>
<section id="settings" class="card"><h2>Settings</h2><p>Developer: <b>@VORTEX_PRIYANSHU</b></p><p>Timezone: <b>Asia/Kolkata</b></p><p>Admin session: <b>Secure HttpOnly cookie</b></p></section>`}
async function gen(){let r=await api('/admin-panel/api/keys',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({plan:plan.value,owner:owner.value,daily_limit:limit.value,note:note.value})});let x=await r.json();if(!r.ok){alert(x.detail||'Error');return}document.getElementById('generated').innerHTML=`<div class="card" style="background:#fff7f8"><b>Key Generated</b><pre id="newkey">${esc(x.key)}</pre><p>${esc(x.plan)} · ${esc(x.owner)} · ${esc(x.expires_at)} · ${x.daily_limit??'Unlimited'}</p><button class="btn" onclick="copyKey('${esc(x.key)}')">COPY</button></div>`;load()}
async function revokeKey(k){if(!confirm('Revoke this key?'))return;let r=await api('/admin-panel/api/keys/'+encodeURIComponent(k)+'/revoke',{method:'POST'});if(r.ok)load();else alert((await r.json()).detail||'Error')}
function copyKey(k){navigator.clipboard.writeText(k).then(()=>alert('Copied!'))}
async function imp(){let f=document.getElementById('csvfile').files[0];if(!f)return alert('Select CSV');if(document.getElementById('imode').value==='replace'&&!confirm('Replace all current keys?'))return;let fd=new FormData();fd.append('file',f);let r=await api('/admin-panel/api/import?mode='+imode.value,{method:'POST',body:fd});let x=await r.json();document.getElementById('csvmsg').textContent=JSON.stringify(x);load()}
async function logout(){await fetch('/admin-panel/logout',{method:'POST'});location.href='/admin-panel'}
load();'''

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
