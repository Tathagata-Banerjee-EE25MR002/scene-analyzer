"""FastAPI server: web UI, accounts (register, sign in, change password), the /analyze endpoint with
per-user daily quotas, and a small admin API.

Every analysis spends Claude API credits, so every route that can cause one needs a signed-in user.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
import os
import secrets
import threading
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .analyzer import SceneAnalyzer
from .auth import (COOKIE_NAME, Sessions, check_password, check_username, hash_password, verify_dummy,
                   verify_password)
from .backends import BackendError, ClaudeBackend
from .config import Settings
from .db import Database, User, UsernameTaken, utc_now
from .preprocess import MAX_UPLOAD_BYTES, ImageError, prepare_image
from .schema import AnalysisResult
from .security import RateLimiter, SecurityHeadersMiddleware, client_ip

log = logging.getLogger("scene_analyzer")
STATIC_DIR = Path(__file__).parent / "static"
NO_STORE = {"Cache-Control": "no-store"}


@lru_cache(maxsize=1)
def get_analyzer() -> SceneAnalyzer:
    """Built on first use so the server can start (and /health can report) without a key."""
    return SceneAnalyzer(ClaudeBackend())


class Credentials(BaseModel):
    username: str = ""
    password: str = ""


class Registration(Credentials):
    code: str = ""


class PasswordChange(BaseModel):
    current: str = ""
    new: str = ""


class ActiveFlag(BaseModel):
    active: bool


def _resets_at() -> str:
    now = utc_now()
    return (dt.datetime(now.year, now.month, now.day) + dt.timedelta(days=1)).isoformat() + "Z"


def create_app(settings: Settings | None = None, db: Database | None = None) -> FastAPI:
    s = settings or Settings.from_env()

    problems = s.problems()
    if problems and s.require_auth:
        raise RuntimeError("Unsafe configuration:\n - " + "\n - ".join(problems))
    if problems:  # local use: fill the gaps with safe-for-localhost defaults and say so
        for p in problems:
            log.warning("Config notice (ignored because REQUIRE_AUTH is off): %s", p)
        s = dataclasses.replace(
            s,
            secret_key=s.secret_key if len(s.secret_key) >= 32 else secrets.token_urlsafe(48),
            open_registration=s.open_registration or not s.registration_code,
        )

    db = db or Database(s.database_url)
    sessions = Sessions(s.secret_key or secrets.token_urlsafe(48), s.session_days * 86400)

    # one-time start-up accounts
    if s.auth_disabled:
        local = db.get_user_by_name("local") or db.create_user("local", hash_password(secrets.token_urlsafe(24)), is_admin=True)
    elif s.admin_username:
        bad = check_username(s.admin_username) or check_password(s.admin_password, s.admin_username)
        if bad:
            raise RuntimeError(f"ADMIN_USERNAME / ADMIN_PASSWORD rejected: {bad}")
        admin = db.get_user_by_name(s.admin_username)
        if admin is None:
            db.create_user(s.admin_username, hash_password(s.admin_password), is_admin=True)
        elif not admin.is_admin or not admin.is_active:
            db.set_admin(admin.id, True)

    user_limit = 0 if s.auth_disabled else s.user_daily_limit
    per_user_rate = RateLimiter(s.rate_limit_per_min)
    everyone_rate = RateLimiter(s.global_rate_limit_per_min)
    register_rate = RateLimiter(10, 3600.0)       # sign-up attempts per IP per hour (also slows invite-code guessing)
    fail_by_ip = RateLimiter(10, 300.0)           # wrong passwords per IP per 5 minutes
    fail_by_user = RateLimiter(10, 900.0)         # wrong passwords per username per 15 minutes
    slots = threading.BoundedSemaphore(max(1, s.max_concurrent))

    app = FastAPI(
        title="Scene Analyzer",
        description="Structured scene analysis of images with a vision-language model (Claude).",
        version="2.0.0",
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.db = db

    @app.exception_handler(BackendError)
    async def _backend_error(_: Request, exc: BackendError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status_code, headers=NO_STORE)

    @app.exception_handler(ImageError)
    async def _image_error(_: Request, exc: ImageError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status_code, headers=NO_STORE)

    # ------------------------------------------------------------------ session helpers

    def user_from_request(request: Request) -> User | None:
        if s.auth_disabled:
            return local
        got = sessions.read(request.cookies.get(COOKIE_NAME))
        if not got:
            return None
        user = db.get_user(got[0])
        if user is None or not user.is_active or user.session_version != got[1]:
            return None
        return user

    def set_session(request: Request, response: Response, user: User) -> None:
        secure = s.cookie_secure if s.cookie_secure is not None else request.url.scheme == "https"
        response.set_cookie(COOKIE_NAME, sessions.issue(user.id, user.session_version), max_age=sessions.max_age_s,
                            httponly=True, samesite="lax", secure=secure, path="/")

    def require_user(request: Request) -> User:
        user = user_from_request(request)
        if user is None:
            raise HTTPException(401, "Sign in to continue.", headers=NO_STORE)
        return user

    def require_admin(user: User = Depends(require_user)) -> User:
        if not user.is_admin:
            raise HTTPException(403, "Administrators only.")
        return user

    def same_origin(request: Request) -> None:
        """Browsers send Origin on cross-site form posts; refuse any that is not this site (on top of SameSite=Lax)."""
        origin = request.headers.get("origin")
        if not origin:
            return
        allowed = {request.headers.get("host", ""), request.headers.get("x-forwarded-host", "")}
        if urlparse(origin).netloc not in allowed:
            raise HTTPException(403, "Cross-site request refused.")

    def me_payload(user: User) -> dict:
        return {
            "username": user.username,
            "is_admin": user.is_admin,
            "used_today": db.used_today(user.id),
            "daily_limit": user_limit or None,
            "resets_at": _resets_at(),
            "auth_disabled": s.auth_disabled,
        }

    # ------------------------------------------------------------------ pages

    def page(name: str) -> FileResponse:
        return FileResponse(STATIC_DIR / name, headers=NO_STORE)

    @app.get("/", include_in_schema=False)
    def index(request: Request):
        return page("index.html") if user_from_request(request) else RedirectResponse("/login", status_code=302)

    @app.get("/login", include_in_schema=False)
    def login_page(request: Request):
        return RedirectResponse("/", status_code=302) if user_from_request(request) else page("login.html")

    @app.get("/admin", include_in_schema=False)
    def admin_page(request: Request):
        user = user_from_request(request)
        if user is None:
            return RedirectResponse("/login", status_code=302)
        return page("admin.html") if user.is_admin else RedirectResponse("/", status_code=302)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": config.DEFAULT_MODEL,
            "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "accounts": not s.auth_disabled,
        }

    # ------------------------------------------------------------------ accounts

    @app.get("/api/config", include_in_schema=False)
    def public_config():
        """What the sign-in page needs to know before anyone is signed in."""
        return {"code_required": bool(s.registration_code), "min_password": 10, "daily_limit": user_limit or None}

    @app.get("/api/me")
    def me(user: User = Depends(require_user)):
        return JSONResponse(me_payload(user), headers=NO_STORE)

    @app.post("/api/register", dependencies=[Depends(same_origin)])
    def register(body: Registration, request: Request, response: Response):
        if s.auth_disabled:
            raise HTTPException(400, "Accounts are turned off on this server.")
        if not register_rate.allow(client_ip(request.scope)):
            raise HTTPException(429, "Too many sign-up attempts. Try again in an hour.")
        if s.registration_code and not secrets.compare_digest(body.code.strip().encode(), s.registration_code.encode()):
            raise HTTPException(403, "That invite code isn't valid. Ask the person who runs this site for the current one.")
        username = body.username.strip()
        bad = check_username(username) or check_password(body.password, username)
        if bad:
            raise HTTPException(400, bad)
        try:
            user = db.create_user(username, hash_password(body.password))
        except UsernameTaken:
            raise HTTPException(409, "That username is taken. Try another.")
        set_session(request, response, user)
        db.touch_login(user.id)
        return {"username": user.username}

    @app.post("/api/login", dependencies=[Depends(same_origin)])
    def login(body: Credentials, request: Request, response: Response):
        ip, name = client_ip(request.scope), body.username.strip().lower()
        if fail_by_ip.blocked(ip) or fail_by_user.blocked(name):
            raise HTTPException(429, "Too many wrong passwords. Wait a few minutes and try again.")
        user = db.get_user_by_name(name) if name else None
        ok = verify_password(body.password, user.password_hash) if user else (verify_dummy(body.password) or False)
        if not (user and ok):
            fail_by_ip.record(ip)
            fail_by_user.record(name)
            raise HTTPException(401, "Wrong username or password.")
        if not user.is_active:
            raise HTTPException(403, "This account has been disabled. Contact the person who runs this site.")
        set_session(request, response, user)
        db.touch_login(user.id)
        return {"username": user.username}

    @app.post("/api/logout", dependencies=[Depends(same_origin)])
    def logout(response: Response):
        response.delete_cookie(COOKIE_NAME, path="/")
        return {"ok": True}

    @app.post("/api/password", dependencies=[Depends(same_origin)])
    def change_password(body: PasswordChange, request: Request, response: Response, user: User = Depends(require_user)):
        if s.auth_disabled:
            raise HTTPException(400, "Accounts are turned off on this server.")
        if not verify_password(body.current, user.password_hash):
            raise HTTPException(403, "The current password is wrong.")
        bad = check_password(body.new, user.username)
        if bad:
            raise HTTPException(400, bad)
        version = db.set_password(user.id, hash_password(body.new))
        set_session(request, response, dataclasses.replace(user, session_version=version))  # other devices are signed out
        return {"ok": True}

    # ------------------------------------------------------------------ analysis

    @app.post("/analyze", response_model=AnalysisResult, dependencies=[Depends(same_origin)])
    async def analyze(
        file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, GIF, BMP, TIFF)."),
        focus: str | None = Form(None, description="Optional: what to pay extra attention to."),
        user: User = Depends(require_user),
        analyzer: SceneAnalyzer = Depends(get_analyzer),
    ):
        if not (per_user_rate.allow(str(user.id)) and everyone_rate.allow("*")):
            raise BackendError("Too many analyses in the last minute. Wait a little and try again.", 429)
        if not slots.acquire(blocking=False):
            raise BackendError("The server is busy with other analyses. Try again in a few seconds.", 503)
        try:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            focus = (focus or "").strip()[:500] or None
            prepared = await run_in_threadpool(prepare_image, data)  # bad files are rejected before they use any allowance

            usage_id, hit = await run_in_threadpool(db.reserve, user.id, user_limit, s.global_daily_limit)
            if hit == "user":
                raise BackendError(f"You've used your {user_limit} analyses for today. The count resets at midnight UTC.", 429)
            if hit == "global":
                raise BackendError("This site has reached its daily capacity. Try again tomorrow.", 429)
            try:
                result = await run_in_threadpool(analyzer.analyze_prepared, prepared, focus)
            except BaseException:
                await run_in_threadpool(db.refund, usage_id)  # a failed analysis doesn't count against the user
                raise
            await run_in_threadpool(db.finish, usage_id, result.meta.input_tokens, result.meta.output_tokens)
            return result
        finally:
            slots.release()

    # ------------------------------------------------------------------ admin

    def target(user_id: int) -> User:
        t = db.get_user(user_id)
        if t is None:
            raise HTTPException(404, "No such user.")
        if t.is_admin:
            raise HTTPException(400, "Administrator accounts can't be changed here.")
        return t

    @app.get("/api/admin/users")
    def admin_users(_: User = Depends(require_admin)):
        return JSONResponse({"users": db.list_users(), "daily_limit": user_limit or None,
                             "global_daily_limit": s.global_daily_limit or None}, headers=NO_STORE)

    @app.post("/api/admin/users/{user_id}/active", dependencies=[Depends(same_origin)])
    def admin_set_active(user_id: int, body: ActiveFlag, _: User = Depends(require_admin)):
        db.set_active(target(user_id).id, body.active)
        return {"ok": True}

    @app.post("/api/admin/users/{user_id}/reset-password", dependencies=[Depends(same_origin)])
    def admin_reset_password(user_id: int, _: User = Depends(require_admin)):
        temp = secrets.token_urlsafe(9)
        db.set_password(target(user_id).id, hash_password(temp))
        return JSONResponse({"temporary_password": temp}, headers=NO_STORE)

    @app.delete("/api/admin/users/{user_id}", dependencies=[Depends(same_origin)])
    def admin_delete(user_id: int, _: User = Depends(require_admin)):
        db.delete_user(target(user_id).id)
        return {"ok": True}

    app.add_middleware(SecurityHeadersMiddleware)
    return app


_default_app: FastAPI | None = None


def __getattr__(name: str):
    """`uvicorn app.main:app` builds the app from the environment on first access, not on import (so importing
    this module for tests does not create a database)."""
    global _default_app
    if name == "app":
        if _default_app is None:
            _default_app = create_app()
        return _default_app
    raise AttributeError(name)
