"""FastAPI server: POST /analyze, GET /health, and a small web UI at /."""
from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .analyzer import SceneAnalyzer
from .backends import BackendError, ClaudeBackend
from .config import Settings
from .preprocess import MAX_UPLOAD_BYTES, ImageError
from .schema import AnalysisResult
from .security import BasicAuthMiddleware, RateLimiter, SecurityHeadersMiddleware, client_ip

STATIC_DIR = Path(__file__).parent / "static"


@lru_cache(maxsize=1)
def get_analyzer() -> SceneAnalyzer:
    """Built on first use so the server can start (and /health can report) without a key."""
    return SceneAnalyzer(ClaudeBackend())


def create_app(settings: Settings | None = None) -> FastAPI:
    s = settings or Settings.from_env()
    if s.require_password and not s.password:
        raise RuntimeError(
            "REQUIRE_PASSWORD is on but APP_PASSWORD is not set. Set APP_PASSWORD to a strong passphrase "
            "so that only people you share it with can spend your Claude API credits."
        )

    app = FastAPI(
        title="Scene Analyzer",
        description="Structured scene analysis of images with a vision-language model (Claude).",
        version="1.0.0",
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    per_ip = RateLimiter(s.rate_limit_per_min)
    everyone = RateLimiter(s.global_rate_limit_per_min)
    slots = threading.BoundedSemaphore(max(1, s.max_concurrent))

    @app.exception_handler(BackendError)
    async def _backend_error(_: Request, exc: BackendError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status_code)

    @app.exception_handler(ImageError)
    async def _image_error(_: Request, exc: ImageError):
        return JSONResponse({"detail": exc.message}, status_code=exc.status_code)

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": config.DEFAULT_MODEL,
            "api_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "password_protected": bool(s.password),
            "max_upload_mb": config.MAX_UPLOAD_MB,
        }

    @app.post("/analyze", response_model=AnalysisResult)
    async def analyze(
        request: Request,
        file: UploadFile = File(..., description="Image file (JPEG, PNG, WebP, GIF, BMP, TIFF)."),
        focus: str | None = Form(None, description="Optional: what to pay extra attention to."),
        analyzer: SceneAnalyzer = Depends(get_analyzer),
    ):
        ip = client_ip(request.scope)
        if not (per_ip.allow(ip) and everyone.allow("*")):
            raise BackendError("Too many analyses in the last minute. Wait a little and try again.", 429)
        if not slots.acquire(blocking=False):
            raise BackendError("The server is busy with other analyses. Try again in a few seconds.", 503)
        try:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            focus = (focus or "").strip()[:500] or None
            return await run_in_threadpool(analyzer.analyze_bytes, data, focus)
        finally:
            slots.release()

    # Last added = outermost, so security headers also cover the 401 responses.
    if s.password:
        app.add_middleware(BasicAuthMiddleware, username=s.username, password=s.password)
    app.add_middleware(SecurityHeadersMiddleware)
    return app


app = create_app()
