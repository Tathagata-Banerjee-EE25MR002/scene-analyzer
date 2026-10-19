"""Rate limiting and security headers."""
from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RateLimiter:
    """In-memory sliding-window counter keyed by string. `max_events <= 0` disables it.

    Best effort: per process, and the client IP comes from the proxy's forwarding headers. The accounts and
    the daily quotas in the database are the real protection.
    """

    def __init__(self, max_events: int, window_s: float = 60.0, clock: Callable[[], float] = time.monotonic):
        self.max_events = max_events
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _live(self, key: str, now: float) -> deque[float]:
        q = self._hits[key]
        while q and now - q[0] >= self.window_s:
            q.popleft()
        if len(self._hits) > 5000:  # drop idle keys so memory stays bounded
            for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window_s]:
                del self._hits[k]
            q = self._hits[key]
        return q

    def allow(self, key: str) -> bool:
        """Record one event for `key`; False (and nothing recorded) if the key is over its limit."""
        if self.max_events <= 0:
            return True
        with self._lock:
            now = self._clock()
            q = self._live(key, now)
            if len(q) >= self.max_events:
                return False
            q.append(now)
            return True

    def blocked(self, key: str) -> bool:
        if self.max_events <= 0:
            return False
        with self._lock:
            return len(self._live(key, self._clock())) >= self.max_events

    def record(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            self._live(key, now).append(now)


def client_ip(scope: Scope) -> str:
    client = scope.get("client")
    return client[0] if client else "unknown"


_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
_NO_CSP_PREFIXES = ("/docs", "/redoc", "/openapi.json")  # the built-in API docs load their UI from a CDN


class SecurityHeadersMiddleware:
    HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
    }

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        csp = not scope["path"].startswith(_NO_CSP_PREFIXES)

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for k, v in self.HEADERS.items():
                    headers.setdefault(k, v)
                if csp:
                    headers.setdefault("Content-Security-Policy", _CSP)
            await send(message)

        await self.app(scope, receive, send_with_headers)
