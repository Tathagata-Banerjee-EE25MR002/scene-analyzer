"""Protection for public deployments: shared-password login, rate limits, security headers.

The password is the real defence. The rate limits are best-effort: they are in memory (per process)
and the client IP comes from the proxy's forwarding headers, so a determined caller can get around
the per-IP limit. The global limit and the concurrency cap still bound how much you can be charged.
"""
from __future__ import annotations

import base64
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Callable

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send


class RateLimiter:
    """Sliding-window counter keyed by string. `max_events <= 0` disables it."""

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
            q = self._live(key, self._clock())
            if len(q) >= self.max_events:
                return False
            q.append(self._clock())
            return True

    def blocked(self, key: str) -> bool:
        if self.max_events <= 0:
            return False
        with self._lock:
            return len(self._live(key, self._clock())) >= self.max_events

    def record(self, key: str) -> None:
        with self._lock:
            self._live(key, self._clock()).append(self._clock())


def client_ip(scope: Scope) -> str:
    client = scope.get("client")
    return client[0] if client else "unknown"


class BasicAuthMiddleware:
    """HTTP Basic login for every path except `exempt` (the health check). Browsers show their own prompt."""

    def __init__(self, app: ASGIApp, username: str, password: str, exempt: tuple[str, ...] = ("/health",)):
        self.app = app
        self._user = username.encode()
        self._pw = password.encode()
        self.exempt = exempt
        self._failures = RateLimiter(10, 300.0)  # 10 wrong attempts per 5 minutes per IP

    def _valid(self, header: str) -> bool:
        if not header.lower().startswith("basic "):
            return False
        try:
            user, _, pw = base64.b64decode(header[6:].strip(), validate=True).decode("utf-8").partition(":")
        except Exception:
            return False
        # evaluate both so timing does not reveal which one was wrong
        ok_user = secrets.compare_digest(user.encode(), self._user)
        ok_pw = secrets.compare_digest(pw.encode(), self._pw)
        return ok_user and ok_pw

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in self.exempt:
            return await self.app(scope, receive, send)

        ip = client_ip(scope)
        if self._failures.blocked(ip):
            return await Response("Too many failed sign-in attempts. Try again in a few minutes.", 429)(scope, receive, send)

        header = Headers(scope=scope).get("authorization", "")
        if self._valid(header):
            return await self.app(scope, receive, send)
        if header:  # credentials were sent and were wrong (a missing header is just the browser's first request)
            self._failures.record(ip)
        challenge = {"WWW-Authenticate": 'Basic realm="Scene Analyzer", charset="UTF-8"'}
        await Response("Sign in to use Scene Analyzer.", 401, headers=challenge)(scope, receive, send)


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

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for k, v in self.HEADERS.items():
                    headers.setdefault(k, v)
            await send(message)

        await self.app(scope, receive, send_with_headers)
