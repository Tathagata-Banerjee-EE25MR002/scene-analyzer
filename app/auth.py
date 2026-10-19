"""Passwords, usernames and session cookies."""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets

from itsdangerous import BadSignature, URLSafeTimedSerializer

COOKIE_NAME = "session"

_N, _R, _P = 2**14, 8, 1  # scrypt cost: about 16 MB and tens of milliseconds per hash

USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$")
MIN_PASSWORD = 10
MAX_PASSWORD = 128


# ---------------------------------------------------------------------------- passwords


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_N, r=_R, p=_P, dklen=32)
    return "scrypt${}${}${}${}${}".format(_N, _R, _P, base64.b64encode(salt).decode(), base64.b64encode(digest).decode())


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        got = hashlib.scrypt(password.encode(), salt=base64.b64decode(salt), n=int(n), r=int(r), p=int(p), dklen=len(expected))
    except Exception:
        return False
    return hmac.compare_digest(got, expected)


# A real hash to compare against when the user does not exist, so that "no such user" and "wrong password"
# take the same time and the response does not reveal which usernames exist.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def verify_dummy(password: str) -> None:
    verify_password(password, _DUMMY_HASH)


def check_username(username: str) -> str | None:
    """Return a problem description, or None if the username is acceptable."""
    if not USERNAME_RE.match(username):
        return "Use 3 to 32 letters, numbers, dots, dashes or underscores, starting with a letter or number."
    return None


def check_password(password: str, username: str = "") -> str | None:
    if len(password) < MIN_PASSWORD:
        return f"Use at least {MIN_PASSWORD} characters. A few random words work well."
    if len(password) > MAX_PASSWORD:
        return f"Use at most {MAX_PASSWORD} characters."
    if username and password.lower() == username.lower():
        return "The password can't be the same as the username."
    return None


# ---------------------------------------------------------------------------- sessions


class Sessions:
    """Signed, expiring cookies holding the user id and the user's session version."""

    def __init__(self, secret_key: str, max_age_s: int):
        self._ser = URLSafeTimedSerializer(secret_key, salt="scene-analyzer-session")
        self.max_age_s = max_age_s

    def issue(self, user_id: int, session_version: int) -> str:
        return self._ser.dumps({"u": user_id, "v": session_version})

    def read(self, token: str | None) -> tuple[int, int] | None:
        if not token:
            return None
        try:
            data = self._ser.loads(token, max_age=self.max_age_s)
            return int(data["u"]), int(data["v"])
        except (BadSignature, KeyError, TypeError, ValueError):
            return None
