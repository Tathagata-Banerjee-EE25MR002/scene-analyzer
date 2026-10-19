"""Runtime configuration, read from the environment (and an optional .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:  # pragma: no cover
    pass

# Claude model used for scene analysis. Override with CLAUDE_MODEL.
DEFAULT_MODEL: str = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")

# Longest image side (px) sent to the model. Larger images are downscaled first.
# Raise for fine detail / small text (costs more tokens), lower for speed.
MAX_IMAGE_EDGE: int = int(os.environ.get("MAX_IMAGE_EDGE", "1568"))

# Largest accepted upload, in MB.
MAX_UPLOAD_MB: int = int(os.environ.get("MAX_UPLOAD_MB", "20"))

# Reject images with more pixels than this (megapixels). Keeps memory bounded on small hosts:
# a PNG/WebP is decoded in full (about 4 bytes per pixel); JPEGs are decoded at reduced size.
MAX_IMAGE_MEGAPIXELS: int = int(os.environ.get("MAX_IMAGE_MEGAPIXELS", "50"))

# Output token budget for one analysis.
MAX_OUTPUT_TOKENS: int = int(os.environ.get("MAX_OUTPUT_TOKENS", "8192"))

# Claude API call limits. Worst case a request takes about timeout x (retries + 1);
# keep that under your host's request timeout.
ANTHROPIC_TIMEOUT_S: float = float(os.environ.get("ANTHROPIC_TIMEOUT_S", "90"))
ANTHROPIC_MAX_RETRIES: int = int(os.environ.get("ANTHROPIC_MAX_RETRIES", "2"))


def _flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """Accounts, quotas and load limits (read at app creation time)."""

    # --- accounts and sessions
    database_url: str = "sqlite:///./data/scene_analyzer.db"
    secret_key: str = ""            # signs the session cookies; required unless auth_disabled
    auth_disabled: bool = False     # local use only: no login, one anonymous user
    require_auth: bool = False      # production guard (set in the container): refuse unsafe config at start-up
    session_days: int = 7
    cookie_secure: bool | None = None  # None = secure whenever the request arrived over https

    # --- who may register
    registration_code: str = ""     # invite code people must enter to create an account
    open_registration: bool = False  # allow sign-up without a code (anyone on the internet can spend your credits)
    admin_username: str = ""        # this account is created at start-up, with admin_password, if it is missing
    admin_password: str = ""

    # --- limits
    user_daily_limit: int = 20      # analyses per user per UTC day, 0 = unlimited
    global_daily_limit: int = 300   # analyses per UTC day across all users (spending backstop), 0 = unlimited
    rate_limit_per_min: int = 10    # analyses per client IP per minute, 0 = unlimited
    global_rate_limit_per_min: int = 60  # analyses per minute across everyone, 0 = unlimited
    max_concurrent: int = 4         # analyses running at once on this instance

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        secure = e("COOKIE_SECURE", "").strip().lower()
        return cls(
            database_url=e("DATABASE_URL", "").strip() or cls.database_url,
            secret_key=e("SECRET_KEY", ""),
            auth_disabled=_flag("AUTH_DISABLED"),
            require_auth=_flag("REQUIRE_AUTH"),
            session_days=int(e("SESSION_DAYS", "7")),
            cookie_secure=None if secure == "" else secure in {"1", "true", "yes", "on"},
            registration_code=e("REGISTRATION_CODE", ""),
            open_registration=_flag("OPEN_REGISTRATION"),
            admin_username=e("ADMIN_USERNAME", "").strip(),
            admin_password=e("ADMIN_PASSWORD", ""),
            user_daily_limit=int(e("USER_DAILY_LIMIT", "20")),
            global_daily_limit=int(e("GLOBAL_DAILY_LIMIT", "300")),
            rate_limit_per_min=int(e("RATE_LIMIT_PER_MIN", "10")),
            global_rate_limit_per_min=int(e("GLOBAL_RATE_LIMIT_PER_MIN", "60")),
            max_concurrent=int(e("MAX_CONCURRENT", "4")),
        )

    def problems(self) -> list[str]:
        """Reasons this configuration is unsafe or unusable. Fatal when `require_auth` is set."""
        out: list[str] = []
        if self.auth_disabled:
            if self.require_auth:
                out.append("AUTH_DISABLED cannot be used together with REQUIRE_AUTH.")
            return out
        if len(self.secret_key) < 32:
            out.append("SECRET_KEY must be set to a random string of at least 32 characters.")
        if not self.registration_code and not self.open_registration:
            out.append("Set REGISTRATION_CODE (an invite code people must enter to register), or set OPEN_REGISTRATION=1 to let anyone register.")
        if self.admin_username and len(self.admin_password) < 10:
            out.append("ADMIN_PASSWORD must be at least 10 characters when ADMIN_USERNAME is set.")
        if bool(self.admin_password) != bool(self.admin_username):
            out.append("Set both ADMIN_USERNAME and ADMIN_PASSWORD, or neither.")
        if self.require_auth and not os.environ.get("DATABASE_URL", "").strip():
            out.append("DATABASE_URL must be set, because accounts have to live in a database that survives restarts.")
        return out
