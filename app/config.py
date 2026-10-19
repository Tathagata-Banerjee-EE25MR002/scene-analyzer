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
    """Server-side access control and load limits (read at app creation time)."""

    username: str = "user"
    password: str = ""            # empty = no login (fine on localhost, not on the internet)
    require_password: bool = False  # refuse to start without a password (set in the container)
    rate_limit_per_min: int = 10    # analyses per client IP per minute, 0 = unlimited
    global_rate_limit_per_min: int = 60  # analyses per minute across everyone, 0 = unlimited
    max_concurrent: int = 4         # analyses running at once on this instance

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            username=os.environ.get("APP_USERNAME", "user"),
            password=os.environ.get("APP_PASSWORD", ""),
            require_password=_flag("REQUIRE_PASSWORD"),
            rate_limit_per_min=int(os.environ.get("RATE_LIMIT_PER_MIN", "10")),
            global_rate_limit_per_min=int(os.environ.get("GLOBAL_RATE_LIMIT_PER_MIN", "60")),
            max_concurrent=int(os.environ.get("MAX_CONCURRENT", "4")),
        )
