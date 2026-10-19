"""VLM backends. ClaudeBackend calls the Claude API; subclass VLMBackend to add others."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

from . import config
from .preprocess import PreparedImage


class BackendError(Exception):
    """A failure talking to the model. `fatal` means retrying other images will not help."""

    def __init__(self, message: str, status_code: int = 502, fatal: bool = False):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.fatal = fatal


@dataclass
class RawResult:
    data: dict
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class VLMBackend(ABC):
    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def analyze_image(
        self,
        image: PreparedImage,
        system: str,
        user_text: str,
        schema: dict,
        max_tokens: int | None = None,
    ) -> RawResult:
        """Send one image plus instructions; return JSON conforming to `schema`."""


class ClaudeBackend(VLMBackend):
    name = "claude"

    def __init__(self, api_key: str | None = None, model: str | None = None, max_tokens: int | None = None):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise BackendError(
                "ANTHROPIC_API_KEY is not set. Put it in scene_analyzer/.env or export it, then restart.",
                status_code=503,
                fatal=True,
            )
        import anthropic

        self._anthropic = anthropic
        self.client = anthropic.Anthropic(
            api_key=key, max_retries=config.ANTHROPIC_MAX_RETRIES, timeout=config.ANTHROPIC_TIMEOUT_S
        )
        self.model = model or config.DEFAULT_MODEL
        self.max_tokens = max_tokens or config.MAX_OUTPUT_TOKENS

    def analyze_image(self, image, system, user_text, schema, max_tokens=None) -> RawResult:
        a = self._anthropic
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                system=system,
                messages=[
                    {
                        "role": "user",
                        # image first, then text: the order Anthropic recommends
                        "content": [image.to_content_block(), {"type": "text", "text": user_text}],
                    }
                ],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except a.AuthenticationError as exc:
            raise BackendError("Anthropic rejected the API key (401). Check ANTHROPIC_API_KEY.", 502, fatal=True) from exc
        except a.PermissionDeniedError as exc:
            raise BackendError("This API key is not allowed to use that model or feature (403).", 502, fatal=True) from exc
        except a.NotFoundError as exc:
            raise BackendError(f"Model '{self.model}' was not found. Check CLAUDE_MODEL.", 502, fatal=True) from exc
        except a.RateLimitError as exc:
            raise BackendError("Anthropic rate limit hit (429). Retry shortly.", 429) from exc
        except a.BadRequestError as exc:
            raise BackendError(f"Anthropic rejected the request: {exc.message}", 502) from exc
        except a.APIConnectionError as exc:
            raise BackendError("Could not reach the Anthropic API (network error or timeout).", 503) from exc
        except a.APIStatusError as exc:
            raise BackendError(f"Anthropic API error {exc.status_code}: {exc.message}", 502) from exc

        if resp.stop_reason == "refusal":
            raise BackendError("The model declined to analyze this image.", 422)
        if resp.stop_reason == "max_tokens":
            raise BackendError("The model output was truncated; increase MAX_OUTPUT_TOKENS.", 502)

        text = next((b.text for b in resp.content if b.type == "text"), None)
        if not text:
            raise BackendError("The model returned no text.", 502)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise BackendError("The model returned invalid JSON.", 502) from exc

        return RawResult(
            data=data,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
