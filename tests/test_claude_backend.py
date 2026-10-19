"""Exercise the real ClaudeBackend against a mocked HTTP transport (no network, no real key)."""
import json

import anthropic
import httpx2 as httpx  # the Anthropic SDK (>=1.x) is built on httpx2, which ships with it
import pytest

from app.analyzer import SceneAnalyzer
from app.backends import BackendError, ClaudeBackend
from fakes import SAMPLE_RAW, png_bytes


def make_backend(handler) -> ClaudeBackend:
    backend = ClaudeBackend(api_key="sk-test")
    backend.client = anthropic.Anthropic(
        api_key="sk-test", max_retries=0, http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    return backend


def message(text="", stop_reason="end_turn"):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": text}], "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1234, "output_tokens": 567},
    }


def test_request_shape_and_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        seen["headers"] = request.headers
        return httpx.Response(200, json=message(json.dumps(SAMPLE_RAW)))

    result = SceneAnalyzer(make_backend(handler)).analyze_bytes(png_bytes(), focus="the ball")

    body = seen["body"]
    assert seen["headers"]["x-api-key"] == "sk-test"
    assert body["model"] == "claude-sonnet-5-5"
    assert "output_config" in body and body["output_config"]["format"]["type"] == "json_schema"
    assert body["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert "tool_choice" not in body  # forced tool use is not supported on Sonnet 5.5
    content = body["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image", "text"]  # image before text
    assert content[0]["source"]["type"] == "base64"
    assert "the ball" in content[1]["text"]
    assert result.analysis.objects[0].label == "golden retriever"
    assert (result.meta.input_tokens, result.meta.output_tokens) == (1234, 567)


def test_auth_error_is_fatal():
    def handler(_):
        return httpx.Response(401, json={"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})

    with pytest.raises(BackendError) as e:
        SceneAnalyzer(make_backend(handler)).analyze_bytes(png_bytes())
    assert e.value.fatal and "API key" in e.value.message


@pytest.mark.parametrize("stop_reason,status", [("refusal", 422), ("max_tokens", 502)])
def test_stop_reasons(stop_reason, status):
    with pytest.raises(BackendError) as e:
        SceneAnalyzer(make_backend(lambda _: httpx.Response(200, json=message("{}", stop_reason)))).analyze_bytes(png_bytes())
    assert e.value.status_code == status


def test_bad_json_from_model():
    with pytest.raises(BackendError):
        SceneAnalyzer(make_backend(lambda _: httpx.Response(200, json=message("not json")))).analyze_bytes(png_bytes())


def test_missing_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(BackendError) as e:
        ClaudeBackend()
    assert e.value.status_code == 503 and e.value.fatal
