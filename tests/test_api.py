import pytest
from fastapi.testclient import TestClient

from app.analyzer import SceneAnalyzer
from app.backends import BackendError
from app.main import app, get_analyzer
from fakes import MockBackend, png_bytes


@pytest.fixture
def client():
    backend = MockBackend()
    app.dependency_overrides[get_analyzer] = lambda: SceneAnalyzer(backend)
    c = TestClient(app)
    c.backend = backend
    yield c
    app.dependency_overrides.clear()


def post(client, data=None, filename="x.png", **form):
    files = {"file": (filename, data if data is not None else png_bytes(), "image/png")}
    return client.post("/analyze", files=files, data=form)


def test_analyze_ok(client):
    r = post(client, focus="the ball")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"analysis", "scene_graph", "meta"}
    assert body["analysis"]["objects"][0]["id"] == "o1"
    assert body["scene_graph"]["edges"][0]["predicate"] == "chasing"
    assert body["meta"]["model"] == "mock-model"


def test_non_image_rejected(client):
    r = post(client, data=b"hello", filename="x.txt")
    assert r.status_code == 400
    assert "not a readable image" in r.json()["detail"]
    assert client.backend.calls == 0


def test_missing_file_is_422(client):
    assert client.post("/analyze").status_code == 422


def test_backend_error_mapped(client):
    app.dependency_overrides[get_analyzer] = lambda: SceneAnalyzer(MockBackend(error=BackendError("slow down", 429)))
    r = post(client)
    assert r.status_code == 429
    assert r.json() == {"detail": "slow down"}


def test_missing_api_key_gives_503(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_analyzer.cache_clear()
    r = TestClient(app).post("/analyze", files={"file": ("x.png", png_bytes(), "image/png")})
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_health_and_ui(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and "model" in h
    page = client.get("/")
    assert page.status_code == 200 and "Scene Analyzer" in page.text
    assert client.get("/openapi.json").status_code == 200
