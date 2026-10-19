import pytest

from app.backends import BackendError
from fakes import MockBackend, analyze, make_client, png_bytes, register


@pytest.fixture
def client(tmp_path):
    c = make_client(tmp_path)
    assert register(c).status_code == 200
    return c


def test_analyze_ok(client):
    r = analyze(client, data=png_bytes(), headers={})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"analysis", "scene_graph", "meta"}
    assert body["analysis"]["objects"][0]["id"] == "o1"
    assert body["scene_graph"]["edges"][0]["predicate"] == "chasing"
    assert body["meta"]["model"] == "mock-model"


def test_focus_is_passed_through(tmp_path):
    seen = {}

    class Spy(MockBackend):
        def analyze_image(self, image, system, user_text, schema, max_tokens=None):
            seen["user_text"] = user_text
            return super().analyze_image(image, system, user_text, schema, max_tokens)

    c = make_client(tmp_path, backend=Spy())
    register(c)
    r = c.post("/analyze", files={"file": ("x.png", png_bytes(), "image/png")}, data={"focus": "the ball"})
    assert r.status_code == 200 and "the ball" in seen["user_text"]


def test_non_image_rejected_before_the_model(tmp_path):
    backend = MockBackend()
    c = make_client(tmp_path, backend=backend)
    register(c)
    r = c.post("/analyze", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400 and "not a readable image" in r.json()["detail"]
    assert backend.calls == 0


def test_missing_file_is_422(client):
    assert client.post("/analyze").status_code == 422


def test_backend_error_mapped(tmp_path):
    c = make_client(tmp_path, backend=MockBackend(error=BackendError("slow down", 429)))
    register(c)
    r = analyze(c)
    assert r.status_code == 429 and r.json() == {"detail": "slow down"}


def test_health_is_public_and_reveals_nothing_secret(tmp_path):
    c = make_client(tmp_path)
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["accounts"] is True
    assert "secret" not in str(h).lower()


def test_pages_for_a_signed_in_user(client):
    page = client.get("/")
    assert page.status_code == 200 and "Scene Analyzer" in page.text
    assert page.headers["cache-control"] == "no-store"
    assert client.get("/openapi.json").status_code == 200
