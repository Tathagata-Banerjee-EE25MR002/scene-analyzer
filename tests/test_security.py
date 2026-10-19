import threading

import pytest
from fastapi.testclient import TestClient

from app.analyzer import SceneAnalyzer
from app.backends import RawResult
from app.config import Settings
from app.main import create_app, get_analyzer
from app.security import RateLimiter
from fakes import SAMPLE_RAW, MockBackend, png_bytes

AUTH = ("user", "correct horse battery staple")


def make_client(backend=None, **settings) -> TestClient:
    app = create_app(Settings(**settings))
    app.dependency_overrides[get_analyzer] = lambda: SceneAnalyzer(backend or MockBackend())
    return TestClient(app)


def post(client, **kw):
    return client.post("/analyze", files={"file": ("x.png", png_bytes(), "image/png")}, **kw)


# ---- RateLimiter
def test_rate_limiter_window():
    now = [0.0]
    rl = RateLimiter(2, 60, clock=lambda: now[0])
    assert rl.allow("a") and rl.allow("a")
    assert not rl.allow("a")          # over the limit
    assert rl.allow("b")              # other key unaffected
    now[0] = 61
    assert rl.allow("a")              # window slid


def test_rate_limiter_disabled():
    rl = RateLimiter(0)
    assert all(rl.allow("a") for _ in range(100)) and not rl.blocked("a")


# ---- password protection
def test_no_password_means_open():
    c = make_client()
    assert c.get("/").status_code == 200
    assert c.get("/health").json()["password_protected"] is False


def test_password_required_for_everything_but_health():
    c = make_client(password=AUTH[1])
    r = c.get("/")
    assert r.status_code == 401 and "Basic" in r.headers["www-authenticate"]
    assert c.get("/docs").status_code == 401
    assert post(c).status_code == 401
    assert c.get("/health").status_code == 200
    assert c.get("/health").json()["password_protected"] is True


def test_correct_and_wrong_credentials():
    c = make_client(password=AUTH[1])
    assert c.get("/", auth=AUTH).status_code == 200
    assert post(c, auth=AUTH).status_code == 200
    assert c.get("/", auth=("user", "wrong")).status_code == 401
    assert c.get("/", auth=("someone", AUTH[1])).status_code == 401


def test_non_ascii_password():
    c = make_client(username="ñandú", password="pässwörd✓")
    assert c.get("/", auth=("ñandú", "pässwörd✓")).status_code == 200
    assert c.get("/", auth=("ñandú", "password")).status_code == 401


def test_repeated_wrong_passwords_lock_out():
    c = make_client(password=AUTH[1])
    for _ in range(10):
        assert c.get("/", auth=("user", "guess")).status_code == 401
    assert c.get("/", auth=AUTH).status_code == 429   # even the right password waits
    assert c.get("/health").status_code == 200         # health checks keep working


def test_missing_header_is_not_counted_as_failure():
    c = make_client(password=AUTH[1])
    for _ in range(30):
        assert c.get("/").status_code == 401
    assert c.get("/", auth=AUTH).status_code == 200


# ---- fail fast
def test_require_password_refuses_to_start_without_one():
    with pytest.raises(RuntimeError, match="APP_PASSWORD"):
        create_app(Settings(require_password=True))
    create_app(Settings(require_password=True, password="x"))  # fine


# ---- limits
def test_per_ip_rate_limit():
    c = make_client(rate_limit_per_min=2)
    assert post(c).status_code == 200 and post(c).status_code == 200
    r = post(c)
    assert r.status_code == 429 and "Too many" in r.json()["detail"]


def test_global_rate_limit():
    c = make_client(rate_limit_per_min=0, global_rate_limit_per_min=1)
    assert post(c).status_code == 200
    assert post(c).status_code == 429


def test_concurrency_cap():
    started, release = threading.Event(), threading.Event()

    class Blocking(MockBackend):
        def analyze_image(self, *a, **kw):
            started.set()
            release.wait(5)
            return RawResult(SAMPLE_RAW, self.model, 1, 1)

    c = make_client(Blocking(), max_concurrent=1, rate_limit_per_min=0)
    first = {}
    t = threading.Thread(target=lambda: first.update(r=post(c)))
    t.start()
    assert started.wait(5)
    busy = post(c)
    assert busy.status_code == 503 and "busy" in busy.json()["detail"]
    release.set(); t.join(5)
    assert first["r"].status_code == 200
    assert post(c).status_code == 200      # slot was released


# ---- headers
def test_security_headers_on_all_responses():
    c = make_client(password=AUTH[1])
    for r in (c.get("/"), c.get("/", auth=AUTH), c.get("/health")):
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
