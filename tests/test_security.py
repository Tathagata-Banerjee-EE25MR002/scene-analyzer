import threading

from app.backends import RawResult
from app.security import RateLimiter
from fakes import SAMPLE_RAW, MockBackend, analyze, another_device, make_client, register


# ---- RateLimiter
def test_rate_limiter_window():
    now = [0.0]
    rl = RateLimiter(2, 60, clock=lambda: now[0])
    assert rl.allow("a") and rl.allow("a")
    assert not rl.allow("a")          # over the limit
    assert rl.allow("b")              # other key unaffected
    now[0] = 61
    assert rl.allow("a")              # window slid


def test_rate_limiter_record_and_blocked():
    rl = RateLimiter(2, 60)
    assert not rl.blocked("a")
    rl.record("a"); rl.record("a")
    assert rl.blocked("a")


def test_rate_limiter_disabled():
    rl = RateLimiter(0)
    assert all(rl.allow("a") for _ in range(100)) and not rl.blocked("a")


# ---- per-user rate limit and the concurrency cap
def test_per_user_rate_limit_is_per_account(tmp_path):
    c = make_client(tmp_path, rate_limit_per_min=2)
    register(c, "alice")
    assert analyze(c).status_code == 200 and analyze(c).status_code == 200
    r = analyze(c)
    assert r.status_code == 429 and "Too many" in r.json()["detail"]
    c2 = another_device(c)
    register(c2, "bob")
    assert analyze(c2).status_code == 200   # bob has his own allowance, even from the same address


def test_global_rate_limit(tmp_path):
    c = make_client(tmp_path, global_rate_limit_per_min=1)
    register(c)
    assert analyze(c).status_code == 200
    assert analyze(c).status_code == 429


def test_concurrency_cap(tmp_path):
    started, release = threading.Event(), threading.Event()

    class Blocking(MockBackend):
        def analyze_image(self, *a, **kw):
            started.set()
            release.wait(5)
            return RawResult(SAMPLE_RAW, self.model, 1, 1)

    c = make_client(tmp_path, backend=Blocking(), max_concurrent=1)
    register(c)
    first = {}
    t = threading.Thread(target=lambda: first.update(r=analyze(c)))
    t.start()
    assert started.wait(5)
    busy = analyze(c)
    assert busy.status_code == 503 and "busy" in busy.json()["detail"]
    release.set(); t.join(5)
    assert first["r"].status_code == 200
    assert analyze(c).status_code == 200      # slot was released


# ---- headers
def test_security_headers_everywhere(tmp_path):
    c = make_client(tmp_path)
    for r in (c.get("/login"), c.get("/health"), c.get("/", follow_redirects=False)):
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
    assert "content-security-policy" not in c.get("/docs").headers   # the built-in docs load from a CDN
