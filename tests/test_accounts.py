import pytest

from app import auth
from app.backends import BackendError
from app.config import Settings
from app.db import Database, normalize_url
from app.main import create_app
from fakes import PW, MockBackend, analyze, another_device, db_url, login, make_client, png_bytes, register


@pytest.fixture
def client(tmp_path):
    return make_client(tmp_path)


# ---------------------------------------------------------------------------- passwords and sessions


def test_password_hashing():
    h = auth.hash_password("s3cret-passphrase")
    assert h.startswith("scrypt$") and "s3cret" not in h
    assert auth.verify_password("s3cret-passphrase", h)
    assert not auth.verify_password("s3cret-passphrasE", h)
    assert auth.hash_password("s3cret-passphrase") != h          # random salt
    for junk in ("", "plain", "scrypt$x$y", "md5$1$2$3$4$5"):
        assert not auth.verify_password("anything", junk)


def test_session_tokens():
    s = auth.Sessions("k" * 40, 3600)
    assert s.read(s.issue(7, 3)) == (7, 3)
    assert s.read(None) is None and s.read("garbage") is None
    assert s.read(s.issue(7, 3) + "x") is None                                      # tampered
    assert auth.Sessions("other" * 10, 3600).read(s.issue(7, 3)) is None           # signed with another key
    assert auth.Sessions("k" * 40, -1).read(s.issue(7, 3)) is None                 # expired


@pytest.mark.parametrize("name,ok", [("alice", True), ("A.b-c_9", True), ("ab", False), ("-bob", False), ("has space", False), ("x" * 33, False), ("ünï", False)])
def test_username_rules(name, ok):
    assert (auth.check_username(name) is None) == ok


def test_password_rules():
    assert auth.check_password("short") and auth.check_password("x" * 200)
    assert auth.check_password("alicealice1", "AliceAlice1")
    assert auth.check_password(PW, "alice") is None


# ---------------------------------------------------------------------------- the signed-out experience


def test_signed_out_visitors_get_nothing_useful(client):
    assert client.get("/", follow_redirects=False).headers["location"] == "/login"
    assert client.get("/admin", follow_redirects=False).headers["location"] == "/login"
    assert client.get("/login").status_code == 200
    assert client.get("/api/me").status_code == 401
    assert analyze(client).status_code == 401
    assert client.get("/api/admin/users").status_code == 401
    assert client.get("/health").status_code == 200
    assert client.get("/api/config").json()["code_required"] is False


def test_no_analysis_happens_without_signing_in(tmp_path):
    backend = MockBackend()
    c = make_client(tmp_path, backend=backend)
    analyze(c)
    assert backend.calls == 0


# ---------------------------------------------------------------------------- register and sign in


def test_register_signs_you_in(client):
    r = register(client)
    assert r.status_code == 200 and r.json() == {"username": "alice"}
    me = client.get("/api/me").json()
    assert me["username"] == "alice" and me["is_admin"] is False
    assert me["used_today"] == 0 and me["daily_limit"] == 20 and me["resets_at"].endswith("Z")
    assert client.get("/login", follow_redirects=False).headers["location"] == "/"     # no need to sign in again
    assert analyze(client).status_code == 200


def test_usernames_are_case_insensitive_and_unique(client):
    assert register(client, "Alice").status_code == 200
    other = another_device(client)
    assert register(other, "alice").status_code == 409
    assert register(other, "ALICE").status_code == 409
    assert login(other, "aLiCe").status_code == 200


@pytest.mark.parametrize("name,pw", [("ab", PW), ("alice", "short"), ("alice", "alice"), ("bad name", PW)])
def test_registration_validation(client, name, pw):
    r = register(client, name, pw)
    assert r.status_code == 400 and isinstance(r.json()["detail"], str)
    assert client.get("/api/me").status_code == 401       # nothing was created


def test_password_is_not_stored_in_clear(tmp_path):
    c = make_client(tmp_path)
    register(c)
    stored = c.app.state.db.get_user_by_name("alice").password_hash
    assert PW not in stored and stored.startswith("scrypt$")


def test_login_and_logout(client):
    register(client)
    client.post("/api/logout")
    assert client.get("/api/me").status_code == 401
    assert login(client, password="wrong password").status_code == 401
    assert login(client).status_code == 200
    assert client.get("/api/me").json()["username"] == "alice"


def test_login_errors_do_not_reveal_which_usernames_exist(client):
    register(client)
    other = another_device(client)
    a = login(other, "alice", "wrong password")
    b = login(other, "nobody", "wrong password")
    assert a.status_code == b.status_code == 401 and a.json() == b.json()


def test_repeated_wrong_passwords_lock_the_account_for_a_while(client):
    register(client)
    other = another_device(client)
    for _ in range(10):
        assert login(other, "alice", "wrong password").status_code == 401
    r = login(other, "alice", PW)
    assert r.status_code == 429 and "Too many" in r.json()["detail"]


def test_forged_cookie_is_rejected(client):
    client.cookies.set("session", "totally.forged.value")
    assert client.get("/api/me").status_code == 401


def test_session_cookie_flags(tmp_path):
    http = make_client(tmp_path / "a")
    cookie = register(http).headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "secure" not in cookie
    https = make_client(tmp_path / "b", base_url="https://testserver")
    assert "secure" in register(https).headers["set-cookie"].lower()


# ---------------------------------------------------------------------------- invite code


def test_invite_code_required_when_configured(tmp_path):
    c = make_client(tmp_path, open_registration=False, registration_code="open-sesame")
    assert c.get("/api/config").json()["code_required"] is True
    assert register(c).status_code == 403
    assert register(c, code="wrong").status_code == 403
    assert register(c, code=" open-sesame ").status_code == 200      # stray spaces from copy and paste are fine
    assert c.get("/api/me").status_code == 200


def test_registration_attempts_are_rate_limited(client):
    for i in range(10):
        assert register(client, f"user{i:02d}", "short").status_code == 400
    assert register(client, "legit-user").status_code == 429


# ---------------------------------------------------------------------------- disabling, passwords, cross-site


def test_disabled_users_are_locked_out_immediately(tmp_path):
    c = make_client(tmp_path)
    register(c)
    db = c.app.state.db
    db.set_active(db.get_user_by_name("alice").id, False)
    assert c.get("/api/me").status_code == 401                      # the existing session stops working
    r = login(c)
    assert r.status_code == 403 and "disabled" in r.json()["detail"]
    assert analyze(c).status_code == 401


def test_change_password(client):
    register(client)
    phone = another_device(client)
    login(phone)
    new = "a brand new passphrase"
    assert client.post("/api/password", json={"current": "wrong", "new": new}).status_code == 403
    assert client.post("/api/password", json={"current": PW, "new": "short"}).status_code == 400
    assert client.post("/api/password", json={"current": PW, "new": new}).status_code == 200
    assert client.get("/api/me").status_code == 200                 # this device stays signed in
    assert phone.get("/api/me").status_code == 401                  # the other device is signed out
    other = another_device(client)
    assert login(other, password=PW).status_code == 401
    assert login(other, password=new).status_code == 200


def test_cross_site_posts_are_refused(client):
    register(client)
    evil = {"Origin": "https://evil.example"}
    assert client.post("/api/login", json={"username": "alice", "password": PW}, headers=evil).status_code == 403
    assert client.post("/api/logout", headers=evil).status_code == 403
    assert analyze(client, headers=evil).status_code == 403
    assert client.get("/api/me").status_code == 200                 # still signed in: the forged logout did nothing
    ok = {"Origin": "http://testserver"}
    assert analyze(client, headers=ok).status_code == 200


# ---------------------------------------------------------------------------- quotas


def test_daily_limit_per_user(tmp_path):
    backend = MockBackend()
    c = make_client(tmp_path, backend=backend, user_daily_limit=2)
    register(c)
    assert analyze(c).status_code == 200 and analyze(c).status_code == 200
    r = analyze(c)
    assert r.status_code == 429 and "2 analyses" in r.json()["detail"] and "midnight UTC" in r.json()["detail"]
    assert backend.calls == 2                                       # the third never reached the model
    assert c.get("/api/me").json()["used_today"] == 2
    bob = another_device(c)
    register(bob, "bob")
    assert analyze(bob).status_code == 200                          # limits are per person


def test_global_daily_limit_is_a_spending_backstop(tmp_path):
    c = make_client(tmp_path, global_daily_limit=2)
    register(c, "alice")
    bob = another_device(c)
    register(bob, "bob")
    assert analyze(c).status_code == 200 and analyze(bob).status_code == 200
    r = analyze(c)
    assert r.status_code == 429 and "capacity" in r.json()["detail"]


def test_failed_and_invalid_analyses_are_free(tmp_path):
    c = make_client(tmp_path, backend=MockBackend(error=BackendError("model down", 503)), user_daily_limit=1)
    register(c)
    assert c.post("/analyze", files={"file": ("x.txt", b"nope", "text/plain")}).status_code == 400
    assert analyze(c).status_code == 503
    assert analyze(c).status_code == 503                            # still allowed: neither attempt used the allowance
    assert c.get("/api/me").json()["used_today"] == 0


def test_usage_records_tokens(tmp_path):
    c = make_client(tmp_path)
    register(c)
    analyze(c)
    from app.db import usage
    with c.app.state.db.engine.connect() as conn:
        row = conn.execute(usage.select()).one()
    assert (row.input_tokens, row.output_tokens) == (120, 60)


def test_unlimited_when_limit_is_zero(tmp_path):
    c = make_client(tmp_path, user_daily_limit=0)
    register(c)
    assert c.get("/api/me").json()["daily_limit"] is None
    assert all(analyze(c).status_code == 200 for _ in range(3))


# ---------------------------------------------------------------------------- admin


@pytest.fixture
def admin_setup(tmp_path):
    c = make_client(tmp_path, admin_username="boss", admin_password="boss-passphrase-1")
    boss = another_device(c)
    assert login(boss, "boss", "boss-passphrase-1").status_code == 200
    register(c, "alice")
    return c, boss


def test_admin_account_is_created_from_settings(admin_setup):
    _, boss = admin_setup
    assert boss.get("/api/me").json()["is_admin"] is True
    assert boss.get("/admin").status_code == 200
    r = register(another_device(boss), "boss")
    assert r.status_code == 409                                     # nobody can grab the admin's name


def test_admin_restart_does_not_reset_the_admin_password(tmp_path):
    kw = dict(database_url=db_url(tmp_path), secret_key="k" * 40, open_registration=True,
              admin_username="boss", admin_password="boss-passphrase-1")
    create_app(Settings(**kw))
    c = make_client(tmp_path, admin_username="boss", admin_password="boss-passphrase-1")
    boss = another_device(c)
    assert login(boss, "boss", "boss-passphrase-1").status_code == 200
    assert boss.post("/api/password", json={"current": "boss-passphrase-1", "new": "changed passphrase 2"}).status_code == 200
    c2 = make_client(tmp_path, admin_username="boss", admin_password="boss-passphrase-1")   # restart with the old env value
    assert login(another_device(c2), "boss", "changed passphrase 2").status_code == 200


def test_regular_users_cannot_reach_admin(admin_setup):
    alice, _ = admin_setup
    assert alice.get("/admin", follow_redirects=False).headers["location"] == "/"
    assert alice.get("/api/admin/users").status_code == 403
    assert alice.post("/api/admin/users/1/active", json={"active": False}).status_code == 403
    assert alice.delete("/api/admin/users/1").status_code == 403


def test_admin_lists_users_with_usage(admin_setup):
    alice, boss = admin_setup
    analyze(alice); analyze(alice)
    d = boss.get("/api/admin/users").json()
    assert d["daily_limit"] == 20
    by = {u["username"]: u for u in d["users"]}
    assert by["alice"]["used_today"] == 2 and by["alice"]["total_used"] == 2 and by["alice"]["is_active"] is True
    assert by["boss"]["is_admin"] is True and by["boss"]["total_used"] == 0
    assert "password" not in str(d).lower()


def test_admin_can_disable_and_enable(admin_setup):
    alice, boss = admin_setup
    uid = next(u["id"] for u in boss.get("/api/admin/users").json()["users"] if u["username"] == "alice")
    assert boss.post(f"/api/admin/users/{uid}/active", json={"active": False}).status_code == 200
    assert alice.get("/api/me").status_code == 401
    assert boss.post(f"/api/admin/users/{uid}/active", json={"active": True}).status_code == 200
    assert login(alice).status_code == 200


def test_admin_password_reset(admin_setup):
    alice, boss = admin_setup
    uid = next(u["id"] for u in boss.get("/api/admin/users").json()["users"] if u["username"] == "alice")
    temp = boss.post(f"/api/admin/users/{uid}/reset-password").json()["temporary_password"]
    assert len(temp) >= 10 and auth.check_password(temp, "alice") is None
    assert alice.get("/api/me").status_code == 401                  # signed out everywhere
    fresh = another_device(alice)
    assert login(fresh, password=PW).status_code == 401
    assert login(fresh, password=temp).status_code == 200


def test_admin_can_delete_users_and_their_history(admin_setup):
    alice, boss = admin_setup
    analyze(alice)
    uid = next(u["id"] for u in boss.get("/api/admin/users").json()["users"] if u["username"] == "alice")
    assert boss.delete(f"/api/admin/users/{uid}").status_code == 200
    assert alice.get("/api/me").status_code == 401
    assert [u["username"] for u in boss.get("/api/admin/users").json()["users"]] == ["boss"]
    from app.db import usage
    with boss.app.state.db.engine.connect() as conn:
        assert conn.execute(usage.select()).first() is None
    assert register(alice).status_code == 200                       # the name is free again


def test_admin_cannot_touch_admins_or_missing_users(admin_setup):
    _, boss = admin_setup
    me_id = next(u["id"] for u in boss.get("/api/admin/users").json()["users"] if u["username"] == "boss")
    assert boss.post(f"/api/admin/users/{me_id}/active", json={"active": False}).status_code == 400
    assert boss.delete(f"/api/admin/users/{me_id}").status_code == 400
    assert boss.post(f"/api/admin/users/{me_id}/reset-password").status_code == 400
    assert boss.delete("/api/admin/users/9999").status_code == 404


def test_admin_actions_refuse_cross_site_requests(admin_setup):
    _, boss = admin_setup
    uid = next(u["id"] for u in boss.get("/api/admin/users").json()["users"] if u["username"] == "alice")
    assert boss.delete(f"/api/admin/users/{uid}", headers={"Origin": "https://evil.example"}).status_code == 403


# ---------------------------------------------------------------------------- configuration


def test_production_mode_refuses_unsafe_settings(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(RuntimeError) as e:
        create_app(Settings(require_auth=True))
    msg = str(e.value)
    assert "SECRET_KEY" in msg and "REGISTRATION_CODE" in msg and "DATABASE_URL" in msg
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app(Settings(require_auth=True, secret_key="short", open_registration=True))
    with pytest.raises(RuntimeError, match="AUTH_DISABLED"):
        create_app(Settings(require_auth=True, auth_disabled=True))
    with pytest.raises(RuntimeError, match="ADMIN_PASSWORD"):
        create_app(Settings(require_auth=True, secret_key="k" * 40, open_registration=True, admin_username="boss"))


def test_production_mode_accepts_a_complete_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p.db")
    app = create_app(Settings(require_auth=True, secret_key="k" * 40, registration_code="code", database_url=f"sqlite:///{tmp_path}/p.db"))
    assert app.title == "Scene Analyzer"


def test_local_defaults_work_without_any_setup(tmp_path):
    c = make_client(tmp_path, secret_key="", open_registration=False, registration_code="")   # nothing configured
    assert register(c).status_code == 200                           # localhost convenience: open sign-up, random key


def test_auth_disabled_mode_for_local_use(tmp_path):
    c = make_client(tmp_path, auth_disabled=True, open_registration=False, secret_key="")
    assert c.get("/").status_code == 200
    me = c.get("/api/me").json()
    assert me["auth_disabled"] is True and me["daily_limit"] is None
    assert analyze(c).status_code == 200
    assert register(c).status_code == 400


# ---------------------------------------------------------------------------- database


def test_database_urls_are_normalized():
    assert normalize_url("postgres://u:p@h:5432/d") == "postgresql+psycopg://u:p@h:5432/d"
    assert normalize_url("postgresql://u:p@h/d?sslmode=require") == "postgresql+psycopg://u:p@h/d?sslmode=require"
    assert normalize_url("postgresql+psycopg://u:p@h/d") == "postgresql+psycopg://u:p@h/d"
    assert normalize_url("sqlite:///x.db") == "sqlite:///x.db"


def test_data_survives_a_restart(tmp_path):
    url = f"sqlite:///{tmp_path}/keep.db"
    db = Database(url)
    db.create_user("alice", auth.hash_password(PW))
    db.engine.dispose()
    assert Database(url).get_user_by_name("alice") is not None


def test_creates_the_database_folder(tmp_path):
    db = Database(f"sqlite:///{tmp_path}/nested/deeper/x.db")
    assert db.ping()
