"""Shared test helpers: a canned model output and a mock backend (no network, no API key)."""
from __future__ import annotations

import copy
import io

from PIL import Image

from app.backends import BackendError, RawResult, VLMBackend

SAMPLE_RAW = {
    "caption": "A golden retriever chases a red ball across a sunny park.",
    "description": "A golden retriever runs across grass toward a red ball. A tree stands behind them. The light is bright and warm.",
    "context": {
        "scene_type": "city park",
        "environment": "outdoor",
        "time_of_day": "afternoon",
        "weather_or_lighting": "sunny",
        "atmosphere": "playful",
        "probable_purpose_or_event": "playing fetch",
        "clues": ["grass", "tree", "long shadows"],
    },
    "objects": [
        {"id": "o1", "label": "golden retriever", "category": "animal", "prominence": "primary",
         "attributes": [{"name": "color", "value": "golden"}, {"name": "pose", "value": "running"}],
         "bbox": [0.1, 0.4, 0.5, 0.9], "confidence": 0.97},
        {"id": "o2", "label": "red ball", "category": "other", "prominence": "secondary",
         "attributes": [{"name": "color", "value": "red"}],
         "bbox": [0.55, 0.7, 0.62, 0.78], "confidence": 0.9},
        {"id": "o3", "label": "oak tree", "category": "plant", "prominence": "background",
         "attributes": [], "bbox": None, "confidence": 0.8},
    ],
    "relationships": [
        {"subject_id": "o1", "predicate": "chasing", "object_id": "o2", "kind": "interaction", "confidence": 0.9},
        {"subject_id": "o1", "predicate": "to the left of", "object_id": "o2", "kind": "spatial", "confidence": 0.95},
        {"subject_id": "o3", "predicate": "behind", "object_id": "o1", "kind": "spatial", "confidence": 0.8},
    ],
    "activities": [
        {"description": "a dog chases a ball", "actor_ids": ["o1"], "target_ids": ["o2"], "confidence": 0.9},
    ],
    "visible_text": [],
    "uncertainties": ["Dog breed is a best guess."],
}


class MockBackend(VLMBackend):
    name = "mock"
    model = "mock-model"

    def __init__(self, data: dict | None = None, error: BackendError | None = None):
        self.data = data if data is not None else SAMPLE_RAW
        self.error = error
        self.calls = 0

    def analyze_image(self, image, system, user_text, schema, max_tokens=None) -> RawResult:
        self.calls += 1
        if self.error:
            raise self.error
        return RawResult(copy.deepcopy(self.data), self.model, 120, 60)


def png_bytes(size=(64, 48), color=(200, 30, 30), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, "PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------- app + accounts helpers

PW = "correct horse battery"


def db_url(tmp_path) -> str:
    """SQLite file per test by default. With TEST_DATABASE_URL (a Postgres maintenance database) each tmp_path
    gets its own Postgres database instead, so the same suite runs against production's database."""
    import hashlib
    import os

    pg = os.environ.get("TEST_DATABASE_URL")
    if not pg:
        return f"sqlite:///{tmp_path}/t.db"
    import sqlalchemy as sa

    name = "t_" + hashlib.md5(str(tmp_path).encode()).hexdigest()[:16]
    admin = sa.create_engine(pg.replace("postgresql://", "postgresql+psycopg://", 1), isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        if not c.execute(sa.text("select 1 from pg_database where datname = :n"), {"n": name}).first():
            c.execute(sa.text(f'create database "{name}"'))
    admin.dispose()
    return pg.rsplit("/", 1)[0] + "/" + name


def make_client(tmp_path, backend=None, base_url="http://testserver", **overrides):
    """A TestClient around a fresh app: its own SQLite file, the mock VLM, and test-friendly limits."""
    from fastapi.testclient import TestClient

    from app.analyzer import SceneAnalyzer
    from app.config import Settings
    from app.main import create_app, get_analyzer

    base = dict(database_url=db_url(tmp_path), secret_key="k" * 40, open_registration=True,
                user_daily_limit=20, global_daily_limit=0, rate_limit_per_min=0, global_rate_limit_per_min=0)
    app = create_app(Settings(**{**base, **overrides}))
    app.dependency_overrides[get_analyzer] = lambda: SceneAnalyzer(backend or MockBackend())
    c = TestClient(app, base_url=base_url)
    c.mock = backend
    return c


def another_device(client):
    """A second browser talking to the same app (its own cookie jar)."""
    from fastapi.testclient import TestClient

    return TestClient(client.app, base_url=str(client.base_url))


def register(client, username="alice", password=PW, **extra):
    return client.post("/api/register", json={"username": username, "password": password, **extra})


def login(client, username="alice", password=PW):
    return client.post("/api/login", json={"username": username, "password": password})


def analyze(client, data=None, **kw):
    return client.post("/analyze", files={"file": ("x.png", data if data is not None else png_bytes(), "image/png")}, **kw)
