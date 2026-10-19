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
