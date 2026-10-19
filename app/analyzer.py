"""Scene-analysis pipeline: preprocess -> VLM -> validate -> clean up -> scene graph."""
from __future__ import annotations

import math
import time
from pathlib import Path

from pydantic import ValidationError

from .backends import BackendError, VLMBackend
from .preprocess import PreparedImage, prepare_image
from .schema import (
    AnalysisResult,
    GraphEdge,
    GraphNode,
    ImageInfo,
    RunMeta,
    SceneAnalysis,
    SceneGraph,
    generation_schema,
)

SYSTEM_PROMPT = """You are a scene-analysis engine. You look at one image and return a factual, structured analysis as JSON matching the provided schema.

Rules
- Describe only what is visible. Never invent objects, text or events. When something is ambiguous, lower the confidence and note it in `uncertainties`.
- Objects: list the salient objects and people (typically 5-25; skip trivial background clutter). Give each a unique id "o1", "o2", ... and a short singular lowercase noun phrase as `label`. If individual instances are impractical to list (e.g. books on a shelf), use one object with an attribute named "count".
- Attributes: record visually grounded properties (color, material, size, state, pose, clothing, expression, orientation, count). Do not guess brands, prices or hidden properties.
- bbox: [x_min, y_min, x_max, y_max] as fractions of image width and height (0 to 1), origin at the top-left, tight around the visible extent. Use null when there is no clear extent (sky, a diffuse crowd).
- Relationships: connect objects by id. Left/right/above/below/behind/in front of are from the viewer's perspective of the image (left = left side of the picture). Cover the most informative relationships (spatial, functional, interaction, part-of), not every pair; at most about 40.
- Activities: describe what is happening (actions, interactions, events), referring to actors and targets by id. Return an empty list for a static scene.
- Context: infer scene type, indoor/outdoor, time of day, weather or lighting, atmosphere and likely purpose or event. Put the visible evidence in `clues`. Use "unclear" when the image does not support a conclusion.
- Text: transcribe legible text (signs, labels, screens) in reading order in `visible_text`. Skip illegible text.
- People: describe them by visible appearance, clothing, pose and action. Do not identify real people from their faces and do not infer sensitive personal attributes.
- caption: one sentence of at most 25 words. description: a coherent paragraph of 3-6 sentences.
- Confidence values are numbers from 0 to 1; use below 0.6 for guesses."""


def build_user_prompt(focus: str | None = None) -> str:
    text = "Analyze this image and fill in every field of the schema."
    if focus:
        text += (
            "\n\nThe user asked you to pay particular attention to the following "
            f"(still obey every rule above):\n{focus.strip()}"
        )
    return text


# --------------------------------------------------------------------------- post-processing


def _clean_bbox(bbox: list[float] | None, width: int, height: int) -> list[float] | None:
    if not bbox or len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        return None
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if max(x1, x2) > 1.5 or max(y1, y2) > 1.5:  # model answered in pixels
        x1, x2, y1, y2 = x1 / width, x2 / width, y1 / height, y2 / height
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1, y1, x2, y2 = (min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))
    if x2 - x1 < 0.005 or y2 - y1 < 0.005:
        return None
    return [round(v, 4) for v in (x1, y1, x2, y2)]


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, float(v))) if math.isfinite(v) else 0.0


def postprocess(analysis: SceneAnalysis, width: int, height: int) -> tuple[SceneAnalysis, list[str]]:
    """Repair what the schema cannot guarantee: unique ids, valid boxes, resolvable references."""
    a = analysis.model_copy(deep=True)
    warnings: list[str] = []

    # unique, non-empty object ids
    all_ids = {o.id for o in a.objects if o.id}
    seen: set[str] = set()
    counter = 0

    def fresh_id() -> str:
        nonlocal counter
        counter += 1
        while f"o{counter}" in all_ids or f"o{counter}" in seen:
            counter += 1
        return f"o{counter}"

    for obj in a.objects:
        if obj.id and obj.id not in seen:
            seen.add(obj.id)
            continue
        new_id = fresh_id()
        warnings.append(f"Duplicate or empty object id '{obj.id}' renamed to '{new_id}'.")
        obj.id = new_id
        seen.add(new_id)

    for obj in a.objects:
        obj.bbox = _clean_bbox(obj.bbox, width, height)
        obj.confidence = _clamp01(obj.confidence)

    # relationships must reference known, distinct objects; drop duplicates
    ids = {o.id for o in a.objects}
    kept, keys = [], set()
    for rel in a.relationships:
        if rel.subject_id not in ids or rel.object_id not in ids:
            warnings.append(f"Dropped relationship '{rel.subject_id} {rel.predicate} {rel.object_id}': unknown object id.")
            continue
        if rel.subject_id == rel.object_id:
            warnings.append(f"Dropped self-relationship on '{rel.subject_id}'.")
            continue
        key = (rel.subject_id, rel.predicate.strip().lower(), rel.object_id)
        if key in keys:
            continue
        keys.add(key)
        rel.confidence = _clamp01(rel.confidence)
        kept.append(rel)
    a.relationships = kept

    for act in a.activities:
        act.actor_ids = [i for i in act.actor_ids if i in ids]
        act.target_ids = [i for i in act.target_ids if i in ids]
        act.confidence = _clamp01(act.confidence)

    return a, warnings


def build_scene_graph(analysis: SceneAnalysis) -> SceneGraph:
    nodes = []
    for o in analysis.objects:
        attrs: dict[str, str] = {}
        for at in o.attributes:
            attrs[at.name] = f"{attrs[at.name]}; {at.value}" if at.name in attrs else at.value
        nodes.append(
            GraphNode(id=o.id, label=o.label, category=o.category, prominence=o.prominence, attributes=attrs)
        )
    edges = [
        GraphEdge(source=r.subject_id, target=r.object_id, predicate=r.predicate, kind=r.kind)
        for r in analysis.relationships
    ]
    return SceneGraph(nodes=nodes, edges=edges)


# --------------------------------------------------------------------------- pipeline


class SceneAnalyzer:
    def __init__(self, backend: VLMBackend):
        self.backend = backend
        self._schema = generation_schema()

    def analyze_prepared(self, image: PreparedImage, focus: str | None = None) -> AnalysisResult:
        start = time.perf_counter()
        raw = self.backend.analyze_image(image, SYSTEM_PROMPT, build_user_prompt(focus), self._schema)
        latency = time.perf_counter() - start

        try:
            analysis = SceneAnalysis.model_validate(raw.data)
        except ValidationError as exc:
            raise BackendError(f"Model output did not match the scene schema: {exc.error_count()} error(s).", 502) from exc

        analysis, warnings = postprocess(analysis, image.width, image.height)
        return AnalysisResult(
            analysis=analysis,
            scene_graph=build_scene_graph(analysis),
            meta=RunMeta(
                model=raw.model,
                latency_s=round(latency, 2),
                input_tokens=raw.input_tokens,
                output_tokens=raw.output_tokens,
                image=ImageInfo(
                    original_width=image.original_width,
                    original_height=image.original_height,
                    processed_width=image.width,
                    processed_height=image.height,
                    media_type=image.media_type,
                    resized=image.resized,
                ),
                warnings=warnings,
            ),
        )

    def analyze_bytes(self, data: bytes, focus: str | None = None) -> AnalysisResult:
        return self.analyze_prepared(prepare_image(data), focus)

    def analyze_path(self, path: str | Path, focus: str | None = None) -> AnalysisResult:
        return self.analyze_bytes(Path(path).read_bytes(), focus)
