"""Structured scene representation (Pydantic models) and the JSON schema sent to the model."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Category = Literal[
    "person",
    "animal",
    "vehicle",
    "furniture",
    "food_drink",
    "plant",
    "building_structure",
    "electronics",
    "clothing_accessory",
    "tool_equipment",
    "sign_or_text",
    "natural_feature",
    "other",
]
Prominence = Literal["primary", "secondary", "background"]
RelationKind = Literal["spatial", "functional", "interaction", "part_of", "comparative", "other"]
Environment = Literal["indoor", "outdoor", "mixed", "unclear"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- model output


class Attribute(_Strict):
    name: str = Field(description="Attribute type: color, material, size, state, pose, clothing, expression, orientation, count, ...")
    value: str


class SceneObject(_Strict):
    id: str = Field(description="Unique id: 'o1', 'o2', ...")
    label: str = Field(description="Short singular lowercase noun phrase, e.g. 'golden retriever', 'wooden bench'.")
    category: Category
    prominence: Prominence
    attributes: list[Attribute]
    bbox: Optional[list[float]] = Field(
        description="[x_min, y_min, x_max, y_max] as fractions (0-1) of image width/height, origin top-left. "
        "null if the object has no clear extent (sky, diffuse crowd)."
    )
    confidence: float = Field(description="0-1; below 0.6 for guesses.")


class Relationship(_Strict):
    subject_id: str = Field(description="id of the subject object.")
    predicate: str = Field(description="Short lowercase phrase: 'on top of', 'to the left of', 'holding', 'looking at', ...")
    object_id: str = Field(description="id of the object of the relationship.")
    kind: RelationKind
    confidence: float


class Activity(_Strict):
    description: str = Field(description="What is happening, e.g. 'a child kicks a ball'.")
    actor_ids: list[str]
    target_ids: list[str]
    confidence: float


class SceneContext(_Strict):
    scene_type: str = Field(description="e.g. 'kitchen', 'city street', 'soccer match'.")
    environment: Environment
    time_of_day: str = Field(description="morning / afternoon / dusk / night / unclear.")
    weather_or_lighting: str
    atmosphere: str = Field(description="Mood or ambience in a few words.")
    probable_purpose_or_event: str = Field(description="Likely event or purpose, or 'unclear'.")
    clues: list[str] = Field(description="Visible evidence supporting the context inference.")


class SceneAnalysis(_Strict):
    caption: str = Field(description="One sentence, at most 25 words.")
    description: str = Field(description="A coherent paragraph (3-6 sentences) covering objects, layout, activity and context.")
    context: SceneContext
    objects: list[SceneObject]
    relationships: list[Relationship]
    activities: list[Activity]
    visible_text: list[str] = Field(description="Legible text in reading order (signs, labels, screens).")
    uncertainties: list[str] = Field(description="Things that are ambiguous, occluded or hard to judge.")


# --------------------------------------------------------------------------- API response


class GraphNode(BaseModel):
    id: str
    label: str
    category: str
    prominence: str
    attributes: dict[str, str]


class GraphEdge(BaseModel):
    source: str
    target: str
    predicate: str
    kind: str


class SceneGraph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ImageInfo(BaseModel):
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    media_type: str
    resized: bool


class RunMeta(BaseModel):
    model: str
    latency_s: float
    input_tokens: int
    output_tokens: int
    image: ImageInfo
    warnings: list[str]


class AnalysisResult(BaseModel):
    analysis: SceneAnalysis
    scene_graph: SceneGraph
    meta: RunMeta


# --------------------------------------------------------------------------- JSON schema

# Structured outputs accept only a subset of JSON Schema: every object needs
# additionalProperties=false and numeric/length bounds are not allowed.
_DROP_KEYS = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "maxItems", "title",
}


def _strictify(node: Any) -> Any:
    if isinstance(node, list):
        return [_strictify(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {k: _strictify(v) for k, v in node.items() if k not in _DROP_KEYS}
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
    if isinstance(out.get("minItems"), int) and out["minItems"] > 1:
        del out["minItems"]
    return out


def generation_schema() -> dict:
    """JSON schema for SceneAnalysis, restricted to what structured outputs supports."""
    return _strictify(SceneAnalysis.model_json_schema())
