import copy

import pytest

from app.analyzer import SceneAnalyzer, build_user_prompt, postprocess
from app.backends import BackendError
from app.schema import SceneAnalysis, generation_schema
from fakes import SAMPLE_RAW, MockBackend, png_bytes


def analysis(**patch) -> SceneAnalysis:
    raw = copy.deepcopy(SAMPLE_RAW)
    raw.update(patch)
    return SceneAnalysis.model_validate(raw)


# ---- schema sent to the API must be within the structured-outputs subset
def _walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_generation_schema_is_strict_subset():
    schema = generation_schema()
    forbidden = {"minimum", "maximum", "minLength", "maxLength", "maxItems", "multipleOf"}
    objects = 0
    for node in _walk(schema):
        assert not (forbidden & node.keys())
        if "properties" in node:
            objects += 1
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])  # nothing optional
    assert objects == 6  # SceneAnalysis, SceneContext, SceneObject, Attribute, Relationship, Activity


def test_sample_validates():
    assert analysis().objects[0].label == "golden retriever"


# ---- post-processing
def test_pixel_bbox_normalised_and_reordered():
    a = analysis()
    a.objects[0].bbox = [320, 180, 64, 36]  # pixels, corners swapped, on a 640x360 image
    out, _ = postprocess(a, 640, 360)
    assert out.objects[0].bbox == [0.1, 0.1, 0.5, 0.5]


def test_bbox_clamped_and_degenerate_dropped():
    a = analysis()
    a.objects[0].bbox = [-0.2, 0.1, 1.3, 0.9]
    a.objects[1].bbox = [0.5, 0.5, 0.5, 0.6]
    a.objects[2].bbox = [0.1, 0.2, 0.3]
    out, _ = postprocess(a, 100, 100)
    assert out.objects[0].bbox == [0.0, 0.1, 1.0, 0.9]
    assert out.objects[1].bbox is None
    assert out.objects[2].bbox is None


def test_duplicate_ids_renamed():
    a = analysis()
    a.objects[1].id = "o1"
    a.objects[2].id = ""
    out, warnings = postprocess(a, 100, 100)
    ids = [o.id for o in out.objects]
    assert len(set(ids)) == 3 and all(ids)
    assert len(warnings) == 2


def test_dangling_and_duplicate_relationships_dropped():
    a = analysis()
    a.relationships.append(a.relationships[0].model_copy(update={"object_id": "o99"}))
    a.relationships.append(a.relationships[0].model_copy())  # exact duplicate
    a.relationships.append(a.relationships[0].model_copy(update={"object_id": "o1"}))  # self
    a.activities[0].target_ids = ["o2", "nope"]
    out, warnings = postprocess(a, 100, 100)
    assert len(out.relationships) == 3
    assert out.activities[0].target_ids == ["o2"]
    assert len(warnings) == 2


def test_confidence_clamped():
    a = analysis()
    a.objects[0].confidence = 1.7
    out, _ = postprocess(a, 100, 100)
    assert out.objects[0].confidence == 1.0


# ---- pipeline
def test_pipeline_end_to_end():
    backend = MockBackend()
    result = SceneAnalyzer(backend).analyze_bytes(png_bytes((200, 100)))
    assert backend.calls == 1
    assert result.analysis.caption.startswith("A golden retriever")
    assert {n.id for n in result.scene_graph.nodes} == {"o1", "o2", "o3"}
    assert result.scene_graph.nodes[0].attributes == {"color": "golden", "pose": "running"}
    assert len(result.scene_graph.edges) == 3
    assert result.meta.image.processed_width == 200
    assert result.meta.input_tokens == 120


def test_invalid_model_output_becomes_backend_error():
    bad = copy.deepcopy(SAMPLE_RAW)
    del bad["context"]
    with pytest.raises(BackendError):
        SceneAnalyzer(MockBackend(data=bad)).analyze_bytes(png_bytes())


def test_focus_in_prompt():
    assert "safety gear" in build_user_prompt("safety gear")
    assert "particular attention" not in build_user_prompt(None)
