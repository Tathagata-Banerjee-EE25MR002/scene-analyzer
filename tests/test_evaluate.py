import json

import evaluate as ev
from app.analyzer import SceneAnalyzer, postprocess
from app.backends import BackendError
from app.schema import SceneAnalysis
from fakes import SAMPLE_RAW, MockBackend, png_bytes


def sample_analysis() -> SceneAnalysis:
    return postprocess(SceneAnalysis.model_validate(SAMPLE_RAW), 100, 100)[0]


def test_canonical_labels():
    assert ev.canonical("golden retriever") == "dog"
    assert ev.canonical("People") == "person"
    assert ev.canonical("a wooden dining table") == ev.canonical("Table") == ev.canonical("dining table")
    assert ev.canonical("red sports car") == "car"
    assert ev.canonical("lamp") == "lamp"


def test_predicate_matching():
    assert ev.predicate_match("on top of", "on")
    assert ev.predicate_match("beside", "next to")
    assert not ev.predicate_match("on", "under")


def test_caption_scores():
    same = ev.caption_scores("a dog runs", ["a dog runs"])
    assert same == {"caption_token_f1": 1.0, "caption_rougeL": 1.0}
    assert ev.caption_scores("a dog runs", ["cat sleeping"])["caption_token_f1"] == 0.0


def test_score_all_metrics():
    gt = {
        "objects": ["dog", "ball", "bench"],
        "scene_type": "park",
        "environment": "outdoor",
        "activities": ["dog chasing a ball"],
        "relationships": [["dog", "chasing", "ball"], ["dog", "on", "bench"]],
        "captions": ["A dog chases a ball in the park."],
    }
    m = ev.score(sample_analysis(), gt)
    assert m["object_recall"] == 2 / 3  # dog, ball found; bench missing
    assert m["object_precision"] == 2 / 3  # dog, ball, tree predicted
    assert m["scene_type_match"] == 1.0
    assert m["environment_match"] == 1.0
    assert m["activity_recall"] == 1.0
    assert m["relationship_triple_recall"] == 0.5
    assert m["relationship_pair_recall"] == 0.5
    assert m["caption_rougeL"] > 0.5


def test_vocab_restricts_precision_only():
    vocab = {ev.canonical("dog"), ev.canonical("ball")}
    m = ev.score(sample_analysis(), {"objects": ["dog"]}, vocab=vocab)
    assert m["object_precision"] == 0.5  # tree ignored; dog correct, ball wrong
    assert m["object_recall"] == 1.0


def test_missing_gt_fields_skipped():
    assert ev.score(sample_analysis(), {"scene_type": "park"}).keys() == {"scene_type_match"}


def make_dataset(tmp_path, n=2):
    ann = {}
    for i in range(n):
        (tmp_path / f"img{i}.png").write_bytes(png_bytes((50 + i, 40)))
        ann[f"img{i}.png"] = {"objects": ["dog", "ball"], "environment": "outdoor"}
    ann["missing.png"] = {"objects": ["x"]}
    (tmp_path / "ann.json").write_text(json.dumps(ann))
    return ev.load_custom(tmp_path, tmp_path / "ann.json")


def test_run_evaluation_and_cache(tmp_path):
    samples = make_dataset(tmp_path)
    assert len(samples) == 2  # missing image skipped
    backend = MockBackend()
    out = tmp_path / "out"
    s = ev.run_evaluation(samples, SceneAnalyzer(backend), out, workers=2)
    assert s["succeeded"] == 2 and s["failed"] == 0
    assert s["metrics"]["object_recall"]["mean"] == 1.0
    assert s["metrics"]["environment_match"]["mean"] == 1.0
    assert backend.calls == 2
    assert (out / "results.json").exists() and (out / "summary.md").exists()

    ev.run_evaluation(samples, SceneAnalyzer(backend), out, workers=2)
    assert backend.calls == 2  # second run served from cache


def test_judge_scores_aggregated(tmp_path):
    samples = make_dataset(tmp_path, 1)
    judge_backend = MockBackend(data={"accuracy": 4, "completeness": 3, "hallucination_free": 5, "rationale": "ok"})
    s = ev.run_evaluation(samples, SceneAnalyzer(MockBackend()), tmp_path / "out", judge_backend=judge_backend)
    assert s["metrics"]["judge_accuracy"]["mean"] == 4
    assert s["metrics"]["judge_hallucination_free"]["mean"] == 5


def test_fatal_error_stops_run(tmp_path):
    samples = make_dataset(tmp_path, 3)
    backend = MockBackend(error=BackendError("bad key", 502, fatal=True))
    s = ev.run_evaluation(samples, SceneAnalyzer(backend), tmp_path / "out", workers=1)
    assert s["failed"] == 3 and s["fatal_error"] == "bad key"
    assert backend.calls == 1  # remaining images skipped
