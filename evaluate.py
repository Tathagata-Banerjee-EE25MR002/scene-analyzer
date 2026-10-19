#!/usr/bin/env python
"""Evaluate the scene analyzer on an annotated image set.

Two annotation sources:

  custom  --images DIR --annotations annotations.json
          {"img1.jpg": {"objects": ["dog", "ball"],
                        "scene_type": "park", "environment": "outdoor",
                        "activities": ["dog chasing a ball"],
                        "relationships": [["dog", "chasing", "ball"]],
                        "captions": ["A dog chases a ball in a park."]}, ...}
          Every key is optional; metrics are computed for whatever is present.

  COCO    --images DIR --coco-instances instances_val2017.json
          [--coco-captions captions_val2017.json] [--limit 100]
          Only images present in DIR are used. Ground-truth objects are the
          COCO categories of instances covering >= --min-area-frac of the image.

Metrics (macro-averaged over images)
  object P/R/F1        canonical label sets (synonyms + head nouns); for COCO, precision only counts
                       predictions that map to a COCO category, since COCO labels just 80 classes
  scene_type, environment   soft match on scene type; exact match on indoor/outdoor
  activity recall      fraction of reference activities covered by a predicted activity or the caption
  relationship recall  reference (subject, predicate, object) triples found; also pair-only recall
  caption token-F1 / ROUGE-L   best over reference captions (crude: wording-sensitive)
  --judge              Claude scores accuracy / completeness / hallucination (1-5) looking at the image
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import statistics
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from app.analyzer import SYSTEM_PROMPT, SceneAnalyzer
from app.backends import BackendError, ClaudeBackend, VLMBackend
from app.preprocess import ImageError, prepare_image
from app.schema import AnalysisResult, SceneAnalysis

# --------------------------------------------------------------------------- text normalisation

_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "with", "is", "are", "was", "were", "be",
    "it", "its", "this", "that", "there", "their", "his", "her", "for", "by", "from", "as", "near", "some",
}


def stem(w: str) -> str:
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "xes", "ches", "shes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 5 and w.endswith("ing"):
        w = w[:-3]
    elif len(w) > 4 and w.endswith("ed"):
        w = w[:-2]
    return w


def words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def content_tokens(text: str) -> list[str]:
    return [stem(w) for w in words(text) if w not in _STOP]


def tok_match(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))


def coverage(ref: list[str], pred: list[str]) -> float:
    """Fraction of `ref` tokens that match some token in `pred`."""
    if not ref:
        return 0.0
    return sum(any(tok_match(r, p) for p in pred) for r in ref) / len(ref)


def norm_label(label: str) -> str:
    return " ".join(stem(w) for w in words(label))


_SYNONYMS = {
    "person": ["man", "woman", "boy", "girl", "child", "kid", "people", "pedestrian", "player", "rider", "baby",
               "toddler", "lady", "guy", "gentleman", "adult", "skier", "surfer", "skateboarder", "cyclist",
               "passenger", "crowd", "men", "women", "children", "snowboarder", "worker", "driver"],
    "car": ["automobile", "sedan", "suv", "taxi", "cab", "hatchback", "jeep"],
    "truck": ["pickup", "lorry", "pickup truck"],
    "bus": ["coach", "minibus", "school bus"],
    "bicycle": ["bike", "cycle"],
    "motorcycle": ["motorbike", "scooter", "moped"],
    "airplane": ["plane", "aircraft", "jet", "airliner"],
    "boat": ["ship", "sailboat", "canoe", "kayak", "yacht", "ferry"],
    "train": ["locomotive", "tram"],
    "couch": ["sofa", "settee", "loveseat"],
    "chair": ["armchair"],
    "tv": ["television", "tv set", "monitor", "tv monitor"],
    "laptop": ["notebook computer", "macbook"],
    "dining table": ["table", "dinner table", "kitchen table"],
    "potted plant": ["plant", "houseplant", "flower pot", "potted flower"],
    "cell phone": ["phone", "smartphone", "mobile phone", "mobile", "iphone"],
    "sports ball": ["ball", "soccer ball", "football", "basketball", "tennis ball", "volleyball"],
    "wine glass": ["wineglass", "goblet"],
    "cup": ["mug", "teacup", "coffee cup"],
    "donut": ["doughnut"],
    "hot dog": ["hotdog"],
    "teddy bear": ["teddy", "stuffed animal", "plush"],
    "handbag": ["purse"],
    "tie": ["necktie"],
    "remote": ["remote control"],
    "hair drier": ["hair dryer", "hairdryer", "blow dryer"],
    "traffic light": ["traffic signal", "stoplight"],
    "fire hydrant": ["hydrant"],
    "tennis racket": ["racket", "racquet", "tennis racquet"],
    "baseball bat": ["bat"],
    "baseball glove": ["mitt", "glove", "baseball mitt"],
    "refrigerator": ["fridge"],
    "backpack": ["rucksack", "knapsack"],
    "suitcase": ["luggage"],
    "umbrella": ["parasol"],
    "frisbee": ["flying disc"],
    "dog": ["puppy", "retriever", "terrier", "poodle"],
    "cat": ["kitten"],
    "bird": ["seagull", "pigeon", "parrot", "eagle", "owl", "sparrow", "duck"],
    "horse": ["pony", "stallion", "mare"],
    "cow": ["cattle", "bull", "calf", "ox"],
    "sheep": ["lamb", "ram"],
    "bear": ["grizzly", "polar bear"],
    "sink": ["basin"],
}
_REV: dict[str, str] = {}
for _canon, _aliases in _SYNONYMS.items():
    _REV[norm_label(_canon)] = norm_label(_canon)
    for _al in _aliases:
        _REV[norm_label(_al)] = norm_label(_canon)


def canonical(label: str) -> str:
    """Canonical form of an object label: synonym map on the full phrase, last two words, or head noun."""
    n = norm_label(label)
    if not n:
        return ""
    if n in _REV:
        return _REV[n]
    parts = n.split()
    if len(parts) > 1 and " ".join(parts[-2:]) in _REV:
        return _REV[" ".join(parts[-2:])]
    return _REV.get(parts[-1], parts[-1])


_PRED_SYN = {
    "beside": "next to", "near": "next to", "by": "next to", "adjacent to": "next to", "alongside": "next to",
    "atop": "on", "on top of": "on", "over": "above", "below": "under", "beneath": "under", "underneath": "under",
    "in front": "in front of", "left": "to the left of", "right": "to the right of",
}


def predicate_match(ref: str, pred: str) -> bool:
    r, p = ref.lower().strip(), pred.lower().strip()
    r, p = _PRED_SYN.get(r, r), _PRED_SYN.get(p, p)
    rt, pt = [t for t in words(r) if t not in {"the", "a", "an"}], [t for t in words(p) if t not in {"the", "a", "an"}]
    rs, ps = [stem(t) for t in rt], [stem(t) for t in pt]
    if not rs or not ps:
        return False
    return max(coverage(rs, ps), coverage(ps, rs)) >= 0.5


def soft_match(ref: str, pred: str) -> bool:
    rt, pt = content_tokens(ref), content_tokens(pred)
    return any(tok_match(a, b) for a in rt for b in pt)


def lcs_len(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def caption_scores(pred: str, refs: list[str]) -> dict[str, float]:
    p = words(pred)
    best_f1 = best_rl = 0.0
    for ref in refs:
        r = words(ref)
        if not p or not r:
            continue
        common = sum((min(p.count(w), r.count(w))) for w in set(p) & set(r))
        if common:
            pr, rc = common / len(p), common / len(r)
            best_f1 = max(best_f1, 2 * pr * rc / (pr + rc))
        lcs = lcs_len(p, r)
        if lcs:
            pr, rc = lcs / len(p), lcs / len(r)
            best_rl = max(best_rl, 2 * pr * rc / (pr + rc))
    return {"caption_token_f1": best_f1, "caption_rougeL": best_rl}


# --------------------------------------------------------------------------- scoring one image


def score(analysis: SceneAnalysis, gt: dict, vocab: set[str] | None = None) -> dict[str, float | None]:
    m: dict[str, float | None] = {}
    labels = {o.id: canonical(o.label) for o in analysis.objects}

    gt_objs = {canonical(x) for x in gt.get("objects") or [] if canonical(x)}
    if gt_objs:
        pred = {c for c in labels.values() if c and (vocab is None or c in vocab)}
        tp = len(gt_objs & pred)
        prec = tp / len(pred) if pred else 0.0
        rec = tp / len(gt_objs)
        m["object_precision"], m["object_recall"] = prec, rec
        m["object_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    if gt.get("scene_type"):
        m["scene_type_match"] = float(soft_match(gt["scene_type"], analysis.context.scene_type))
    if gt.get("environment"):
        m["environment_match"] = float(gt["environment"].strip().lower() == analysis.context.environment)

    if gt.get("activities"):
        pool = [a.description for a in analysis.activities] + [analysis.caption]
        pool_tokens = [content_tokens(t) for t in pool]
        hits = []
        for ref in gt["activities"]:
            rt = content_tokens(ref)
            hits.append(any(coverage(rt, pt) >= 0.5 for pt in pool_tokens))
        m["activity_recall"] = sum(hits) / len(hits)

    if gt.get("relationships"):
        pair_hits = triple_hits = 0
        for s, p, o in gt["relationships"]:
            cs, co = canonical(s), canonical(o)
            pairs = [r for r in analysis.relationships if labels.get(r.subject_id) == cs and labels.get(r.object_id) == co]
            pair_hits += bool(pairs)
            triple_hits += any(predicate_match(p, r.predicate) for r in pairs)
        n = len(gt["relationships"])
        m["relationship_pair_recall"], m["relationship_triple_recall"] = pair_hits / n, triple_hits / n

    refs = gt.get("captions") or ([gt["caption"]] if gt.get("caption") else [])
    if refs:
        m.update(caption_scores(analysis.caption, refs))
    return m


# --------------------------------------------------------------------------- datasets


@dataclass
class Sample:
    path: Path
    gt: dict
    vocab: set[str] | None = None


def load_custom(images_dir: Path, ann_path: Path) -> list[Sample]:
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    samples = []
    for name, gt in ann.items():
        p = images_dir / name
        if p.exists():
            samples.append(Sample(p, gt))
        else:
            print(f"warning: {p} not found, skipped", file=sys.stderr)
    return samples


def load_coco(images_dir: Path, instances: Path, captions: Path | None, limit: int, seed: int, min_area_frac: float) -> list[Sample]:
    inst = json.loads(instances.read_text(encoding="utf-8"))
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    vocab = {canonical(n) for n in cats.values()}
    imgs = {i["id"]: i for i in inst["images"]}
    objs: dict[int, set[str]] = defaultdict(set)
    for a in inst["annotations"]:
        im = imgs[a["image_id"]]
        if a["area"] / (im["width"] * im["height"]) >= min_area_frac:
            objs[a["image_id"]].add(cats[a["category_id"]])
    caps: dict[int, list[str]] = defaultdict(list)
    if captions:
        for a in json.loads(captions.read_text(encoding="utf-8"))["annotations"]:
            caps[a["image_id"]].append(a["caption"].strip())
    ids = sorted(i for i in imgs if (images_dir / imgs[i]["file_name"]).exists())
    random.Random(seed).shuffle(ids)
    samples = []
    for i in ids[:limit]:
        gt = {"objects": sorted(objs[i])}
        if caps[i]:
            gt["captions"] = caps[i]
        samples.append(Sample(images_dir / imgs[i]["file_name"], gt, vocab))
    return samples


# --------------------------------------------------------------------------- LLM judge

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy": {"type": "integer"},
        "completeness": {"type": "integer"},
        "hallucination_free": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "required": ["accuracy", "completeness", "hallucination_free", "rationale"],
    "additionalProperties": False,
}
JUDGE_SYSTEM = (
    "You grade an automatic scene analysis of the attached image. Look at the image yourself; reference "
    "annotations, when given, are incomplete hints, not the full truth. Score each of these as an integer 1-5:\n"
    "accuracy: are the stated objects, attributes, relationships, activities and context correct?\n"
    "completeness: does it cover what matters in the scene?\n"
    "hallucination_free: 5 = nothing invented, 1 = many invented or wrong claims.\n"
    "Keep the rationale to two sentences."
)


def judge(backend: VLMBackend, image, analysis: SceneAnalysis, gt: dict) -> dict:
    parts = [f"Caption: {analysis.caption}", f"Description: {analysis.description}"]
    parts.append("Objects: " + "; ".join(
        f"{o.label} [{', '.join(f'{a.name}={a.value}' for a in o.attributes)}]" for o in analysis.objects))
    labels = {o.id: o.label for o in analysis.objects}
    parts.append("Relationships: " + "; ".join(
        f"{labels[r.subject_id]} {r.predicate} {labels[r.object_id]}" for r in analysis.relationships))
    parts.append("Activities: " + "; ".join(a.description for a in analysis.activities))
    c = analysis.context
    parts.append(f"Context: {c.scene_type}, {c.environment}, {c.time_of_day}, {c.weather_or_lighting}, {c.probable_purpose_or_event}")
    refs = gt.get("captions") or ([gt["caption"]] if gt.get("caption") else [])
    if refs:
        parts.append("Reference captions: " + " | ".join(refs))
    raw = backend.analyze_image(image, JUDGE_SYSTEM, "\n".join(parts), JUDGE_SCHEMA, max_tokens=512)
    return {k: raw.data[k] for k in ("accuracy", "completeness", "hallucination_free", "rationale")}


# --------------------------------------------------------------------------- run


@dataclass
class Record:
    image: str
    metrics: dict = field(default_factory=dict)
    judge: dict | None = None
    latency_s: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    error: str | None = None


def _cache_key(path: Path, model: str, focus: str | None) -> str:
    h = hashlib.sha256()
    h.update(SYSTEM_PROMPT.encode())
    h.update(model.encode())
    h.update((focus or "").encode())
    h.update(path.read_bytes())
    return h.hexdigest()[:20]


def run_evaluation(
    samples: list[Sample],
    analyzer: SceneAnalyzer,
    out_dir: Path,
    workers: int = 4,
    judge_backend: VLMBackend | None = None,
    focus: str | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(exist_ok=True)
    abort = threading.Event()
    fatal: list[str] = []

    def work(s: Sample) -> Record:
        rec = Record(image=s.path.name)
        if abort.is_set():
            rec.error = "skipped after fatal error"
            return rec
        try:
            cache_file = cache_dir / f"{s.path.stem}-{_cache_key(s.path, analyzer.backend.model, focus)}.json"
            cached = json.loads(cache_file.read_text()) if cache_file.exists() else None
            prepared = prepare_image(s.path.read_bytes())
            if cached:
                result = AnalysisResult.model_validate(cached["result"])
                jd = cached.get("judge")
                rec.cached = True
            else:
                result = analyzer.analyze_prepared(prepared, focus)
                jd = None
            if judge_backend is not None and jd is None:
                jd = judge(judge_backend, prepared, result.analysis, s.gt)
            if not cached or (jd and not cached.get("judge")):
                cache_file.write_text(json.dumps({"result": result.model_dump(), "judge": jd}))
            rec.metrics = score(result.analysis, s.gt, s.vocab)
            rec.judge = jd
            rec.latency_s = result.meta.latency_s
            rec.input_tokens, rec.output_tokens = result.meta.input_tokens, result.meta.output_tokens
        except (BackendError, ImageError, OSError) as exc:
            rec.error = getattr(exc, "message", str(exc))
            if getattr(exc, "fatal", False):
                fatal.append(rec.error)
                abort.set()
        return rec

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        records = list(pool.map(work, samples))

    summary = summarize(records)
    (out_dir / "results.json").write_text(
        json.dumps({"summary": summary, "images": [r.__dict__ for r in records]}, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    if fatal:
        summary["fatal_error"] = fatal[0]
    return summary


def summarize(records: list[Record]) -> dict:
    ok = [r for r in records if not r.error]
    agg: dict[str, list[float]] = defaultdict(list)
    for r in ok:
        for k, v in r.metrics.items():
            if v is not None:
                agg[k].append(v)
        if r.judge:
            for k in ("accuracy", "completeness", "hallucination_free"):
                agg[f"judge_{k}"].append(r.judge[k])
    lat = [r.latency_s for r in ok if r.latency_s is not None and not r.cached]
    return {
        "images": len(records),
        "succeeded": len(ok),
        "failed": len(records) - len(ok),
        "metrics": {k: {"mean": round(statistics.fmean(v), 4), "n": len(v)} for k, v in sorted(agg.items())},
        "mean_latency_s": round(statistics.fmean(lat), 2) if lat else None,
        "input_tokens": sum(r.input_tokens for r in ok),
        "output_tokens": sum(r.output_tokens for r in ok),
        "errors": sorted({r.error for r in records if r.error}),
    }


def summary_markdown(s: dict) -> str:
    lines = ["# Evaluation summary", "", f"Images: {s['images']} ({s['succeeded']} ok, {s['failed']} failed)", "",
             "| Metric | Mean | n |", "|---|---|---|"]
    lines += [f"| {k} | {v['mean']:.3f} | {v['n']} |" for k, v in s["metrics"].items()]
    lines += ["", f"Mean latency: {s['mean_latency_s']} s · tokens in/out: {s['input_tokens']}/{s['output_tokens']}"]
    if s["errors"]:
        lines += ["", "Errors:"] + [f"- {e}" for e in s["errors"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", type=Path, required=True, help="Directory containing the images.")
    p.add_argument("--annotations", type=Path, help="Custom annotation JSON.")
    p.add_argument("--coco-instances", type=Path)
    p.add_argument("--coco-captions", type=Path)
    p.add_argument("--min-area-frac", type=float, default=0.01, help="COCO: ignore instances smaller than this fraction of the image.")
    p.add_argument("--limit", type=int, default=50, help="COCO: number of images to sample.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("eval_out"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--focus", help="Pass the same focus text to every image.")
    p.add_argument("--judge", action="store_true", help="Add LLM-judge scores (one extra API call per image).")
    p.add_argument("--judge-model", help="Model for the judge (default: same as the analyzer).")
    p.add_argument("--model", help="Override CLAUDE_MODEL for the analyzer.")
    args = p.parse_args(argv)

    if bool(args.annotations) == bool(args.coco_instances):
        p.error("give exactly one of --annotations or --coco-instances")
    samples = (
        load_custom(args.images, args.annotations)
        if args.annotations
        else load_coco(args.images, args.coco_instances, args.coco_captions, args.limit, args.seed, args.min_area_frac)
    )
    if not samples:
        print("No images with annotations found.", file=sys.stderr)
        return 1

    try:
        backend = ClaudeBackend(model=args.model)
        judge_backend = ClaudeBackend(model=args.judge_model or args.model) if args.judge else None
    except BackendError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1

    print(f"Evaluating {len(samples)} images with {backend.model} ...", file=sys.stderr)
    summary = run_evaluation(samples, SceneAnalyzer(backend), args.out, args.workers, judge_backend, args.focus)
    print(summary_markdown(summary))
    print(f"Per-image results: {args.out / 'results.json'}", file=sys.stderr)
    if summary.get("fatal_error"):
        print(f"Stopped early: {summary['fatal_error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
