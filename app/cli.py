"""Command-line entry point:  python -m app.cli photo.jpg [--focus "..."] [--out result.json]"""
from __future__ import annotations

import argparse
import sys

from .analyzer import SceneAnalyzer
from .backends import BackendError, ClaudeBackend
from .preprocess import ImageError


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Analyze an image with Claude and print structured JSON.")
    p.add_argument("image")
    p.add_argument("--focus", help="Optional: what to pay extra attention to.")
    p.add_argument("--model", help="Override CLAUDE_MODEL.")
    p.add_argument("--out", help="Write JSON here instead of stdout.")
    args = p.parse_args(argv)

    try:
        analyzer = SceneAnalyzer(ClaudeBackend(model=args.model))
        result = analyzer.analyze_path(args.image, args.focus)
    except (BackendError, ImageError, OSError) as exc:
        print(f"error: {getattr(exc, 'message', exc)}", file=sys.stderr)
        return 1

    text = result.model_dump_json(indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
