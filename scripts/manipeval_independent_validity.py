#!/usr/bin/env python3
"""Aggregate multi-rater labels and paired VQA controls offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.independent_validity import (
    IndependentValidityError,
    build_synthetic_validity_demonstration,
    summarize_independent_validity,
)


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndependentValidityError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise IndependentValidityError(f"input must be a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=Path)
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Run a deterministic no-human, no-provider, no-ACT fixture.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.synthetic_demo == (args.study is not None):
            raise IndependentValidityError(
                "choose exactly one of --study or --synthetic-demo"
            )
        result = (
            build_synthetic_validity_demonstration()
            if args.synthetic_demo
            else {
                "study": _read(args.study),
                "summary": summarize_independent_validity(_read(args.study)),
            }
        )
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except IndependentValidityError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
