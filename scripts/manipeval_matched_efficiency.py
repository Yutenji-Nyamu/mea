#!/usr/bin/env python3
"""Audit a preregistered fixed/adaptive pair without starting any runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.matched_efficiency_protocol import (
    MatchedEfficiencyError,
    build_synthetic_demonstrations,
    compare_matched_results,
)


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MatchedEfficiencyError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatchedEfficiencyError(f"input must be a JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--fixed-result", type=Path)
    parser.add_argument("--adaptive-result", type=Path)
    parser.add_argument(
        "--synthetic-demo",
        action="store_true",
        help="Emit two 0-ACT functional fixtures: one saving and one zero-saving pair.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.synthetic_demo:
            if any(
                value is not None
                for value in (
                    args.preregistration,
                    args.fixed_result,
                    args.adaptive_result,
                )
            ):
                raise MatchedEfficiencyError(
                    "--synthetic-demo cannot be combined with pair inputs"
                )
            summary = build_synthetic_demonstrations()
        else:
            if any(
                value is None
                for value in (
                    args.preregistration,
                    args.fixed_result,
                    args.adaptive_result,
                )
            ):
                raise MatchedEfficiencyError(
                    "pair audit requires --preregistration, --fixed-result, and --adaptive-result"
                )
            summary = compare_matched_results(
                _read(args.preregistration),
                _read(args.fixed_result),
                _read(args.adaptive_result),
            )
        if args.output is not None:
            _write(args.output.expanduser().resolve(), summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except MatchedEfficiencyError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
