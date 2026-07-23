#!/usr/bin/env python3
"""Aggregate one explicit real/proxy paper-claim manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.paper_claim_demo import (  # noqa: E402
    PaperClaimDemoError,
    evaluate_paper_claim_manifest,
)


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperClaimDemoError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PaperClaimDemoError(f"input must be a JSON object: {path}")
    return value


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate an explicitly supplied live/proxy paper-claim manifest. "
            "No synthetic/default result mode is provided."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_paper_claim_manifest(
            _read(args.input.expanduser().resolve())
        )
        if args.output is not None:
            _write(args.output.expanduser().resolve(), result)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except PaperClaimDemoError as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
