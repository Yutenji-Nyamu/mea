#!/usr/bin/env python3
"""Summarize completed real-simulator clean/clutter Execution VQA evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.simulator_vqa_validation import (
    SimulatorVQAValidationError,
    summarize_simulator_vqa_suite,
)


def _output_path(repo_root: Path, value: Path) -> Path:
    path = value.expanduser()
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise SimulatorVQAValidationError("output path escapes repo root") from exc
    if resolved.exists():
        raise SimulatorVQAValidationError(f"output already exists: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    try:
        suite_path = args.suite.expanduser().resolve()
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        summary = summarize_simulator_vqa_suite(root, suite)
        output = _output_path(root, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except (
        OSError,
        json.JSONDecodeError,
        SimulatorVQAValidationError,
    ) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
