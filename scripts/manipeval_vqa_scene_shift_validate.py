#!/usr/bin/env python3
"""Audit completed simulator-native texture/light Execution VQA cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.scene_shift_vqa_validation import (
    SceneShiftVQAValidationError,
    summarize_scene_shift_vqa_suite,
)


def _safe_existing_file(root: Path, value: Path, *, field: str) -> Path:
    path = value.expanduser()
    lexical = path if path.is_absolute() else root / path
    try:
        lexical_relative = lexical.absolute().relative_to(root)
    except ValueError as exc:
        raise SceneShiftVQAValidationError(f"{field} escapes repo root") from exc
    current = root
    for part in lexical_relative.parts:
        current = current / part
        if current.is_symlink():
            raise SceneShiftVQAValidationError(f"{field} contains a symlink")
    resolved = lexical.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise SceneShiftVQAValidationError(f"{field} escapes repo root") from exc
    if not resolved.is_file():
        raise SceneShiftVQAValidationError(f"{field} is missing")
    return resolved


def _safe_new_output(root: Path, value: Path) -> Path:
    path = value.expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SceneShiftVQAValidationError("output escapes repo root") from exc
    if resolved.exists():
        raise SceneShiftVQAValidationError(f"output already exists: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    try:
        suite_path = _safe_existing_file(root, args.suite, field="suite")
        suite = json.loads(suite_path.read_text(encoding="utf-8"))
        summary = summarize_scene_shift_vqa_suite(root, suite)
        output = _safe_new_output(root, args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    except (OSError, json.JSONDecodeError, SceneShiftVQAValidationError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
