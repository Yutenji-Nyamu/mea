#!/usr/bin/env python3
"""Inventory completed scene-shift Agent evidence without starting runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.scene_shift_collector import (
    SceneShiftCollectionError,
    collect_scene_shift_candidates,
)


def _repo_path(root: Path, value: Path, *, field: str, must_exist: bool) -> Path:
    candidate = value.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise SceneShiftCollectionError(f"{field} must stay inside --repo-root") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SceneShiftCollectionError(f"{field} contains a symlink component")
    resolved = candidate.resolve(strict=must_exist)
    if not resolved.is_relative_to(root):
        raise SceneShiftCollectionError(f"{field} must stay inside --repo-root")
    return resolved


def _read_labels(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SceneShiftCollectionError("--labels must contain a JSON object")
    return value


def _report(result: dict) -> str:
    lines = [
        "# Scene-shift Evidence Collection",
        "",
        f"- scanned evaluations: `{result['evaluation_count_scanned']}`",
        f"- completed evaluations: `{result['completed_evaluation_count']}`",
        f"- relevant evaluations: `{result['relevant_evaluation_count']}`",
        f"- ready candidates: `{result['ready_candidate_count']}/{result['candidate_count']}`",
        f"- diagnostics: `{result['diagnostic_count']}`",
        f"- inventory SHA-256: `{result['inventory_sha256']}`",
        f"- label status: `{result['label_status']}`",
        "- runtime calls: provider=`false`, simulator=`false`, ACT=`false`",
        "- paper-table eligible: `false`",
        "",
        "| Condition | Candidates | Ready | Unique seeds |",
        "|---|---:|---:|---:|",
    ]
    for condition, count in result["counts_by_condition"].items():
        lines.append(
            f"| {condition} | {count['candidate_count']} | "
            f"{count['ready_count']} | {count['unique_seed_count']} |"
        )
    lines.extend(
        [
            "",
            "This collector hashes existing evidence only. It never turns VQA "
            "predictions into gold labels and does not claim Tables 7–8.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-id", action="append", dest="evaluation_ids")
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--reviewer-id")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    try:
        output = _repo_path(root, args.output_dir, field="--output-dir", must_exist=False)
        if output.exists():
            raise SceneShiftCollectionError(f"output directory already exists: {output}")
        labels = None
        if args.labels is not None:
            labels = _read_labels(
                _repo_path(root, args.labels, field="--labels", must_exist=True)
            )
        result = collect_scene_shift_candidates(
            root,
            evaluation_ids=args.evaluation_ids,
            labels=labels,
            reviewer_id=args.reviewer_id,
        )
        output.mkdir(parents=True, exist_ok=False)
        (output / "collection.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "report.md").write_text(_report(result), encoding="utf-8")
        if result["suite_draft"] is not None:
            (output / "suite_draft.json").write_text(
                json.dumps(result["suite_draft"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, json.JSONDecodeError, SceneShiftCollectionError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
