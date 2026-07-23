#!/usr/bin/env python3
"""Review, install, and inspect exact generated-Task registrations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.taskgen.reviewed_registry import (
    ReviewedTaskRegistryError,
    build_task_review_manifest_template,
    find_reviewed_task,
    install_reviewed_task,
)


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReviewedTaskRegistryError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewedTaskRegistryError(f"{label} must be a JSON object")
    return value


def _resolution_query(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project a TaskResolution artifact onto the registry lookup contract."""

    semantic_key = value.get("semantic_key")
    semantic_key_sha256 = value.get("semantic_key_sha256")
    if not isinstance(semantic_key, Mapping):
        raise ReviewedTaskRegistryError("TaskResolution.semantic_key is missing")
    if not isinstance(semantic_key_sha256, str) or len(semantic_key_sha256) != 64:
        raise ReviewedTaskRegistryError(
            "TaskResolution.semantic_key_sha256 must be a 64-character hash"
        )
    return {
        "schema_version": 1,
        "semantic_key": dict(semantic_key),
        "semantic_key_sha256": semantic_key_sha256,
    }


def _source_query(
    source_run: Path, task_resolution: Path | None = None
) -> dict[str, Any]:
    resolution = _read_json(
        task_resolution.expanduser().resolve()
        if task_resolution is not None
        else source_run.expanduser().resolve() / "generation/task_resolution.json",
        label="TaskResolution",
    )
    return _resolution_query(resolution)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a pending review manifest, install an explicitly approved "
            "generated Task, or verify an exact registry lookup."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser(
        "template", help="Print a pending manifest; never approves a Task."
    )
    template.add_argument("--source-run", type=Path, required=True)
    template.add_argument(
        "--task-resolution",
        type=Path,
        help="Optional external identity for a legacy run lacking this artifact.",
    )

    install = subparsers.add_parser(
        "install", help="Install only when the supplied review says approved."
    )
    install.add_argument("--source-run", type=Path, required=True)
    install.add_argument(
        "--task-resolution",
        type=Path,
        help="Optional external identity for a legacy run lacking this artifact.",
    )
    install.add_argument("--review-manifest", type=Path, required=True)
    install.add_argument("--reviewed-registry", type=Path, required=True)

    find = subparsers.add_parser(
        "find", help="Verify an exact TaskResolution-to-registry match."
    )
    find.add_argument("--task-resolution", type=Path, required=True)
    find.add_argument("--reviewed-registry", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        if args.command == "template":
            source_run = args.source_run.expanduser().resolve()
            result = build_task_review_manifest_template(
                source_run,
                _source_query(source_run, args.task_resolution)["semantic_key"],
                repo_root=REPO_ROOT,
            )
        elif args.command == "install":
            source_run = args.source_run.expanduser().resolve()
            result = {
                "status": "installed",
                "registration": install_reviewed_task(
                    source_run,
                    _source_query(source_run, args.task_resolution)["semantic_key"],
                    args.review_manifest,
                    args.reviewed_registry,
                    repo_root=REPO_ROOT,
                ),
            }
        else:
            query = _resolution_query(
                _read_json(args.task_resolution, label="TaskResolution")
            )
            match = find_reviewed_task(
                args.reviewed_registry, query, repo_root=REPO_ROOT
            )
            result = {"status": "matched" if match else "not_found", "match": match}
    except ReviewedTaskRegistryError as exc:
        raise SystemExit(f"reviewed task registry error: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
