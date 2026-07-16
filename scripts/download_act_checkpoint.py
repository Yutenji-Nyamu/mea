#!/usr/bin/env python3
"""Selectively download official RoboTwin 2.0 ACT task checkpoints."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_ID = "TianxingChen/RoboTwin2.0"
DEFAULT_REVISION = "9dc9299c163db059931898a9f0852098a61155a1"
_TASK_NAME = re.compile(r"[a-z0-9][a-z0-9_]*\Z")


def checkpoint_patterns(task_names: Iterable[str]) -> list[str]:
    """Return the two required Hub paths for each canonical task name."""

    ordered_tasks: list[str] = []
    for raw_name in task_names:
        name = str(raw_name).strip()
        if not _TASK_NAME.fullmatch(name):
            raise ValueError(f"invalid RoboTwin task name: {raw_name!r}")
        if name not in ordered_tasks:
            ordered_tasks.append(name)
    if not ordered_tasks:
        raise ValueError("at least one task name is required")

    patterns: list[str] = []
    for name in ordered_tasks:
        base = f"act_ckpt/act-{name}/demo_clean-50"
        patterns.extend(
            [
                f"{base}/policy_last.ckpt",
                f"{base}/dataset_stats.pkl",
            ]
        )
    return patterns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download only the policy_last.ckpt and dataset_stats.pkl files "
            "needed by selected RoboTwin ACT tasks."
        )
    )
    parser.add_argument("tasks", nargs="+", help="canonical task names")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "policy/ACT",
        help="directory that will contain act_ckpt/ (default: repo policy/ACT)",
    )
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_workers <= 0:
        raise SystemExit("--max-workers must be positive")
    try:
        patterns = checkpoint_patterns(args.tasks)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    local_dir = args.local_dir.expanduser().resolve()
    plan = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "revision": args.revision,
        "local_dir": str(local_dir),
        "allow_patterns": patterns,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required; install it with "
            "`python -m pip install huggingface_hub`"
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=patterns,
            local_dir=str(local_dir),
            max_workers=args.max_workers,
        )
    except Exception as exc:
        raise SystemExit(
            "checkpoint download failed "
            f"({type(exc).__name__}: {exc}). On AutoDL, enable academic "
            "acceleration and, for slow large files, set "
            "HF_HUB_DOWNLOAD_TIMEOUT=300; otherwise configure a server-side "
            "HF_ENDPOINT mirror. "
            "Do not relay routine checkpoints through a local workstation."
        ) from None

    missing = [path for path in patterns if not (local_dir / path).is_file()]
    if missing:
        raise SystemExit(
            "checkpoint download completed without required files: "
            + ", ".join(missing)
        )
    plan["files"] = [
        {
            "path": path,
            "bytes": (local_dir / path).stat().st_size,
        }
        for path in patterns
    ]
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
