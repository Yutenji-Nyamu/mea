#!/usr/bin/env python3
"""Regenerate or publish the compact illustrated MEA evidence report."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.feedback import write_evidence_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument(
        "--publish-dir",
        type=Path,
        help=(
            "Optional repo-relative destination such as "
            "docs/evidence_runs/<evaluation_id>. Writes README.md plus small "
            "real assets suitable for GitHub."
        ),
    )
    parser.add_argument("--max-video-mb", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"eval_[A-Za-z0-9_]+", args.evaluation_id):
        raise SystemExit("--evaluation-id must match eval_[A-Za-z0-9_]+")
    if args.max_video_mb < 0:
        raise SystemExit("--max-video-mb must be non-negative")
    root = args.repo_root.expanduser().resolve()
    evaluation = root / "mea/evaluation_runs" / args.evaluation_id
    if not (evaluation / "manifest.json").is_file():
        raise SystemExit(f"evaluation does not exist: {evaluation}")
    if args.publish_dir is None:
        destination = evaluation / "evidence_report.md"
        publish = False
    else:
        publish_root = (
            args.publish_dir.expanduser().resolve()
            if args.publish_dir.is_absolute()
            else (root / args.publish_dir).resolve()
        )
        try:
            publish_root.relative_to(root)
        except ValueError as exc:
            raise SystemExit("--publish-dir must remain inside --repo-root") from exc
        destination = publish_root / "README.md"
        publish = True
    result = write_evidence_report(
        root,
        evaluation,
        destination=destination,
        publish=publish,
        max_video_bytes=int(args.max_video_mb * 1_000_000),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
