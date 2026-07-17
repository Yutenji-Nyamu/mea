#!/usr/bin/env python3
"""Validate cached TaskGen functional evidence without running an experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.taskgen.acceptance import (
    DEFAULT_ACCEPTANCE_RUNS,
    TaskGenAcceptanceError,
    build_cached_taskgen_acceptance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--official-run-id", default=DEFAULT_ACCEPTANCE_RUNS["official_reuse"]
    )
    parser.add_argument(
        "--overlay-run-id", default=DEFAULT_ACCEPTANCE_RUNS["click_overlay"]
    )
    parser.add_argument(
        "--codegen-run-id", default=DEFAULT_ACCEPTANCE_RUNS["bbh_codegen"]
    )
    parser.add_argument(
        "--reflection-run-id",
        default=DEFAULT_ACCEPTANCE_RUNS["scene_error_repair"],
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = build_cached_taskgen_acceptance(
            args.repo_root,
            official_run_id=args.official_run_id,
            overlay_run_id=args.overlay_run_id,
            codegen_run_id=args.codegen_run_id,
            reflection_run_id=args.reflection_run_id,
        )
    except TaskGenAcceptanceError as exc:
        raise SystemExit(str(exc)) from exc
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
