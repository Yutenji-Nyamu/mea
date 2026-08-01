#!/usr/bin/env python3
"""Retry only the final answer from completed cached MEA evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.cached_finalization import finalize_cached_evaluation
from mea.providers import OpenAICompatibleProvider, resolve_model_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finish answer/report artifacts from cached rounds and Aggregate; "
            "never reruns TaskGen, ToolGen, simulator, or policy inference."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-id", required=True)
    parser.add_argument("--base-url")
    parser.add_argument("--feedback-model")
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"eval_[A-Za-z0-9_]+", args.evaluation_id):
        raise SystemExit("--evaluation-id must match eval_[A-Za-z0-9_]+")
    root = args.repo_root.expanduser().resolve()
    manifest_path = (
        root / "mea" / "evaluation_runs" / args.evaluation_id / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    models = resolve_model_profile(manifest.get("model_profile", "balanced"))
    feedback_model = args.feedback_model or models["feedback"]
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=feedback_model,
        timeout=args.timeout,
    )
    result = finalize_cached_evaluation(
        root,
        args.evaluation_id,
        provider=provider,
        feedback_model=feedback_model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
