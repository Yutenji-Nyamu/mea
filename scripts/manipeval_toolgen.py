#!/usr/bin/env python3
"""Generate and differentially validate one offline trajectory tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.providers import OpenAICompatibleProvider
from mea.toolgen import ToolGenPrototype
from mea.toolgen.examples import EXAMPLE_CATALOG
from mea.toolgen.targets import COMPOSITE_TARGETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a pure offline Tool over TrajectoryView and compare it "
            "with an exact or composition-based validation oracle on multiple "
            "episodes."
        )
    )
    parser.add_argument("--request", required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--reference-tool",
        choices=sorted(EXAMPLE_CATALOG),
        help="Regenerate an existing exact Trusted Tool metric.",
    )
    target.add_argument(
        "--target-metric",
        choices=sorted(COMPOSITE_TARGETS),
        help="Generate a genuinely new metric with a private composition oracle.",
    )
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        help="Episode telemetry directory; pass at least one positive and one negative episode.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tool-name")
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--model",
        default="gpt-4o-2024-11-20",
    )
    parser.add_argument("--base-url")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=args.model,
        timeout=120.0,
        max_retries=2,
    )
    result = ToolGenPrototype(
        REPO_ROOT,
        provider,
        model=args.model,
    ).generate(
        args.request,
        reference_tool=args.reference_tool,
        target_metric=args.target_metric,
        episode_dirs=args.trajectory,
        output_dir=args.output_dir,
        tool_name=args.tool_name,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
