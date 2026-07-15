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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a pure offline Tool over TrajectoryView and compare it "
            "with a Trusted Tool oracle on multiple episodes."
        )
    )
    parser.add_argument("--request", required=True)
    parser.add_argument(
        "--reference-tool",
        required=True,
        choices=sorted(EXAMPLE_CATALOG),
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
        episode_dirs=args.trajectory,
        output_dir=args.output_dir,
        tool_name=args.tool_name,
        max_attempts=args.max_attempts,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
