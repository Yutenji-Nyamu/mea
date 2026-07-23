#!/usr/bin/env python3
"""Generate, validate, register, and reuse one query-induced telemetry Tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.providers import OpenAICompatibleProvider
from mea.toolgen.query_induced import run_query_induced_toolgen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--telemetry",
        required=True,
        help="Recorded real episode directory; there is no default.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--registry-dir", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--base-url")
    parser.add_argument(
        "--reuse-only",
        action="store_true",
        help=(
            "Do not construct or call a provider. This succeeds only when the "
            "exact normalized Query is already bound to a compatible validated "
            "run-local registration."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = None
    if not args.reuse_only:
        provider = OpenAICompatibleProvider(
            base_url=args.base_url,
            text_model=args.model,
            timeout=120.0,
            max_retries=2,
        )
    result = run_query_induced_toolgen(
        query=args.query,
        episode_dir=args.telemetry,
        output_dir=args.output_dir,
        registry_dir=args.registry_dir,
        provider=provider,
        model=args.model,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
