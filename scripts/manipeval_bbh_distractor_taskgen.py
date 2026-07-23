#!/usr/bin/env python3
"""Materialize one provider-written BBH target/distractor task candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.providers import OpenAICompatibleProvider
from mea.taskgen.bbh_distractor import (
    default_bbh_distractor_proposal,
    materialize_bbh_distractor_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--proposal-json",
        type=Path,
        help=(
            "Optional bounded proposal JSON. The committed target/distractor "
            "proposal is used when omitted."
        ),
    )
    parser.add_argument("--model", default="gpt-4o-2024-11-20")
    parser.add_argument("--base-url")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proposal = (
        json.loads(args.proposal_json.read_text(encoding="utf-8"))
        if args.proposal_json is not None
        else default_bbh_distractor_proposal()
    )
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=args.model,
        timeout=120.0,
        max_retries=2,
    )
    manifest = materialize_bbh_distractor_candidate(
        repo_root=args.repo_root,
        run_id=args.run_id,
        proposal=proposal,
        provider=provider,
        model=args.model,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
