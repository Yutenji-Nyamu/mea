#!/usr/bin/env python3
"""Generate or proxy-review a live-provider, zero-ACT Table 3 micro-ablation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.module_ablation_live import (
    LiveModuleAblationError,
    generate_live_module_ablation,
    review_live_module_ablation,
)
from mea.providers import OpenAICompatibleProvider


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise LiveModuleAblationError(f"input must be a JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--schedule", type=Path, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--item-id", action="append", dest="item_ids")
    generate.add_argument("--model", required=True)
    generate.add_argument("--base-url")
    review = sub.add_parser("review")
    review.add_argument("--run-dir", type=Path, required=True)
    review.add_argument("--labels", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "generate":
            provider = OpenAICompatibleProvider(
                base_url=args.base_url,
                text_model=args.model,
                vision_model=args.model,
                timeout=180.0,
            )
            result = generate_live_module_ablation(
                root,
                _read((root / args.schedule).resolve() if not args.schedule.is_absolute() else args.schedule),
                output_dir=args.output_dir,
                provider=provider,
                model=args.model,
                schedule_item_ids=args.item_ids,
            )
        else:
            run_dir = (root / args.run_dir).resolve() if not args.run_dir.is_absolute() else args.run_dir.resolve()
            labels = (root / args.labels).resolve() if not args.labels.is_absolute() else args.labels.resolve()
            if not run_dir.is_relative_to(root) or not labels.is_relative_to(root):
                raise LiveModuleAblationError("review paths must stay inside repository")
            result = review_live_module_ablation(run_dir, _read(labels))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, json.JSONDecodeError, LiveModuleAblationError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
