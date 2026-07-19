#!/usr/bin/env python3
"""Plan or audit a two-task MEA cross-task portfolio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.portfolio import (
    PortfolioError,
    build_portfolio_command_plan,
    build_reused_portfolio,
    render_portfolio_report,
)


def _output_dir(root: Path, value: Path) -> Path:
    raw = value.expanduser()
    candidate = raw if raw.is_absolute() else root / raw
    try:
        relative = candidate.absolute().relative_to(root)
    except ValueError as exc:
        raise PortfolioError("--output-dir must stay inside --repo-root") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PortfolioError("--output-dir contains a symlink component")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise PortfolioError("--output-dir must be a child of --repo-root")
    if resolved.exists():
        raise PortfolioError(f"output directory already exists: {resolved}")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--portfolio-id", required=True)
    plan.add_argument("--query", required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--start-seed", type=int, default=100403)
    plan.add_argument("--gpu", type=int, default=0)
    plan.add_argument("--model-profile", default="economy")
    plan.add_argument("--python-executable", default="python")

    reuse = subparsers.add_parser("reuse")
    reuse.add_argument("--portfolio-id", required=True)
    reuse.add_argument("--query", required=True)
    reuse.add_argument("--output-dir", type=Path, required=True)
    reuse.add_argument("--click-bell-evaluation-id", required=True)
    reuse.add_argument("--bbh-evaluation-id", required=True)

    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    try:
        destination = _output_dir(root, args.output_dir)
        if args.command == "plan":
            result = build_portfolio_command_plan(
                root,
                portfolio_id=args.portfolio_id,
                user_query=args.query,
                start_seed=args.start_seed,
                gpu=args.gpu,
                model_profile=args.model_profile,
                python_executable=args.python_executable,
            )
            json_name = "command_plan.json"
        else:
            result = build_reused_portfolio(
                root,
                portfolio_id=args.portfolio_id,
                user_query=args.query,
                child_evaluation_ids={
                    "click_bell": args.click_bell_evaluation_id,
                    "beat_block_hammer": args.bbh_evaluation_id,
                },
            )
            json_name = "summary.json"
        destination.mkdir(parents=True, exist_ok=False)
        (destination / json_name).write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (destination / "report.md").write_text(
            render_portfolio_report(result), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (OSError, PortfolioError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
