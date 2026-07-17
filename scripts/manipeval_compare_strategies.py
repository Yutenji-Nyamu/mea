#!/usr/bin/env python3
"""Compare cached fixed-suite and dynamic-evidence ACT evaluations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.strategy_comparison import StrategyComparisonError, compare_fixed_dynamic


def _report(summary: dict) -> str:
    fixed = summary["strategies"]["fixed_predeclared_v1"]["totals"]
    dynamic = summary["strategies"]["dynamic_evidence_v1"]["totals"]
    return "\n".join(
        [
            "# ACT Fixed-Suite vs Dynamic-Evidence Micro-Pilot",
            "",
            f"- claim scope: `{summary['claim_scope']}`",
            f"- paper-table eligible: `{str(summary['paper_table_eligible']).lower()}`",
            f"- rollout savings: `{summary['rollout_savings']}`",
            "- Table 2 consistency: unavailable at N=1",
            "",
            "| Strategy | ACT rollouts | Successes | Policy steps | Physics steps |",
            "|---|---:|---:|---:|---:|",
            (
                "| fixed_predeclared_v1 | "
                f"{fixed['act_rollouts']} | {fixed['successes']} | "
                f"{fixed['policy_steps']} | {fixed['physics_steps']} |"
            ),
            (
                "| dynamic_evidence_v1 | "
                f"{dynamic['act_rollouts']} | {dynamic['successes']} | "
                f"{dynamic['policy_steps']} | {dynamic['physics_steps']} |"
            ),
            "",
            "This artifact-only report starts no simulator or ACT rollout.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    destination = args.output_dir.expanduser().resolve()
    if not destination.is_relative_to(root):
        raise SystemExit("--output-dir must stay inside --repo-root")
    if destination.exists():
        raise SystemExit(f"output directory already exists: {destination}")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        summary = compare_fixed_dynamic(root, config)
    except (OSError, json.JSONDecodeError, StrategyComparisonError) as exc:
        raise SystemExit(str(exc)) from exc
    destination.mkdir(parents=True)
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (destination / "report.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
