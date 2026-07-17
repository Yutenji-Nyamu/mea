#!/usr/bin/env python3
"""Compare cached fixed/dynamic runs against their evidence preregistration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.evidence_manifest import EvidenceManifestError, read_repo_json
from mea.strategy_plan import StrategyPlanError, compare_registered_strategies


def _inside(root: Path, value: Path, *, must_exist: bool) -> Path:
    candidate = value.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        lexical = candidate.absolute()
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise StrategyPlanError("path must stay inside --repo-root") from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise StrategyPlanError("path may not traverse a symlink")
    try:
        resolved = candidate.resolve(strict=must_exist)
    except OSError as exc:
        raise StrategyPlanError(f"cannot resolve path: {exc}") from exc
    if not resolved.is_relative_to(root):
        raise StrategyPlanError("path must stay inside --repo-root")
    return resolved


def _report(result: dict) -> str:
    comparison = result["comparison"]
    fixed = comparison["strategies"]["fixed_predeclared_v1"]["totals"]
    dynamic = comparison["strategies"]["dynamic_evidence_v1"]["totals"]
    return "\n".join(
        [
            "# Registered ACT fixed/dynamic micro-pilot",
            "",
            f"- registered identity match: `{str(result['registered_identity_match']).lower()}`",
            f"- paper-table eligible: `{str(result['paper_table_eligible']).lower()}`",
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
            "The existing strict strategy comparator produced these metrics; this wrapper additionally binds them to the preregistered hashes and N=1 schedule.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    try:
        plan_path = _inside(root, args.command_plan, must_exist=True)
        plan = read_repo_json(
            root, plan_path.relative_to(root).as_posix(), label="command plan"
        )
        output = _inside(root, args.output_dir, must_exist=False)
        if output.exists():
            raise StrategyPlanError(f"output directory already exists: {output}")
        result = compare_registered_strategies(root, plan)
        output.parent.mkdir(parents=True, exist_ok=True)
        _inside(root, output, must_exist=False)
        output.mkdir()
        (output / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (output / "report.md").write_text(_report(result), encoding="utf-8")
    except (EvidenceManifestError, StrategyPlanError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
