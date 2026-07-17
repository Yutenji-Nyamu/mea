#!/usr/bin/env python3
"""Materialize an auditable fixed/dynamic command plan without running it."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.evidence_manifest import EvidenceManifestError, read_repo_json
from mea.strategy_plan import StrategyPlanError, build_matched_strategy_plan


def _inside(root: Path, path: Path, *, must_exist: bool) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    try:
        absolute = candidate.absolute()
        relative = absolute.relative_to(root)
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


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _report(plan: dict) -> str:
    evidence = plan["evidence"]["manifest_path"]
    fixed = shlex.join(plan["strategies"]["fixed_predeclared_v1"]["argv"])
    dynamic = shlex.join(plan["strategies"]["dynamic_evidence_v1"]["argv"])
    compare = shlex.join(plan["posthoc"]["comparison_argv"])
    python_executable = plan["strategies"]["fixed_predeclared_v1"]["argv"][0]
    validate = shlex.join(
        [
            python_executable,
            "scripts/manipeval_evidence_manifest.py",
            "--repo-root",
            ".",
            "validate",
            "--manifest",
            evidence,
        ]
    )
    return "\n".join(
        [
            "# Matched ACT fixed/dynamic command plan",
            "",
            "This file is an inert command record. Generating it started 0 ACT rollouts.",
            "Run only after reviewing the registered evidence identity and budget.",
            "",
            f"- claim scope: `{plan['claim_scope']}`",
            f"- maximum pair budget: `{plan['schedule']['pair_max_act_rollouts']}` ACT rollouts",
            "- Table 2 consistency: unavailable at N=1",
            "",
            "## Audited sequence",
            "",
            "```sh",
            "set -euo pipefail",
            f"{validate}",
            f"{fixed}",
            f"{dynamic}",
            f"{validate}",
            f"{compare}",
            "```",
            "",
            "Both runs consume the same frozen registered route; Agent preflight and post-hoc comparison reject identity drift.",
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
    try:
        config_path = _inside(root, args.config, must_exist=True)
        config = read_repo_json(
            root, config_path.relative_to(root).as_posix(), label="strategy plan config"
        )
        plan = build_matched_strategy_plan(root, config)
        output = _inside(root, args.output_dir, must_exist=False)
        expected = root / "mea/validation_runs" / plan["plan_id"]
        if output != expected:
            raise StrategyPlanError(
                "--output-dir must equal "
                f"mea/validation_runs/{plan['plan_id']} so recorded commands remain exact"
            )
        if output.exists():
            raise StrategyPlanError(f"output directory already exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        _inside(root, output, must_exist=False)
        output.mkdir()
        _write_json(output / "command_plan.json", plan)
        _write_json(
            output / "registered_route.json",
            plan["registered_route"]["payload"],
        )
        _write_json(
            output / "strategy_comparison_config.json",
            plan["posthoc"]["comparison_config"],
        )
        (output / "commands.md").write_text(_report(plan), encoding="utf-8")
    except (EvidenceManifestError, StrategyPlanError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "status": "planned_not_started",
                "output_dir": output.relative_to(root).as_posix(),
                "act_rollouts_started": 0,
                "provider_calls_started": 0,
                "pair_max_act_rollouts": plan["schedule"]["pair_max_act_rollouts"],
                "paper_table_eligible": False,
                "table2_consistency": None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
