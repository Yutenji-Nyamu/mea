#!/usr/bin/env python3
"""Assess an explicit query-sufficiency contract from cached evidence.

This offline utility performs no provider, simulator, expert, probe, or ACT
calls and does not mutate the live planner.  It reads only the two explicitly
named JSON inputs, binds their candidate universe and budget to a ready
``BoundTaskPlanSession``, and writes one deterministic assessment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mea.planner import (
    BoundTaskPlanSession,
    build_act_catalog,
    validate_query_sufficiency_contract,
)


_COMPACT_FIXTURE_KEYS = {
    "schema_version",
    "source_kind",
    "candidate_evidence",
    "completed_rounds",
}


def _inside(root: Path, value: Path, name: str) -> Path:
    path = value.expanduser().resolve() if value.is_absolute() else (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{name} must remain inside --repo-root") from exc
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read JSON {path}: {exc}") from exc


def _candidate_fixture(value: Any) -> tuple[list[dict[str, Any]], int, str]:
    """Normalize either a bare evidence list or a compact cached fixture."""

    if isinstance(value, list):
        evidence = value
        completed_rounds = len(evidence)
        source_kind = "explicit_candidate_evidence_list"
    elif isinstance(value, dict):
        if set(value) != _COMPACT_FIXTURE_KEYS:
            raise SystemExit(
                "compact evidence fixture fields must be exactly "
                f"{sorted(_COMPACT_FIXTURE_KEYS)}"
            )
        if value.get("schema_version") != 1:
            raise SystemExit("compact evidence fixture schema_version must be 1")
        if value.get("source_kind") != "cached_compact_evidence":
            raise SystemExit(
                "compact evidence fixture source_kind must be "
                "cached_compact_evidence"
            )
        evidence = value.get("candidate_evidence")
        completed_rounds = value.get("completed_rounds")
        source_kind = "cached_compact_evidence"
    else:
        raise SystemExit(
            "candidate evidence JSON must be a list or compact fixture object"
        )
    if not isinstance(evidence, list) or any(
        not isinstance(item, dict) for item in evidence
    ):
        raise SystemExit("candidate_evidence must be a list of objects")
    if (
        isinstance(completed_rounds, bool)
        or not isinstance(completed_rounds, int)
        or completed_rounds < 0
    ):
        raise SystemExit("completed_rounds must be a non-negative integer")
    return evidence, completed_rounds, source_kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic cached/offline query-sufficiency assessment "
            "without starting ACT."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--contract-json", type=Path, required=True)
    parser.add_argument("--candidate-evidence-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.expanduser().resolve()
    contract_path = _inside(root, args.contract_json, "--contract-json")
    evidence_path = _inside(
        root, args.candidate_evidence_json, "--candidate-evidence-json"
    )
    output_path = _inside(root, args.output_json, "--output-json")
    if output_path.exists():
        raise SystemExit(f"output already exists: {output_path}")

    raw_contract = _read_json(contract_path)
    if not isinstance(raw_contract, dict):
        raise SystemExit("contract JSON must be an object")
    try:
        contract = validate_query_sufficiency_contract(raw_contract)
        evidence, completed_rounds, source_kind = _candidate_fixture(
            _read_json(evidence_path)
        )
        session = BoundTaskPlanSession.from_catalog(
            build_act_catalog(root),
            args.task_name,
            max_rounds=contract["round_budget"],
        )
        assessment = session.assess_query_sufficiency(
            contract,
            evidence,
            completed_rounds=completed_rounds,
        )
    except ValueError as exc:
        raise SystemExit(f"offline query sufficiency validation failed: {exc}") from exc

    output = {
        "schema_version": 1,
        "status": "offline_query_sufficiency_assessment",
        "execution_mode": "cached_offline_0_act",
        "task_name": session.target["task_name"],
        "checkpoint": session.target["checkpoint"],
        "contract_source": str(contract_path.relative_to(root)).replace("\\", "/"),
        "candidate_evidence_source": str(
            evidence_path.relative_to(root)
        ).replace("\\", "/"),
        "candidate_evidence_source_kind": source_kind,
        "live_planner_changed": False,
        "provider_calls_started": 0,
        "simulator_calls_started": 0,
        "act_rollouts_started": 0,
        "assessment": assessment,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
