"""Deterministic functional execution for module-ablation schedules.

This module deliberately implements a *development functional smoke*, not the
paper's human-reviewed Table 3 experiment.  It consumes a hash-bound prepared
schedule, applies every declared module switch to a small deterministic codegen
contract driver, and writes append-only evidence showing which modules actually
ran.  It never calls a provider, simulator, or policy.

The deterministic driver is useful for proving control flow and evidence
integrity before spending model/simulator budget.  Its binary outcome must not
be presented as a generation success rate from the paper.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from mea.module_ablation_protocol import (
    _canonical_relative,
    _canonical_sha256,
    _contract_payload,
    _item_payload,
    _safe_child_path,
)


CLAIM_SCOPE = "development deterministic functional-switch smoke only"
DRIVER_ID = "deterministic_codegen_contract_v1"
MANIFEST_PROTOCOL = "table3_module_ablation_development_execution_v1"
OUTCOME_TYPE = "module_ablation_development_functional_outcome_v1"
TRACE_TYPE = "module_ablation_functional_execution_trace_v1"


class FunctionalSwitchExecutionError(RuntimeError):
    """A schedule, switch, or append-only artifact violated the smoke contract."""


@dataclass(frozen=True)
class ResolvedSwitches:
    """Canonical switch view plus its paper/compatibility classification."""

    component: str
    condition: str
    declared: dict[str, bool]
    normalized: dict[str, bool]
    contract: str
    paper_table3_condition: bool


_PAPER_TABLE3: dict[str, dict[str, dict[str, bool]]] = {
    "taskgen": {
        "complete": {
            "rag": True,
            "visual_self_check": True,
            "readme_agent": True,
        },
        "no_rag": {
            "rag": False,
            "visual_self_check": True,
            "readme_agent": True,
        },
        "no_visual_self_check": {
            "rag": True,
            "visual_self_check": False,
            "readme_agent": True,
        },
        "no_readme_agent": {
            "rag": True,
            "visual_self_check": True,
            "readme_agent": False,
        },
        "base": {
            "rag": False,
            "visual_self_check": False,
            "readme_agent": False,
        },
    },
    "toolgen": {
        "complete": {"rag": True},
        "no_rag": {"rag": False},
    },
}

# These switch shapes are accepted only for explicit migration/compatibility
# tests. Schedule protocol v1 itself is not executable by this v2 executor.
_LEGACY: dict[str, dict[str, tuple[dict[str, bool], dict[str, bool]]]] = {
    "taskgen": {
        "complete": (
            {"rag": True, "visual_gate": True},
            {"rag": True, "visual_self_check": True, "readme_agent": True},
        ),
        "no_rag": (
            {"rag": False, "visual_gate": True},
            {"rag": False, "visual_self_check": True, "readme_agent": True},
        ),
        "no_visual_gate": (
            {"rag": True, "visual_gate": False},
            {"rag": True, "visual_self_check": False, "readme_agent": True},
        ),
    },
    "toolgen": {
        "complete": (
            {"tool_validation": True},
            {"rag": True, "tool_validation": True},
        ),
        "no_tool_validation": (
            {"tool_validation": False},
            {"rag": True, "tool_validation": False},
        ),
    },
}


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FunctionalSwitchExecutionError(
            f"append-only artifact already exists: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _boolean_switches(value: Any, *, field: str) -> dict[str, bool]:
    if not isinstance(value, Mapping) or not value:
        raise FunctionalSwitchExecutionError(f"{field} must be a non-empty object")
    result = dict(value)
    if any(not isinstance(key, str) or not isinstance(flag, bool) for key, flag in result.items()):
        raise FunctionalSwitchExecutionError(
            f"{field} keys must be strings and values must be booleans"
        )
    return result


def resolve_condition_switches(
    component: str, condition: str, module_switches: Mapping[str, Any]
) -> ResolvedSwitches:
    """Resolve an exact paper Table 3 or explicitly legacy switch contract."""

    declared = _boolean_switches(
        module_switches, field=f"{component}.{condition}.module_switches"
    )
    paper = _PAPER_TABLE3.get(component, {}).get(condition)
    if paper is not None and declared == paper:
        normalized = dict(paper)
        # Table 3 removes ToolGen RAG only. Candidate validation remains a
        # common invariant in both the complete and w/o-RAG branches.
        if component == "toolgen":
            normalized["tool_validation"] = True
        return ResolvedSwitches(
            component=component,
            condition=condition,
            declared=declared,
            normalized=normalized,
            contract="paper_table3_switch_shape",
            paper_table3_condition=True,
        )
    legacy = _LEGACY.get(component, {}).get(condition)
    if legacy is not None and declared == legacy[0]:
        return ResolvedSwitches(
            component=component,
            condition=condition,
            declared=declared,
            normalized=dict(legacy[1]),
            contract="legacy_non_paper_compatibility",
            paper_table3_condition=False,
        )
    raise FunctionalSwitchExecutionError(
        f"unregistered switch contract: {component}/{condition} {declared}"
    )


def _condition_contract_switches(
    schedule: Mapping[str, Any], component: str, condition: str
) -> dict[str, bool]:
    contracts = schedule.get("condition_contracts")
    if not isinstance(contracts, Mapping):
        raise FunctionalSwitchExecutionError("schedule condition_contracts is invalid")
    rows = contracts.get(component)
    if not isinstance(rows, list):
        raise FunctionalSwitchExecutionError(
            f"schedule has no condition contracts for {component}"
        )
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and row.get("condition") == condition
    ]
    if len(matches) != 1:
        raise FunctionalSwitchExecutionError(
            f"schedule requires exactly one condition contract for {component}/{condition}"
        )
    return _boolean_switches(
        matches[0].get("module_switches"),
        field=f"condition_contracts.{component}.{condition}.module_switches",
    )


def _validate_execution_schedule(
    repo_root: Path, schedule: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Validate the hash and identity bindings needed before any execution."""

    if not isinstance(schedule, Mapping):
        raise FunctionalSwitchExecutionError("schedule must be an object")
    if schedule.get("protocol") != "table3_module_ablation_schedule_v2":
        raise FunctionalSwitchExecutionError("unsupported schedule protocol")
    if schedule.get("status") != "prepared" or schedule.get("mode") != "prepare_only":
        raise FunctionalSwitchExecutionError("schedule must be prepared and prepare-only")
    contract_hash = schedule.get("schedule_contract_sha256")
    if contract_hash != _canonical_sha256(_contract_payload(schedule)):
        raise FunctionalSwitchExecutionError("schedule contract hash mismatch")
    artifact_root = _canonical_relative(
        schedule.get("artifact_root"), field="schedule.artifact_root"
    )
    _safe_child_path(repo_root, artifact_root, field="schedule.artifact_root")
    items = schedule.get("items")
    if not isinstance(items, list) or not items:
        raise FunctionalSwitchExecutionError("schedule items must be non-empty")
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, Mapping):
            raise FunctionalSwitchExecutionError("schedule item must be an object")
        item = dict(raw)
        item_id = item.get("schedule_item_id")
        if not isinstance(item_id, str) or not item_id or item_id in seen:
            raise FunctionalSwitchExecutionError(
                f"invalid or duplicate schedule item id: {item_id!r}"
            )
        seen.add(item_id)
        if item.get("status") != "scheduled":
            raise FunctionalSwitchExecutionError(f"item is not scheduled: {item_id}")
        if item.get("schedule_contract_sha256") != contract_hash:
            raise FunctionalSwitchExecutionError(
                f"schedule contract hash mismatch in item: {item_id}"
            )
        if item.get("schedule_item_sha256") != _canonical_sha256(_item_payload(item)):
            raise FunctionalSwitchExecutionError(f"schedule item hash mismatch: {item_id}")
        input_identity = item.get("input_identity")
        execution_identity = item.get("execution_identity")
        if not isinstance(input_identity, Mapping) or not isinstance(
            execution_identity, Mapping
        ):
            raise FunctionalSwitchExecutionError(
                f"item identities must be objects: {item_id}"
            )
        if item.get("input_identity_sha256") != _canonical_sha256(input_identity):
            raise FunctionalSwitchExecutionError(
                f"input identity hash mismatch: {item_id}"
            )
        if item.get("execution_identity_sha256") != _canonical_sha256(
            execution_identity
        ):
            raise FunctionalSwitchExecutionError(
                f"execution identity hash mismatch: {item_id}"
            )
        component = item.get("component")
        condition = item.get("condition")
        if not isinstance(component, str) or not isinstance(condition, str):
            raise FunctionalSwitchExecutionError(
                f"item component/condition must be strings: {item_id}"
            )
        declared = _boolean_switches(
            item.get("module_switches"), field=f"{item_id}.module_switches"
        )
        if declared != _condition_contract_switches(schedule, component, condition):
            raise FunctionalSwitchExecutionError(
                f"item switches differ from condition contract: {item_id}"
            )
        resolve_condition_switches(component, condition, declared)
        artifact_dir = _safe_child_path(
            repo_root, item.get("artifact_dir"), field=f"{item_id}.artifact_dir"
        )
        expected_manifest = _safe_child_path(
            repo_root,
            item.get("expected_manifest"),
            field=f"{item_id}.expected_manifest",
        )
        if expected_manifest != artifact_dir / "manifest.json":
            raise FunctionalSwitchExecutionError(
                f"non-canonical manifest path: {item_id}"
            )
        validated.append(item)
    return validated


class DeterministicCodegenContractDriver:
    """Small in-process driver whose calls expose whether switches took effect."""

    def __init__(self) -> None:
        self.call_counts: dict[str, int] = {}

    def _call(self, module: str) -> None:
        self.call_counts[module] = self.call_counts.get(module, 0) + 1

    def _taskgen(
        self, switches: Mapping[str, bool], input_identity: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if switches["rag"]:
            self._call("rag")
            api_contract_bound = True
        else:
            api_contract_bound = bool(input_identity.get("few_shot_api_contract", False))
        if switches["readme_agent"]:
            self._call("readme_agent")
            task_invariants_bound = True
        else:
            task_invariants_bound = bool(
                input_identity.get("inline_task_invariants", False)
            )
        self._call("codegen")
        scene_aligned = bool(input_identity.get("raw_scene_aligned", False))
        if switches["visual_self_check"]:
            self._call("visual_self_check")
            # The deterministic repair represents the existing observe/repair
            # contract, without claiming that a VLM or simulator ran.
            scene_aligned = True
        candidate = {
            "component": "taskgen",
            "api_contract_bound": api_contract_bound,
            "task_invariants_bound": task_invariants_bound,
            "scene_aligned": scene_aligned,
        }
        self._call("independent_judge")
        checks = {
            "api_contract_bound": api_contract_bound,
            "task_invariants_bound": task_invariants_bound,
            "scene_aligned": scene_aligned,
        }
        return candidate, checks

    def _toolgen(
        self, switches: Mapping[str, bool], input_identity: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if switches["rag"]:
            self._call("rag")
            metric_contract_bound = True
        else:
            metric_contract_bound = bool(
                input_identity.get("few_shot_metric_contract", False)
            )
        self._call("codegen")
        pure_function = not bool(input_identity.get("force_impure_candidate", False))
        if switches.get("tool_validation", False):
            self._call("tool_validation")
        candidate = {
            "component": "toolgen",
            "metric_contract_bound": metric_contract_bound,
            "pure_function": pure_function,
        }
        self._call("independent_judge")
        checks = {
            "metric_contract_bound": metric_contract_bound,
            "pure_function": pure_function,
        }
        return candidate, checks

    def execute(
        self,
        *,
        component: str,
        switches: Mapping[str, bool],
        input_identity: Mapping[str, Any],
    ) -> dict[str, Any]:
        if component == "taskgen":
            candidate, checks = self._taskgen(switches, input_identity)
            expected_modules = {
                "rag",
                "readme_agent",
                "codegen",
                "visual_self_check",
                "independent_judge",
            }
        elif component == "toolgen":
            candidate, checks = self._toolgen(switches, input_identity)
            expected_modules = {
                "rag",
                "codegen",
                "tool_validation",
                "independent_judge",
            }
        else:
            raise FunctionalSwitchExecutionError(
                f"unsupported execution component: {component}"
            )
        failed_checks = sorted(name for name, passed in checks.items() if not passed)
        call_counts = {
            module: self.call_counts.get(module, 0)
            for module in sorted(expected_modules)
        }
        return {
            "candidate": candidate,
            "judge": {
                "passed": not failed_checks,
                "checks": checks,
                "failed_checks": failed_checks,
                "source": "deterministic_independent_contract_judge",
            },
            "call_counts": call_counts,
        }


def _execute_item(
    repo_root: Path,
    schedule: Mapping[str, Any],
    item: Mapping[str, Any],
    development_root: Path,
) -> dict[str, Any]:
    item_id = str(item["schedule_item_id"])
    artifact_dir = (
        development_root
        / str(item["component"])
        / str(item["condition"])
        / str(item["case_id"])
    )
    if artifact_dir.exists():
        raise FunctionalSwitchExecutionError(
            f"append-only artifact directory already exists: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True, exist_ok=False)
    resolved = resolve_condition_switches(
        str(item["component"]),
        str(item["condition"]),
        item["module_switches"],
    )
    driver = DeterministicCodegenContractDriver()
    result = driver.execute(
        component=resolved.component,
        switches=resolved.normalized,
        input_identity=item["input_identity"],
    )
    candidate_path = artifact_dir / "candidate.json"
    _write_new(candidate_path, result["candidate"])
    candidate_sha256 = _file_sha256(candidate_path)
    success = bool(result["judge"]["passed"])
    trace = {
        "schema_version": 1,
        "evidence_type": TRACE_TYPE,
        "study_id": schedule["study_id"],
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        "schedule_item_id": item_id,
        "schedule_item_sha256": item["schedule_item_sha256"],
        "component": resolved.component,
        "condition": resolved.condition,
        "input_identity_sha256": item["input_identity_sha256"],
        "execution_identity_sha256": item["execution_identity_sha256"],
        "declared_module_switches": resolved.declared,
        "normalized_module_switches": resolved.normalized,
        "condition_contract": resolved.contract,
        "paper_table3_condition": resolved.paper_table3_condition,
        "driver": DRIVER_ID,
        "call_counts": result["call_counts"],
        "candidate_path": "candidate.json",
        "candidate_sha256": candidate_sha256,
        "judge": result["judge"],
        "runtime": {
            "provider_called": False,
            "simulator_called": False,
            "act_rollouts_started": 0,
        },
        "claim_scope": CLAIM_SCOPE,
        "paper_table_eligible": False,
    }
    trace_path = artifact_dir / "execution_trace.json"
    _write_new(trace_path, trace)
    trace_sha256 = _file_sha256(trace_path)
    outcome = {
        "schema_version": 1,
        "evidence_type": OUTCOME_TYPE,
        "study_id": schedule["study_id"],
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        "schedule_item_id": item_id,
        "schedule_item_sha256": item["schedule_item_sha256"],
        "component": resolved.component,
        "condition": resolved.condition,
        "case_id": item["case_id"],
        "input_identity_sha256": item["input_identity_sha256"],
        "execution_identity_sha256": item["execution_identity_sha256"],
        "applied_module_switches": resolved.declared,
        "measurement_kind": "development_functional_smoke",
        "success": success,
        "failure_stage": None if success else "independent_functional_judge",
        "execution_trace_path": "execution_trace.json",
        "execution_trace_sha256": trace_sha256,
        "claim_scope": CLAIM_SCOPE,
        "paper_table_eligible": False,
    }
    outcome_path = artifact_dir / "outcome.json"
    _write_new(outcome_path, outcome)
    outcome_sha256 = _file_sha256(outcome_path)
    manifest = {
        "schema_version": 1,
        "protocol": MANIFEST_PROTOCOL,
        "status": "completed",
        "study_id": schedule["study_id"],
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        "schedule_item_id": item_id,
        "schedule_item_sha256": item["schedule_item_sha256"],
        "component": resolved.component,
        "condition": resolved.condition,
        "case_id": item["case_id"],
        "input_identity_sha256": item["input_identity_sha256"],
        "execution_identity": item["execution_identity"],
        "execution_identity_sha256": item["execution_identity_sha256"],
        "applied_module_switches": resolved.declared,
        "execution_mode": DRIVER_ID,
        "claim_scope": CLAIM_SCOPE,
        "paper_table_eligible": False,
        "runtime": {
            "provider_called": False,
            "simulator_called": False,
            "act_rollouts_started": 0,
        },
        "result": {
            "measurement_kind": "development_functional_smoke",
            "success": success,
            "failure_stage": outcome["failure_stage"],
            "outcome_evidence_path": "outcome.json",
            "outcome_evidence_sha256": outcome_sha256,
            "execution_trace_path": "execution_trace.json",
            "execution_trace_sha256": trace_sha256,
        },
        "artifacts": [
            {
                "kind": "outcome_evidence",
                "path": "outcome.json",
                "sha256": outcome_sha256,
            },
            {
                "kind": "supporting_evidence",
                "path": "execution_trace.json",
                "sha256": trace_sha256,
            },
            {
                "kind": "supporting_evidence",
                "path": "candidate.json",
                "sha256": candidate_sha256,
            },
        ],
        "completed_at": _now(),
    }
    manifest_path = artifact_dir / "manifest.json"
    _write_new(manifest_path, manifest)
    return {
        "schedule_item_id": item_id,
        "component": resolved.component,
        "condition": resolved.condition,
        "success": success,
        "manifest": str(manifest_path.relative_to(repo_root)).replace("\\", "/"),
        "manifest_sha256": _file_sha256(manifest_path),
        "call_counts": result["call_counts"],
        "paper_table3_condition": resolved.paper_table3_condition,
        "paper_table_eligible": False,
    }


def execute_module_ablation_schedule(
    repo_root: str | Path,
    schedule: Mapping[str, Any],
    *,
    schedule_item_ids: Sequence[str] | None = None,
    development_artifact_root: str | Path | None = None,
) -> dict[str, Any]:
    """Execute selected scheduled items using the zero-runtime smoke driver."""

    root = Path(repo_root).expanduser().resolve()
    items = _validate_execution_schedule(root, schedule)
    formal_root = _safe_child_path(
        root, schedule["artifact_root"], field="schedule.artifact_root"
    )
    if development_artifact_root is None:
        development = root / f"{schedule['artifact_root']}__development_smoke"
    else:
        raw = Path(development_artifact_root).expanduser()
        development = raw if raw.is_absolute() else root / raw
    development = development.resolve()
    if not development.is_relative_to(root) or development == root:
        raise FunctionalSwitchExecutionError(
            "development_artifact_root must stay inside repo_root"
        )
    if (
        development == formal_root
        or development.is_relative_to(formal_root)
        or formal_root.is_relative_to(development)
    ):
        raise FunctionalSwitchExecutionError(
            "development smoke and formal schedule artifact_root must be disjoint"
        )
    cursor = root
    for part in development.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise FunctionalSwitchExecutionError(
                "development_artifact_root contains a symlink component"
            )
    if development.exists():
        raise FunctionalSwitchExecutionError(
            f"append-only development artifact root already exists: {development}"
        )
    by_id = {str(item["schedule_item_id"]): item for item in items}
    selected = list(schedule_item_ids) if schedule_item_ids is not None else list(by_id)
    if not selected or any(not isinstance(item_id, str) for item_id in selected):
        raise FunctionalSwitchExecutionError(
            "schedule_item_ids must be a non-empty string sequence"
        )
    if len(selected) != len(set(selected)):
        raise FunctionalSwitchExecutionError("schedule_item_ids contains duplicates")
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise FunctionalSwitchExecutionError(f"unknown schedule item ids: {unknown}")
    results = [
        _execute_item(root, schedule, by_id[item_id], development)
        for item_id in selected
    ]
    return {
        "schema_version": 1,
        "execution_mode": DRIVER_ID,
        "status": "completed",
        "study_id": schedule["study_id"],
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        "selected_item_count": len(results),
        "formal_schedule_artifact_root": schedule["artifact_root"],
        "development_artifact_root": str(development.relative_to(root)).replace(
            "\\", "/"
        ),
        "claim_scope": CLAIM_SCOPE,
        "paper_table_eligible": False,
        "runtime": {
            "provider_called": False,
            "simulator_called": False,
            "act_rollouts_started": 0,
        },
        "items": results,
    }


__all__ = [
    "CLAIM_SCOPE",
    "DeterministicCodegenContractDriver",
    "FunctionalSwitchExecutionError",
    "ResolvedSwitches",
    "execute_module_ablation_schedule",
    "resolve_condition_switches",
]
