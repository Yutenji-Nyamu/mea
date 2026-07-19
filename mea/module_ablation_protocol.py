"""Bounded, artifact-only TaskGen/ToolGen module-ablation protocol.

This module has two deliberately separate phases:

* :func:`prepare_module_ablation_schedule` freezes a complete matched matrix.
* :func:`audit_module_ablation_artifacts` verifies already-completed artifacts.

Neither phase calls a model provider, RoboTwin, or ACT.  Historical runtime is
self-attested by each completed manifest and is reported separately.  Effects
are functional-only and are available only for a complete set of matched typed
generation outcomes; provenance is never interpreted as an outcome.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Mapping


class ModuleAblationError(RuntimeError):
    """Raised when a schedule or completed artifact violates the protocol."""


CONDITION_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    "taskgen": {
        "complete": {
            "description": "RAG, visual self-check, and README.Agent are enabled.",
            "module_switches": {
                "rag": True,
                "visual_self_check": True,
                "readme_agent": True,
            },
        },
        "no_rag": {
            "description": "Disable only TaskGen retrieval/RAG.",
            "module_switches": {
                "rag": False,
                "visual_self_check": True,
                "readme_agent": True,
            },
        },
        "no_visual_self_check": {
            "description": "Disable only TaskGen visual self-check and repair.",
            "module_switches": {
                "rag": True,
                "visual_self_check": False,
                "readme_agent": True,
            },
        },
        "no_readme_agent": {
            "description": "Disable only TaskGen README.Agent guidance.",
            "module_switches": {
                "rag": True,
                "visual_self_check": True,
                "readme_agent": False,
            },
        },
        "base": {
            "description": "Disable RAG, visual self-check, and README.Agent.",
            "module_switches": {
                "rag": False,
                "visual_self_check": False,
                "readme_agent": False,
            },
        },
        "no_visual_gate": {
            "description": (
                "Legacy engineering compatibility: disable the old visual gate."
            ),
            "module_switches": {"rag": True, "visual_gate": False},
        },
    },
    "toolgen": {
        "complete": {
            "description": "Tool retrieval/RAG is enabled.",
            "module_switches": {"rag": True},
        },
        "no_rag": {
            "description": "Disable only ToolGen retrieval/RAG.",
            "module_switches": {"rag": False},
        },
        "no_tool_validation": {
            "description": (
                "Legacy engineering compatibility: disable generated-tool validation."
            ),
            "module_switches": {"tool_validation": False},
        },
    },
}

EXECUTION_IDENTITY_FIELDS = (
    "base_commit",
    "runner",
    "runner_sha256",
    "provider_model",
    "config_sha256",
    "seed",
)

_SCHEDULE_PROTOCOL = "table3_module_ablation_schedule_v2"
_ARTIFACT_PROTOCOL = "table3_module_ablation_completed_artifact_v1"
_AUDIT_PROTOCOL = "table3_module_ablation_completed_artifact_audit_v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_ITEM_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,511}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{7,64}")


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModuleAblationError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ModuleAblationError(
            f"{field} fields do not match the exact contract; "
            f"missing={missing}, extra={extra}"
        )


def _identifier(value: Any, *, field: str, item: bool = False) -> str:
    pattern = _ITEM_IDENTIFIER if item else _IDENTIFIER
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ModuleAblationError(f"{field} has an invalid identifier: {value!r}")
    return value


def _canonical_relative(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ModuleAblationError(f"{field} must be a canonical POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
        or path.as_posix() != value
    ):
        raise ModuleAblationError(f"{field} must be a canonical POSIX relative path")
    return value


def _safe_child_path(root: Path, value: Any, *, field: str) -> Path:
    """Resolve a canonical relative path and reject every symlink component."""

    canonical = _canonical_relative(value, field=field)
    cursor = root
    for part in PurePosixPath(canonical).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ModuleAblationError(f"{field} contains a symlink component: {part}")
    resolved = cursor.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ModuleAblationError(f"{field} escapes or equals its root")
    return resolved


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModuleAblationError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModuleAblationError(f"JSON artifact must be an object: {path}")
    return value


def _execution_identity(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModuleAblationError(f"{field} must be an object")
    _require_exact_keys(value, set(EXECUTION_IDENTITY_FIELDS), field=field)
    normalized = {name: value[name] for name in EXECUTION_IDENTITY_FIELDS}
    base_commit = normalized["base_commit"]
    if base_commit is not None and (
        not isinstance(base_commit, str) or _COMMIT.fullmatch(base_commit) is None
    ):
        raise ModuleAblationError(f"{field}.base_commit must be a commit hash or null")
    runner = normalized["runner"]
    runner_sha256 = normalized["runner_sha256"]
    if runner is not None:
        normalized["runner"] = _canonical_relative(runner, field=f"{field}.runner")
    if runner_sha256 is not None and (
        not isinstance(runner_sha256, str) or _SHA256.fullmatch(runner_sha256) is None
    ):
        raise ModuleAblationError(f"{field}.runner_sha256 must be SHA256 or null")
    if (runner is None) != (runner_sha256 is None):
        raise ModuleAblationError(f"{field}.runner and runner_sha256 must be paired")
    provider_model = normalized["provider_model"]
    if provider_model is not None and (
        not isinstance(provider_model, str) or not provider_model.strip()
    ):
        raise ModuleAblationError(f"{field}.provider_model must be non-empty or null")
    config_sha256 = normalized["config_sha256"]
    if config_sha256 is not None and (
        not isinstance(config_sha256, str) or _SHA256.fullmatch(config_sha256) is None
    ):
        raise ModuleAblationError(f"{field}.config_sha256 must be SHA256 or null")
    seed = normalized["seed"]
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ModuleAblationError(f"{field}.seed must be an integer or null")
    return normalized


def _condition_contracts(
    components: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    contracts: dict[str, list[dict[str, Any]]] = {}
    cases: list[dict[str, Any]] = []
    seen_cases: set[tuple[str, str]] = set()
    for component, component_config in components.items():
        if component not in CONDITION_SPECS:
            raise ModuleAblationError(f"unsupported component: {component!r}")
        if not isinstance(component_config, Mapping):
            raise ModuleAblationError(f"{component} config must be an object")
        conditions = component_config.get("conditions")
        if (
            not isinstance(conditions, list)
            or not conditions
            or any(not isinstance(condition, str) for condition in conditions)
            or len(conditions) != len(set(conditions))
        ):
            raise ModuleAblationError(
                f"{component} conditions must be a unique non-empty string list"
            )
        unsupported = sorted(set(conditions) - set(CONDITION_SPECS[component]))
        if unsupported:
            raise ModuleAblationError(
                f"unsupported {component} conditions: {unsupported}; allowed: "
                f"{sorted(CONDITION_SPECS[component])}"
            )
        if "complete" not in conditions or len(conditions) < 2:
            raise ModuleAblationError(
                f"{component} requires complete and at least one module-off condition"
            )
        contracts[component] = [
            {
                "condition": condition,
                "description": CONDITION_SPECS[component][condition]["description"],
                "module_switches": dict(
                    CONDITION_SPECS[component][condition]["module_switches"]
                ),
            }
            for condition in conditions
        ]
        raw_cases = component_config.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ModuleAblationError(f"{component} cases must be a non-empty list")
        for raw_case in raw_cases:
            if not isinstance(raw_case, Mapping):
                raise ModuleAblationError(f"{component} case must be an object")
            _require_exact_keys(
                raw_case,
                {"case_id", "input_identity", "execution_identity"},
                field=f"{component} case",
            )
            case_id = _identifier(raw_case.get("case_id"), field=f"{component}.case_id")
            key = (component, case_id)
            if key in seen_cases:
                raise ModuleAblationError(f"duplicate case: {component}/{case_id}")
            seen_cases.add(key)
            input_identity = raw_case.get("input_identity")
            if not isinstance(input_identity, Mapping) or not input_identity:
                raise ModuleAblationError(
                    f"{component}/{case_id} input_identity must be non-empty"
                )
            input_identity = dict(input_identity)
            execution_identity = _execution_identity(
                raw_case.get("execution_identity"),
                field=f"{component}/{case_id}.execution_identity",
            )
            cases.append(
                {
                    "component": component,
                    "case_id": case_id,
                    "conditions": list(conditions),
                    "input_identity": input_identity,
                    "input_identity_sha256": _canonical_sha256(input_identity),
                    "execution_identity": execution_identity,
                    "execution_identity_sha256": _canonical_sha256(execution_identity),
                }
            )
    return contracts, cases


def _contract_payload(schedule: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": _SCHEDULE_PROTOCOL,
        "study_id": schedule["study_id"],
        "artifact_root": schedule["artifact_root"],
        "condition_contracts": schedule["condition_contracts"],
        "matched_sets": schedule["matched_sets"],
    }


def _item_payload(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "protocol": _SCHEDULE_PROTOCOL,
        "schedule_contract_sha256": item["schedule_contract_sha256"],
        "schedule_item_id": item["schedule_item_id"],
        "component": item["component"],
        "condition": item["condition"],
        "case_id": item["case_id"],
        "input_identity_sha256": item["input_identity_sha256"],
        "execution_identity_sha256": item["execution_identity_sha256"],
        "module_switches": item["module_switches"],
        "artifact_dir": item["artifact_dir"],
        "expected_manifest": item["expected_manifest"],
    }


def prepare_module_ablation_schedule(
    repo_root: str | Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Freeze a complete prepare-only matrix without running an experiment."""

    root = Path(repo_root).expanduser().resolve()
    if config.get("schema_version") != 1:
        raise ModuleAblationError("config schema_version must be 1")
    study_id = _identifier(config.get("study_id"), field="study_id")
    artifact_root = _canonical_relative(
        config.get("artifact_root"), field="artifact_root"
    )
    _safe_child_path(root, artifact_root, field="artifact_root")
    components = config.get("components")
    if not isinstance(components, Mapping) or not components:
        raise ModuleAblationError("components must be a non-empty object")
    condition_contracts, cases = _condition_contracts(components)

    matched_sets: list[dict[str, Any]] = []
    for case in cases:
        component = case["component"]
        case_id = case["case_id"]
        conditions = case["conditions"]
        matched_sets.append(
            {
                "component": component,
                "case_id": case_id,
                "conditions": list(conditions),
                "schedule_item_ids": {
                    condition: f"{component}.{condition}.{case_id}"
                    for condition in conditions
                },
                "input_identity": case["input_identity"],
                "input_identity_sha256": case["input_identity_sha256"],
                "execution_identity": case["execution_identity"],
                "execution_identity_sha256": case["execution_identity_sha256"],
            }
        )
    contract_holder = {
        "study_id": study_id,
        "artifact_root": artifact_root,
        "condition_contracts": condition_contracts,
        "matched_sets": matched_sets,
    }
    schedule_contract_sha256 = _canonical_sha256(_contract_payload(contract_holder))

    items: list[dict[str, Any]] = []
    for case in cases:
        component = case["component"]
        case_id = case["case_id"]
        for condition in case["conditions"]:
            artifact_dir = f"{artifact_root}/{component}/{condition}/{case_id}"
            expected_manifest = f"{artifact_dir}/manifest.json"
            _safe_child_path(root, artifact_dir, field="artifact_dir")
            item = {
                "schedule_item_id": f"{component}.{condition}.{case_id}",
                "schedule_contract_sha256": schedule_contract_sha256,
                "status": "scheduled",
                "component": component,
                "condition": condition,
                "case_id": case_id,
                "input_identity": case["input_identity"],
                "input_identity_sha256": case["input_identity_sha256"],
                "execution_identity": case["execution_identity"],
                "execution_identity_sha256": case["execution_identity_sha256"],
                "module_switches": dict(
                    CONDITION_SPECS[component][condition]["module_switches"]
                ),
                "artifact_dir": artifact_dir,
                "expected_manifest": expected_manifest,
            }
            item["schedule_item_sha256"] = _canonical_sha256(_item_payload(item))
            items.append(item)

    return {
        "schema_version": 1,
        "protocol": _SCHEDULE_PROTOCOL,
        "status": "prepared",
        "mode": "prepare_only",
        "study_id": study_id,
        "artifact_root": artifact_root,
        "schedule_contract_sha256": schedule_contract_sha256,
        "claim_scope": (
            "matched TaskGen/ToolGen functional generation outcomes only; "
            "zero ACT and not a paper Table 3 result"
        ),
        "paper_table_eligible": False,
        "runtime": {
            "provider_called": False,
            "simulator_called": False,
            "act_rollouts_started": 0,
        },
        "condition_contracts": condition_contracts,
        "items": items,
        "matched_sets": matched_sets,
    }


def _validate_condition_contracts(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping) or not value:
        raise ModuleAblationError("condition_contracts must be a non-empty object")
    conditions_by_component: dict[str, list[str]] = {}
    for component, rows in value.items():
        if component not in CONDITION_SPECS:
            raise ModuleAblationError(f"unsupported contract component: {component!r}")
        if not isinstance(rows, list) or not rows:
            raise ModuleAblationError(f"{component} condition contract must be a list")
        conditions: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ModuleAblationError(f"{component} condition contract row invalid")
            _require_exact_keys(
                row,
                {"condition", "description", "module_switches"},
                field=f"{component} condition contract",
            )
            condition = row.get("condition")
            if condition not in CONDITION_SPECS[component]:
                raise ModuleAblationError(
                    f"unsupported contract condition: {component}/{condition}"
                )
            if condition in conditions:
                raise ModuleAblationError(
                    f"duplicate contract condition: {component}/{condition}"
                )
            spec = CONDITION_SPECS[component][condition]
            if row.get("description") != spec["description"] or row.get(
                "module_switches"
            ) != spec["module_switches"]:
                raise ModuleAblationError(
                    f"condition contract does not match registry: {component}/{condition}"
                )
            conditions.append(condition)
        if "complete" not in conditions or len(conditions) < 2:
            raise ModuleAblationError(
                f"{component} contract requires complete and a module-off condition"
            )
        conditions_by_component[component] = conditions
    return conditions_by_component


def _validate_schedule(
    root: Path, schedule: Mapping[str, Any]
) -> list[dict[str, Any]]:
    expected_schedule_fields = {
        "schema_version",
        "protocol",
        "status",
        "mode",
        "study_id",
        "artifact_root",
        "schedule_contract_sha256",
        "claim_scope",
        "paper_table_eligible",
        "runtime",
        "condition_contracts",
        "items",
        "matched_sets",
    }
    _require_exact_keys(schedule, expected_schedule_fields, field="schedule")
    if schedule.get("schema_version") != 1 or schedule.get("protocol") != _SCHEDULE_PROTOCOL:
        raise ModuleAblationError("unsupported schedule schema/protocol")
    if schedule.get("status") != "prepared" or schedule.get("mode") != "prepare_only":
        raise ModuleAblationError("schedule must be a prepared prepare-only artifact")
    if schedule.get("paper_table_eligible") is not False:
        raise ModuleAblationError("prepare-only schedule cannot be paper-table eligible")
    if schedule.get("runtime") != {
        "provider_called": False,
        "simulator_called": False,
        "act_rollouts_started": 0,
    }:
        raise ModuleAblationError("schedule runtime must declare zero external calls")
    _identifier(schedule.get("study_id"), field="schedule.study_id")
    artifact_root = _canonical_relative(
        schedule.get("artifact_root"), field="schedule.artifact_root"
    )
    _safe_child_path(root, artifact_root, field="schedule.artifact_root")
    conditions_by_component = _validate_condition_contracts(
        schedule.get("condition_contracts")
    )

    matched_sets = schedule.get("matched_sets")
    if not isinstance(matched_sets, list) or not matched_sets:
        raise ModuleAblationError("matched_sets must be a non-empty list")
    matched_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    expected_item_keys: set[tuple[str, str, str]] = set()
    matched_fields = {
        "component",
        "case_id",
        "conditions",
        "schedule_item_ids",
        "input_identity",
        "input_identity_sha256",
        "execution_identity",
        "execution_identity_sha256",
    }
    for raw in matched_sets:
        if not isinstance(raw, Mapping):
            raise ModuleAblationError("matched set must be an object")
        _require_exact_keys(raw, matched_fields, field="matched set")
        component = raw.get("component")
        if component not in conditions_by_component:
            raise ModuleAblationError(f"matched set component is not contracted: {component}")
        case_id = _identifier(raw.get("case_id"), field=f"{component}.case_id")
        key = (component, case_id)
        if key in matched_by_key:
            raise ModuleAblationError(f"duplicate matched set: {component}/{case_id}")
        conditions = raw.get("conditions")
        if conditions != conditions_by_component[component]:
            raise ModuleAblationError(
                f"matched set is not the full condition contract: {component}/{case_id}"
            )
        expected_ids = {
            condition: f"{component}.{condition}.{case_id}"
            for condition in conditions_by_component[component]
        }
        if raw.get("schedule_item_ids") != expected_ids:
            raise ModuleAblationError(
                f"matched set schedule item IDs mismatch: {component}/{case_id}"
            )
        input_identity = raw.get("input_identity")
        if not isinstance(input_identity, Mapping) or not input_identity:
            raise ModuleAblationError(f"matched input identity invalid: {component}/{case_id}")
        input_sha256 = _canonical_sha256(input_identity)
        if raw.get("input_identity_sha256") != input_sha256:
            raise ModuleAblationError(f"matched input identity hash mismatch: {component}/{case_id}")
        execution_identity = _execution_identity(
            raw.get("execution_identity"),
            field=f"{component}/{case_id}.execution_identity",
        )
        execution_sha256 = _canonical_sha256(execution_identity)
        if raw.get("execution_identity_sha256") != execution_sha256:
            raise ModuleAblationError(
                f"matched execution identity hash mismatch: {component}/{case_id}"
            )
        normalized = dict(raw)
        normalized["input_identity"] = dict(input_identity)
        normalized["execution_identity"] = execution_identity
        matched_by_key[key] = normalized
        expected_item_keys.update(
            (component, case_id, condition)
            for condition in conditions_by_component[component]
        )
    if set(conditions_by_component) != {key[0] for key in matched_by_key}:
        raise ModuleAblationError("every contracted component needs at least one matched set")

    schedule_contract_sha256 = schedule.get("schedule_contract_sha256")
    if not isinstance(schedule_contract_sha256, str) or _SHA256.fullmatch(
        schedule_contract_sha256
    ) is None:
        raise ModuleAblationError("schedule_contract_sha256 is invalid")
    if schedule_contract_sha256 != _canonical_sha256(_contract_payload(schedule)):
        raise ModuleAblationError("schedule contract hash mismatch")

    items = schedule.get("items")
    if not isinstance(items, list) or not items:
        raise ModuleAblationError("schedule items must be a non-empty list")
    item_fields = {
        "schedule_item_id",
        "schedule_item_sha256",
        "schedule_contract_sha256",
        "status",
        "component",
        "condition",
        "case_id",
        "input_identity",
        "input_identity_sha256",
        "execution_identity",
        "execution_identity_sha256",
        "module_switches",
        "artifact_dir",
        "expected_manifest",
    }
    validated: list[dict[str, Any]] = []
    seen_item_keys: set[tuple[str, str, str]] = set()
    for raw in items:
        if not isinstance(raw, Mapping):
            raise ModuleAblationError("schedule item must be an object")
        _require_exact_keys(raw, item_fields, field="schedule item")
        item = dict(raw)
        component = item.get("component")
        condition = item.get("condition")
        case_id = item.get("case_id")
        if not isinstance(component, str) or not isinstance(condition, str):
            raise ModuleAblationError("schedule item component/condition must be strings")
        case_id = _identifier(case_id, field="schedule item case_id")
        key = (component, case_id, condition)
        if key not in expected_item_keys:
            raise ModuleAblationError(f"schedule item is outside the matched matrix: {key}")
        if key in seen_item_keys:
            raise ModuleAblationError(f"duplicate schedule item: {key}")
        seen_item_keys.add(key)
        matched = matched_by_key.get((component, case_id))
        if matched is None:
            raise ModuleAblationError(f"schedule item has no matched set: {key}")
        item_id = _identifier(
            item.get("schedule_item_id"), field="schedule_item_id", item=True
        )
        if item_id != matched["schedule_item_ids"][condition] or item.get(
            "status"
        ) != "scheduled":
            raise ModuleAblationError(f"schedule item identity/status mismatch: {key}")
        if item.get("schedule_contract_sha256") != schedule_contract_sha256:
            raise ModuleAblationError(f"schedule contract hash mismatch in item: {item_id}")
        for identity_name in ("input_identity", "input_identity_sha256"):
            if item.get(identity_name) != matched[identity_name]:
                raise ModuleAblationError(
                    f"cross-condition input identity mismatch: {component}/{case_id}"
                )
        execution_identity = _execution_identity(
            item.get("execution_identity"), field=f"{item_id}.execution_identity"
        )
        if execution_identity != matched["execution_identity"] or item.get(
            "execution_identity_sha256"
        ) != matched["execution_identity_sha256"]:
            raise ModuleAblationError(
                f"cross-condition execution identity mismatch: {component}/{case_id}"
            )
        switches = CONDITION_SPECS[component][condition]["module_switches"]
        if item.get("module_switches") != switches:
            raise ModuleAblationError(f"module switches mismatch: {item_id}")
        artifact_dir = f"{artifact_root}/{component}/{condition}/{case_id}"
        manifest = f"{artifact_dir}/manifest.json"
        if item.get("artifact_dir") != artifact_dir or item.get(
            "expected_manifest"
        ) != manifest:
            raise ModuleAblationError(f"non-canonical artifact path: {item_id}")
        _safe_child_path(root, artifact_dir, field=f"{item_id}.artifact_dir")
        _safe_child_path(root, manifest, field=f"{item_id}.expected_manifest")
        if item.get("schedule_item_sha256") != _canonical_sha256(_item_payload(item)):
            raise ModuleAblationError(f"schedule item hash mismatch: {item_id}")
        item["execution_identity"] = execution_identity
        validated.append(item)
    if seen_item_keys != expected_item_keys:
        missing = sorted(expected_item_keys - seen_item_keys)
        raise ModuleAblationError(f"schedule matrix is incomplete; missing={missing}")
    return validated


def _missing_audit_row(item: Mapping[str, Any], manifest: str, status: str) -> dict[str, Any]:
    return {
        "schedule_item_id": item["schedule_item_id"],
        "component": item["component"],
        "condition": item["condition"],
        "case_id": item["case_id"],
        "manifest": manifest,
        "artifact_status": status,
        "completed": False,
        "effect_eligible": False,
        "measurement_kind": None,
        "success": None,
        "unavailable_reason": (
            "completed artifact manifest is missing"
            if status == "missing"
            else "artifact status is not completed"
        ),
        "historical_runtime": None,
        "verified_artifacts": [],
    }


def _validate_runtime(value: Any, *, item_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ModuleAblationError(f"completed artifact {item_id} has no runtime")
    _require_exact_keys(
        value,
        {"provider_called", "simulator_called", "act_rollouts_started"},
        field=f"completed artifact {item_id} runtime",
    )
    provider_called = value.get("provider_called")
    simulator_called = value.get("simulator_called")
    act_rollouts = value.get("act_rollouts_started")
    if not isinstance(provider_called, bool) or not isinstance(simulator_called, bool):
        raise ModuleAblationError(f"completed artifact {item_id} runtime booleans invalid")
    if isinstance(act_rollouts, bool) or not isinstance(act_rollouts, int):
        raise ModuleAblationError(f"completed artifact {item_id} ACT count invalid")
    if act_rollouts != 0:
        raise ModuleAblationError(
            f"completed artifact {item_id} violates the zero-ACT protocol"
        )
    return {
        "provider_called": provider_called,
        "simulator_called": simulator_called,
        "act_rollouts_started": act_rollouts,
    }


def _validate_artifact_refs(
    root: Path,
    artifact_dir: Path,
    manifest_path: Path,
    refs: Any,
    *,
    item_id: str,
) -> tuple[list[dict[str, str]], dict[str, str]]:
    if not isinstance(refs, list) or not refs:
        raise ModuleAblationError(f"completed artifact {item_id} has no evidence files")
    verified: list[dict[str, str]] = []
    outcome_refs: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in refs:
        if not isinstance(raw, Mapping):
            raise ModuleAblationError(f"completed artifact {item_id} evidence ref invalid")
        _require_exact_keys(
            raw, {"kind", "path", "sha256"}, field=f"{item_id} evidence ref"
        )
        kind = raw.get("kind")
        if kind not in {"outcome_evidence", "supporting_evidence"}:
            raise ModuleAblationError(f"completed artifact {item_id} evidence kind invalid")
        relative = _canonical_relative(raw.get("path"), field=f"{item_id} evidence path")
        if relative in seen:
            raise ModuleAblationError(f"completed artifact {item_id} duplicate evidence path")
        seen.add(relative)
        evidence_path = _safe_child_path(
            artifact_dir, relative, field=f"{item_id} evidence path"
        )
        if evidence_path == manifest_path or not evidence_path.is_file():
            raise ModuleAblationError(f"completed artifact {item_id} evidence is missing")
        expected_sha256 = raw.get("sha256")
        if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
            raise ModuleAblationError(f"completed artifact {item_id} evidence hash invalid")
        actual_sha256 = _file_sha256(evidence_path)
        if actual_sha256 != expected_sha256:
            raise ModuleAblationError(f"completed artifact {item_id} evidence hash mismatch")
        row = {
            "kind": kind,
            "path": str(evidence_path.relative_to(root)).replace("\\", "/"),
            "relative_path": relative,
            "sha256": actual_sha256,
        }
        verified.append(row)
        if kind == "outcome_evidence":
            outcome_refs.append(row)
    if len(outcome_refs) != 1:
        raise ModuleAblationError(
            f"completed artifact {item_id} requires exactly one typed outcome evidence"
        )
    return verified, outcome_refs[0]


def _validate_typed_outcome(
    evidence: Mapping[str, Any],
    *,
    schedule: Mapping[str, Any],
    item: Mapping[str, Any],
    measurement_kind: str,
    success: bool | None,
) -> None:
    fields = {
        "schema_version",
        "evidence_type",
        "study_id",
        "schedule_contract_sha256",
        "schedule_item_id",
        "schedule_item_sha256",
        "component",
        "condition",
        "case_id",
        "input_identity_sha256",
        "execution_identity_sha256",
        "applied_module_switches",
        "measurement_kind",
        "success",
    }
    _require_exact_keys(evidence, fields, field=f"{item['schedule_item_id']} outcome evidence")
    evidence_type = (
        "module_ablation_generation_outcome_v1"
        if measurement_kind == "generation_outcome"
        else "module_ablation_provenance_only_v1"
    )
    expected = {
        "schema_version": 1,
        "evidence_type": evidence_type,
        "study_id": schedule["study_id"],
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        "schedule_item_id": item["schedule_item_id"],
        "schedule_item_sha256": item["schedule_item_sha256"],
        "component": item["component"],
        "condition": item["condition"],
        "case_id": item["case_id"],
        "input_identity_sha256": item["input_identity_sha256"],
        "execution_identity_sha256": item["execution_identity_sha256"],
        "applied_module_switches": item["module_switches"],
        "measurement_kind": measurement_kind,
        "success": success,
    }
    if dict(evidence) != expected:
        raise ModuleAblationError(
            f"typed outcome evidence is not bound to result/condition/identity/switches: "
            f"{item['schedule_item_id']}"
        )


def _audit_completed_item(
    root: Path, schedule: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    item_id = str(item["schedule_item_id"])
    artifact_dir = _safe_child_path(
        root, item["artifact_dir"], field=f"{item_id}.artifact_dir"
    )
    manifest_path = _safe_child_path(
        root, item["expected_manifest"], field=f"{item_id}.expected_manifest"
    )
    manifest_relative = str(manifest_path.relative_to(root)).replace("\\", "/")
    if manifest_path.parent != artifact_dir:
        raise ModuleAblationError(f"manifest must be directly inside {item_id} artifact dir")
    if not manifest_path.exists():
        return _missing_audit_row(item, manifest_relative, "missing")
    if not manifest_path.is_file():
        raise ModuleAblationError(f"manifest is not a regular file: {item_id}")
    manifest = _read_object(manifest_path)
    if manifest.get("status") != "completed":
        return _missing_audit_row(
            item, manifest_relative, str(manifest.get("status") or "unknown")
        )
    manifest_fields = {
        "schema_version",
        "protocol",
        "status",
        "study_id",
        "schedule_contract_sha256",
        "schedule_item_id",
        "schedule_item_sha256",
        "component",
        "condition",
        "case_id",
        "input_identity_sha256",
        "execution_identity",
        "execution_identity_sha256",
        "applied_module_switches",
        "runtime",
        "result",
        "artifacts",
    }
    _require_exact_keys(manifest, manifest_fields, field=f"completed manifest {item_id}")
    expected_scalar = {
        "schema_version": 1,
        "protocol": _ARTIFACT_PROTOCOL,
        "status": "completed",
        "study_id": schedule["study_id"],
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        "schedule_item_id": item_id,
        "schedule_item_sha256": item["schedule_item_sha256"],
        "component": item["component"],
        "condition": item["condition"],
        "case_id": item["case_id"],
        "input_identity_sha256": item["input_identity_sha256"],
        "execution_identity_sha256": item["execution_identity_sha256"],
    }
    for field, expected in expected_scalar.items():
        if manifest.get(field) != expected:
            raise ModuleAblationError(
                f"completed artifact {item_id} {field} mismatch: "
                f"{manifest.get(field)!r} != {expected!r}"
            )
    execution_identity = _execution_identity(
        manifest.get("execution_identity"), field=f"{item_id}.execution_identity"
    )
    if execution_identity != item["execution_identity"] or _canonical_sha256(
        execution_identity
    ) != manifest["execution_identity_sha256"]:
        raise ModuleAblationError(f"completed artifact {item_id} execution identity mismatch")
    if manifest.get("applied_module_switches") != item["module_switches"]:
        raise ModuleAblationError(f"completed artifact {item_id} applied switches mismatch")
    runtime = _validate_runtime(manifest.get("runtime"), item_id=item_id)
    result = manifest.get("result")
    if not isinstance(result, Mapping):
        raise ModuleAblationError(f"completed artifact {item_id} has no result")
    _require_exact_keys(
        result,
        {
            "measurement_kind",
            "success",
            "outcome_evidence_path",
            "outcome_evidence_sha256",
        },
        field=f"completed artifact {item_id} result",
    )
    measurement_kind = result.get("measurement_kind")
    if measurement_kind not in {"generation_outcome", "provenance_only"}:
        raise ModuleAblationError(
            f"completed artifact {item_id} measurement_kind is unsupported"
        )
    success_value = result.get("success")
    if measurement_kind == "generation_outcome":
        if not isinstance(success_value, bool):
            raise ModuleAblationError(
                f"completed artifact {item_id} generation success must be boolean"
            )
        success: bool | None = success_value
    else:
        if success_value is not None:
            raise ModuleAblationError(
                f"completed artifact {item_id} provenance success must be null"
            )
        success = None
    outcome_path = _canonical_relative(
        result.get("outcome_evidence_path"), field=f"{item_id}.outcome_evidence_path"
    )
    outcome_sha256 = result.get("outcome_evidence_sha256")
    if not isinstance(outcome_sha256, str) or _SHA256.fullmatch(outcome_sha256) is None:
        raise ModuleAblationError(f"completed artifact {item_id} outcome hash invalid")
    verified, outcome_ref = _validate_artifact_refs(
        root,
        artifact_dir,
        manifest_path,
        manifest.get("artifacts"),
        item_id=item_id,
    )
    if outcome_ref["relative_path"] != outcome_path or outcome_ref["sha256"] != outcome_sha256:
        raise ModuleAblationError(
            f"completed artifact {item_id} result does not bind its outcome evidence ref"
        )
    outcome_file = _safe_child_path(
        artifact_dir, outcome_path, field=f"{item_id}.outcome_evidence_path"
    )
    outcome_evidence = _read_object(outcome_file)
    _validate_typed_outcome(
        outcome_evidence,
        schedule=schedule,
        item=item,
        measurement_kind=measurement_kind,
        success=success,
    )
    return {
        "schedule_item_id": item_id,
        "component": item["component"],
        "condition": item["condition"],
        "case_id": item["case_id"],
        "manifest": manifest_relative,
        "manifest_sha256": _file_sha256(manifest_path),
        "artifact_status": "completed",
        "completed": True,
        "effect_eligible": measurement_kind == "generation_outcome",
        "measurement_kind": measurement_kind,
        "success": success,
        "unavailable_reason": (
            None
            if measurement_kind == "generation_outcome"
            else "provenance-only evidence is not an ablation outcome"
        ),
        "historical_runtime": runtime,
        "runtime_attestation": "self_attested_by_completed_manifest",
        "verified_artifacts": verified,
    }


def audit_module_ablation_artifacts(
    repo_root: str | Path, schedule: Mapping[str, Any]
) -> dict[str, Any]:
    """Audit completed artifacts and estimate only fully matched effects."""

    root = Path(repo_root).expanduser().resolve()
    items = _validate_schedule(root, schedule)
    audits = [_audit_completed_item(root, schedule, item) for item in items]
    audit_by_key = {
        (row["component"], row["case_id"], row["condition"]): row
        for row in audits
    }
    conditions_by_component = {
        component: [row["condition"] for row in contracts]
        for component, contracts in schedule["condition_contracts"].items()
    }
    cases_by_component: dict[str, list[str]] = {}
    for matched in schedule["matched_sets"]:
        cases_by_component.setdefault(matched["component"], []).append(matched["case_id"])

    comparisons: list[dict[str, Any]] = []
    for component, conditions in conditions_by_component.items():
        cases = cases_by_component[component]
        for condition in conditions:
            if condition == "complete":
                continue
            eligible_pairs: list[dict[str, Any]] = []
            ineligible_pairs: list[dict[str, Any]] = []
            for case_id in cases:
                complete = audit_by_key.get((component, case_id, "complete"))
                module_off = audit_by_key.get((component, case_id, condition))
                if complete is None or module_off is None:
                    raise ModuleAblationError(
                        f"audited matrix unexpectedly incomplete: {component}/{case_id}/{condition}"
                    )
                pair = {
                    "case_id": case_id,
                    "complete_status": complete["artifact_status"],
                    "condition_status": module_off["artifact_status"],
                    "complete_measurement_kind": complete["measurement_kind"],
                    "condition_measurement_kind": module_off["measurement_kind"],
                    "complete_success": complete["success"],
                    "condition_success": module_off["success"],
                }
                if complete["effect_eligible"] and module_off["effect_eligible"]:
                    eligible_pairs.append(pair)
                else:
                    ineligible_pairs.append(pair)
            effect: dict[str, Any] | None = None
            unavailable_reason: str | None = None
            if ineligible_pairs:
                unavailable_reason = (
                    "one or more matched completed generation-outcome artifacts "
                    "are missing or ineligible"
                )
            elif not eligible_pairs:
                unavailable_reason = "no matched generation outcomes"
            else:
                count = len(eligible_pairs)
                complete_successes = sum(
                    pair["complete_success"] is True for pair in eligible_pairs
                )
                condition_successes = sum(
                    pair["condition_success"] is True for pair in eligible_pairs
                )
                complete_rate = complete_successes / count
                condition_rate = condition_successes / count
                effect = {
                    "estimand": "complete_success_rate_minus_condition_success_rate",
                    "matched_case_count": count,
                    "complete_successes": complete_successes,
                    "condition_successes": condition_successes,
                    "complete_success_rate": complete_rate,
                    "condition_success_rate": condition_rate,
                    "absolute_success_rate_difference": complete_rate - condition_rate,
                }
            comparisons.append(
                {
                    "component": component,
                    "reference_condition": "complete",
                    "module_off_condition": condition,
                    "scheduled_case_count": len(cases),
                    "eligible_matched_case_count": len(eligible_pairs),
                    "effect": effect,
                    "unavailable_reason": unavailable_reason,
                    "ineligible_pairs": ineligible_pairs,
                }
            )

    historical = [
        row["historical_runtime"]
        for row in audits
        if isinstance(row["historical_runtime"], Mapping)
    ]
    all_effects_available = bool(comparisons) and all(
        comparison["effect"] is not None for comparison in comparisons
    )
    table3_rates = None
    if all_effects_available:
        table3_rates = [
            {
                "component": comparison["component"],
                "reference_condition": comparison["reference_condition"],
                "module_off_condition": comparison["module_off_condition"],
                **comparison["effect"],
            }
            for comparison in comparisons
        ]
    return {
        "schema_version": 1,
        "protocol": _AUDIT_PROTOCOL,
        "status": "completed",
        "mode": "cached_completed_artifact_audit",
        "study_id": schedule["study_id"],
        "schedule_contract_sha256": schedule["schedule_contract_sha256"],
        "schedule_artifact_sha256": _canonical_sha256(schedule),
        "claim_scope": (
            "functional-only, fully matched generation outcomes; provenance and "
            "incomplete pairs never estimate an effect"
        ),
        "paper_table_eligible": False,
        "runtime": {
            "provider_called": False,
            "simulator_called": False,
            "act_rollouts_started": 0,
        },
        "historical_artifact_runtime": {
            "attestation": (
                "self_attested_by_completed_manifests_not_independently_observed"
            ),
            "completed_artifacts": len(historical),
            "any_provider_called": any(row["provider_called"] for row in historical),
            "any_simulator_called": any(row["simulator_called"] for row in historical),
            "act_rollouts_started": sum(row["act_rollouts_started"] for row in historical),
        },
        "artifact_audit": {
            "scheduled": len(audits),
            "completed": sum(row["completed"] for row in audits),
            "effect_eligible": sum(row["effect_eligible"] for row in audits),
            "provenance_only": sum(
                row["measurement_kind"] == "provenance_only" for row in audits
            ),
            "missing_or_incomplete": sum(not row["completed"] for row in audits),
            "rows": audits,
        },
        "comparisons": comparisons,
        "all_effects_available": all_effects_available,
        "table3_success_rates": table3_rates,
        "limitations": [
            "These are functional-only effects, not paper-scale success rates.",
            "Historical provider/simulator runtime is self-attested by manifests.",
            "Provenance confirms lineage only and never estimates module effect.",
            "The audit process starts no provider, simulator, or ACT call.",
        ],
    }


__all__ = [
    "CONDITION_SPECS",
    "EXECUTION_IDENTITY_FIELDS",
    "ModuleAblationError",
    "audit_module_ablation_artifacts",
    "prepare_module_ablation_schedule",
]
