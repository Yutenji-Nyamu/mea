"""Trusted ACT evaluation catalog for the open-query Plan Agent.

Only task/profile/aspect combinations with a readable RoboTwin TaskSchema and
the two required ACT checkpoint artifacts are exposed to the model.  Paths,
seeds, gates, and executable modules remain runtime-owned data.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .click_bell import (
    CLICK_BELL_ADAPTIVE_ASPECTS,
    CLICK_BELL_ADAPTIVE_TEMPLATES,
)
from .prototype import EXPECTED_POLICY, MAX_ROUNDS, SUB_ASPECT_CATALOG


class ACTCatalogError(ValueError):
    """Raised when a trusted ACT catalog is missing or has been changed."""


ACT_ROUTE_TASKS = ("beat_block_hammer", "click_bell")

_TASK_PROFILE = {
    "beat_block_hammer": "generated",
    "click_bell": "adaptive_properties",
}
_PLANNER_KIND = {
    "beat_block_hammer": "bounded_bbh_v1",
    "click_bell": "model_click_bell_adaptive_v1",
}
_CLICK_CAPABILITY = {
    "object_position": "object_position.fixed_xy",
    "object_instance": "object_instance.official_id",
}
_CLICK_METRIC = {
    "object_position": "bell_active_tcp_min_xy_error",
    "object_instance": "official_check_success",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _catalog_sha256(value: Mapping[str, Any]) -> str:
    payload = {key: deepcopy(item) for key, item in value.items() if key != "catalog_sha256"}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _bbh_aspects() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for template_id, template in SUB_ASPECT_CATALOG.items():
        result.append(
            {
                "aspect_id": str(template["sub_aspect"]),
                "description": str(template["rationale"]),
                "template_ids": [template_id],
                # Every committed BBH round preserves the validated blue-block
                # variant; later aspects reuse it rather than inventing a new
                # generation capability.
                "taskgen_capability_id": "object_appearance.color",
                "taskgen_route": str(template["route"]),
                "default_metric": str(template["tool_metric"]),
            }
        )
    return result


def _click_aspects() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for aspect_id, aspect in CLICK_BELL_ADAPTIVE_ASPECTS.items():
        template_ids = list(aspect["template_ids"])
        if any(
            CLICK_BELL_ADAPTIVE_TEMPLATES[template_id]["aspect_id"] != aspect_id
            for template_id in template_ids
        ):
            raise ACTCatalogError(
                f"click_bell template catalog does not preserve aspect {aspect_id!r}"
            )
        result.append(
            {
                "aspect_id": aspect_id,
                "description": str(aspect["description"]),
                "template_ids": template_ids,
                "taskgen_capability_id": _CLICK_CAPABILITY[aspect_id],
                "taskgen_route": "reuse",
                "default_metric": _CLICK_METRIC[aspect_id],
            }
        )
    return result


def _trusted_task_entry(task_name: str, task_family: str) -> dict[str, Any]:
    aspects = _bbh_aspects() if task_name == "beat_block_hammer" else _click_aspects()
    return {
        "task_name": task_name,
        "task_family": task_family,
        "task_profile": _TASK_PROFILE[task_name],
        "planner_kind": _PLANNER_KIND[task_name],
        "max_rounds": MAX_ROUNDS,
        "checkpoint": {
            "policy_name": "ACT",
            "checkpoint_setting": "demo_clean",
            "expert_data_num": 50,
            "checkpoint_id": f"act-{task_name}/demo_clean-50",
            "ready": True,
        },
        "aspects": aspects,
    }


def _read_task_family(schema_path: Path, task_name: str) -> tuple[str | None, str | None]:
    if not schema_path.is_file():
        return None, "task_schema_missing"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "task_schema_invalid"
    if not isinstance(schema, dict) or schema.get("task_name") != task_name:
        return None, "task_schema_identity_mismatch"
    family = schema.get("task_family")
    if not isinstance(family, str) or not family.strip():
        return None, "task_family_missing"
    return family.strip(), None


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def build_act_catalog(repo_root: str | Path) -> dict[str, Any]:
    """Build the deterministic public catalog from trusted local artifacts."""

    root = Path(repo_root).expanduser().resolve()
    tasks: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for task_name in ACT_ROUTE_TASKS:
        missing: list[str] = []
        family, schema_issue = _read_task_family(
            root / f"mea/toolkit/schemas/{task_name}.json", task_name
        )
        if schema_issue:
            missing.append(schema_issue)
        checkpoint_dir = (
            root
            / "policy/ACT/act_ckpt"
            / f"act-{task_name}"
            / "demo_clean-50"
        )
        if not _nonempty_file(checkpoint_dir / "dataset_stats.pkl"):
            missing.append("dataset_stats_missing")
        if not _nonempty_file(checkpoint_dir / "policy_last.ckpt"):
            missing.append("policy_weights_missing")
        if missing:
            excluded.append(
                {"task_name": task_name, "missing_requirements": sorted(missing)}
            )
            continue
        assert family is not None
        tasks.append(_trusted_task_entry(task_name, family))

    catalog: dict[str, Any] = {
        "schema_version": 1,
        "policy": deepcopy(EXPECTED_POLICY),
        "tasks": tasks,
        "excluded_tasks": excluded,
    }
    catalog["catalog_sha256"] = _catalog_sha256(catalog)
    return validate_act_catalog(catalog)


def validate_act_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate catalog provenance and every model-visible routing choice."""

    required = {
        "schema_version",
        "policy",
        "tasks",
        "excluded_tasks",
        "catalog_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ACTCatalogError(f"ACT catalog fields must be exactly {sorted(required)}")
    catalog = deepcopy(dict(value))
    if catalog.get("schema_version") != 1:
        raise ACTCatalogError("ACT catalog schema_version must be 1")
    if catalog.get("policy") != EXPECTED_POLICY:
        raise ACTCatalogError("ACT catalog policy contract changed")
    if catalog.get("catalog_sha256") != _catalog_sha256(catalog):
        raise ACTCatalogError("ACT catalog sha256 does not match its contents")

    tasks = catalog.get("tasks")
    excluded = catalog.get("excluded_tasks")
    if not isinstance(tasks, list) or not isinstance(excluded, list):
        raise ACTCatalogError("ACT catalog tasks and excluded_tasks must be lists")
    seen: set[str] = set()
    task_keys = {
        "task_name",
        "task_family",
        "task_profile",
        "planner_kind",
        "max_rounds",
        "checkpoint",
        "aspects",
    }
    for task in tasks:
        if not isinstance(task, dict) or set(task) != task_keys:
            raise ACTCatalogError("ACT catalog task fields changed")
        task_name = task.get("task_name")
        family = task.get("task_family")
        if task_name not in ACT_ROUTE_TASKS or task_name in seen:
            raise ACTCatalogError(f"unknown or duplicate ACT route task: {task_name!r}")
        if not isinstance(family, str) or not family.strip():
            raise ACTCatalogError("task_family must be non-empty")
        expected = _trusted_task_entry(str(task_name), family)
        if task != expected:
            raise ACTCatalogError(f"trusted task entry changed for {task_name!r}")
        seen.add(str(task_name))

    excluded_seen: set[str] = set()
    for item in excluded:
        if not isinstance(item, dict) or set(item) != {
            "task_name",
            "missing_requirements",
        }:
            raise ACTCatalogError("excluded task fields changed")
        task_name = item.get("task_name")
        missing = item.get("missing_requirements")
        if (
            task_name not in ACT_ROUTE_TASKS
            or task_name in seen
            or task_name in excluded_seen
            or not isinstance(missing, list)
            or not missing
            or missing != sorted(set(missing))
            or any(not isinstance(reason, str) or not reason for reason in missing)
        ):
            raise ACTCatalogError(f"invalid excluded task entry: {item!r}")
        excluded_seen.add(str(task_name))
    if seen | excluded_seen != set(ACT_ROUTE_TASKS):
        raise ACTCatalogError("every allowlisted task must be ready or explicitly excluded")
    return catalog


def catalog_task(catalog: Mapping[str, Any], task_name: str) -> dict[str, Any]:
    """Return one ready task entry from a validated catalog."""

    normalized = validate_act_catalog(catalog)
    for task in normalized["tasks"]:
        if task["task_name"] == task_name:
            return deepcopy(task)
    raise ACTCatalogError(f"task is not ACT-ready in this catalog: {task_name!r}")


__all__ = [
    "ACTCatalogError",
    "ACT_ROUTE_TASKS",
    "build_act_catalog",
    "catalog_task",
    "validate_act_catalog",
]
