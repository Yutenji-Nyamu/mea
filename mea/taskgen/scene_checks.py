"""Proposal-derived visual gate contracts for generated RoboTwin scenes.

The contract deliberately separates visual plausibility from simulator-owned
facts.  It is small enough to persist beside every TaskGen run and does not
grant the vision model authority over coordinates, instance ids, or success.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping

from .success_spec import SuccessSpecError, success_spec_validation_report


class SceneCheckSpecError(ValueError):
    """Raised when a scene-check contract is inconsistent with its task."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _click_bell_authorities(changes: Mapping[str, Any]) -> list[str]:
    bell = changes.get("bell")
    randomization = changes.get("domain_randomization")
    if isinstance(bell, Mapping):
        if bell.get("instance_mode") == "fixed":
            return ["simulator_task_attribute.bell_id", "official_check_success"]
        return ["simulator_tracked_actor.bell_xy", "official_check_success"]
    if isinstance(randomization, Mapping):
        if "cluttered_table" in randomization:
            return ["simulator_task_info.clutter_count", "official_check_success"]
        if "random_background" in randomization:
            return [
                "simulator_task_info.background_texture_split",
                "official_check_success",
            ]
        if "random_light" in randomization:
            return [
                "simulator_task_info.light_configuration",
                "official_check_success",
            ]
    return ["simulator_task_state", "official_check_success"]


def _inferred_aspect(task_name: str, spec: Mapping[str, Any]) -> str:
    controlled = spec.get("controlled_axis")
    if isinstance(controlled, str) and controlled.strip():
        return controlled.strip()
    changes = spec.get("changes")
    if task_name == "click_bell" and isinstance(changes, Mapping):
        if not changes or spec.get("generation_mode") in {
            "official",
            "official_passthrough",
        }:
            return "task_execution"
        bell = changes.get("bell")
        if isinstance(bell, Mapping) and bell.get("instance_mode") == "fixed":
            return "object_instance"
        if "domain_randomization" in changes:
            randomization = changes.get("domain_randomization") or {}
            if "cluttered_table" in randomization:
                return "robustness.scene_clutter"
            if "random_background" in randomization:
                return "scene_background_texture"
            if "random_light" in randomization:
                return "scene_lighting"
        return "object_position"
    if task_name == "beat_block_hammer":
        return "object_appearance"
    return "task_execution"


def build_scene_check_spec(
    variant_spec: Mapping[str, Any],
    *,
    task_proposal: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one bounded visual checklist from a proposal or VariantSpec."""

    if not isinstance(variant_spec, Mapping):
        raise SceneCheckSpecError("variant_spec must be an object")
    task_name = str(variant_spec.get("task_name") or "").strip()
    if not task_name:
        raise SceneCheckSpecError("variant_spec.task_name must be non-empty")
    changes = variant_spec.get("changes")
    if not isinstance(changes, Mapping):
        raise SceneCheckSpecError("variant_spec.changes must be an object")

    proposal_id = None
    source = "variant_spec"
    aspect_id = _inferred_aspect(task_name, variant_spec)
    success_semantics = "official_check_success"
    if task_proposal is not None:
        if not isinstance(task_proposal, Mapping):
            raise SceneCheckSpecError("task_proposal must be an object")
        proposal_task = str(task_proposal.get("task_name") or "").strip()
        if proposal_task != task_name:
            raise SceneCheckSpecError("TaskProposal cannot change SceneCheckSpec task")
        proposal_changes = task_proposal.get("changes")
        if proposal_changes != changes:
            raise SceneCheckSpecError(
                "TaskProposal changes must equal the materialized VariantSpec"
            )
        proposal_aspect = str(task_proposal.get("aspect_id") or "").strip()
        if proposal_aspect:
            aspect_id = proposal_aspect
        proposal_id = str(task_proposal.get("proposal_id") or "").strip() or None
        source = "task_proposal"
        if task_proposal.get("preserve_success_semantics") is False:
            if (
                task_name == "beat_block_hammer"
                and task_proposal.get("capability_id")
                == "robustness.distractor_avoidance"
                and task_proposal.get("aspect_id")
                == "robustness.distractor_avoidance"
            ):
                success_semantics = "provider_generated_python"
            else:
                try:
                    success_report = success_spec_validation_report(
                        task_proposal.get("success_spec")
                    )
                except SuccessSpecError as exc:
                    raise SceneCheckSpecError(
                        f"TaskProposal replacement SuccessSpec is invalid: {exc}"
                    ) from exc
                if (
                    not success_report["act_eligible"]
                    or not success_report["experimental_bounded"]
                ):
                    raise SceneCheckSpecError(
                        "SceneCheckSpec only accepts experimental bounded "
                        "ACT semantics"
                    )
                success_semantics = "experimental_bounded_success_spec"

    if task_name == "beat_block_hammer":
        target_actor = "block"
        if success_semantics == "provider_generated_python":
            visual_checks = [
                "target_actor_visible",
                "lookalike_distractor_visible",
                "scene_is_physically_plausible",
            ]
            simulator_authorities = [
                "simulator_actor_identity",
                "simulator_rule_check",
                "provider_checker_semantic_fixtures",
                "expert_solvability",
            ]
            repair_policy = {
                "mode": "regenerate_scene_checker_code",
                "handler": "regenerate_scene_and_checker",
                "max_repairs_supported": 1,
            }
        else:
            visual_checks = [
                "target_actor_visible",
                "requested_appearance_is_plausible",
                "no_obvious_unrequested_scene_change",
            ]
            simulator_authorities = [
                "simulator_actor_identity",
                "simulator_rule_check",
                (
                    "compiled_success_spec"
                    if success_semantics == "experimental_bounded_success_spec"
                    else "official_check_success"
                ),
            ]
            repair_policy = {
                "mode": "regenerate_scene_code",
                "handler": "regenerate_load_actors",
                "max_repairs_supported": 5,
            }
    elif task_name == "click_bell":
        target_actor = "bell"
        visual_checks = [
            "target_actor_visible",
            "scene_is_physically_plausible",
            "no_obvious_unrequested_scene_change",
        ]
        simulator_authorities = _click_bell_authorities(changes)
        repair_policy = {
            "mode": "validate_only",
            "handler": None,
            "max_repairs_supported": 0,
        }
    else:
        target_actor = task_name
        visual_checks = ["scene_is_physically_plausible"]
        simulator_authorities = ["official_check_success"]
        repair_policy = {
            "mode": "validate_only",
            "handler": None,
            "max_repairs_supported": 0,
        }

    result = {
        "schema_version": 1,
        "task_name": task_name,
        "aspect_id": aspect_id,
        "source": source,
        "proposal_id": proposal_id,
        "proposal_sha256": (
            _canonical_sha256(task_proposal) if task_proposal is not None else None
        ),
        "target_actor": target_actor,
        "requested_changes": deepcopy(dict(changes)),
        "visual_checks": visual_checks,
        "simulator_authorities": simulator_authorities,
        "success_semantics": success_semantics,
        "repair_policy": repair_policy,
    }
    return validate_scene_check_spec(result)


def validate_scene_check_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "task_name",
        "aspect_id",
        "source",
        "proposal_id",
        "proposal_sha256",
        "target_actor",
        "requested_changes",
        "visual_checks",
        "simulator_authorities",
        "success_semantics",
        "repair_policy",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise SceneCheckSpecError(
            f"SceneCheckSpec fields must be exactly {sorted(required)}"
        )
    result = deepcopy(dict(value))
    if result.get("schema_version") != 1:
        raise SceneCheckSpecError("SceneCheckSpec.schema_version must be 1")
    for field in ("task_name", "aspect_id", "source", "target_actor"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise SceneCheckSpecError(f"SceneCheckSpec.{field} must be non-empty")
    if result["source"] not in {"variant_spec", "task_proposal"}:
        raise SceneCheckSpecError("SceneCheckSpec.source is unsupported")
    if result["source"] == "task_proposal" and (
        not result.get("proposal_id") or not result.get("proposal_sha256")
    ):
        raise SceneCheckSpecError("proposal-derived SceneCheckSpec lacks identity")
    if not isinstance(result.get("requested_changes"), Mapping):
        raise SceneCheckSpecError("requested_changes must be an object")
    for field in ("visual_checks", "simulator_authorities"):
        items = result.get(field)
        if (
            not isinstance(items, list)
            or not items
            or any(not isinstance(item, str) or not item for item in items)
            or len(items) != len(set(items))
        ):
            raise SceneCheckSpecError(f"{field} must be a non-empty unique list")
    if result.get("success_semantics") not in {
        "official_check_success",
        "experimental_bounded_success_spec",
        "provider_generated_python",
    }:
        raise SceneCheckSpecError("SceneCheckSpec success semantics are unsupported")
    if result["success_semantics"] == "experimental_bounded_success_spec" and (
        result["task_name"] != "beat_block_hammer"
        or result["source"] != "task_proposal"
        or not result.get("proposal_sha256")
        or "compiled_success_spec" not in result["simulator_authorities"]
    ):
        raise SceneCheckSpecError(
            "experimental SceneCheckSpec lacks bounded proposal authority"
        )
    if result["success_semantics"] == "provider_generated_python" and (
        result["task_name"] != "beat_block_hammer"
        or result["aspect_id"] != "robustness.distractor_avoidance"
        or result["source"] != "task_proposal"
        or not result.get("proposal_sha256")
        or "provider_checker_semantic_fixtures"
        not in result["simulator_authorities"]
    ):
        raise SceneCheckSpecError(
            "provider checker SceneCheckSpec lacks proposal/fixture authority"
        )
    policy = result.get("repair_policy")
    if not isinstance(policy, Mapping) or set(policy) != {
        "mode",
        "handler",
        "max_repairs_supported",
    }:
        raise SceneCheckSpecError("repair_policy fields are invalid")
    if policy["mode"] == "validate_only":
        if policy["handler"] is not None or policy["max_repairs_supported"] != 0:
            raise SceneCheckSpecError("validate_only cannot install a repair handler")
    elif policy["mode"] == "regenerate_scene_code":
        if (
            policy["handler"] != "regenerate_load_actors"
            or not isinstance(policy["max_repairs_supported"], int)
            or not 1 <= policy["max_repairs_supported"] <= 5
        ):
            raise SceneCheckSpecError("scene-code repair policy is invalid")
    elif policy["mode"] == "regenerate_scene_checker_code":
        if (
            policy["handler"] != "regenerate_scene_and_checker"
            or policy["max_repairs_supported"] != 1
            or result["success_semantics"] != "provider_generated_python"
        ):
            raise SceneCheckSpecError(
                "provider scene+checker repair policy is invalid"
            )
    else:
        raise SceneCheckSpecError("unsupported repair policy")
    return result


__all__ = [
    "SceneCheckSpecError",
    "build_scene_check_spec",
    "validate_scene_check_spec",
]
