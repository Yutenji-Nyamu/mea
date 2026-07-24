"""Fail-closed, no-execution protocols for the next paper-evidence runs.

The builders in this module preregister experiments.  The evaluators only
consume receipts from runs that happened elsewhere; they never start a
provider, simulator, expert, probe, or policy rollout.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from mea.paper_claim_demo import evaluate_policy_ranking
from mea.planner.query_contract import (
    assess_query_sufficiency,
    build_query_sufficiency_contract,
)
from mea.taskgen.click_bell import compile_click_bell_overlay


class LivePaperProtocolError(ValueError):
    """Raised when a paper-evidence manifest violates its frozen contract."""


EFFICIENCY_PROTOCOL = "click_bell_independent_live_efficiency_v1"
RANKING_PROTOCOL = "act_dp3_exact_seed_pair_v1"
TABLE3_PROTOCOL = "table3_real_codegen_ablation_v1"
PROXY_PROTOCOL = "plan_vqa_development_proxy_manifest_v1"

CLICK_BELL_CANDIDATES = (
    {
        "candidate_id": "object_position.left_fixed",
        "task_name": "click_bell",
        "axis_id": "object_position",
        "variant_hint": {
            "bell": {"position_mode": "fixed", "xy": [-0.20, -0.08]}
        },
    },
    {
        "candidate_id": "object_position.right_fixed",
        "task_name": "click_bell",
        "axis_id": "object_position",
        "variant_hint": {
            "bell": {"position_mode": "fixed", "xy": [0.20, -0.08]}
        },
    },
    {
        "candidate_id": "object_instance.base0",
        "task_name": "click_bell",
        "axis_id": "object_instance",
        "variant_hint": {
            "bell": {
                "position_mode": "official_random",
                "instance_mode": "fixed",
                "bell_id": 0,
            }
        },
    },
    {
        "candidate_id": "object_instance.base1",
        "task_name": "click_bell",
        "axis_id": "object_instance",
        "variant_hint": {
            "bell": {
                "position_mode": "official_random",
                "instance_mode": "fixed",
                "bell_id": 1,
            }
        },
    },
)
_CANDIDATE_IDS = tuple(row["candidate_id"] for row in CLICK_BELL_CANDIDATES)
_EFFICIENCY_AXIS_PAIRS = {
    "object_position": (
        "object_position.left_fixed",
        "object_position.right_fixed",
    ),
    "object_instance": (
        "object_instance.base0",
        "object_instance.base1",
    ),
}
_EFFICIENCY_MODES = {
    "smoke_3act": {
        "fixed_candidates": _EFFICIENCY_AXIS_PAIRS["object_position"],
        "adaptive_candidates": _EFFICIENCY_AXIS_PAIRS["object_position"],
        "adaptive_min": 1,
        "adaptive_max": 1,
        "total_min": 3,
        "total_max": 3,
        "claim_scope": "three_act_mechanism_smoke_not_dense_reference",
        "query": (
            "Does at least one of the two frozen click_bell position "
            "candidates fail?"
        ),
        "query_sufficient_rule": "at_least_one_completed_failure",
    },
    "toy_5to7act": {
        "fixed_candidates": _CANDIDATE_IDS,
        "adaptive_candidates": _CANDIDATE_IDS,
        "adaptive_min": 2,
        "adaptive_max": 3,
        "total_min": 6,
        "total_max": 7,
        "claim_scope": "independent_live_toy_not_paper_tables_1_2",
        "query": (
            "Does at least one of the four frozen click_bell candidates fail, "
            "and which paired position/instance axis is directly contrasted?"
        ),
        "query_sufficient_rule": (
            "completed_failure_with_its_frozen_axis_pair_observed"
        ),
    },
    "position_universal_3to4act": {
        "fixed_candidates": _EFFICIENCY_AXIS_PAIRS["object_position"],
        "adaptive_candidates": _EFFICIENCY_AXIS_PAIRS["object_position"],
        "adaptive_min": 1,
        "adaptive_max": 2,
        "total_min": 3,
        "total_max": 4,
        "claim_scope": (
            "independent_live_finite_position_universal_toy_not_paper_tables_1_2"
        ),
        "query": (
            "Does this ACT checkpoint succeed on every candidate in the "
            "preregistered two-position click_bell domain?"
        ),
        "query_sufficient_rule": (
            "finite_universal_refuted_by_one_failure_or_supported_by_full_coverage"
        ),
    },
}

TABLE3_CONDITIONS = (
    "complete",
    "base",
    "no_rag",
    "no_visual_self_check",
    "no_readme_agent",
)
TABLE3_PROPOSALS = (
    {
        "proposal_id": "u01_bbh_blue_block",
        "task_name": "beat_block_hammer",
        "prompt": "Use a blue target block while preserving official pose, yaw, and scale.",
        "changes": {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.0,
                "color": [0.0, 0.2, 1.0],
            }
        },
    },
    {
        "proposal_id": "u02_bbh_green_block",
        "task_name": "beat_block_hammer",
        "prompt": "Use a green target block while preserving official pose, yaw, and scale.",
        "changes": {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.0,
                "color": [0.1, 0.8, 0.2],
            }
        },
    },
    {
        "proposal_id": "u03_bbh_yellow_block",
        "task_name": "beat_block_hammer",
        "prompt": "Use a yellow target block while preserving official pose, yaw, and scale.",
        "changes": {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.0,
                "color": [0.9, 0.8, 0.1],
            }
        },
    },
    {
        "proposal_id": "u04_bbh_scale_0_8",
        "task_name": "beat_block_hammer",
        "prompt": "Use a 0.8-scale target block while preserving official pose, yaw, and color.",
        "changes": {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 0.8,
                "color": [0.0, 0.0, 1.0],
            }
        },
    },
    {
        "proposal_id": "u05_bbh_scale_1_2",
        "task_name": "beat_block_hammer",
        "prompt": "Use a 1.2-scale target block while preserving official pose, yaw, and color.",
        "changes": {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.2,
                "color": [0.0, 0.0, 1.0],
            }
        },
    },
)
TABLE3_SWITCHES = {
    "complete": {"rag": True, "visual_self_check": True, "readme_agent": True},
    "base": {"rag": False, "visual_self_check": False, "readme_agent": False},
    "no_rag": {"rag": False, "visual_self_check": True, "readme_agent": True},
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
}

CLICK_BELL_TASK_MODULE = "mea.tasks.click_bell"
CLICK_BELL_ACT_CHECKPOINT_REF = (
    "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt"
)
BBH_ACT_CHECKPOINT_REF = (
    "policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50/policy_last.ckpt"
)
BBH_DP3_CHECKPOINT_REF = (
    "policy/DP3/3D-Diffusion-Policy/checkpoints/"
    "beat_block_hammer-demo_clean-50_0/3000.ckpt"
)
ROBOTWIN_PYTHON = "/root/autodl-tmp/conda/envs/RoboTwin/bin/python"
DP3_PYTHON = "/root/autodl-tmp/conda/envs/RoboTwin-DP3/bin/python"
PAPER_VQA_CONDITIONS = (
    "clean",
    "scene_clutter",
    "background_texture",
    "lighting",
)


def canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LivePaperProtocolError(f"value is not canonical JSON: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_ref(value: Any, *, field: str) -> str:
    text = _text(value, field=field).replace("\\", "/")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts:
        raise LivePaperProtocolError(f"{field} must be a repository-relative path")
    return path.as_posix()


def _bound_path(repo_root: Path, ref: Any, *, field: str, must_exist: bool = True) -> Path:
    relative = _relative_ref(ref, field=field)
    path = (repo_root / relative).resolve()
    if not path.is_relative_to(repo_root):
        raise LivePaperProtocolError(f"{field} escapes repository")
    if must_exist and not path.is_file():
        raise LivePaperProtocolError(f"{field} is missing: {relative}")
    return path


def _checkpoint_artifact_path(
    repo_root: Path, ref: Any, *, field: str
) -> Path:
    """Resolve a repo-bound checkpoint or its server-side model-store link."""

    relative = _relative_ref(ref, field=field)
    lexical_path = repo_root / relative
    if not lexical_path.exists():
        raise LivePaperProtocolError(f"{field} does not exist")
    resolved = lexical_path.resolve()
    server_asset_roots = [
        (repo_root.parent / name).resolve()
        for name in ("models", "RoboTwin")
        if (repo_root.parent / name).exists()
    ]
    if not (
        resolved.is_relative_to(repo_root)
        or any(
            resolved.is_relative_to(asset_root)
            for asset_root in server_asset_roots
        )
    ):
        raise LivePaperProtocolError(
            f"{field} resolves outside the repository and server model store"
        )
    return resolved


def _write_bound_file(repo_root: Path, ref: str, payload: bytes) -> None:
    path = _bound_path(repo_root, ref, field="artifact_ref", must_exist=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise LivePaperProtocolError(f"append-only artifact already exists: {ref}")
    path.write_bytes(payload)


def _object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LivePaperProtocolError(f"{field} must be an object")
    return dict(value)


def _items(value: Any, *, field: str, minimum: int = 0) -> list[Any]:
    if not isinstance(value, list) or len(value) < minimum:
        raise LivePaperProtocolError(f"{field} must be a list with >= {minimum} items")
    return list(value)


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise LivePaperProtocolError(f"{field} must be non-empty text")
    return value.strip()


def _identifier(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if not all(ch.isalnum() or ch in "._-" for ch in text):
        raise LivePaperProtocolError(f"{field} must be an identifier")
    return text


def _sha256(value: Any, *, field: str) -> str:
    text = _text(value, field=field)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise LivePaperProtocolError(f"{field} must be 64 lowercase hex characters")
    return text


def _integer(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LivePaperProtocolError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: Any, *, field: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise LivePaperProtocolError(f"{field} must be finite and >= {minimum}")
    return float(value)


def _utc(value: Any, *, field: str) -> datetime:
    text = _text(value, field=field)
    if not text.endswith("Z"):
        raise LivePaperProtocolError(f"{field} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise LivePaperProtocolError(f"{field} is not a timestamp") from exc
    if parsed.tzinfo != timezone.utc:
        raise LivePaperProtocolError(f"{field} must use UTC")
    return parsed


def _checkpoint(
    value: Any, *, field: str, artifact_ref: str | None = None
) -> dict[str, str]:
    row = _object(value, field=field)
    resolved_ref = artifact_ref or row.get("artifact_ref")
    return {
        "checkpoint_id": _text(row.get("checkpoint_id"), field=f"{field}.checkpoint_id"),
        "artifact_ref": _relative_ref(
            resolved_ref, field=f"{field}.artifact_ref"
        ),
        "artifact_sha256": _sha256(
            row.get("artifact_sha256"), field=f"{field}.artifact_sha256"
        ),
    }


def _seal(value: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result[hash_field] = canonical_sha256(result)
    return result


def _verify_seal(value: Mapping[str, Any], *, hash_field: str) -> dict[str, Any]:
    result = deepcopy(dict(value))
    supplied = _sha256(result.pop(hash_field, None), field=hash_field)
    expected = canonical_sha256(result)
    if supplied != expected:
        raise LivePaperProtocolError(f"{hash_field} mismatch")
    result[hash_field] = supplied
    return result


def _click_overlay(candidate: Mapping[str, Any]) -> dict[str, Any]:
    variant_hint = deepcopy(dict(candidate["variant_hint"]))
    return compile_click_bell_overlay(variant_hint)


def _click_variant_binding(
    candidate: Mapping[str, Any], *, artifact_root_ref: str
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    variant_root = f"{artifact_root_ref}/variants/{candidate_id}"
    overlay = _click_overlay(candidate)
    overlay_ref = f"{variant_root}/overlay.yml"
    overlay_sha256 = _bytes_sha256(_json_bytes(overlay))
    variant_manifest = {
        "schema_version": 1,
        "kind": "click_bell_single_axis_variant_v1",
        "variant_id": f"click_bell.{candidate_id}",
        "candidate_id": candidate_id,
        "task_name": "click_bell",
        "task_module": CLICK_BELL_TASK_MODULE,
        "axis_id": str(candidate["axis_id"]),
        "variant_hint": deepcopy(dict(candidate["variant_hint"])),
        "overlay_ref": overlay_ref,
        "overlay_sha256": overlay_sha256,
    }
    manifest_ref = f"{variant_root}/variant_manifest.json"
    return {
        "variant_id": variant_manifest["variant_id"],
        "task_module": CLICK_BELL_TASK_MODULE,
        "axis_id": variant_manifest["axis_id"],
        "variant_hint": deepcopy(variant_manifest["variant_hint"]),
        "overlay_ref": overlay_ref,
        "overlay_sha256": overlay_sha256,
        "variant_manifest_ref": manifest_ref,
        "variant_manifest_sha256": _bytes_sha256(_json_bytes(variant_manifest)),
    }


def _seed_manifest(
    *, task_name: str, seed: int, checkpoint_setting: str = "demo_clean"
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol": "exact_seed_paired_v1",
        "task_name": task_name,
        "seeds": [seed],
        "conditions": [
            {"id": "clean", "task_config": "demo_clean"},
            {"id": "unused", "task_config": "demo_randomized"},
        ],
        "checkpoint_setting": checkpoint_setting,
        "expert_data_num": 50,
        "policy_seed": 0,
    }


def _efficiency_command_binding(
    *,
    study_id: str,
    arm: str,
    candidate: Mapping[str, Any],
    seed: int,
    checkpoint: Mapping[str, Any],
    artifact_root_ref: str,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    variant = candidate["variant_binding"]
    command_root = f"{artifact_root_ref}/commands/{arm}/{candidate_id}"
    live_root = f"{artifact_root_ref}/live_runs/{arm}/{candidate_id}"
    seed_manifest = _seed_manifest(task_name="click_bell", seed=seed)
    seed_manifest_ref = f"{command_root}/seed_manifest.json"
    seed_manifest_sha256 = _bytes_sha256(_json_bytes(seed_manifest))
    command = {
        "schema_version": 1,
        "kind": "click_bell_exact_n1_act_command_v1",
        "study_id": study_id,
        "arm": arm,
        "candidate_id": candidate_id,
        "variant_id": variant["variant_id"],
        "variant_manifest_ref": variant["variant_manifest_ref"],
        "variant_manifest_sha256": variant["variant_manifest_sha256"],
        "checkpoint": deepcopy(dict(checkpoint)),
        "seed": seed,
        "cwd": ".",
        "environment": {"PYTHON_BIN": ROBOTWIN_PYTHON},
        "argv": [
            "bash",
            "policy/ACT/eval_mea.sh",
            "click_bell",
            "demo_clean",
            "demo_clean",
            "50",
            "0",
            "0",
            "1",
            CLICK_BELL_TASK_MODULE,
            variant["overlay_ref"],
            "",
            f"{live_root}/telemetry",
            "balanced_v1",
            seed_manifest_ref,
            f"{live_root}/seed_results.json",
            f"{live_root}/eval_output",
        ],
        "seed_manifest_ref": seed_manifest_ref,
        "seed_manifest_sha256": seed_manifest_sha256,
        "expected_seed_results_ref": f"{live_root}/seed_results.json",
        "expected_telemetry_episode_ref": (
            f"{live_root}/telemetry/episode_000_seed_{seed}/episode.json"
        ),
        "receipt_ref": f"{live_root}/rollout_receipt.json",
    }
    command_ref = f"{command_root}/command.json"
    return {
        "candidate_id": candidate_id,
        "variant_id": variant["variant_id"],
        "command_ref": command_ref,
        "command_sha256": _bytes_sha256(_json_bytes(command)),
        "seed_manifest_ref": seed_manifest_ref,
        "seed_manifest_sha256": seed_manifest_sha256,
        "receipt_ref": command["receipt_ref"],
        "expected_seed_results_ref": command["expected_seed_results_ref"],
        "expected_telemetry_episode_ref": command[
            "expected_telemetry_episode_ref"
        ],
    }


def build_click_bell_efficiency_preregistration(
    *,
    study_id: str,
    mode: str,
    checkpoint: Mapping[str, Any],
    seed: int,
    created_at_utc: str,
    artifact_root_ref: str | None = None,
) -> dict[str, Any]:
    if mode not in _EFFICIENCY_MODES:
        raise LivePaperProtocolError(f"unknown efficiency mode: {mode}")
    spec = _EFFICIENCY_MODES[mode]
    _utc(created_at_utc, field="created_at_utc")
    resolved_study_id = _identifier(study_id, field="study_id")
    resolved_artifact_root = _relative_ref(
        artifact_root_ref
        or f"mea/protocol_runs/{resolved_study_id}/efficiency_artifacts",
        field="artifact_root_ref",
    )
    resolved_checkpoint = _checkpoint(
        checkpoint,
        field="checkpoint",
        artifact_ref=CLICK_BELL_ACT_CHECKPOINT_REF,
    )
    candidates = []
    for raw in CLICK_BELL_CANDIDATES:
        candidate = deepcopy(dict(raw))
        candidate["variant_binding"] = _click_variant_binding(
            candidate, artifact_root_ref=resolved_artifact_root
        )
        candidates.append(candidate)
    by_candidate = {item["candidate_id"]: item for item in candidates}
    execution_schedule = {
        "fixed": [
            _efficiency_command_binding(
                study_id=resolved_study_id,
                arm="fixed",
                candidate=by_candidate[candidate_id],
                seed=seed,
                checkpoint=resolved_checkpoint,
                artifact_root_ref=resolved_artifact_root,
            )
            for candidate_id in spec["fixed_candidates"]
        ],
        "adaptive": [
            _efficiency_command_binding(
                study_id=resolved_study_id,
                arm="adaptive",
                candidate=by_candidate[candidate_id],
                seed=seed,
                checkpoint=resolved_checkpoint,
                artifact_root_ref=resolved_artifact_root,
            )
            for candidate_id in spec["adaptive_candidates"]
        ],
    }
    query_contract = (
        build_query_sufficiency_contract(
            spec["query"],
            candidate_universe=spec["adaptive_candidates"],
            required_candidate_ids=spec["adaptive_candidates"],
            round_budget=spec["adaptive_max"],
            claim_type="universal",
        )
        if mode == "position_universal_3to4act"
        else None
    )
    body = {
        "schema_version": 1,
        "protocol": EFFICIENCY_PROTOCOL,
        "study_id": resolved_study_id,
        "created_at_utc": created_at_utc,
        "artifact_root_ref": resolved_artifact_root,
        "evidence_requirement": "independent_live_rollout_only",
        "mode": mode,
        "claim_scope": spec["claim_scope"],
        "query": spec["query"],
        "query_sufficiency_contract": query_contract,
        "checkpoint": resolved_checkpoint,
        "seed": _integer(seed, field="seed"),
        "candidate_universe": candidates,
        "fixed_contract": {
            "candidate_ids": list(spec["fixed_candidates"]),
            "stop_reason": "fixed_suite_complete",
        },
        "adaptive_contract": {
            "candidate_ids": list(spec["adaptive_candidates"]),
            "min_episode_starts": spec["adaptive_min"],
            "max_episode_starts": spec["adaptive_max"],
            "query_sufficient_rule": spec["query_sufficient_rule"],
            "allowed_stop_reasons": ["query_sufficient", "budget_exhausted"],
        },
        "total_episode_start_contract": {
            "minimum": spec["total_min"],
            "maximum": spec["total_max"],
        },
        "conclusion_contract": {
            "score_semantics": "official_success_boolean",
            "overall_verdicts": [
                "weakness_observed",
                "frozen_suite_all_succeeded",
                "inconclusive",
            ],
            "axis_rule": "paired_binary_score_difference",
            "comparison_fields": (
                ["claim_verdict"]
                if mode == "position_universal_3to4act"
                else ["overall_verdict", "weakness_axes"]
            ),
        },
        "provenance_contract": {
            "forbidden_designs": [
                "cached_prefix_counterfactual",
                "posthoc_arm_split",
                "shared_rollout_receipt",
            ],
            "required_attempt_fields": [
                "attempt_id",
                "candidate_id",
                "receipt_ref",
                "receipt_sha256",
            ],
        },
        "execution_schedule": execution_schedule,
        "execution_entrypoint": "policy/ACT/eval_mea.sh",
        "calls_started_by_preregistration": {
            "provider": 0,
            "simulator": 0,
            "expert": 0,
            "probe": 0,
            "act": 0,
        },
    }
    return _seal(body, hash_field="preregistration_sha256")


def validate_click_bell_efficiency_preregistration(
    value: Any,
    *,
    repo_root: str | Path | None = None,
    require_materialized: bool = False,
) -> dict[str, Any]:
    row = _verify_seal(_object(value, field="preregistration"), hash_field="preregistration_sha256")
    if row.get("schema_version") != 1 or row.get("protocol") != EFFICIENCY_PROTOCOL:
        raise LivePaperProtocolError("unsupported efficiency preregistration")
    mode = row.get("mode")
    if mode not in _EFFICIENCY_MODES:
        raise LivePaperProtocolError("efficiency mode is not frozen")
    rebuilt = build_click_bell_efficiency_preregistration(
        study_id=row.get("study_id"),
        mode=mode,
        checkpoint=row.get("checkpoint"),
        seed=row.get("seed"),
        created_at_utc=row.get("created_at_utc"),
        artifact_root_ref=row.get("artifact_root_ref"),
    )
    if rebuilt != row:
        raise LivePaperProtocolError("preregistration contract was modified")
    if require_materialized:
        if repo_root is None:
            raise LivePaperProtocolError("repo_root is required for materialized validation")
        root = Path(repo_root).expanduser().resolve()
        checkpoint_path = _checkpoint_artifact_path(
            root,
            row["checkpoint"]["artifact_ref"],
            field="checkpoint.artifact_ref",
        )
        if _file_sha256(checkpoint_path) != row["checkpoint"]["artifact_sha256"]:
            raise LivePaperProtocolError("checkpoint artifact hash mismatch")
        for candidate in row["candidate_universe"]:
            binding = candidate["variant_binding"]
            overlay_path = _bound_path(
                root, binding["overlay_ref"], field="overlay_ref"
            )
            if _file_sha256(overlay_path) != binding["overlay_sha256"]:
                raise LivePaperProtocolError("overlay hash mismatch")
            manifest_path = _bound_path(
                root,
                binding["variant_manifest_ref"],
                field="variant_manifest_ref",
            )
            if _file_sha256(manifest_path) != binding["variant_manifest_sha256"]:
                raise LivePaperProtocolError("variant manifest hash mismatch")
        for arm_rows in row["execution_schedule"].values():
            for binding in arm_rows:
                seed_path = _bound_path(
                    root,
                    binding["seed_manifest_ref"],
                    field="seed_manifest_ref",
                )
                command_path = _bound_path(
                    root, binding["command_ref"], field="command_ref"
                )
                if _file_sha256(seed_path) != binding["seed_manifest_sha256"]:
                    raise LivePaperProtocolError("seed manifest hash mismatch")
                if _file_sha256(command_path) != binding["command_sha256"]:
                    raise LivePaperProtocolError("command spec hash mismatch")
    return row


def materialize_click_bell_efficiency_preregistration(
    repo_root: str | Path, preregistration: Any
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    prereg = validate_click_bell_efficiency_preregistration(preregistration)
    for candidate in prereg["candidate_universe"]:
        binding = candidate["variant_binding"]
        overlay = _click_overlay(candidate)
        _write_bound_file(root, binding["overlay_ref"], _json_bytes(overlay))
        variant_manifest = {
            "schema_version": 1,
            "kind": "click_bell_single_axis_variant_v1",
            "variant_id": binding["variant_id"],
            "candidate_id": candidate["candidate_id"],
            "task_name": "click_bell",
            "task_module": binding["task_module"],
            "axis_id": binding["axis_id"],
            "variant_hint": deepcopy(binding["variant_hint"]),
            "overlay_ref": binding["overlay_ref"],
            "overlay_sha256": binding["overlay_sha256"],
        }
        _write_bound_file(
            root,
            binding["variant_manifest_ref"],
            _json_bytes(variant_manifest),
        )
    by_candidate = {
        candidate["candidate_id"]: candidate
        for candidate in prereg["candidate_universe"]
    }
    for arm, arm_rows in prereg["execution_schedule"].items():
        for binding in arm_rows:
            candidate = by_candidate[binding["candidate_id"]]
            seed_manifest = _seed_manifest(
                task_name="click_bell", seed=prereg["seed"]
            )
            _write_bound_file(
                root, binding["seed_manifest_ref"], _json_bytes(seed_manifest)
            )
            command = _efficiency_command_binding(
                study_id=prereg["study_id"],
                arm=arm,
                candidate=candidate,
                seed=prereg["seed"],
                checkpoint=prereg["checkpoint"],
                artifact_root_ref=prereg["artifact_root_ref"],
            )
            command_body = {
                "schema_version": 1,
                "kind": "click_bell_exact_n1_act_command_v1",
                "study_id": prereg["study_id"],
                "arm": arm,
                "candidate_id": binding["candidate_id"],
                "variant_id": binding["variant_id"],
                "variant_manifest_ref": candidate["variant_binding"][
                    "variant_manifest_ref"
                ],
                "variant_manifest_sha256": candidate["variant_binding"][
                    "variant_manifest_sha256"
                ],
                "checkpoint": prereg["checkpoint"],
                "seed": prereg["seed"],
                "cwd": ".",
                "environment": {"PYTHON_BIN": ROBOTWIN_PYTHON},
                "argv": [
                    "bash",
                    "policy/ACT/eval_mea.sh",
                    "click_bell",
                    "demo_clean",
                    "demo_clean",
                    "50",
                    "0",
                    "0",
                    "1",
                    CLICK_BELL_TASK_MODULE,
                    candidate["variant_binding"]["overlay_ref"],
                    "",
                    (
                        f"{prereg['artifact_root_ref']}/live_runs/{arm}/"
                        f"{binding['candidate_id']}/telemetry"
                    ),
                    "balanced_v1",
                    binding["seed_manifest_ref"],
                    binding["expected_seed_results_ref"],
                    (
                        f"{prereg['artifact_root_ref']}/live_runs/{arm}/"
                        f"{binding['candidate_id']}/eval_output"
                    ),
                ],
                "seed_manifest_ref": binding["seed_manifest_ref"],
                "seed_manifest_sha256": binding["seed_manifest_sha256"],
                "expected_seed_results_ref": binding["expected_seed_results_ref"],
                "expected_telemetry_episode_ref": binding[
                    "expected_telemetry_episode_ref"
                ],
                "receipt_ref": binding["receipt_ref"],
            }
            if command["command_sha256"] != binding["command_sha256"]:
                raise LivePaperProtocolError("internal command materialization mismatch")
            _write_bound_file(root, binding["command_ref"], _json_bytes(command_body))
    return validate_click_bell_efficiency_preregistration(
        prereg, repo_root=root, require_materialized=True
    )


def _live_attempt(
    value: Any,
    *,
    field: str,
    arm: str,
    arm_run_id: str,
    prereg: Mapping[str, Any],
    allowed_candidates: set[str],
    repo_root: Path,
) -> dict[str, Any]:
    row = _object(value, field=field)
    expected = {"attempt_id", "candidate_id", "receipt_ref", "receipt_sha256"}
    if set(row) != expected:
        raise LivePaperProtocolError(f"{field} fields must be exactly {sorted(expected)}")
    candidate = _identifier(row["candidate_id"], field=f"{field}.candidate_id")
    if candidate not in allowed_candidates:
        raise LivePaperProtocolError(f"{field} candidate is outside frozen arm")
    schedule_rows = prereg["execution_schedule"][arm]
    matches = [item for item in schedule_rows if item["candidate_id"] == candidate]
    if len(matches) != 1:
        raise LivePaperProtocolError(f"{field} has no unique command binding")
    command_binding = matches[0]
    receipt_ref = _relative_ref(row["receipt_ref"], field=f"{field}.receipt_ref")
    if receipt_ref != command_binding["receipt_ref"]:
        raise LivePaperProtocolError(f"{field} receipt path differs from preregistration")
    receipt_path = _bound_path(repo_root, receipt_ref, field=f"{field}.receipt_ref")
    supplied_receipt_sha256 = _sha256(
        row["receipt_sha256"], field=f"{field}.receipt_sha256"
    )
    if _file_sha256(receipt_path) != supplied_receipt_sha256:
        raise LivePaperProtocolError(f"{field} receipt hash mismatch")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LivePaperProtocolError(f"{field} receipt is invalid JSON: {exc}") from exc
    receipt = _object(receipt, field=f"{field}.receipt")
    expected_receipt_fields = {
        "schema_version",
        "protocol",
        "preregistration_sha256",
        "arm",
        "arm_run_id",
        "attempt_id",
        "candidate_id",
        "variant_id",
        "variant_manifest_sha256",
        "command_sha256",
        "checkpoint_sha256",
        "seed",
        "evidence_source",
        "started_at_utc",
        "ended_at_utc",
        "wall_seconds",
        "status",
        "success",
        "seed_results_ref",
        "seed_results_sha256",
        "telemetry_episode_ref",
        "telemetry_episode_sha256",
    }
    if set(receipt) != expected_receipt_fields:
        raise LivePaperProtocolError(f"{field} receipt fields are not exact")
    candidate_binding = next(
        item["variant_binding"]
        for item in prereg["candidate_universe"]
        if item["candidate_id"] == candidate
    )
    expected_identity = {
        "schema_version": 1,
        "protocol": "click_bell_bound_live_rollout_receipt_v1",
        "preregistration_sha256": prereg["preregistration_sha256"],
        "arm": arm,
        "arm_run_id": arm_run_id,
        "attempt_id": _identifier(row["attempt_id"], field=f"{field}.attempt_id"),
        "candidate_id": candidate,
        "variant_id": candidate_binding["variant_id"],
        "variant_manifest_sha256": candidate_binding["variant_manifest_sha256"],
        "command_sha256": command_binding["command_sha256"],
        "checkpoint_sha256": prereg["checkpoint"]["artifact_sha256"],
        "seed": prereg["seed"],
        "evidence_source": "live_policy_rollout",
    }
    if any(receipt.get(key) != value for key, value in expected_identity.items()):
        raise LivePaperProtocolError(f"{field} receipt identity mismatch")
    preregistered_at = _utc(prereg["created_at_utc"], field="created_at_utc")
    start = _utc(receipt["started_at_utc"], field=f"{field}.started_at_utc")
    end = _utc(receipt["ended_at_utc"], field=f"{field}.ended_at_utc")
    if start < preregistered_at or end < start:
        raise LivePaperProtocolError(f"{field} timestamps violate preregistration order")
    wall = _number(receipt["wall_seconds"], field=f"{field}.wall_seconds")
    if wall > (end - start).total_seconds() + 1.0:
        raise LivePaperProtocolError(f"{field}.wall_seconds exceeds elapsed time")
    status = receipt["status"]
    if status not in {"completed", "runtime_error"}:
        raise LivePaperProtocolError(f"{field}.status is invalid")
    success = receipt["success"]
    if status == "completed" and not isinstance(success, bool):
        raise LivePaperProtocolError(f"{field}.success must be boolean when completed")
    if status == "runtime_error" and success is not None:
        raise LivePaperProtocolError(f"{field}.success must be null on runtime_error")
    policy_steps: int | None = None
    if status == "completed":
        seed_results_ref = _relative_ref(
            receipt["seed_results_ref"], field=f"{field}.seed_results_ref"
        )
        telemetry_ref = _relative_ref(
            receipt["telemetry_episode_ref"],
            field=f"{field}.telemetry_episode_ref",
        )
        if (
            seed_results_ref != command_binding["expected_seed_results_ref"]
            or telemetry_ref != command_binding["expected_telemetry_episode_ref"]
        ):
            raise LivePaperProtocolError(f"{field} output paths differ from command")
        seed_results_path = _bound_path(
            repo_root, seed_results_ref, field=f"{field}.seed_results_ref"
        )
        telemetry_path = _bound_path(
            repo_root, telemetry_ref, field=f"{field}.telemetry_episode_ref"
        )
        if _file_sha256(seed_results_path) != _sha256(
            receipt["seed_results_sha256"],
            field=f"{field}.seed_results_sha256",
        ):
            raise LivePaperProtocolError(f"{field} seed results hash mismatch")
        if _file_sha256(telemetry_path) != _sha256(
            receipt["telemetry_episode_sha256"],
            field=f"{field}.telemetry_episode_sha256",
        ):
            raise LivePaperProtocolError(f"{field} telemetry hash mismatch")
        try:
            seed_results = json.loads(seed_results_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LivePaperProtocolError(
                f"{field} seed results are invalid JSON: {exc}"
            ) from exc
        measurements = seed_results.get("seed_measurements")
        if (
            seed_results.get("requested_seeds") != [prereg["seed"]]
            or seed_results.get("requested_count") != 1
            or seed_results.get("evaluated_count") != 1
            or not isinstance(measurements, list)
            or len(measurements) != 1
            or measurements[0].get("seed") != prereg["seed"]
            or measurements[0].get("policy_success") is not success
        ):
            raise LivePaperProtocolError(f"{field} seed results do not prove exact N=1")
        try:
            telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LivePaperProtocolError(
                f"{field} telemetry episode is invalid JSON: {exc}"
            ) from exc
        required_telemetry = {
            "task_name": "click_bell",
            "task_module": CLICK_BELL_TASK_MODULE,
            "task_config": "demo_clean",
            "policy_name": "ACT",
            "seed": prereg["seed"],
            "episode_index": 0,
            "success": success,
            "error": None,
        }
        if any(
            telemetry.get(key) != expected
            for key, expected in required_telemetry.items()
        ):
            raise LivePaperProtocolError(
                f"{field} telemetry policy/task/seed/outcome binding mismatch"
            )
        policy_steps = _integer(
            telemetry.get("policy_steps"),
            field=f"{field}.telemetry_episode.policy_steps",
            minimum=1,
        )
    else:
        if any(
            receipt.get(key) is not None
            for key in (
                "seed_results_ref",
                "seed_results_sha256",
                "telemetry_episode_ref",
                "telemetry_episode_sha256",
            )
        ):
            raise LivePaperProtocolError(
                f"{field} runtime_error cannot attest completed outputs"
            )
    return {
        "attempt_id": expected_identity["attempt_id"],
        "candidate_id": candidate,
        "seed": prereg["seed"],
        "evidence_source": "live_policy_rollout",
        "receipt_ref": receipt_ref,
        "receipt_sha256": supplied_receipt_sha256,
        "variant_id": candidate_binding["variant_id"],
        "started_at_utc": receipt["started_at_utc"],
        "ended_at_utc": receipt["ended_at_utc"],
        "wall_seconds": wall,
        "status": status,
        "success": success,
        "policy_steps": policy_steps,
    }


def _efficiency_arm(
    value: Any,
    *,
    arm: str,
    prereg: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    row = _object(value, field=f"{arm}_result")
    expected = {
        "schema_version",
        "protocol",
        "arm",
        "arm_run_id",
        "preregistration_sha256",
        "stop_reason",
        "attempts",
    }
    if set(row) != expected:
        raise LivePaperProtocolError(f"{arm} result fields must be exactly {sorted(expected)}")
    if row["schema_version"] != 1 or row["protocol"] != f"{EFFICIENCY_PROTOCOL}_arm":
        raise LivePaperProtocolError(f"unsupported {arm} result")
    if row["arm"] != arm or row["preregistration_sha256"] != prereg["preregistration_sha256"]:
        raise LivePaperProtocolError(f"{arm} result is not bound to preregistration")
    arm_run_id = _identifier(row["arm_run_id"], field=f"{arm}.arm_run_id")
    allowed = (
        set(prereg["fixed_contract"]["candidate_ids"])
        if arm == "fixed"
        else set(prereg["adaptive_contract"]["candidate_ids"])
    )
    attempts = [
        _live_attempt(
            item,
            field=f"{arm}.attempts[{index}]",
            arm=arm,
            arm_run_id=arm_run_id,
            prereg=prereg,
            allowed_candidates=allowed,
            repo_root=repo_root,
        )
        for index, item in enumerate(_items(row["attempts"], field=f"{arm}.attempts", minimum=1))
    ]
    identities = [item["attempt_id"] for item in attempts]
    refs = [item["receipt_ref"] for item in attempts]
    candidates = [item["candidate_id"] for item in attempts]
    if len(identities) != len(set(identities)) or len(refs) != len(set(refs)):
        raise LivePaperProtocolError(f"{arm} attempt ids and rollout refs must be unique")
    if len(candidates) != len(set(candidates)):
        raise LivePaperProtocolError(f"{arm} cannot retry or reuse a candidate in this bounded pilot")
    completed = {item["candidate_id"] for item in attempts if item["status"] == "completed"}
    if arm == "fixed":
        required = set(prereg["fixed_contract"]["candidate_ids"])
        if completed != required or len(attempts) != len(required):
            raise LivePaperProtocolError("fixed arm must complete its exact preregistered suite")
        if row["stop_reason"] != "fixed_suite_complete":
            raise LivePaperProtocolError("fixed arm must stop with fixed_suite_complete")
    else:
        contract = prereg["adaptive_contract"]
        if not contract["min_episode_starts"] <= len(attempts) <= contract["max_episode_starts"]:
            raise LivePaperProtocolError("adaptive arm start count violates frozen budget")
        if prereg["mode"] == "position_universal_3to4act":
            query_assessment = _efficiency_query_assessment(prereg, attempts)
            if row["stop_reason"] == "query_sufficient":
                if query_assessment["evidence_sufficient"] is not True:
                    raise LivePaperProtocolError(
                        "query_sufficient does not satisfy the frozen universal "
                        "truth condition"
                    )
            elif row["stop_reason"] == "budget_exhausted":
                if (
                    len(attempts) != contract["max_episode_starts"]
                    or query_assessment["evidence_sufficient"] is True
                ):
                    raise LivePaperProtocolError(
                        "budget_exhausted does not match universal query evidence"
                    )
            else:
                raise LivePaperProtocolError("invalid adaptive stop reason")
            return {
                "arm": arm,
                "arm_run_id": arm_run_id,
                "wall_seconds": sum(item["wall_seconds"] for item in attempts),
                "policy_steps": sum(item["policy_steps"] or 0 for item in attempts),
                "stop_reason": row["stop_reason"],
                "attempts": attempts,
                "query_assessment": query_assessment,
            }
        has_failure = any(
            item["status"] == "completed" and item["success"] is False
            for item in attempts
        )
        completed_scores = {
            item["candidate_id"]: item["success"]
            for item in attempts
            if item["status"] == "completed"
        }
        paired_failure = any(
            completed_scores.get(left) is False
            or completed_scores.get(right) is False
            for left, right in _EFFICIENCY_AXIS_PAIRS.values()
            if left in completed_scores and right in completed_scores
        )
        if row["stop_reason"] == "query_sufficient":
            sufficient = (
                has_failure
                if prereg["mode"] == "smoke_3act"
                else has_failure and paired_failure
            )
            if not sufficient:
                raise LivePaperProtocolError(
                    "query_sufficient requires a completed failure and its "
                    "frozen paired-axis contrast"
                )
        elif row["stop_reason"] == "budget_exhausted":
            evidence_sufficient = (
                has_failure
                if prereg["mode"] == "smoke_3act"
                else paired_failure
            )
            if (
                len(attempts) != contract["max_episode_starts"]
                or evidence_sufficient
            ):
                raise LivePaperProtocolError("budget_exhausted does not match adaptive evidence")
        else:
            raise LivePaperProtocolError("invalid adaptive stop reason")
    return {
        "arm": arm,
        "arm_run_id": arm_run_id,
        "wall_seconds": sum(item["wall_seconds"] for item in attempts),
        "policy_steps": sum(item["policy_steps"] or 0 for item in attempts),
        "stop_reason": row["stop_reason"],
        "attempts": attempts,
    }


def _efficiency_query_assessment(
    prereg: Mapping[str, Any],
    attempts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    contract = prereg.get("query_sufficiency_contract")
    if not isinstance(contract, Mapping):
        raise LivePaperProtocolError(
            "universal efficiency mode has no query-sufficiency contract"
        )
    evidence = [
        {
            "candidate_id": item["candidate_id"],
            "outcome": "pass" if item["success"] is True else "fail",
            "score": 1.0 if item["success"] is True else 0.0,
            "diagnosis": None,
        }
        for item in attempts
        if item["status"] == "completed"
    ]
    return assess_query_sufficiency(
        contract,
        evidence,
        completed_rounds=len(attempts),
    )


def _efficiency_conclusion(
    arm: Mapping[str, Any],
    *,
    prereg: Mapping[str, Any],
) -> dict[str, Any]:
    scores = {
        item["candidate_id"]: item["success"]
        for item in arm["attempts"]
        if item["status"] == "completed"
    }
    failures = sorted(key for key, value in scores.items() if value is False)
    if failures:
        verdict = "weakness_observed"
    elif set(scores) == set(_CANDIDATE_IDS):
        verdict = "frozen_suite_all_succeeded"
    else:
        verdict = "inconclusive"
    axes: list[str] = []
    for axis, (left, right) in _EFFICIENCY_AXIS_PAIRS.items():
        if left in scores and right in scores and scores[left] != scores[right]:
            axes.append(axis)
    result = {
        "overall_verdict": verdict,
        "weakness_axes": axes,
        "observed_failure_candidates": failures,
        "tested_candidates": sorted(scores),
    }
    if prereg["mode"] == "position_universal_3to4act":
        assessment = (
            deepcopy(dict(arm["query_assessment"]))
            if isinstance(arm.get("query_assessment"), Mapping)
            else _efficiency_query_assessment(prereg, arm["attempts"])
        )
        result.update(
            {
                "claim_type": "universal",
                "claim_verdict": assessment["claim_verdict"],
                "evidence_sufficient": assessment["evidence_sufficient"],
                "query_assessment": assessment,
            }
        )
    return result


def evaluate_click_bell_efficiency(
    preregistration: Any,
    fixed_result: Any,
    adaptive_result: Any,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    prereg = validate_click_bell_efficiency_preregistration(
        preregistration, repo_root=root, require_materialized=True
    )
    fixed = _efficiency_arm(
        fixed_result, arm="fixed", prereg=prereg, repo_root=root
    )
    adaptive = _efficiency_arm(
        adaptive_result, arm="adaptive", prereg=prereg, repo_root=root
    )
    if fixed["arm_run_id"] == adaptive["arm_run_id"]:
        raise LivePaperProtocolError("arms must have independent run ids")
    fixed_ids = {item["attempt_id"] for item in fixed["attempts"]}
    adaptive_ids = {item["attempt_id"] for item in adaptive["attempts"]}
    fixed_refs = {item["receipt_ref"] for item in fixed["attempts"]}
    adaptive_refs = {item["receipt_ref"] for item in adaptive["attempts"]}
    if fixed_ids & adaptive_ids or fixed_refs & adaptive_refs:
        raise LivePaperProtocolError("arms cannot share rollout receipts")
    total_starts = len(fixed["attempts"]) + len(adaptive["attempts"])
    budget = prereg["total_episode_start_contract"]
    if not budget["minimum"] <= total_starts <= budget["maximum"]:
        raise LivePaperProtocolError("pair violates frozen total ACT-start budget")
    conclusions = {
        "fixed": _efficiency_conclusion(fixed, prereg=prereg),
        "adaptive": _efficiency_conclusion(adaptive, prereg=prereg),
    }
    fields = prereg["conclusion_contract"]["comparison_fields"]
    agrees = all(conclusions["fixed"][field] == conclusions["adaptive"][field] for field in fields)
    act_saving = len(fixed["attempts"]) - len(adaptive["attempts"])
    wall_saving = fixed["wall_seconds"] - adaptive["wall_seconds"]
    policy_step_saving = fixed["policy_steps"] - adaptive["policy_steps"]
    technical_errors = sum(
        item["status"] != "completed"
        for arm in (fixed, adaptive)
        for item in arm["attempts"]
    )
    eligible_toy = (
        prereg["mode"]
        in {"toy_5to7act", "position_universal_3to4act"}
        and technical_errors == 0
        and agrees
        and act_saving > 0
        and wall_saving > 0
        and policy_step_saving > 0
    )
    return {
        "schema_version": 1,
        "protocol": f"{EFFICIENCY_PROTOCOL}_result",
        "study_id": prereg["study_id"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "comparison_design": "independent_live_arms",
        "cached_prefix_used": False,
        "mode": prereg["mode"],
        "claim_scope": prereg["claim_scope"],
        "arms": {"fixed": fixed, "adaptive": adaptive},
        "conclusions": conclusions,
        "conclusion_comparison_fields": fields,
        "original_query_conclusion_agrees": agrees,
        "resource_measurement": {
            "fixed_act_episode_starts": len(fixed["attempts"]),
            "adaptive_act_episode_starts": len(adaptive["attempts"]),
            "act_episode_start_saving": act_saving,
            "fixed_wall_seconds": fixed["wall_seconds"],
            "adaptive_wall_seconds": adaptive["wall_seconds"],
            "measured_wall_second_saving": wall_saving,
            "fixed_policy_steps": fixed["policy_steps"],
            "adaptive_policy_steps": adaptive["policy_steps"],
            "policy_step_saving": policy_step_saving,
            "technical_runtime_errors": technical_errors,
        },
        "toy_efficiency_evidence_passed": eligible_toy,
        "paper_tables_1_2_eligible": False,
        "limitations": [
            "The three-ACT mode is a mechanism smoke, not a dense reference.",
            "The five-to-seven-ACT mode is one task, one checkpoint, and one seed.",
            "The three-to-four-ACT universal mode covers only two frozen positions.",
            "Policy steps are the shared simulator-sample proxy for this toy.",
            "This protocol does not reproduce the paper trial or agent-run counts.",
        ],
    }


def _ranking_command_binding(
    *,
    study_id: str,
    policy_id: str,
    seed: int,
    checkpoint: Mapping[str, Any],
    artifact_root_ref: str,
) -> dict[str, Any]:
    command_root = f"{artifact_root_ref}/commands/{policy_id}/seed_{seed}"
    live_root = f"{artifact_root_ref}/live_runs/{policy_id}/seed_{seed}"
    seed_manifest = _seed_manifest(task_name="beat_block_hammer", seed=seed)
    seed_manifest_ref = f"{command_root}/seed_manifest.json"
    common_outputs = {
        "seed_manifest_ref": seed_manifest_ref,
        "seed_manifest_sha256": _bytes_sha256(_json_bytes(seed_manifest)),
        "expected_seed_results_ref": f"{live_root}/seed_results.json",
        "expected_telemetry_episode_ref": (
            f"{live_root}/telemetry/episode_000_seed_{seed}/episode.json"
        ),
        "expected_output_dir": f"{live_root}/eval_output",
    }
    if policy_id == "act":
        environment = {"PYTHON_BIN": ROBOTWIN_PYTHON}
        argv = [
            "bash",
            "policy/ACT/eval_mea.sh",
            "beat_block_hammer",
            "demo_clean",
            "demo_clean",
            "50",
            "0",
            "0",
            "1",
            "envs.beat_block_hammer",
            "",
            "",
            f"{live_root}/telemetry",
            "balanced_v1",
            seed_manifest_ref,
            common_outputs["expected_seed_results_ref"],
            common_outputs["expected_output_dir"],
        ]
        entrypoint = "policy/ACT/eval_mea.sh"
    elif policy_id == "dp3":
        environment = {"CUDA_VISIBLE_DEVICES": "0"}
        argv = [
            DP3_PYTHON,
            "script/eval_policy.py",
            "--config",
            "policy/DP3/deploy_policy.yml",
            "--overrides",
            "--task_name",
            "beat_block_hammer",
            "--task_module",
            "envs.beat_block_hammer",
            "--task_config",
            "demo_clean",
            "--ckpt_setting",
            "demo_clean",
            "--expert_data_num",
            "50",
            "--seed",
            "0",
            "--policy_name",
            "DP3",
            "--config_name",
            "robot_dp3",
            "--checkpoint_num",
            "3000",
            "--use_rgb",
            "False",
            "--num_episodes",
            "1",
            "--seed_manifest",
            seed_manifest_ref,
            "--telemetry_dir",
            f"{live_root}/telemetry",
            "--telemetry_profile",
            "balanced_v1",
            "--seed_results_path",
            common_outputs["expected_seed_results_ref"],
            "--output_dir",
            common_outputs["expected_output_dir"],
        ]
        entrypoint = "script/eval_policy.py"
    else:  # pragma: no cover - only the frozen policies call this helper.
        raise LivePaperProtocolError(f"unsupported ranking policy: {policy_id}")
    command = {
        "schema_version": 1,
        "kind": "exact_seed_policy_n1_command_v1",
        "study_id": study_id,
        "policy_id": policy_id,
        "seed": seed,
        "checkpoint": deepcopy(dict(checkpoint)),
        "entrypoint": entrypoint,
        "python_environment": (
            ROBOTWIN_PYTHON if policy_id == "act" else DP3_PYTHON
        ),
        "cwd": ".",
        "environment": environment,
        "argv": argv,
        **common_outputs,
    }
    return {
        "policy_id": policy_id,
        "seed": seed,
        "command_ref": f"{command_root}/command.json",
        "command_sha256": _bytes_sha256(_json_bytes(command)),
        **common_outputs,
    }


def build_ranking_preregistration(
    *,
    study_id: str,
    act_checkpoint: Mapping[str, Any],
    dp3_checkpoint: Mapping[str, Any],
    seeds: Sequence[int],
    created_at_utc: str,
    reference_source_ref: str,
    reference_scores: Mapping[str, float],
    artifact_root_ref: str | None = None,
) -> dict[str, Any]:
    _utc(created_at_utc, field="created_at_utc")
    normalized_seeds = [_integer(seed, field="seeds[]") for seed in seeds]
    if len(normalized_seeds) != 3 or len(set(normalized_seeds)) != 3:
        raise LivePaperProtocolError("ranking pilot requires exactly three unique seeds")
    if set(reference_scores) != {"act", "dp3"}:
        raise LivePaperProtocolError("reference_scores must contain exactly act and dp3")
    resolved_study_id = _identifier(study_id, field="study_id")
    resolved_artifact_root = _relative_ref(
        artifact_root_ref
        or f"mea/protocol_runs/{resolved_study_id}/ranking_artifacts",
        field="artifact_root_ref",
    )
    policies = {
        "act": _checkpoint(
            act_checkpoint,
            field="act_checkpoint",
            artifact_ref=BBH_ACT_CHECKPOINT_REF,
        ),
        "dp3": _checkpoint(
            dp3_checkpoint,
            field="dp3_checkpoint",
            artifact_ref=BBH_DP3_CHECKPOINT_REF,
        ),
    }
    commands = {
        policy_id: [
            _ranking_command_binding(
                study_id=resolved_study_id,
                policy_id=policy_id,
                seed=seed,
                checkpoint=policies[policy_id],
                artifact_root_ref=resolved_artifact_root,
            )
            for seed in normalized_seeds
        ]
        for policy_id in ("act", "dp3")
    }
    body = {
        "schema_version": 1,
        "protocol": RANKING_PROTOCOL,
        "study_id": resolved_study_id,
        "created_at_utc": created_at_utc,
        "artifact_root_ref": resolved_artifact_root,
        "candidate_id": "bbh_official_demo_clean",
        "seeds": normalized_seeds,
        "policies": policies,
        "reference_source_ref": _text(reference_source_ref, field="reference_source_ref"),
        "reference_scores": {
            key: _number(reference_scores[key], field=f"reference_scores.{key}")
            for key in ("act", "dp3")
        },
        "execution_entrypoints": {
            "act": "policy/ACT/eval_mea.sh",
            "dp3": "script/eval_policy.py",
        },
        "execution_schedule": commands,
        "rollout_contract": {
            "exact_trials_per_policy": 3,
            "exact_total_policy_rollouts": 6,
            "evidence_source": "live_policy_rollout",
            "tie_rule": "spearman_null_and_inconclusive",
        },
        "claim_scope": "two_policy_three_seed_pair_order_pilot_not_table9",
        "calls_started_by_preregistration": {
            "provider": 0,
            "simulator": 0,
            "expert": 0,
            "probe": 0,
            "act": 0,
        },
    }
    return _seal(body, hash_field="preregistration_sha256")


def validate_ranking_preregistration(
    value: Any,
    *,
    repo_root: str | Path | None = None,
    require_materialized: bool = False,
) -> dict[str, Any]:
    row = _verify_seal(_object(value, field="ranking preregistration"), hash_field="preregistration_sha256")
    if row.get("schema_version") != 1 or row.get("protocol") != RANKING_PROTOCOL:
        raise LivePaperProtocolError("unsupported ranking preregistration")
    rebuilt = build_ranking_preregistration(
        study_id=row.get("study_id"),
        act_checkpoint=_object(row.get("policies"), field="policies").get("act"),
        dp3_checkpoint=_object(row.get("policies"), field="policies").get("dp3"),
        seeds=row.get("seeds"),
        created_at_utc=row.get("created_at_utc"),
        reference_source_ref=row.get("reference_source_ref"),
        reference_scores=row.get("reference_scores"),
        artifact_root_ref=row.get("artifact_root_ref"),
    )
    if rebuilt != row:
        raise LivePaperProtocolError("ranking preregistration contract was modified")
    if require_materialized:
        if repo_root is None:
            raise LivePaperProtocolError("repo_root is required for materialized ranking")
        root = Path(repo_root).expanduser().resolve()
        for policy_id, checkpoint in row["policies"].items():
            checkpoint_path = _checkpoint_artifact_path(
                root,
                checkpoint["artifact_ref"],
                field=f"policies.{policy_id}.artifact_ref",
            )
            if _file_sha256(checkpoint_path) != checkpoint["artifact_sha256"]:
                raise LivePaperProtocolError(
                    f"{policy_id} checkpoint artifact hash mismatch"
                )
        for policy_rows in row["execution_schedule"].values():
            for binding in policy_rows:
                seed_path = _bound_path(
                    root, binding["seed_manifest_ref"], field="seed_manifest_ref"
                )
                command_path = _bound_path(
                    root, binding["command_ref"], field="command_ref"
                )
                if _file_sha256(seed_path) != binding["seed_manifest_sha256"]:
                    raise LivePaperProtocolError("ranking seed manifest hash mismatch")
                if _file_sha256(command_path) != binding["command_sha256"]:
                    raise LivePaperProtocolError("ranking command hash mismatch")
    return row


def materialize_ranking_preregistration(
    repo_root: str | Path, preregistration: Any
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    prereg = validate_ranking_preregistration(preregistration)
    for policy_id, rows in prereg["execution_schedule"].items():
        for binding in rows:
            seed = binding["seed"]
            _write_bound_file(
                root,
                binding["seed_manifest_ref"],
                _json_bytes(
                    _seed_manifest(task_name="beat_block_hammer", seed=seed)
                ),
            )
            rebuilt = _ranking_command_binding(
                study_id=prereg["study_id"],
                policy_id=policy_id,
                seed=seed,
                checkpoint=prereg["policies"][policy_id],
                artifact_root_ref=prereg["artifact_root_ref"],
            )
            live_root = (
                f"{prereg['artifact_root_ref']}/live_runs/{policy_id}/seed_{seed}"
            )
            if policy_id == "act":
                environment = {"PYTHON_BIN": ROBOTWIN_PYTHON}
                argv = [
                    "bash",
                    "policy/ACT/eval_mea.sh",
                    "beat_block_hammer",
                    "demo_clean",
                    "demo_clean",
                    "50",
                    "0",
                    "0",
                    "1",
                    "envs.beat_block_hammer",
                    "",
                    "",
                    f"{live_root}/telemetry",
                    "balanced_v1",
                    binding["seed_manifest_ref"],
                    binding["expected_seed_results_ref"],
                    binding["expected_output_dir"],
                ]
                entrypoint = "policy/ACT/eval_mea.sh"
            else:
                environment = {"CUDA_VISIBLE_DEVICES": "0"}
                argv = [
                    DP3_PYTHON,
                    "script/eval_policy.py",
                    "--config",
                    "policy/DP3/deploy_policy.yml",
                    "--overrides",
                    "--task_name",
                    "beat_block_hammer",
                    "--task_module",
                    "envs.beat_block_hammer",
                    "--task_config",
                    "demo_clean",
                    "--ckpt_setting",
                    "demo_clean",
                    "--expert_data_num",
                    "50",
                    "--seed",
                    "0",
                    "--policy_name",
                    "DP3",
                    "--config_name",
                    "robot_dp3",
                    "--checkpoint_num",
                    "3000",
                    "--use_rgb",
                    "False",
                    "--num_episodes",
                    "1",
                    "--seed_manifest",
                    binding["seed_manifest_ref"],
                    "--telemetry_dir",
                    f"{live_root}/telemetry",
                    "--telemetry_profile",
                    "balanced_v1",
                    "--seed_results_path",
                    binding["expected_seed_results_ref"],
                    "--output_dir",
                    binding["expected_output_dir"],
                ]
                entrypoint = "script/eval_policy.py"
            command = {
                "schema_version": 1,
                "kind": "exact_seed_policy_n1_command_v1",
                "study_id": prereg["study_id"],
                "policy_id": policy_id,
                "seed": seed,
                "checkpoint": prereg["policies"][policy_id],
                "entrypoint": entrypoint,
                "python_environment": (
                    ROBOTWIN_PYTHON if policy_id == "act" else DP3_PYTHON
                ),
                "cwd": ".",
                "environment": environment,
                "argv": argv,
                "seed_manifest_ref": binding["seed_manifest_ref"],
                "seed_manifest_sha256": binding["seed_manifest_sha256"],
                "expected_seed_results_ref": binding[
                    "expected_seed_results_ref"
                ],
                "expected_telemetry_episode_ref": binding[
                    "expected_telemetry_episode_ref"
                ],
                "expected_output_dir": binding["expected_output_dir"],
            }
            if rebuilt["command_sha256"] != binding["command_sha256"]:
                raise LivePaperProtocolError("internal ranking command mismatch")
            _write_bound_file(root, binding["command_ref"], _json_bytes(command))
    return validate_ranking_preregistration(
        prereg, repo_root=root, require_materialized=True
    )


def _read_ranking_json(path: Path, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LivePaperProtocolError(f"{field} is invalid JSON: {exc}") from exc
    return _object(value, field=field)


def _ranking_command_for_seed(
    prereg: Mapping[str, Any],
    *,
    policy_id: str,
    seed: int,
) -> dict[str, Any]:
    matches = [
        row
        for row in prereg["execution_schedule"][policy_id]
        if row.get("policy_id") == policy_id and row.get("seed") == seed
    ]
    if len(matches) != 1:
        raise LivePaperProtocolError(
            f"{policy_id} seed {seed} has no unique preregistered command"
        )
    return dict(matches[0])


def _ranking_seed_result(
    value: Mapping[str, Any],
    *,
    policy_id: str,
    seed: int,
    expected_episode_dir: Path,
) -> bool:
    field = f"{policy_id}.seed_{seed}.seed_results"
    required_identity = {
        "schema_version": 1,
        "protocol": "exact_seed_paired_v1",
        "task_name": "beat_block_hammer",
        "task_config": "demo_clean",
        "condition_id": "clean",
        "requested_seeds": [seed],
        "requested_count": 1,
        "eligible_count": 1,
        "evaluated_count": 1,
        "all_eligible": True,
        "no_seed_replacement": True,
    }
    if any(value.get(key) != expected for key, expected in required_identity.items()):
        raise LivePaperProtocolError(
            f"{field} does not prove the exact eligible N=1 contract"
        )
    measurements = value.get("seed_measurements")
    if not isinstance(measurements, list) or len(measurements) != 1:
        raise LivePaperProtocolError(f"{field} must contain exactly one measurement")
    measurement = _object(measurements[0], field=f"{field}.seed_measurements[0]")
    success = measurement.get("policy_success")
    if not isinstance(success, bool):
        raise LivePaperProtocolError(f"{field} policy_success must be boolean")
    expected_status = "success" if success else "failure"
    if (
        measurement.get("requested_index") != 0
        or measurement.get("seed") != seed
        or measurement.get("eligibility_status") != "passed"
        or measurement.get("policy_executed") is not True
        or measurement.get("execution_attempted") is not True
        or measurement.get("policy_status") != expected_status
        or "policy_error" in measurement
        or "eligibility_error" in measurement
    ):
        raise LivePaperProtocolError(
            f"{field} does not prove one completed policy execution"
        )
    telemetry_episode_dir = measurement.get("telemetry_episode_dir")
    if not isinstance(telemetry_episode_dir, str) or not telemetry_episode_dir:
        raise LivePaperProtocolError(f"{field} is missing telemetry_episode_dir")
    observed_episode_dir = Path(telemetry_episode_dir).expanduser()
    if not observed_episode_dir.is_absolute():
        raise LivePaperProtocolError(
            f"{field} telemetry_episode_dir must be absolute"
        )
    if observed_episode_dir.resolve() != expected_episode_dir.resolve():
        raise LivePaperProtocolError(
            f"{field} telemetry directory differs from preregistration"
        )
    if (
        value.get("success_count") != int(success)
        or value.get("success_rate_evaluated") != float(success)
    ):
        raise LivePaperProtocolError(f"{field} aggregate success differs from episode")
    return success


def _ranking_episode(
    value: Mapping[str, Any],
    *,
    policy_id: str,
    seed: int,
    success: bool,
) -> tuple[float, int]:
    field = f"{policy_id}.seed_{seed}.telemetry_episode"
    expected_policy_name = {"act": "ACT", "dp3": "DP3"}[policy_id]
    required_identity = {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "task_module": "envs.beat_block_hammer",
        "task_config": "demo_clean",
        "checkpoint_setting": "demo_clean",
        "policy_name": expected_policy_name,
        "seed": seed,
        "episode_index": 0,
        "success": success,
        "error": None,
    }
    if any(value.get(key) != expected for key, expected in required_identity.items()):
        raise LivePaperProtocolError(
            f"{field} policy/task/seed/outcome binding mismatch"
        )
    policy_steps = _integer(
        value.get("policy_steps"), field=f"{field}.policy_steps", minimum=1
    )
    wall_seconds = _number(
        value.get("wall_duration_seconds"),
        field=f"{field}.wall_duration_seconds",
    )
    if value.get("recorder_schema_version") not in {2, 3}:
        raise LivePaperProtocolError(f"{field} recorder schema is unsupported")
    return wall_seconds, policy_steps


def evaluate_exact_seed_ranking(
    preregistration: Any,
    result_manifest: Any,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    prereg = validate_ranking_preregistration(
        preregistration,
        repo_root=root,
        require_materialized=True,
    )
    result = _object(result_manifest, field="ranking result")
    expected_result_fields = {
        "schema_version",
        "protocol",
        "preregistration_sha256",
        "policies",
    }
    if set(result) != expected_result_fields:
        raise LivePaperProtocolError(
            "ranking result fields must be exactly "
            f"{sorted(expected_result_fields)}"
        )
    if result.get("schema_version") != 1 or result.get("protocol") != f"{RANKING_PROTOCOL}_runs":
        raise LivePaperProtocolError("unsupported ranking result manifest")
    if result.get("preregistration_sha256") != prereg["preregistration_sha256"]:
        raise LivePaperProtocolError("ranking result is not bound to preregistration")
    policy_rows = _items(result.get("policies"), field="policies", minimum=2)
    if {row.get("policy_id") for row in policy_rows if isinstance(row, Mapping)} != {"act", "dp3"}:
        raise LivePaperProtocolError("ranking result must contain exactly ACT and DP3")
    converted: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    trial_ids: set[str] = set()
    seed_result_refs: set[str] = set()
    telemetry_refs: set[str] = set()
    seed_result_hashes: set[str] = set()
    telemetry_hashes: set[str] = set()
    total_wall = 0.0
    for raw_policy in policy_rows:
        policy = _object(raw_policy, field="policy")
        expected_policy_fields = {"policy_id", "checkpoint", "run_id", "trials"}
        if set(policy) != expected_policy_fields:
            raise LivePaperProtocolError(
                f"policy fields must be exactly {sorted(expected_policy_fields)}"
            )
        policy_id = policy.get("policy_id")
        checkpoint = _checkpoint(policy.get("checkpoint"), field=f"{policy_id}.checkpoint")
        if checkpoint != prereg["policies"][policy_id]:
            raise LivePaperProtocolError(f"{policy_id} checkpoint differs from preregistration")
        run_id = _identifier(policy.get("run_id"), field=f"{policy_id}.run_id")
        if run_id in run_ids:
            raise LivePaperProtocolError("policy run ids must be independent")
        run_ids.add(run_id)
        trials = _items(policy.get("trials"), field=f"{policy_id}.trials", minimum=3)
        if len(trials) != 3:
            raise LivePaperProtocolError("each policy requires exactly three trials")
        seen_seeds: set[int] = set()
        converted_trials: list[dict[str, Any]] = []
        for index, raw_trial in enumerate(trials):
            trial = _object(raw_trial, field=f"{policy_id}.trials[{index}]")
            expected_trial_fields = {
                "trial_id",
                "seed",
                "evidence_source",
                "status",
                "error",
                "seed_results_ref",
                "seed_results_sha256",
                "telemetry_episode_ref",
                "telemetry_episode_sha256",
            }
            if set(trial) != expected_trial_fields:
                raise LivePaperProtocolError(
                    f"{policy_id}.trials[{index}] fields must be exactly "
                    f"{sorted(expected_trial_fields)}"
                )
            seed = _integer(trial.get("seed"), field="trial.seed")
            if seed not in prereg["seeds"] or seed in seen_seeds:
                raise LivePaperProtocolError("policy trials must cover exact unique seeds")
            seen_seeds.add(seed)
            if trial.get("evidence_source") != "live_policy_rollout":
                raise LivePaperProtocolError("ranking trials must be live, never cached")
            if trial.get("status") != "completed" or trial.get("error") is not None:
                raise LivePaperProtocolError("ranking requires six completed trials")
            trial_id = _identifier(trial.get("trial_id"), field="trial.trial_id")
            if trial_id in trial_ids:
                raise LivePaperProtocolError("ranking trial ids must be unique")
            trial_ids.add(trial_id)
            command = _ranking_command_for_seed(
                prereg, policy_id=policy_id, seed=seed
            )
            seed_results_ref = _relative_ref(
                trial.get("seed_results_ref"), field="trial.seed_results_ref"
            )
            telemetry_ref = _relative_ref(
                trial.get("telemetry_episode_ref"),
                field="trial.telemetry_episode_ref",
            )
            if (
                seed_results_ref != command["expected_seed_results_ref"]
                or telemetry_ref != command["expected_telemetry_episode_ref"]
            ):
                raise LivePaperProtocolError(
                    "ranking output paths differ from preregistered command"
                )
            if (
                seed_results_ref in seed_result_refs
                or telemetry_ref in telemetry_refs
            ):
                raise LivePaperProtocolError(
                    "ranking trials cannot share evidence files"
                )
            seed_result_refs.add(seed_results_ref)
            telemetry_refs.add(telemetry_ref)
            seed_results_path = _bound_path(
                root, seed_results_ref, field="trial.seed_results_ref"
            )
            telemetry_path = _bound_path(
                root, telemetry_ref, field="trial.telemetry_episode_ref"
            )
            supplied_seed_sha = _sha256(
                trial.get("seed_results_sha256"),
                field="trial.seed_results_sha256",
            )
            supplied_telemetry_sha = _sha256(
                trial.get("telemetry_episode_sha256"),
                field="trial.telemetry_episode_sha256",
            )
            if _file_sha256(seed_results_path) != supplied_seed_sha:
                raise LivePaperProtocolError("ranking seed results hash mismatch")
            if _file_sha256(telemetry_path) != supplied_telemetry_sha:
                raise LivePaperProtocolError("ranking telemetry hash mismatch")
            if (
                supplied_seed_sha in seed_result_hashes
                or supplied_telemetry_sha in telemetry_hashes
            ):
                raise LivePaperProtocolError(
                    "ranking trials must have unique evidence hashes"
                )
            seed_result_hashes.add(supplied_seed_sha)
            telemetry_hashes.add(supplied_telemetry_sha)
            success = _ranking_seed_result(
                _read_ranking_json(
                    seed_results_path, field="trial.seed_results"
                ),
                policy_id=policy_id,
                seed=seed,
                expected_episode_dir=telemetry_path.parent,
            )
            wall, policy_steps = _ranking_episode(
                _read_ranking_json(
                    telemetry_path, field="trial.telemetry_episode"
                ),
                policy_id=policy_id,
                seed=seed,
                success=success,
            )
            total_wall += wall
            score = float(success)
            converted_trials.append(
                {
                    "trial_id": trial_id,
                    "candidate_id": prereg["candidate_id"],
                    "seed": seed,
                    "rollout_ref": telemetry_ref,
                    "episode_status": "completed",
                    "score": score,
                    "policy_steps": policy_steps,
                    "seed_results_ref": seed_results_ref,
                    "seed_results_sha256": supplied_seed_sha,
                    "telemetry_episode_sha256": supplied_telemetry_sha,
                }
            )
        if seen_seeds != set(prereg["seeds"]):
            raise LivePaperProtocolError("policy is missing an exact seed")
        converted.append(
            {
                "policy_id": policy_id,
                "checkpoint_id": checkpoint["checkpoint_id"],
                "run_id": run_id,
                "trials": converted_trials,
            }
        )
    ranking = evaluate_policy_ranking(
        {
            "schema_version": 1,
            "protocol": "paper_claim_policy_ranking_v1",
            "evidence_source": "live_policy_rollout",
            "study_id": prereg["study_id"],
            "candidate_universe": [prereg["candidate_id"]],
            "seeds": prereg["seeds"],
            "reference_source_ref": prereg["reference_source_ref"],
            "reference_scores": prereg["reference_scores"],
            "policies": converted,
        }
    )
    ranking.update(
        {
            "preregistration_sha256": prereg["preregistration_sha256"],
            "exact_seed_pair": True,
            "exact_trials_per_policy": 3,
            "exact_total_policy_rollouts": 6,
            "measured_trial_wall_seconds_total": total_wall,
            "paper_table9_eligible": False,
            "scope_limitation": (
                "Two policies, one task, and three seeds; a tie leaves Spearman null."
            ),
        }
    )
    return ranking


def _table3_success_spec() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "task_name": "beat_block_hammer",
        "envelope_id": "bbh.experimental_bounded_act",
        "logic": "all",
        "predicates": [
            {
                "predicate": "planar_axis_distance",
                "left": {"actor": "hammer", "functional_point_id": 0},
                "right": {"actor": "block", "functional_point_id": 1},
                "axes": [0, 1],
                "thresholds_m": [0.025, 0.025],
                "comparison": "strict_lt",
            },
            {
                "predicate": "physical_contact",
                "actors": ["hammer", "block"],
            },
        ],
    }


def _table3_task_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    changes = deepcopy(proposal["changes"])
    is_scale = float(changes["block"]["scale"]) != 1.0
    return {
        "schema_version": 2,
        "proposal_id": proposal["proposal_id"],
        "task_name": "beat_block_hammer",
        "aspect_id": "object_scale" if is_scale else "object_appearance.color",
        "intent": proposal["prompt"],
        "capability_id": "object_scale" if is_scale else "object_appearance.color",
        # TaskProposal schema remains reuse-first; the preregistered runner's
        # explicit ``force_codegen`` mode supplies the ablation override.
        "reuse_first": True,
        "changes": changes,
        "preserve_success_semantics": False,
        "success_spec": _table3_success_spec(),
    }


def _table3_runner(
    *,
    study_id: str,
    proposal: Mapping[str, Any],
    condition: str,
    proposal_ref: str,
    proposal_sha256: str,
    artifact_root_ref: str,
    text_model: str,
    vision_model: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell_id = f"{proposal['proposal_id']}__{condition}"
    run_token = "".join(
        ch if ch.isalnum() or ch == "_" else "_" for ch in study_id
    )
    run_id = f"run_{run_token}_{cell_id}"
    switches = deepcopy(TABLE3_SWITCHES[condition])
    task_proposal = _table3_task_proposal(proposal)
    argv = [
        ROBOTWIN_PYTHON,
        "scripts/manipeval_taskgen.py",
        "--repo-root",
        ".",
        "--request",
        proposal["prompt"],
        "--run-id",
        run_id,
        "--task-name",
        "beat_block_hammer",
        "--mode",
        "force_codegen",
        "--task-proposal-json",
        json.dumps(task_proposal, ensure_ascii=False, sort_keys=True),
        "--taskgen-ablation-json",
        json.dumps(switches, ensure_ascii=False, sort_keys=True),
        "--text-model",
        text_model,
        "--vision-model",
        vision_model,
        "--seed",
        str(seed),
        "--num-episodes",
        "1",
        "--gpu",
        "0",
        "--expert",
        "--accept-task-only",
        "--max-reflections",
        "2",
    ]
    if switches["visual_self_check"]:
        argv.append("--vision-check")
    run_root = f"mea/generated_tasks/{run_id}"
    stage_receipts = {
        "codegen": {
            "artifact_ref": f"{run_root}/task.py",
            "manifest_ref": f"{run_root}/manifest.json",
        },
        "compile": {"receipt_ref": f"{run_root}/validation/static.json"},
        "render": {"receipt_ref": f"{run_root}/validation/scene.json"},
        "simulator": {"receipt_ref": f"{run_root}/validation/scene.json"},
        "oracle": {
            "receipt_ref": (
                f"{run_root}/validation/task_generation_attempts/"
                "task_generation_attempt_summary.json"
            )
        },
    }
    runner = {
        "schema_version": 1,
        "kind": "table3_real_taskgen_cell_runner_v1",
        "study_id": study_id,
        "cell_id": cell_id,
        "proposal_id": proposal["proposal_id"],
        "condition": condition,
        "module_switches": switches,
        "proposal_ref": proposal_ref,
        "proposal_sha256": proposal_sha256,
        "provider_models": {"text": text_model, "vision": vision_model},
        "required_environment": ["UIUI_API_KEY"],
        "cwd": ".",
        "argv": argv,
        "run_id": run_id,
        "act_rollout_budget": 0,
        "expected_stage_receipts": stage_receipts,
    }
    runner_ref = f"{artifact_root_ref}/cells/{cell_id}/runner.json"
    binding = {
        "cell_id": cell_id,
        "proposal_id": proposal["proposal_id"],
        "condition": condition,
        "module_switches": switches,
        "proposal_ref": proposal_ref,
        "proposal_sha256": proposal_sha256,
        "runner_ref": runner_ref,
        "runner_sha256": _bytes_sha256(_json_bytes(runner)),
        "run_id": run_id,
        "expected_stage_receipts": stage_receipts,
    }
    return binding, runner


def build_table3_codegen_preregistration(
    *,
    study_id: str,
    created_at_utc: str,
    artifact_root_ref: str | None = None,
    text_model: str = "gpt-4o-2024-11-20",
    vision_model: str = "gpt-4o-2024-11-20",
) -> dict[str, Any]:
    _utc(created_at_utc, field="created_at_utc")
    resolved_study_id = _identifier(study_id, field="study_id")
    resolved_artifact_root = _relative_ref(
        artifact_root_ref
        or f"mea/protocol_runs/{resolved_study_id}/table3_artifacts",
        field="artifact_root_ref",
    )
    resolved_text_model = _text(text_model, field="text_model")
    resolved_vision_model = _text(vision_model, field="vision_model")
    proposals: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for proposal_index, raw in enumerate(TABLE3_PROPOSALS):
        proposal = deepcopy(dict(raw))
        task_proposal = _table3_task_proposal(proposal)
        proposal_ref = (
            f"{resolved_artifact_root}/proposals/"
            f"{proposal['proposal_id']}.json"
        )
        proposal_sha256 = _bytes_sha256(_json_bytes(task_proposal))
        proposal["task_proposal_ref"] = proposal_ref
        proposal["task_proposal_sha256"] = proposal_sha256
        proposals.append(proposal)
        for condition in TABLE3_CONDITIONS:
            binding, _ = _table3_runner(
                study_id=resolved_study_id,
                proposal=proposal,
                condition=condition,
                proposal_ref=proposal_ref,
                proposal_sha256=proposal_sha256,
                artifact_root_ref=resolved_artifact_root,
                text_model=resolved_text_model,
                vision_model=resolved_vision_model,
                seed=100700 + proposal_index,
            )
            cells.append(binding)
    body = {
        "schema_version": 1,
        "protocol": TABLE3_PROTOCOL,
        "study_id": resolved_study_id,
        "created_at_utc": created_at_utc,
        "artifact_root_ref": resolved_artifact_root,
        "provider_models": {
            "text": resolved_text_model,
            "vision": resolved_vision_model,
        },
        "unseen_proposals": proposals,
        "conditions": list(TABLE3_CONDITIONS),
        "cells": cells,
        "required_downstream_stages": [
            "codegen",
            "compile",
            "render",
            "simulator",
            "oracle",
        ],
        "success_rule": "all_five_downstream_stages_pass",
        "oracle_fixture_minimum": {"positive": 1, "negative": 1},
        "act_rollout_budget": 0,
        "execution_contract": {
            "runner": "scripts/manipeval_taskgen.py",
            "taskgen_ablation_switch_argument": "--taskgen-ablation-json",
            "one_command_per_cell": True,
            "provider_generation_calls": 25,
            "simulator_acceptance_calls": 25,
        },
        "claim_scope": "five_unseen_proposals_per_condition_micro_ablation_not_table3",
    }
    return _seal(body, hash_field="preregistration_sha256")


def validate_table3_codegen_preregistration(
    value: Any,
    *,
    repo_root: str | Path | None = None,
    require_materialized: bool = False,
) -> dict[str, Any]:
    row = _verify_seal(_object(value, field="table3 preregistration"), hash_field="preregistration_sha256")
    if row.get("schema_version") != 1 or row.get("protocol") != TABLE3_PROTOCOL:
        raise LivePaperProtocolError("unsupported Table 3 preregistration")
    rebuilt = build_table3_codegen_preregistration(
        study_id=row.get("study_id"),
        created_at_utc=row.get("created_at_utc"),
        artifact_root_ref=row.get("artifact_root_ref"),
        text_model=_object(row.get("provider_models"), field="provider_models").get(
            "text"
        ),
        vision_model=_object(
            row.get("provider_models"), field="provider_models"
        ).get("vision"),
    )
    if rebuilt != row:
        raise LivePaperProtocolError("Table 3 preregistration contract was modified")
    if require_materialized:
        if repo_root is None:
            raise LivePaperProtocolError("repo_root is required for materialized Table 3")
        root = Path(repo_root).expanduser().resolve()
        for proposal in row["unseen_proposals"]:
            path = _bound_path(
                root, proposal["task_proposal_ref"], field="task_proposal_ref"
            )
            if _file_sha256(path) != proposal["task_proposal_sha256"]:
                raise LivePaperProtocolError("Table 3 proposal hash mismatch")
        for cell in row["cells"]:
            path = _bound_path(root, cell["runner_ref"], field="runner_ref")
            if _file_sha256(path) != cell["runner_sha256"]:
                raise LivePaperProtocolError("Table 3 runner hash mismatch")
    return row


def materialize_table3_codegen_preregistration(
    repo_root: str | Path, preregistration: Any
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    prereg = validate_table3_codegen_preregistration(preregistration)
    by_proposal = {
        proposal["proposal_id"]: proposal
        for proposal in prereg["unseen_proposals"]
    }
    proposal_templates = {
        proposal["proposal_id"]: proposal for proposal in TABLE3_PROPOSALS
    }
    for proposal in prereg["unseen_proposals"]:
        task_proposal = _table3_task_proposal(
            proposal_templates[proposal["proposal_id"]]
        )
        _write_bound_file(
            root, proposal["task_proposal_ref"], _json_bytes(task_proposal)
        )
    for cell in prereg["cells"]:
        proposal = proposal_templates[cell["proposal_id"]]
        proposal_binding = by_proposal[cell["proposal_id"]]
        proposal_index = list(proposal_templates).index(cell["proposal_id"])
        rebuilt, runner = _table3_runner(
            study_id=prereg["study_id"],
            proposal=proposal,
            condition=cell["condition"],
            proposal_ref=proposal_binding["task_proposal_ref"],
            proposal_sha256=proposal_binding["task_proposal_sha256"],
            artifact_root_ref=prereg["artifact_root_ref"],
            text_model=prereg["provider_models"]["text"],
            vision_model=prereg["provider_models"]["vision"],
            seed=100700 + proposal_index,
        )
        if rebuilt != cell:
            raise LivePaperProtocolError("internal Table 3 cell mismatch")
        _write_bound_file(root, cell["runner_ref"], _json_bytes(runner))
    return validate_table3_codegen_preregistration(
        prereg, repo_root=root, require_materialized=True
    )


def evaluate_table3_codegen(
    preregistration: Any,
    result_manifest: Any,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    prereg = validate_table3_codegen_preregistration(
        preregistration, repo_root=root, require_materialized=True
    )
    result = _object(result_manifest, field="table3 result")
    if result.get("schema_version") != 1 or result.get("protocol") != f"{TABLE3_PROTOCOL}_runs":
        raise LivePaperProtocolError("unsupported Table 3 result")
    if result.get("preregistration_sha256") != prereg["preregistration_sha256"]:
        raise LivePaperProtocolError("Table 3 result is not bound to preregistration")
    raw_cells = _items(result.get("cells"), field="cells", minimum=25)
    expected = {cell["cell_id"]: cell for cell in prereg["cells"]}
    if len(raw_cells) != 25 or {cell.get("cell_id") for cell in raw_cells if isinstance(cell, Mapping)} != set(expected):
        raise LivePaperProtocolError("Table 3 requires the exact 5x5 cell grid")
    rows: list[dict[str, Any]] = []
    for raw in raw_cells:
        cell = _object(raw, field="cell")
        cell_id = cell.get("cell_id")
        frozen = expected[cell_id]
        if cell.get("proposal_id") != frozen["proposal_id"] or cell.get("condition") != frozen["condition"]:
            raise LivePaperProtocolError(f"cell identity differs from preregistration: {cell_id}")
        stages = _object(cell.get("stages"), field=f"{cell_id}.stages")
        if set(stages) != set(prereg["required_downstream_stages"]):
            raise LivePaperProtocolError(f"{cell_id} is missing downstream stages")
        codegen = _object(stages["codegen"], field=f"{cell_id}.codegen")
        if codegen.get("generated_by_provider") is not True:
            raise LivePaperProtocolError(f"{cell_id} is proposal-only, not real codegen")
        if (
            codegen.get("scene_generated_by_model") is not True
            or codegen.get("checker_generated_by_model") is not True
        ):
            raise LivePaperProtocolError(
                f"{cell_id} must contain model-generated scene and checker code"
            )
        if codegen.get("module_switches") != frozen["module_switches"]:
            raise LivePaperProtocolError(
                f"{cell_id} codegen switches differ from preregistration"
            )
        codegen_ref = _relative_ref(
            codegen.get("artifact_ref"), field=f"{cell_id}.codegen.artifact_ref"
        )
        expected_codegen_ref = frozen["expected_stage_receipts"]["codegen"][
            "artifact_ref"
        ]
        if codegen_ref != expected_codegen_ref:
            raise LivePaperProtocolError(f"{cell_id} codegen artifact path mismatch")
        codegen_path = _bound_path(
            root, codegen_ref, field=f"{cell_id}.codegen.artifact_ref"
        )
        if _file_sha256(codegen_path) != _sha256(
            codegen.get("artifact_sha256"),
            field=f"{cell_id}.codegen.artifact_sha256",
        ):
            raise LivePaperProtocolError(f"{cell_id} codegen artifact hash mismatch")
        try:
            tree = ast.parse(codegen_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            raise LivePaperProtocolError(
                f"{cell_id} generated task code cannot be parsed: {exc}"
            ) from exc
        defined_functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        missing_functions = {"load_actors", "check_success"} - defined_functions
        if missing_functions:
            raise LivePaperProtocolError(
                f"{cell_id} generated task code is missing "
                f"{sorted(missing_functions)}"
            )
        review = _object(
            cell.get("blind_proxy_review"), field=f"{cell_id}.blind_proxy_review"
        )
        if (
            review.get("annotator_kind") != "development_agent_proxy"
            or review.get("blind_to_condition") is not True
            or not isinstance(review.get("passed"), bool)
            or review.get("human_reviewer_count") != 0
        ):
            raise LivePaperProtocolError(
                f"{cell_id} requires a condition-blind development proxy review"
            )
        stage_pass = [True]
        for stage_name in ("compile", "render", "simulator", "oracle"):
            stage = _object(stages[stage_name], field=f"{cell_id}.{stage_name}")
            if not isinstance(stage.get("passed"), bool):
                raise LivePaperProtocolError(f"{cell_id}.{stage_name}.passed must be boolean")
            receipt_ref = _relative_ref(
                stage.get("receipt_ref"),
                field=f"{cell_id}.{stage_name}.receipt_ref",
            )
            expected_ref = frozen["expected_stage_receipts"][stage_name][
                "receipt_ref"
            ]
            if receipt_ref != expected_ref:
                raise LivePaperProtocolError(
                    f"{cell_id}.{stage_name} receipt path mismatch"
                )
            receipt_path = _bound_path(
                root, receipt_ref, field=f"{cell_id}.{stage_name}.receipt_ref"
            )
            if _file_sha256(receipt_path) != _sha256(
                stage.get("receipt_sha256"),
                field=f"{cell_id}.{stage_name}.receipt_sha256",
            ):
                raise LivePaperProtocolError(
                    f"{cell_id}.{stage_name} receipt hash mismatch"
                )
            stage_pass.append(stage["passed"])
            if stage_name == "oracle":
                if _integer(stage.get("positive_fixture_count"), field="positive_fixture_count", minimum=1) < 1:
                    raise LivePaperProtocolError("oracle requires a positive fixture")
                if _integer(stage.get("negative_fixture_count"), field="negative_fixture_count", minimum=1) < 1:
                    raise LivePaperProtocolError("oracle requires a negative fixture")
        rows.append(
            {
                "cell_id": cell_id,
                "proposal_id": frozen["proposal_id"],
                "condition": frozen["condition"],
                "success": all(stage_pass) and review["passed"],
                "blind_proxy_review_passed": review["passed"],
            }
        )
    rates = {
        condition: sum(row["success"] for row in rows if row["condition"] == condition) / 5.0
        for condition in TABLE3_CONDITIONS
    }
    return {
        "schema_version": 1,
        "protocol": f"{TABLE3_PROTOCOL}_result",
        "study_id": prereg["study_id"],
        "preregistration_sha256": prereg["preregistration_sha256"],
        "cell_count": 25,
        "provider_generation_count": 25,
        "development_proxy_review_count": 25,
        "human_reviewer_count": 0,
        "act_rollouts_started": 0,
        "rows": rows,
        "success_rates": rates,
        "paper_table3_eligible": False,
        "claim_scope": prereg["claim_scope"],
        "limitations": [
            "The review is a condition-blind development-agent proxy, not independent human gold.",
            "A five-proposal micro-ablation is not the paper-scale Table 3 experiment.",
        ],
    }


def validate_proxy_gold_manifest(repo_root: str | Path, value: Any) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    row = _object(value, field="proxy manifest")
    if row.get("schema_version") != 1 or row.get("protocol") != PROXY_PROTOCOL:
        raise LivePaperProtocolError("unsupported proxy manifest")
    if row.get("annotator_kind") != "development_agent_proxy":
        raise LivePaperProtocolError("proxy manifest must not impersonate human gold")
    if row.get("human_reviewer_count") != 0 or row.get("paper_eligible") is not False:
        raise LivePaperProtocolError("development proxy must declare zero humans and paper_eligible=false")
    query_ref = _text(row.get("query_manifest_ref"), field="query_manifest_ref")
    query_path = (root / query_ref).resolve()
    if not query_path.is_relative_to(root) or not query_path.is_file():
        raise LivePaperProtocolError("query manifest ref is missing or outside repository")
    try:
        query_manifest = json.loads(query_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LivePaperProtocolError(f"cannot read query manifest: {exc}") from exc
    cases = _items(query_manifest.get("cases"), field="query cases", minimum=20)
    if not 20 <= len(cases) <= 30:
        raise LivePaperProtocolError("Plan proxy suite must contain 20-30 queries")
    for case in cases:
        annotation = _object(case.get("annotation"), field="query annotation")
        if (
            annotation.get("source") != "development_agent_proxy"
            or annotation.get("paper_eligible") is not False
            or annotation.get("human_votes") != []
        ):
            raise LivePaperProtocolError("query proxy labels must remain explicitly non-human")
    clips = _items(row.get("clip_slots"), field="clip_slots", minimum=8)
    expected = {(condition, polarity) for condition in PAPER_VQA_CONDITIONS for polarity in ("positive", "negative")}
    observed: set[tuple[str, str]] = set()
    materialized = 0
    source_label_audited = 0
    for index, clip in enumerate(clips):
        clip = _object(clip, field=f"clip_slots[{index}]")
        pair = (clip.get("condition"), clip.get("polarity"))
        if pair not in expected or pair in observed:
            raise LivePaperProtocolError("clip slots must be the exact four-condition polarity grid")
        observed.add(pair)
        if not isinstance(clip.get("proxy_gold_observed"), bool):
            raise LivePaperProtocolError("clip proxy label must be boolean")
        if clip.get("label_source") != "development_agent_proxy":
            raise LivePaperProtocolError("clip label source must remain development proxy")
        is_materialized = clip.get("materialized")
        if not isinstance(is_materialized, bool):
            raise LivePaperProtocolError("clip materialized must be boolean")
        source_or_recipe_ref = _text(
            clip.get("source_or_recipe_ref"), field="source_or_recipe_ref"
        )
        if is_materialized:
            artifact = (root / source_or_recipe_ref).resolve()
            if not artifact.is_relative_to(root) or not artifact.is_file():
                raise LivePaperProtocolError(
                    "materialized clip ref is missing or outside repository"
                )
            source_label_ref = _relative_ref(
                clip.get("source_label_ref"), field="source_label_ref"
            )
            source_label_path = _bound_path(
                root, source_label_ref, field="source_label_ref"
            )
            try:
                source_label = json.loads(
                    source_label_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise LivePaperProtocolError(
                    f"cannot read source label receipt: {exc}"
                ) from exc
            phenomenon_id = _text(
                clip.get("source_phenomenon_id"),
                field="source_phenomenon_id",
            )
            phenomena = (
                (source_label.get("vqa") or {}).get("phenomena")
                if isinstance(source_label, Mapping)
                else None
            )
            matches = [
                item
                for item in phenomena or []
                if isinstance(item, Mapping)
                and item.get("id") == phenomenon_id
                and isinstance(item.get("observed"), bool)
            ]
            if len(matches) != 1:
                raise LivePaperProtocolError(
                    "source label receipt has no unique boolean phenomenon"
                )
            observed_label = matches[0]["observed"]
            expected_label = clip["proxy_gold_observed"]
            expected_polarity = "positive" if observed_label else "negative"
            if observed_label is not expected_label or clip["polarity"] != expected_polarity:
                raise LivePaperProtocolError(
                    "materialized clip label/polarity conflicts with its source receipt"
                )
            source_label_audited += 1
        elif (
            clip.get("source_label_ref") is not None
            or clip.get("source_phenomenon_id") is not None
        ):
            raise LivePaperProtocolError(
                "unmaterialized clip slots cannot claim a source label receipt"
            )
        materialized += int(is_materialized)
    if observed != expected or len(clips) != 8:
        raise LivePaperProtocolError("clip slots must contain exactly 8 entries")
    return {
        "schema_version": 1,
        "protocol": f"{PROXY_PROTOCOL}_validation",
        "query_count": len(cases),
        "clip_slot_count": len(clips),
        "materialized_clip_count": materialized,
        "source_label_audited_count": source_label_audited,
        "conditions": list(PAPER_VQA_CONDITIONS),
        "annotation_scope": "development_agent_proxy_not_human_gold",
        "human_reviewer_count": 0,
        "paper_plan_validity_eligible": False,
        "paper_vqa_robustness_eligible": False,
        "ready_for_proxy_smoke": materialized == len(clips),
    }
