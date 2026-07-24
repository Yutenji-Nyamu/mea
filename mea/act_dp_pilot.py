"""Read-only readiness contract for the low-cost ACT-versus-DP pilot.

The paper compares several policies, but the current reproduction deliberately
freezes the smallest useful pair.  This module never downloads, trains, or runs
a policy.  It reports whether the exact ACT+DP N=3 experiment can be launched
without substituting DP3 or silently training a missing DP checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


PROTOCOL = "act_dp_exact_seed_n3_readiness_v1"
DEFAULT_SEEDS = (100600, 100601, 100602)
ACT_CHECKPOINT_REF = (
    "policy/ACT/act_ckpt/act-beat_block_hammer/"
    "demo_clean-50/policy_last.ckpt"
)
ACT_STATS_REF = (
    "policy/ACT/act_ckpt/act-beat_block_hammer/"
    "demo_clean-50/dataset_stats.pkl"
)
DP_CHECKPOINT_REF = (
    "policy/DP/checkpoints/"
    "beat_block_hammer-demo_clean-50-0/600.ckpt"
)
DP_CONFIG_REF = "policy/DP/deploy_policy.yml"
DP_ENTRYPOINT_REF = "script/eval_policy.py"
DEFAULT_ACT_PYTHON = "/root/autodl-tmp/conda/envs/RoboTwin/bin/python"
DEFAULT_DP_PYTHON = "/root/autodl-tmp/conda/envs/RoboTwin-DP/bin/python"


class ActDpPilotError(ValueError):
    """Raised when the frozen readiness request is malformed."""


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact(root: Path, ref: str) -> dict[str, Any]:
    path = root / ref
    present = path.is_file()
    return {
        "artifact_ref": ref,
        "present": present,
        "size_bytes": path.stat().st_size if present else None,
        "sha256": _file_sha256(path) if present else None,
    }


def _environment(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    return {
        "python_path": path.as_posix(),
        "present": path.is_file(),
    }


def _normalize_seeds(values: Iterable[int]) -> list[int]:
    seeds = list(values)
    if (
        len(seeds) != 3
        or len(set(seeds)) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds)
    ):
        raise ActDpPilotError("the pilot requires exactly three distinct integer seeds")
    return seeds


def _command_templates(
    *,
    seeds: list[int],
    act_python: str,
    dp_python: str,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"act": [], "dp": []}
    for seed in seeds:
        seed_manifest = (
            "mea/protocol_runs/act_dp_exact_seed_n3/"
            f"commands/seed_{seed}/seed_manifest.json"
        )
        for policy_id in ("act", "dp"):
            live_root = (
                "mea/protocol_runs/act_dp_exact_seed_n3/"
                f"live_runs/{policy_id}/seed_{seed}"
            )
            if policy_id == "act":
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
                    seed_manifest,
                    f"{live_root}/seed_results.json",
                    f"{live_root}/eval_output",
                ]
                python_environment = act_python
            else:
                argv = [
                    dp_python,
                    DP_ENTRYPOINT_REF,
                    "--config",
                    DP_CONFIG_REF,
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
                    "DP",
                    "--checkpoint_num",
                    "600",
                    "--num_episodes",
                    "1",
                    "--seed_manifest",
                    seed_manifest,
                    "--telemetry_dir",
                    f"{live_root}/telemetry",
                    "--telemetry_profile",
                    "balanced_v1",
                    "--seed_results_path",
                    f"{live_root}/seed_results.json",
                    "--output_dir",
                    f"{live_root}/eval_output",
                ]
                python_environment = dp_python
            result[policy_id].append(
                {
                    "policy_id": policy_id,
                    "seed": seed,
                    "python_environment": python_environment,
                    "argv": argv,
                }
            )
    return result


def build_act_dp_readiness(
    repo_root: str | Path,
    *,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    act_python: str = DEFAULT_ACT_PYTHON,
    dp_python: str = DEFAULT_DP_PYTHON,
) -> dict[str, Any]:
    """Inspect exact prerequisites and return a sealed no-execution report."""

    root = Path(repo_root).expanduser().resolve()
    normalized_seeds = _normalize_seeds(seeds)
    artifacts = {
        "act_checkpoint": _artifact(root, ACT_CHECKPOINT_REF),
        "act_dataset_stats": _artifact(root, ACT_STATS_REF),
        "dp_checkpoint": _artifact(root, DP_CHECKPOINT_REF),
        "dp_config": _artifact(root, DP_CONFIG_REF),
        "shared_eval_entrypoint": _artifact(root, DP_ENTRYPOINT_REF),
    }
    environments = {
        "act": _environment(act_python),
        "dp": _environment(dp_python),
    }
    required = {
        "act_checkpoint": artifacts["act_checkpoint"]["present"],
        "act_dataset_stats": artifacts["act_dataset_stats"]["present"],
        "act_environment": environments["act"]["present"],
        "dp_checkpoint": artifacts["dp_checkpoint"]["present"],
        "dp_config": artifacts["dp_config"]["present"],
        "dp_environment": environments["dp"]["present"],
        "shared_eval_entrypoint": artifacts["shared_eval_entrypoint"]["present"],
    }
    missing = sorted(key for key, present in required.items() if not present)
    ready = not missing
    body = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "task_name": "beat_block_hammer",
        "task_config": "demo_clean",
        "policy_ids": ["act", "dp"],
        "seeds": normalized_seeds,
        "rollouts_per_policy": 3,
        "maximum_policy_rollouts": 6,
        "same_seed_pairing_required": True,
        "substitution_forbidden": ["dp3", "rdt", "pi0"],
        "artifacts": artifacts,
        "environments": environments,
        "required_checks": required,
        "missing_requirements": missing,
        "status": "ready" if ready else "blocked_missing_prerequisites",
        "live_execution_authorized": ready,
        "command_templates": (
            _command_templates(
                seeds=normalized_seeds,
                act_python=act_python,
                dp_python=dp_python,
            )
            if ready
            else None
        ),
        "official_reference": {
            "benchmark": "RoboTwin 2.0 beat_block_hammer Easy",
            "act_success_percent": 56.0,
            "dp_success_percent": 42.0,
            "frozen_expected_pair_order": "act_above_dp",
            "source": "https://robotwin-platform.github.io/leaderboard",
        },
        "checkpoint_acquisition_contract": {
            "dp_checkpoint_must_be_user_supplied_or_locally_trained": True,
            "do_not_treat_dp3_as_dp": True,
            "do_not_start_600_epoch_training_as_a_readiness_check": True,
        },
        "calls_started": {
            "provider": 0,
            "simulator": 0,
            "act": 0,
            "dp": 0,
            "training": 0,
            "download": 0,
        },
        "claim_scope": (
            "readiness_and_exact_command_contract_only_not_policy_performance"
        ),
        "paper_table9_eligible": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


__all__ = [
    "ACT_CHECKPOINT_REF",
    "ACT_STATS_REF",
    "ActDpPilotError",
    "DEFAULT_ACT_PYTHON",
    "DEFAULT_DP_PYTHON",
    "DEFAULT_SEEDS",
    "DP_CHECKPOINT_REF",
    "PROTOCOL",
    "build_act_dp_readiness",
]
