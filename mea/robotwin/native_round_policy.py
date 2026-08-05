"""Policy invocation for one prepared native RoboTwin round."""

from __future__ import annotations

from typing import Any, Mapping

from mea.method_runtime import RolloutRequest

from .native_round_contracts import NativeRoundPreparation


def invoke_robotwin_policy(
    prepared: NativeRoundPreparation,
    *,
    round_plan: Mapping[str, Any],
    evaluation_id: str,
    policy_backend: str,
) -> Any:
    return prepared.runtime.rollout(
        prepared.candidate,
        RolloutRequest(
            round_id=str(round_plan["round_id"]),
            seed=prepared.seed,
            output_dir=prepared.rollout_dir,
            provenance={
                "evaluation_id": evaluation_id,
                "policy_backend": policy_backend,
            },
        ),
    )
