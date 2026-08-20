"""Policy invocation for one prepared native RoboTwin round."""

from __future__ import annotations

from typing import Any, Mapping

from mea.method_runtime import RolloutRequest

from .native_round_contracts import (
    NativeAgentRoundError,
    NativeRoundPreparation,
)


def invoke_robotwin_policy(
    prepared: NativeRoundPreparation,
    *,
    round_plan: Mapping[str, Any],
    evaluation_id: str,
    policy_backend: str,
) -> tuple[Any, ...]:
    rollouts: list[Any] = []
    for trial_index, seed in enumerate(prepared.seeds):
        rollout = prepared.runtime.rollout(
            prepared.candidate,
            RolloutRequest(
                round_id=str(round_plan["round_id"]),
                seed=seed,
                output_dir=(
                    prepared.rollout_dir
                    / f"episode_{trial_index:03d}_seed_{seed}"
                ),
                provenance={
                    "evaluation_id": evaluation_id,
                    "policy_backend": policy_backend,
                    "trial_index": trial_index,
                    "trial_count": len(prepared.seeds),
                },
            ),
        )
        if rollout.seed != seed:
            raise NativeAgentRoundError(
                "native policy rollout changed the requested seed"
            )
        if (
            policy_backend == "act"
            and rollout.metadata.get("actual_seeds") != [seed]
        ):
            raise NativeAgentRoundError(
                "native ACT rollout did not report the exact requested seed"
            )
        rollouts.append(rollout)
    return tuple(rollouts)
