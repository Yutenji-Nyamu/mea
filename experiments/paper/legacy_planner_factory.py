"""Legacy and paper-ablation planner construction.

The production ClaimFirst entry point must not import task-specific planners.
This module keeps the historical catalog, fixed-suite, and position planners
available to explicit paper protocols while caller migration is completed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mea.planner import (
    CatalogPlanAgent,
    ClickBellFixedSuitePlanAgent,
    ClickBellPositionPlanAgent,
)


def build_legacy_planner(
    repo_root: str | Path,
    *,
    task_name: str,
    task_profile: str,
    provider: Any,
    model: str,
    task_module: str | None,
    start_seed: int | None,
    num_episodes: int,
    telemetry_profile: str,
    max_rounds: int,
    execution_backend: str,
) -> Any:
    """Construct one historical planner for an explicit compatibility run."""

    root = Path(repo_root).expanduser().resolve()
    if task_profile == "fixed_suite":
        if provider is None:
            raise ValueError("fixed_suite requires a provider")
        return ClickBellFixedSuitePlanAgent(
            root,
            provider,
            model=model,
            start_seed=start_seed if start_seed is not None else 100401,
            num_episodes=num_episodes,
            telemetry_profile=telemetry_profile,
            max_rounds=max_rounds,
        )
    if task_profile == "position_lr":
        return ClickBellPositionPlanAgent(
            root,
            start_seed=start_seed if start_seed is not None else 100401,
            num_episodes=num_episodes,
            telemetry_profile=telemetry_profile,
            max_rounds=max_rounds,
        )
    return CatalogPlanAgent(
        root,
        task_name=task_name,
        provider=provider,
        model=model,
        task_module=task_module,
        start_seed=start_seed,
        num_episodes=num_episodes,
        telemetry_profile=telemetry_profile,
        max_rounds=max_rounds,
        execution_backend=execution_backend,
        task_profile=task_profile,
    )


__all__ = ["build_legacy_planner"]
