"""Lazy boundary for preregistered paper-protocol execution.

Normal ClaimFirst evaluation does not need the strategy-plan or evidence-
manifest stack. Keep that stack importable for frozen fixed/adaptive paper
protocols without loading it in the production method path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


class RegisteredExecutionAdapterError(RuntimeError):
    """A paper-protocol registration failed before execution."""


def load_registered_execution_for_cli(
    repo_root: str | Path,
    *,
    evidence_manifest_path: str,
    command_plan_path: str,
    registered_route_path: str,
    strategy: str,
    evaluation_id: str,
    observed_argv: Sequence[str],
) -> dict[str, Any]:
    """Validate one explicitly requested preregistered paper execution."""

    from mea.strategy_plan import StrategyPlanError, load_registered_execution

    try:
        return load_registered_execution(
            repo_root,
            evidence_manifest_path=evidence_manifest_path,
            command_plan_path=command_plan_path,
            registered_route_path=registered_route_path,
            strategy=strategy,
            evaluation_id=evaluation_id,
            observed_argv=observed_argv,
        )
    except StrategyPlanError as exc:
        raise RegisteredExecutionAdapterError(str(exc)) from exc


__all__ = [
    "RegisteredExecutionAdapterError",
    "load_registered_execution_for_cli",
]
