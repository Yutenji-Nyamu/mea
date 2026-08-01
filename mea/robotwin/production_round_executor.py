"""Production RoboTwin RoundExecutor assembly.

All production policy backends use the native RoboTwin ``MethodRuntime``
adapter.  Legacy ACT child-process TaskGen remains a compatibility concern and
is deliberately unavailable from this factory.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

from mea.robotwin.native_agent_round import (
    execute_act_method_round,
    execute_hyvla_method_round,
    execute_smolvla_method_round,
)
from mea.round_executor import RoundExecutionServices, RoundExecutor
from mea.taskgen.runtime import create_generic_provider_taskgen_run


PRODUCTION_POLICY_BACKENDS = frozenset({"act", "smolvla", "hyvla"})


def production_native_policy_rounds() -> dict[str, Callable[..., Any]]:
    """Return the complete production backend registry by value."""

    materializer = create_generic_provider_taskgen_run
    return {
        "act": partial(
            execute_act_method_round,
            generated_task_materializer=materializer,
        ),
        "smolvla": partial(
            execute_smolvla_method_round,
            generated_task_materializer=materializer,
        ),
        "hyvla": partial(
            execute_hyvla_method_round,
            generated_task_materializer=materializer,
        ),
    }


def _compat_subprocess_unavailable(*_args: Any, **_kwargs: Any) -> Any:
    """Reject accidental entry into the legacy child-process transport."""

    raise RuntimeError(
        "production RoboTwin rounds require a native MethodRuntime backend; "
        "the legacy TaskGen subprocess is available only through paper compat"
    )


def build_production_round_executor() -> RoundExecutor:
    """Build the native-only executor used by the production Plan Agent."""

    # Import lazily so this assembly module can later be called by
    # PlanAgentApplication without creating an import cycle during module load.
    from mea.plan_agent_application import update_manifest

    return RoundExecutor(
        RoundExecutionServices(
            update_manifest=update_manifest,
            build_taskgen_command=_compat_subprocess_unavailable,
            run_logged=_compat_subprocess_unavailable,
            native_policy_rounds=production_native_policy_rounds(),
        )
    )


__all__ = [
    "PRODUCTION_POLICY_BACKENDS",
    "build_production_round_executor",
    "production_native_policy_rounds",
]
