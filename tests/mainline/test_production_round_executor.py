from __future__ import annotations

from functools import partial

from mea.robotwin import production_round_executor as production
from mea.robotwin.native_agent_round import (
    execute_act_method_round,
    execute_hyvla_method_round,
    execute_smolvla_method_round,
)
from mea.round_executor import RoundExecutor
from mea.taskgen.runtime import create_generic_provider_taskgen_run


def test_production_executor_registers_only_native_method_backends() -> None:
    registry = production.production_native_policy_rounds()

    assert set(registry) == production.PRODUCTION_POLICY_BACKENDS
    assert {
        name: runner.func
        for name, runner in registry.items()
        if isinstance(runner, partial)
    } == {
        "act": execute_act_method_round,
        "smolvla": execute_smolvla_method_round,
        "hyvla": execute_hyvla_method_round,
    }
    assert all(
        runner.keywords.get("generated_task_materializer")
        is create_generic_provider_taskgen_run
        for runner in registry.values()
    )

    executor = production.build_production_round_executor()
    assert isinstance(executor, RoundExecutor)
    assert set(executor._services.native_policy_rounds) == {
        "act",
        "smolvla",
        "hyvla",
    }
