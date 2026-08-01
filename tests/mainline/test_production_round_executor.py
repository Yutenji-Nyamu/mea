from __future__ import annotations

import ast
from functools import partial
from pathlib import Path

import pytest

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
    assert (
        executor._services.build_taskgen_command
        is production._compat_subprocess_unavailable
    )
    assert (
        executor._services.run_logged
        is production._compat_subprocess_unavailable
    )
    with pytest.raises(RuntimeError, match="legacy TaskGen subprocess"):
        executor._services.run_logged([])

    source_path = Path(production.__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    imported_modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "subprocess" not in imported_modules
    assert "mea.taskgen.round_materialization" not in imported_modules
