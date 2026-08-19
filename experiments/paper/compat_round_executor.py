"""Legacy ACT subprocess transport for frozen paper/compat protocols.

Production RoboTwin rounds use ``mea.round_executor.RoundExecutor`` with a
native ``MethodRuntime`` backend.  This module keeps the old child-process
TaskGen transport available only to historical experiment wrappers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from mea.round_executor import (
    RoundExecutionRequest,
    RoundExecutionServices,
    RoundExecutor,
    _PolicyRoundArtifacts,
    _write_json,
)


@dataclass(frozen=True)
class LegacySubprocessServices:
    update_manifest: Callable[..., Mapping[str, Any]]
    build_taskgen_command: Callable[..., tuple[list[str], str]]
    run_logged: Callable[..., int]
    native_policy_rounds: Mapping[str, Callable[..., Mapping[str, Any]]]
    base_url: str | None = None


class LegacySubprocessRoundExecutor(RoundExecutor):
    """Run historical ACT TaskGen subprocess rounds when no native hook exists."""

    def __init__(
        self,
        services: LegacySubprocessServices,
    ) -> None:
        super().__init__(
            RoundExecutionServices(
                update_manifest=services.update_manifest,
                native_policy_rounds=services.native_policy_rounds,
            )
        )
        self._legacy_services = services

    def _execute_policy(
        self, request: RoundExecutionRequest
    ) -> _PolicyRoundArtifacts:
        if request.policy_backend in self._services.native_policy_rounds:
            return super()._execute_policy(request)
        if request.policy_backend != "act":
            raise RuntimeError(
                f"unsupported legacy policy backend: {request.policy_backend!r}"
            )
        command, run_id = self._legacy_services.build_taskgen_command(
            request.repo_root,
            request.evaluation_id,
            request.round_plan,
            text_model=request.text_model,
            vision_model=request.vision_model,
            base_url=self._legacy_services.base_url,
            gpu=request.gpu,
            max_reflections=request.max_reflections,
            telemetry_profile=request.telemetry_profile,
            run_id_suffix="",
        )
        round_id = request.round_plan["round_id"]
        execution_dir = request.evaluation_dir / "execution" / round_id
        _write_json(
            execution_dir / "taskgen_command.json",
            {"command": command, "child_run_id": run_id},
        )
        self._legacy_services.update_manifest(
            request.evaluation_dir,
            status=f"executing_{round_id}",
            active_child_run_id=run_id,
        )
        returncode = self._legacy_services.run_logged(
            command,
            cwd=request.repo_root,
            log_path=execution_dir / "taskgen.log",
        )
        child_dir = request.repo_root / "mea/generated_tasks" / run_id
        return _PolicyRoundArtifacts(
            child_manifest={"run_id": run_id},
            child_dir=child_dir,
            execution_dir=execution_dir,
            child_manifest_path=child_dir / "manifest.json",
            run_id=run_id,
            returncode=returncode,
            semantic_ready=True,
            native=None,
        )

__all__ = [
    "LegacySubprocessRoundExecutor",
    "LegacySubprocessServices",
]
