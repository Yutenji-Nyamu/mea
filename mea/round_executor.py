"""Production round orchestration independent from the Agent CLI.

``RoundExecutor`` owns one complete execution round after the Plan Agent has
selected a Proposal: launch the policy backend, route Rule/VQA tools, aggregate
evidence, project the shared MethodRuntime view, and persist the round summary.

The CLI supplies a small service bundle for repository-specific materializers
and validators.  This keeps argument parsing, route/session planning, and paper
compatibility flags out of the execution boundary while allowing another
simulator backend to reuse the same outer round lifecycle.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from mea.planner import build_evidence_aggregate
from mea.proposals import tool_request_from_proposal
from mea.toolgen import route_tool_request


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _round_requests_execution_vqa(
    round_plan: Mapping[str, Any] | None,
) -> bool:
    """Keep legacy VQA behavior unless the Proposal explicitly omits it."""

    semantic_needs = (round_plan or {}).get("semantic_need_execution")
    if not isinstance(semantic_needs, Mapping):
        return True
    vqa_need = semantic_needs.get("vqa_tool")
    return not (
        isinstance(vqa_need, Mapping)
        and vqa_need.get("requested") is False
    )


@dataclass(frozen=True)
class RoundExecutionServices:
    """Migration seams not yet owned by the backend-neutral lifecycle.

    Generic artifact writing and existing Tool/MethodRuntime APIs are imported
    directly in this module.  The remaining callbacks are the child TaskGen
    transport and summary/VQA helpers that still live in the legacy CLI; they
    can move independently without changing the typed round contract.
    """

    update_manifest: Callable[..., Mapping[str, Any]]
    build_taskgen_command: Callable[..., tuple[list[str], str]]
    run_logged: Callable[..., int]
    materialize_open_world_tool_request: Callable[..., Mapping[str, Any]]
    reuse_bound_child_checker_tool: Callable[..., Mapping[str, Any] | None]
    execute_tool_request: Callable[..., Mapping[str, Any]]
    aggregate_round_results: Callable[..., Mapping[str, Any]]
    run_round_execution_vqa: Callable[..., Mapping[str, Any]]
    summarize_round: Callable[..., dict[str, Any]]
    native_policy_rounds: Mapping[str, Callable[..., Mapping[str, Any]]]


@dataclass(frozen=True)
class RoundExecutionRequest:
    """All explicit inputs needed to execute one planned round."""

    repo_root: Path
    evaluation_dir: Path
    evaluation_id: str
    round_plan: dict[str, Any]
    text_model: str
    vision_model: str
    base_url: str | None
    gpu: int
    max_reflections: int
    provider: Any
    toolgen_model: str
    telemetry_profile: str = "balanced_v1"
    reviewed_task_registry: Path | None = None
    reviewed_tool_registry: Path | None = None
    reviewed_vqa_registry: Path | None = None
    registration_identity: dict[str, Any] | None = None
    policy_backend: str = "act"
    runtime_target: Mapping[str, Any] | None = None
    smolvla_port: int = 18771


@dataclass(frozen=True)
class RoundExecutionResult:
    """Typed result returned to the route/session loop."""

    child_manifest: dict[str, Any]
    child_dir: Path
    round_summary: dict[str, Any]
    tool_evaluation: dict[str, Any]
    returncode: int

    def as_legacy_tuple(
        self,
    ) -> tuple[dict[str, Any], Path, dict[str, Any], dict[str, Any], int]:
        """Preserve the historical import contract during caller migration."""

        return (
            self.child_manifest,
            self.child_dir,
            self.round_summary,
            self.tool_evaluation,
            self.returncode,
        )


class RoundExecutor:
    """Execute one Proposal through policy, evidence tools, and aggregation."""

    def __init__(self, services: RoundExecutionServices) -> None:
        self._services = services

    def execute(self, request: RoundExecutionRequest) -> RoundExecutionResult:
        services = self._services
        repo_root = request.repo_root
        evaluation_dir = request.evaluation_dir
        evaluation_id = request.evaluation_id
        round_plan = request.round_plan
        round_id = round_plan["round_id"]

        native_policy_round = services.native_policy_rounds.get(
            request.policy_backend
        )
        if native_policy_round is not None:
            if request.runtime_target is None:
                raise RuntimeError(
                    "native policy execution requires the bound runtime target"
                )
            services.update_manifest(
                evaluation_dir,
                status=f"executing_{round_id}",
                active_child_run_id=None,
                policy_backend=request.policy_backend,
            )
            native = native_policy_round(
                repo_root=repo_root,
                evaluation_dir=evaluation_dir,
                evaluation_id=evaluation_id,
                round_plan=round_plan,
                runtime_target=request.runtime_target,
                telemetry_profile=request.telemetry_profile,
                policy_server_port=request.smolvla_port,
                gpu=request.gpu,
                provider=request.provider,
                text_model=request.text_model,
                vision_model=request.vision_model,
                max_reflections=request.max_reflections,
            )
            child_manifest = native["child_manifest"]
            child_dir = native["child_dir"]
            execution_dir = evaluation_dir / "execution" / round_id
            child_manifest_path = native["manifest_path"]
            run_id = child_manifest["run_id"]
            returncode = 0
            semantic_ready = native["semantic_telemetry_ready"]
        elif request.policy_backend != "act":
            raise RuntimeError(
                f"unsupported policy backend: {request.policy_backend!r}"
            )
        else:
            semantic_ready = True
            native = None
            command, run_id = services.build_taskgen_command(
                repo_root,
                evaluation_id,
                round_plan,
                text_model=request.text_model,
                vision_model=request.vision_model,
                base_url=request.base_url,
                gpu=request.gpu,
                max_reflections=request.max_reflections,
                telemetry_profile=request.telemetry_profile,
                reviewed_task_registry=request.reviewed_task_registry,
                registration_identity=request.registration_identity,
                run_id_suffix="",
            )
            execution_dir = evaluation_dir / "execution" / round_id
            _write_json(
                execution_dir / "taskgen_command.json",
                {"command": command, "child_run_id": run_id},
            )
            services.update_manifest(
                evaluation_dir,
                status=f"executing_{round_id}",
                active_child_run_id=run_id,
            )
            returncode = services.run_logged(
                command,
                cwd=repo_root,
                log_path=execution_dir / "taskgen.log",
            )
            child_dir = repo_root / "mea/generated_tasks" / run_id
            child_manifest_path = child_dir / "manifest.json"

        if not child_manifest_path.is_file():
            raise RuntimeError(
                f"child TaskGen manifest 不存在: {child_manifest_path}"
            )
        child_manifest = json.loads(
            child_manifest_path.read_text(encoding="utf-8")
        )
        if (
            native is None
            and request.registration_identity is not None
            and child_manifest.get("registration_identity")
            != request.registration_identity
        ):
            raise RuntimeError(
                f"child registration identity mismatch: {run_id}"
            )
        _write_json(
            execution_dir / "child_run.json",
            {
                "run_id": run_id,
                "returncode": returncode,
                "manifest_path": str(
                    child_manifest_path.relative_to(repo_root)
                ),
                "status": child_manifest.get("status"),
                "policy_backend": request.policy_backend,
            },
        )

        tool_evaluation = self._execute_rule_tool(
            request=request,
            child_manifest=child_manifest,
            child_dir=child_dir,
            execution_dir=execution_dir,
            returncode=returncode,
            semantic_ready=semantic_ready,
            native_execution=native is not None,
        )
        self._project_tool_state(
            round_plan=round_plan,
            tool_evaluation=tool_evaluation,
        )
        aggregate_result = services.aggregate_round_results(
            round_plan,
            child_manifest,
            tool_evaluation,
            execution_dir / "aggregate_result.json",
        )
        execution_vqa = self._execute_vqa(
            request=request,
            child_manifest=child_manifest,
            child_dir=child_dir,
            execution_dir=execution_dir,
            tool_evaluation=tool_evaluation,
            semantic_ready=semantic_ready,
        )
        self._project_vqa_state(
            round_plan=round_plan,
            execution_vqa=execution_vqa,
        )
        round_summary = services.summarize_round(
            round_plan,
            child_manifest,
            child_dir,
            tool_evaluation,
            aggregate_result,
            execution_vqa,
            returncode,
        )
        round_summary["round_attempt_index"] = 1
        round_summary["execution_artifact_dir"] = str(
            execution_dir.relative_to(repo_root)
        ).replace("\\", "/")
        self._attach_native_method_runtime(
            request=request,
            native=native,
            semantic_ready=semantic_ready,
            round_summary=round_summary,
        )
        _write_json(
            execution_dir / "evidence_aggregate.json",
            round_summary["observations"]["evidence_aggregate"],
        )
        _write_json(
            evaluation_dir / "summary" / f"{round_id}.json",
            round_summary,
        )
        return RoundExecutionResult(
            child_manifest=dict(child_manifest),
            child_dir=child_dir,
            round_summary=round_summary,
            tool_evaluation=dict(tool_evaluation),
            returncode=returncode,
        )

    def _execute_rule_tool(
        self,
        *,
        request: RoundExecutionRequest,
        child_manifest: dict[str, Any],
        child_dir: Path,
        execution_dir: Path,
        returncode: int,
        semantic_ready: bool,
        native_execution: bool,
    ) -> dict[str, Any]:
        services = self._services
        round_plan = request.round_plan
        if (
            child_manifest.get("status")
            in {
                "completed",
                "completed_without_act",
            }
            and returncode == 0
            and semantic_ready
        ):
            tool_kwargs: dict[str, Any] = {
                "provider": request.provider,
                "model": request.toolgen_model,
            }
            if request.reviewed_tool_registry is not None:
                tool_kwargs["reviewed_registry_dir"] = (
                    request.reviewed_tool_registry
                )
            if round_plan.get("open_tool_request_deferred") is True:
                tool_bundle = services.materialize_open_world_tool_request(
                    request.repo_root,
                    execution_dir,
                    round_plan=round_plan,
                    child_dir=child_dir,
                    provider=request.provider,
                    toolgen_model=request.toolgen_model,
                    reviewed_tool_registry=request.reviewed_tool_registry,
                )
                if tool_bundle.get("status") == "unsupported":
                    round_plan["open_tool_request_deferred"] = False
                    semantic_execution = round_plan.get(
                        "semantic_need_execution"
                    )
                    if isinstance(semantic_execution, dict):
                        tool_execution = semantic_execution.get("rule_tool")
                        if isinstance(tool_execution, dict):
                            tool_execution.update(
                                {
                                    "route": "unsupported",
                                    "status": "unsupported",
                                    "request_artifact": str(
                                        (
                                            execution_dir
                                            / "open_tool_request/"
                                            "tool_request_bundle.json"
                                        ).relative_to(request.repo_root)
                                    ).replace("\\", "/"),
                                }
                            )
                    return {
                        "schema_version": 1,
                        "status": "unsupported",
                        "requested_route": "auto",
                        "route": None,
                        "reference_tool": None,
                        "tool_request": None,
                        "route_decision": {
                            "status": "unsupported",
                            "requested_route": "auto",
                            "resolved_route": None,
                            "reason": tool_bundle.get("reason"),
                            "reason_code": tool_bundle.get("reason_code"),
                            "provider_required": True,
                            "provider_called": bool(
                                (tool_bundle.get("provider") or {}).get(
                                    "called"
                                )
                            ),
                        },
                        "source": {
                            "artifact": str(
                                (
                                    execution_dir
                                    / "open_tool_request/"
                                    "tool_request_bundle.json"
                                ).relative_to(request.repo_root)
                            ).replace("\\", "/")
                        },
                        "episodes": [],
                        "validation": {
                            "passed": False,
                            "reason": tool_bundle.get("reason"),
                            "reason_code": tool_bundle.get("reason_code"),
                        },
                        "artifacts": {},
                    }
                round_plan["tool_request"] = deepcopy(
                    tool_bundle["tool_request"]
                )
                round_plan["open_tool_request_deferred"] = False
                semantic_execution = round_plan.get(
                    "semantic_need_execution"
                )
                if isinstance(semantic_execution, dict):
                    tool_execution = semantic_execution.get("rule_tool")
                    if isinstance(tool_execution, dict):
                        tool_execution.update(
                            {
                                "route": route_tool_request(
                                    tool_bundle["tool_request"]
                                )["route_decision"]["resolved_route"],
                                "status": "selected",
                                "request_artifact": str(
                                    (
                                        execution_dir
                                        / "open_tool_request/"
                                        "tool_request_bundle.json"
                                    ).relative_to(request.repo_root)
                                ).replace("\\", "/"),
                            }
                        )
            proposed_request = (
                tool_request_from_proposal(
                    round_plan["tool_proposal"]
                )
                if round_plan.get("tool_proposal") is not None
                else round_plan["tool_request"]
            )
            if round_plan.get("task_proposal") is not None:
                tool_kwargs["task_proposal"] = round_plan["task_proposal"]
            planned_tool_dir = execution_dir / "planned_tool"
            tool_evaluation = services.reuse_bound_child_checker_tool(
                request.repo_root,
                child_manifest,
                planned_tool_dir,
                proposed_request,
            )
            if tool_evaluation is None:
                tool_evaluation = services.execute_tool_request(
                    request.repo_root,
                    child_dir,
                    planned_tool_dir,
                    proposed_request,
                    **tool_kwargs,
                )
        else:
            skip_reason = (
                str(
                    (
                        child_manifest.get("unsupported_capability") or {}
                    ).get("reason")
                )
                if child_manifest.get("status") == "unsupported"
                else (
                    "TaskSchema unavailable; Rule Tool and VQA evidence were "
                    "not executed."
                )
                if not semantic_ready
                else f"child TaskGen exited with code {returncode}"
                if returncode != 0
                else "child TaskGen pipeline did not complete"
            )
            tool_evaluation = {
                "schema_version": 1,
                "status": "skipped",
                "requested_route": "auto",
                "route": None,
                "reference_tool": None,
                "tool_request": (
                    tool_request_from_proposal(
                        round_plan["tool_proposal"]
                    )
                    if round_plan.get("tool_proposal") is not None
                    else round_plan["tool_request"]
                ),
                "route_decision": {
                    "status": "skipped",
                    "requested_route": "auto",
                    "resolved_route": None,
                    "reason": skip_reason,
                    "provider_required": None,
                    "provider_called": False,
                },
                "source": {},
                "episodes": [],
                "validation": {"reason": skip_reason},
                "artifacts": {},
            }
            _write_json(
                execution_dir / "planned_tool_skipped.json",
                tool_evaluation,
            )

        if (
            native_execution
            and semantic_ready
            and tool_evaluation.get("status") == "passed"
        ):
            telemetry_root = child_dir / "evaluation/telemetry"
            trusted_episodes = []
            for episode in tool_evaluation.get("episodes", []):
                normalized_episode = deepcopy(dict(episode))
                episode_path = Path(str(episode["episode_dir"]))
                telemetry_root_resolved = telemetry_root.resolve()
                if episode_path.is_absolute():
                    relative_episode = episode_path.resolve().relative_to(
                        telemetry_root_resolved
                    )
                else:
                    repo_candidate = (
                        request.repo_root / episode_path
                    ).resolve()
                    if repo_candidate.is_relative_to(
                        telemetry_root_resolved
                    ):
                        relative_episode = repo_candidate.relative_to(
                            telemetry_root_resolved
                        )
                    else:
                        relative_episode = (
                            telemetry_root_resolved / episode_path
                        ).resolve().relative_to(telemetry_root_resolved)
                normalized_episode["episode_dir"] = (
                    relative_episode.as_posix()
                )
                trusted_episodes.append(normalized_episode)
            child_manifest["trusted_tool_evaluation"].update(
                {
                    "status": "passed",
                    "episode_count": len(trusted_episodes),
                    "episodes": trusted_episodes,
                    "artifact": (
                        tool_evaluation.get("artifacts") or {}
                    ).get("tool_execution"),
                }
            )
            _write_json(
                child_dir / "manifest.json",
                child_manifest,
            )
        return dict(tool_evaluation)

    @staticmethod
    def _project_tool_state(
        *,
        round_plan: dict[str, Any],
        tool_evaluation: Mapping[str, Any],
    ) -> None:
        semantic_execution = round_plan.get("semantic_need_execution")
        if not isinstance(semantic_execution, dict):
            return
        rule_execution = semantic_execution.get("rule_tool")
        if not (
            isinstance(rule_execution, dict)
            and rule_execution.get("requested") is True
        ):
            return
        route_decision = tool_evaluation.get("route_decision")
        route_decision = (
            route_decision if isinstance(route_decision, Mapping) else {}
        )
        rule_execution.update(
            {
                "status": str(tool_evaluation.get("status") or "missing"),
                "route": (
                    tool_evaluation.get("route")
                    or route_decision.get("resolved_route")
                ),
            }
        )

    def _execute_vqa(
        self,
        *,
        request: RoundExecutionRequest,
        child_manifest: Mapping[str, Any],
        child_dir: Path,
        execution_dir: Path,
        tool_evaluation: Mapping[str, Any],
        semantic_ready: bool,
    ) -> dict[str, Any]:
        services = self._services
        if not _round_requests_execution_vqa(request.round_plan):
            execution_vqa = {
                "schema_version": 1,
                "status": "skipped",
                "reason": (
                    "The Proposal did not request a VQA Tool; visual evidence "
                    "was not required for this round."
                ),
                "evidence_conflict": False,
            }
            _write_json(
                execution_dir / "execution_vqa_skipped.json",
                execution_vqa,
            )
        elif semantic_ready:
            execution_vqa = services.run_round_execution_vqa(
                repo_root=request.repo_root,
                child_manifest=child_manifest,
                child_dir=child_dir,
                tool_evaluation=tool_evaluation,
                execution_dir=execution_dir,
                provider=request.provider,
                model=request.vision_model,
                round_plan=request.round_plan,
                reviewed_vqa_registry=request.reviewed_vqa_registry,
            )
        else:
            execution_vqa = {
                "schema_version": 1,
                "status": "skipped",
                "reason": (
                    "TaskSchema unavailable or the requested capability is "
                    "unsupported; VQA was not executed."
                ),
                "evidence_conflict": False,
            }
            _write_json(
                execution_dir / "execution_vqa_skipped.json",
                execution_vqa,
            )
        return dict(execution_vqa)

    @staticmethod
    def _project_vqa_state(
        *,
        round_plan: dict[str, Any],
        execution_vqa: Mapping[str, Any],
    ) -> None:
        semantic_execution = round_plan.get("semantic_need_execution")
        if not isinstance(semantic_execution, dict):
            return
        vqa_execution = semantic_execution.get("vqa_tool")
        if not (
            isinstance(vqa_execution, dict)
            and vqa_execution.get("requested") is True
        ):
            return
        vqa_execution.update(
            {
                "status": str(execution_vqa.get("status") or "missing"),
                "route": "run_local_query_vqa",
            }
        )

    def _attach_native_method_runtime(
        self,
        *,
        request: RoundExecutionRequest,
        native: Mapping[str, Any] | None,
        semantic_ready: bool,
        round_summary: dict[str, Any],
    ) -> None:
        if native is not None:
            round_summary["observations"].update(
                {
                    "execution_backend": (
                        "SmolVLA"
                        if request.policy_backend == "smolvla"
                        else request.policy_backend.upper()
                    ),
                    "policy_backend": request.policy_backend,
                    "semantic_telemetry_ready": semantic_ready,
                    "method_runtime": {
                        "status": (
                            "unsupported"
                            if native.get("unsupported") is True
                            else "validated"
                        ),
                        "runtime": "MethodRuntime",
                        "backend": "RoboTwinMethodBackend",
                        "policy_backend": request.policy_backend,
                        "candidate_id": native["candidate_id"],
                        "outcome": native["evidence_outcome"],
                        "artifact": str(
                            native["method_runtime_path"].relative_to(
                                request.repo_root
                            )
                        ).replace("\\", "/"),
                    },
                }
            )
            round_summary["observations"]["evidence_aggregate"] = (
                build_evidence_aggregate(
                    request.round_plan,
                    round_summary,
                )
            )


__all__ = [
    "RoundExecutionRequest",
    "RoundExecutionResult",
    "RoundExecutionServices",
    "RoundExecutor",
]
