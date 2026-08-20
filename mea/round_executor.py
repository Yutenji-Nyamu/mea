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

from mea.execution_vqa.runtime import run_round_execution_vqa
from mea.planner.evidence_policy import build_round_evidence
from mea.proposals import tool_request_from_proposal
from mea.round_evidence import aggregate_round_results
from mea.round_summary import summarize_round
from mea.round_tools import (
    materialize_open_world_tool_request,
    reuse_bound_child_checker_tool,
)
from mea.toolgen import execute_tool_request, route_tool_request


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _round_requests_execution_vqa(
    round_plan: Mapping[str, Any] | None,
) -> bool:
    """Run execution VQA only when the round contract explicitly requests it."""

    plan = round_plan or {}
    semantic_needs = plan.get("semantic_need_execution")
    vqa_need = (
        semantic_needs.get("vqa_tool")
        if isinstance(semantic_needs, Mapping)
        else None
    )
    if isinstance(vqa_need, Mapping) and vqa_need.get("requested") is True:
        return True
    observations = plan.get("observations")
    if isinstance(observations, list) and any(
        item in {"execution_vqa", "dynamic_vqa"} for item in observations
    ):
        return True
    phenomenon_ids = plan.get("vqa_phenomenon_ids")
    return isinstance(phenomenon_ids, list) and bool(phenomenon_ids)


@dataclass(frozen=True)
class RoundExecutionServices:
    """Native policy backends and the manifest writer used by one round."""

    update_manifest: Callable[..., Mapping[str, Any]]
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
    gpu: int
    max_reflections: int
    provider: Any
    toolgen_model: str
    telemetry_profile: str = "balanced_v1"
    policy_backend: str = "act"
    runtime_target: Mapping[str, Any] | None = None
    policy_server_port: int = 18771


@dataclass(frozen=True)
class RoundExecutionResult:
    """Typed result returned to the route/session loop."""

    child_manifest: dict[str, Any]
    child_dir: Path
    round_summary: dict[str, Any]
    tool_evaluation: dict[str, Any]
    returncode: int


@dataclass(frozen=True)
class _PolicyRoundArtifacts:
    """Simulator/backend artifacts consumed by the shared evidence lifecycle."""

    child_manifest: dict[str, Any]
    child_dir: Path
    execution_dir: Path
    child_manifest_path: Path
    run_id: str
    returncode: int
    semantic_ready: bool
    native: Mapping[str, Any] | None


class RoundExecutor:
    """Execute one Proposal through policy, evidence tools, and aggregation."""

    def __init__(self, services: RoundExecutionServices) -> None:
        self._services = services

    def _execute_policy(
        self, request: RoundExecutionRequest
    ) -> _PolicyRoundArtifacts:
        """Execute one native MethodRuntime backend.

        Legacy ACT TaskGen subprocess transport lives in
        ``experiments.paper.compat_round_executor`` and is intentionally not a
        production fallback.
        """

        native_policy_round = self._services.native_policy_rounds.get(
            request.policy_backend
        )
        if native_policy_round is None:
            raise RuntimeError(
                "production rounds require a native MethodRuntime backend: "
                f"{request.policy_backend!r}"
            )
        if request.runtime_target is None:
            raise RuntimeError(
                "native policy execution requires the bound runtime target"
            )
        round_id = request.round_plan["round_id"]
        self._services.update_manifest(
            request.evaluation_dir,
            status=f"executing_{round_id}",
            active_child_run_id=None,
            policy_backend=request.policy_backend,
        )
        native = native_policy_round(
            repo_root=request.repo_root,
            evaluation_dir=request.evaluation_dir,
            evaluation_id=request.evaluation_id,
            round_plan=request.round_plan,
            runtime_target=request.runtime_target,
            telemetry_profile=request.telemetry_profile,
            policy_server_port=request.policy_server_port,
            gpu=request.gpu,
            provider=request.provider,
            text_model=request.text_model,
            vision_model=request.vision_model,
            max_reflections=request.max_reflections,
        )
        child_manifest = dict(native["child_manifest"])
        return _PolicyRoundArtifacts(
            child_manifest=child_manifest,
            child_dir=native["child_dir"],
            execution_dir=(
                request.evaluation_dir / "execution" / round_id
            ),
            child_manifest_path=native["manifest_path"],
            run_id=str(child_manifest["run_id"]),
            returncode=0,
            semantic_ready=bool(native["semantic_telemetry_ready"]),
            native=native,
        )

    def execute(self, request: RoundExecutionRequest) -> RoundExecutionResult:
        services = self._services
        repo_root = request.repo_root
        evaluation_dir = request.evaluation_dir
        evaluation_id = request.evaluation_id
        round_plan = request.round_plan
        round_id = round_plan["round_id"]

        policy = self._execute_policy(request)
        child_manifest = policy.child_manifest
        child_dir = policy.child_dir
        execution_dir = policy.execution_dir
        child_manifest_path = policy.child_manifest_path
        run_id = policy.run_id
        returncode = policy.returncode
        semantic_ready = policy.semantic_ready
        native = policy.native

        if not child_manifest_path.is_file():
            raise RuntimeError(
                f"child TaskGen manifest 不存在: {child_manifest_path}"
            )
        child_manifest = json.loads(
            child_manifest_path.read_text(encoding="utf-8")
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
        aggregate_result = aggregate_round_results(
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
        round_summary = summarize_round(
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
        round_summary["observations"]["round_evidence"] = (
            build_round_evidence(round_plan, round_summary)
        )
        _write_json(
            execution_dir / "round_evidence.json",
            round_summary["observations"]["round_evidence"],
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
            if round_plan.get("open_tool_request_deferred") is True:
                tool_bundle = materialize_open_world_tool_request(
                    request.repo_root,
                    execution_dir,
                    round_plan=round_plan,
                    child_dir=child_dir,
                    provider=request.provider,
                    toolgen_model=request.toolgen_model,
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
            tool_evaluation = reuse_bound_child_checker_tool(
                request.repo_root,
                child_manifest,
                planned_tool_dir,
                proposed_request,
            )
            if tool_evaluation is None:
                tool_evaluation = execute_tool_request(
                    request.repo_root,
                    child_dir,
                    planned_tool_dir,
                    proposed_request,
                    **tool_kwargs,
                )
        else:
            candidate_rejection = child_manifest.get(
                "candidate_unexecutable"
            )
            taskgen_failure = child_manifest.get(
                "taskgen_materialization_failed"
            )
            skip_reason = (
                str(candidate_rejection.get("diagnosis"))
                if child_manifest.get("status") == "candidate_unexecutable"
                and isinstance(candidate_rejection, Mapping)
                else str(taskgen_failure.get("diagnosis"))
                if child_manifest.get("status")
                == "taskgen_materialization_failed"
                and isinstance(taskgen_failure, Mapping)
                else
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
            self._persist_native_planned_tool(
                repo_root=request.repo_root,
                child_dir=child_dir,
                child_manifest=child_manifest,
                tool_evaluation=tool_evaluation,
            )
        return dict(tool_evaluation)

    @staticmethod
    def _persist_native_planned_tool(
        *,
        repo_root: Path,
        child_dir: Path,
        child_manifest: dict[str, Any],
        tool_evaluation: Mapping[str, Any],
    ) -> None:
        """Persist a planned Tool without replacing outcome authority.

        Native TaskGen records the official/generated checker in
        ``trusted_tool_evaluation`` before the planned Rule Tool runs.  The
        latter is an additional observation from the same rollout, not a new
        success predicate.  Keeping it separate preserves checker semantics;
        the Aggregate still consumes the returned ``tool_evaluation`` directly.
        """

        telemetry_root = (child_dir / "evaluation/telemetry").resolve()
        planned_episodes = []
        for episode in tool_evaluation.get("episodes", []):
            normalized_episode = deepcopy(dict(episode))
            episode_path = Path(str(episode["episode_dir"]))
            if episode_path.is_absolute():
                relative_episode = episode_path.resolve().relative_to(
                    telemetry_root
                )
            else:
                repo_candidate = (repo_root / episode_path).resolve()
                if repo_candidate.is_relative_to(telemetry_root):
                    relative_episode = repo_candidate.relative_to(
                        telemetry_root
                    )
                else:
                    relative_episode = (
                        telemetry_root / episode_path
                    ).resolve().relative_to(telemetry_root)
            normalized_episode["episode_dir"] = relative_episode.as_posix()
            planned_episodes.append(normalized_episode)
        child_manifest["planned_tool_evaluation"] = {
            **deepcopy(dict(tool_evaluation)),
            "episode_count": len(planned_episodes),
            "episodes": planned_episodes,
            "artifact": (
                tool_evaluation.get("artifacts") or {}
            ).get("tool_execution"),
        }
        trusted_outcome = child_manifest.get("trusted_tool_evaluation")
        if (
            isinstance(trusted_outcome, dict)
            and not trusted_outcome.get("episodes")
        ):
            # Official native rounds may not have a TaskGen-owned outcome Tool.
            # Retain the historical representative-episode fallback for VQA,
            # while never replacing an already recorded checker outcome.
            trusted_outcome.update(
                {
                    "status": "passed",
                    "episode_count": len(planned_episodes),
                    "episodes": deepcopy(planned_episodes),
                    "artifact": child_manifest[
                        "planned_tool_evaluation"
                    ]["artifact"],
                }
            )
        _write_json(child_dir / "manifest.json", child_manifest)

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
            execution_vqa = run_round_execution_vqa(
                repo_root=request.repo_root,
                child_manifest=child_manifest,
                child_dir=child_dir,
                tool_evaluation=tool_evaluation,
                execution_dir=execution_dir,
                provider=request.provider,
                model=request.vision_model,
                round_plan=request.round_plan,
            )
        else:
            candidate_rejection = child_manifest.get(
                "candidate_unexecutable"
            )
            taskgen_failure = child_manifest.get(
                "taskgen_materialization_failed"
            )
            execution_vqa = {
                "schema_version": 1,
                "status": "skipped",
                "reason": (
                    str(candidate_rejection.get("diagnosis"))
                    if child_manifest.get("status")
                    == "candidate_unexecutable"
                    and isinstance(candidate_rejection, Mapping)
                    else str(taskgen_failure.get("diagnosis"))
                    if child_manifest.get("status")
                    == "taskgen_materialization_failed"
                    and isinstance(taskgen_failure, Mapping)
                    else (
                        "TaskSchema unavailable or the requested capability "
                        "is unsupported; VQA was not executed."
                    )
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
            planning_observation = native.get("planning_observation")
            if not isinstance(planning_observation, Mapping):
                native_manifest = native.get("child_manifest", {})
                candidate_rejection = (
                    native_manifest.get("planning_observation")
                    if isinstance(native_manifest, Mapping)
                    else None
                )
                if not isinstance(candidate_rejection, Mapping):
                    candidate_rejection = (
                        native_manifest.get("candidate_unexecutable")
                        if isinstance(native_manifest, Mapping)
                        else None
                    )
                planning_observation = (
                    candidate_rejection
                    if isinstance(candidate_rejection, Mapping)
                    else None
                )
            method_status = (
                str(planning_observation.get("kind"))
                if isinstance(planning_observation, Mapping)
                and planning_observation.get("kind")
                else "taskgen_materialization_failed"
                if native.get("taskgen_materialization_failed") is True
                else "unsupported"
                if native.get("unsupported") is True
                else "validated"
            )
            round_summary["observations"].update(
                {
                    "execution_backend": {
                        "smolvla": "SmolVLA",
                        "hyvla": "Hy-VLA",
                    }.get(request.policy_backend, request.policy_backend.upper()),
                    "policy_backend": request.policy_backend,
                    "semantic_telemetry_ready": semantic_ready,
                    "method_runtime": {
                        "status": method_status,
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
            if planning_observation is not None:
                reason_code = str(
                    planning_observation.get("reason_code")
                    or "taskgen_planning_observation_before_policy"
                )
                round_summary["failure_stage"] = str(
                    planning_observation.get("failure_stage")
                    or "taskgen_expert_gate"
                )
                round_summary["observations"].update(
                    {
                        "planning_observation": deepcopy(
                            dict(planning_observation)
                        ),
                        "policy_success": None,
                        "policy_outcome": {
                            "metric": None,
                            "authority": None,
                            "binding": None,
                            "value": None,
                            "official_equivalent": None,
                            "execution_scope": "not_executed",
                            "outcome_semantics": {
                                "schema_version": 1,
                                "status": "non_comparable",
                                "evidence_conflict": False,
                                "official_equivalent": None,
                                "outcome_authority": None,
                                "episodes": [],
                                "reason_codes": [
                                    reason_code
                                ],
                            },
                        },
                        "outcome_semantics": {
                            "schema_version": 1,
                            "status": "non_comparable",
                            "evidence_conflict": False,
                            "official_equivalent": None,
                            "outcome_authority": None,
                            "episodes": [],
                            "reason_codes": [
                                reason_code
                            ],
                        },
                    }
                )


__all__ = [
    "RoundExecutionRequest",
    "RoundExecutionResult",
    "RoundExecutionServices",
    "RoundExecutor",
]
