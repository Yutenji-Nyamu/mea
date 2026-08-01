"""Materialize one Plan Agent Proposal into an executable TaskGen round.

This module owns the paper-level hand-off from a typed Proposal to the TaskGen
request and round contract.  The Agent CLI selects and sequences Proposals; it
does not encode TaskGen command-line details or reconstruct semantic needs.
"""

from __future__ import annotations

import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.agent_evidence import round_execution_backend
from mea.capability_adapter import taskgen_route
from mea.plan_artifacts import PROPOSAL_FILENAME, PROPOSAL_MATERIALIZATION
from mea.planner.experiment_candidate import validate_experiment_candidate
from mea.proposals import ProposalError, validate_task_proposal
from mea.round_contract import validate_round_capability_contract


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def child_run_id(evaluation_id: str, round_id: str) -> str:
    return f"run_{evaluation_id.removeprefix('eval_')}_{round_id}"


def build_taskgen_command(
    repo_root: Path,
    evaluation_id: str,
    round_plan: dict[str, Any],
    *,
    text_model: str,
    vision_model: str,
    base_url: str | None,
    gpu: int,
    max_reflections: int,
    telemetry_profile: str = "balanced_v1",
    reviewed_task_registry: Path | None = None,
    registration_identity: dict[str, Any] | None = None,
    run_id_suffix: str = "",
) -> tuple[list[str], str]:
    capability_contract = validate_round_capability_contract(round_plan)
    if run_id_suffix and re.fullmatch(r"_[A-Za-z0-9_]+", run_id_suffix) is None:
        raise ValueError("run_id_suffix must be empty or a safe underscore suffix")
    run_id = child_run_id(evaluation_id, round_plan["round_id"]) + run_id_suffix
    execution = round_plan["execution"]
    seed = execution["seeds"][0]
    raw_task_name = round_plan.get("task_name")
    if not isinstance(raw_task_name, str) or not raw_task_name.strip():
        raise ValueError("round task_name must be explicit")
    task_name = (
        capability_contract["task_name"]
        if capability_contract is not None
        else raw_task_name.strip()
    )
    task_proposal = round_plan.get("task_proposal")
    variant_hint = round_plan.get("variant_hint") or {}
    if task_proposal is not None:
        try:
            normalized_task_proposal = validate_task_proposal(
                task_proposal, expected_task_name=task_name
            )
        except ProposalError as exc:
            raise ValueError(f"invalid TaskProposal before TaskGen: {exc}") from exc
        variant_hint = normalized_task_proposal["changes"]
    task_module = round_plan.get("task_module")
    route = (
        taskgen_route(capability_contract)
        if capability_contract is not None
        else str(round_plan["route"])
    )
    execution_backend = round_execution_backend(round_plan)
    command = [
        sys.executable,
        str(repo_root / "scripts/manipeval_taskgen.py"),
        "--repo-root",
        str(repo_root),
        "--request",
        round_plan["task_instruction"],
        "--run-id",
        run_id,
        "--task-name",
        task_name,
        "--mode",
        route,
        "--text-model",
        text_model,
        "--vision-model",
        vision_model,
        "--seed",
        str(seed),
        "--num-episodes",
        str(execution["num_episodes"]),
        "--gpu",
        str(gpu),
        "--telemetry-profile",
        telemetry_profile,
        "--probe",
        "--max-reflections",
        str(max_reflections),
    ]
    if task_module:
        command.extend(["--task-module", str(task_module)])
    if variant_hint:
        command.extend(
            [
                "--variant-hint-json",
                json.dumps(
                    variant_hint,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    experiment_candidate = (
        round_plan.get("proposal")
        or round_plan.get("experiment_candidate")
    )
    if (
        experiment_candidate is not None
        and route == "generic_provider_scene_checker_codegen"
    ):
        command.extend(
            [
                "--proposal-json",
                json.dumps(
                    experiment_candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    task_variant_id = round_plan.get("task_variant_id")
    if task_variant_id:
        command.extend(["--variant-id", str(task_variant_id)])
    elif (
        round_plan.get("template_id")
        and round_plan.get("capability_contract") is None
    ):
        # Compatibility for hand-authored legacy plans that predate the
        # capability adapter's template/task-variant identity split.
        command.extend(["--variant-id", str(round_plan["template_id"])])
    if round_plan.get("capability_contract") is not None:
        command.extend(
            [
                "--capability-contract-json",
                json.dumps(
                    round_plan["capability_contract"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    if task_proposal is not None:
        command.extend(
            [
                "--task-proposal-json",
                json.dumps(
                    normalized_task_proposal,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    if route == "official":
        if execution_backend in {"expert", "both"}:
            command.append("--expert")
        if execution_backend in {"act", "both"}:
            command.append("--run-act")
    elif route == "generic_provider_scene_checker_codegen":
        # Generic TaskGen already performs live initial-negative, render, and
        # official-expert-positive preflight inside its single repair loop.
        command.append("--run-act")
    elif route == "provider_scene_checker_codegen":
        # The proposal-derived visual contract checks that both intended
        # blocks are visible before the expert and ACT gates.
        command.extend(["--expert", "--vision-check", "--run-act"])
    else:
        # The bounded generated-task prototype keeps its original expert
        # solvability gate before the ACT policy rollout.
        command.extend(["--expert", "--vision-check", "--run-act"])
    if base_url:
        command.extend(["--base-url", base_url])
    if reviewed_task_registry is not None:
        command.extend(["--reviewed-task-registry", str(reviewed_task_registry)])
    if registration_identity is not None:
        command.extend(
            [
                "--registration-identity-json",
                json.dumps(
                    registration_identity,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    return command, run_id


def materialize_open_world_round(
    repo_root: Path,
    evaluation_dir: Path,
    *,
    round_number: int,
    candidate: Mapping[str, Any],
    control_execution: Mapping[str, Any],
    policy_backend: str = "act",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize only the TaskGen and ToolGen stages requested by a candidate."""

    normalized = validate_experiment_candidate(candidate)
    scene_need = normalized["scene_need"]
    checker_need = normalized["checker_need"]
    rule_tool_need = normalized["rule_tool_need"]
    vqa_tool_need = normalized["vqa_tool_need"]
    taskgen_requested = scene_need is not None or checker_need is not None
    rule_tool_requested = rule_tool_need is not None
    toolgen_requested = bool(
        rule_tool_need is not None and rule_tool_need.get("kind") != "reuse"
    )
    vqa_tool_requested = vqa_tool_need is not None
    route = (
        "generic_provider_scene_checker_codegen"
        if taskgen_requested
        else "official"
    )
    outcome_metric = (
        "generated_check_success"
        if checker_need is not None
        else "official_check_success"
    )
    deferred_tool_request = {
        "schema_version": 1,
        "task_name": str(normalized["base_task"]),
        "metric": outcome_metric,
        "question": (
            "Did the rollout satisfy the official RoboTwin success check?"
            if outcome_metric == "official_check_success"
            else "Did the rollout satisfy the generated success predicate?"
        ),
    }
    tool_bundle = {
        "schema_version": 1,
        "source": (
            "deferred_until_executed_telemetry_schema"
            if toolgen_requested
            else "official_checker_reuse"
            if rule_tool_requested
            else "vqa_only_no_rule_tool_requested"
            if vqa_tool_requested
            else "task_checker_evidence_no_new_tool_requested"
        ),
        "tool_request": deferred_tool_request,
    }
    artifact_dir = (
        evaluation_dir
        / PROPOSAL_MATERIALIZATION
        / f"round_{round_number:02d}"
    )
    _write_json(artifact_dir / PROPOSAL_FILENAME, normalized)
    _write_json(artifact_dir / "tool_request_bundle.json", tool_bundle)
    execution = deepcopy(dict(control_execution))
    if policy_backend not in {"act", "smolvla", "hyvla"}:
        raise ValueError(f"unsupported production policy backend: {policy_backend!r}")
    # ``backend`` remains the legacy policy-vs-expert evidence selector until
    # its compatibility readers are migrated.  ``policy_backend`` is the
    # actual runner used by the native MethodRuntime.
    execution["backend"] = "act"
    execution["policy_backend"] = policy_backend
    execution["gates"] = (
        [
            "ast",
            "render",
            "visual_diagnosis",
            "expert",
            policy_backend,
            "toolkit",
        ]
        if taskgen_requested
        else ["render", policy_backend, "toolkit"]
    )
    if rule_tool_requested:
        execution["gates"].append("planned_tool")
    if vqa_tool_requested:
        execution["gates"].append("dynamic_vqa")
    execution["gates"].append("aggregate")
    candidate_id = str(normalized["candidate_id"])
    sub_aspect = str(normalized["semantic_concern"]).split(":", 1)[0].strip()

    def need_description(need: Mapping[str, Any] | None) -> str:
        return (
            str(need["description"])
            if need is not None
            else "reuse the official implementation"
        )

    round_plan = {
        "round_id": f"round_{round_number}",
        "template_id": None,
        "candidate_id": candidate_id,
        "proposal": normalized,
        "sub_aspect": sub_aspect,
        "rationale": (
            "Materialize only the Query-derived Task or Tool needs; no catalog "
            "template authorizes this round."
        ),
        "task_instruction": (
            f"{normalized['source_query']}\nScene need: "
            f"{need_description(scene_need)}\nChecker need: "
            f"{need_description(checker_need)}"
        ),
        "task_name": str(normalized["base_task"]),
        "task_module": (
            None
            if taskgen_requested
            else f"envs.{normalized['base_task']}"
        ),
        "telemetry_profile": "balanced_v1",
        "route": route,
        "variant_hint": {},
        "execution": execution,
        "observations": (
            ["scene_alignment", "expert_solvable", "trusted_tools"]
            + (["planned_tool"] if rule_tool_requested else [])
            + (["dynamic_vqa"] if vqa_tool_requested else [])
            + ["aggregate"]
        ),
        "tool_request": deepcopy(deferred_tool_request),
        "open_tool_request_deferred": toolgen_requested,
        "vqa_phenomenon_ids": [],
        "semantic_need_execution": {
            "schema_version": 2,
            "candidate_id": candidate_id,
            "task": {
                "requested": scene_need is not None,
                "description": (
                    str(scene_need["description"])
                    if scene_need is not None
                    else None
                ),
                "route": (
                    "generic_provider_scene_checker_codegen"
                    if scene_need is not None
                    else "official_scene_reuse"
                ),
                "status": (
                    "selected" if scene_need is not None else "not_requested"
                ),
            },
            "checker": {
                "requested": checker_need is not None,
                "description": (
                    str(checker_need["description"])
                    if checker_need is not None
                    else None
                ),
                "route": (
                    "provider_written_python"
                    if checker_need is not None
                    else "official_checker_reuse"
                ),
                "status": (
                    "selected" if checker_need is not None else "not_requested"
                ),
            },
            "rule_tool": {
                "requested": rule_tool_requested,
                "description": (
                    str(rule_tool_need["description"])
                    if rule_tool_need is not None
                    else None
                ),
                "route": (
                    "after_executed_telemetry_schema"
                    if toolgen_requested
                    else "trusted_official_checker_reuse"
                    if rule_tool_requested
                    else "task_checker_evidence"
                ),
                "status": (
                    "pending"
                    if rule_tool_requested
                    else "not_requested"
                ),
            },
            "vqa_tool": {
                "requested": vqa_tool_need is not None,
                "description": (
                    str(vqa_tool_need["description"])
                    if vqa_tool_need is not None
                    else None
                ),
                "route": (
                    "task_owned_or_generated_question"
                    if vqa_tool_need is not None
                    else "not_requested"
                ),
                "status": (
                    "pending"
                    if vqa_tool_need is not None
                    else "not_requested"
                ),
            },
        },
    }
    return round_plan, tool_bundle


__all__ = [
    "build_taskgen_command",
    "child_run_id",
    "materialize_open_world_round",
]
