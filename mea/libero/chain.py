"""LIBERO backend composition for the shared MEA outer method runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from mea.feedback.answer_scope import build_answer_scope
from mea.method_runtime import (
    CandidateRequest,
    EvidenceRequest,
    MethodRuntime,
    RolloutRequest,
    BackendBindingRequest,
)
from mea.planner.claim_first import ClaimFirstOpenQueryAgent
from mea.providers import OpenAICompatibleProvider
from mea.toolkit.aggregate import aggregate_tool_executions

from .benchmark import (
    BATCH23_PARITY_ACTION_STEPS,
    BATCH23_PARITY_HORIZON_STEPS,
    BATCH23_PARITY_OBSERVATION_SIZE,
    LiberoBenchmarkAdapter,
    TaskContract,
    build_official_task_contract,
)
from .policy import LeRobotPolicyAdapter
from .runtime import LiberoMethodBackend
from .retrieval import (
    BDDLRetrieval,
    BDDLTaskIndex,
    ControlledChangeContract,
    PolicyTaskCompatibility,
    authorize_controlled_change,
    pending_controlled_change,
    smolvla_policy_compatibility,
)
from .taskgen import LiberoTaskGenBackend
from .tool import LiberoPredicateToolBackend


DEFAULT_CHECKPOINT = Path("/root/autodl-tmp/checkpoints/libero/smolvla_libero")
DEFAULT_SEED = 100800
HORIZON_STEPS = BATCH23_PARITY_HORIZON_STEPS
OBSERVATION_SIZE = BATCH23_PARITY_OBSERVATION_SIZE
N_ACTION_STEPS = BATCH23_PARITY_ACTION_STEPS
_BOUND_TASK_RE = re.compile(r"^(libero_[a-z0-9_]+)/task([0-9]+)$")


def parse_bound_libero_task(value: str | None) -> tuple[str, int]:
    match = _BOUND_TASK_RE.fullmatch((value or "").strip().casefold())
    if not match:
        raise ValueError(
            "checkpoint task scope is unknown; bind an official control explicitly "
            "with --bound-task-name libero_object/task0 (or another suite/task)"
        )
    return match.group(1), int(match.group(2))


def _write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _open_query_retrieval(
    *,
    request: str,
    checkpoint: Path,
    official: TaskContract,
) -> tuple[PolicyTaskCompatibility, BDDLRetrieval, ControlledChangeContract]:
    index = BDDLTaskIndex.from_libero_suite(official.suite)
    task = next(
        item
        for item in index.tasks
        if item.task_id == official.official_task_id
        and item.problem_name == official.problem_name
    )
    compatibility = smolvla_policy_compatibility(
        checkpoint=checkpoint,
        explicit_task_binding=task,
    )
    retrieval = index.retrieve_nearest(request, compatibility=compatibility)
    return compatibility, retrieval, pending_controlled_change(retrieval)


def _gate0(
    *,
    checkpoint: Path,
    official: TaskContract,
    compatibility: PolicyTaskCompatibility,
    retrieval: BDDLRetrieval,
    change_contract: ControlledChangeContract,
) -> dict[str, Any]:
    bddl_path = Path(official.bddl_path)
    init_path = Path(official.initial_state_source)
    official_env = LiberoBenchmarkAdapter(
        episode_length=HORIZON_STEPS,
        observation_size=OBSERVATION_SIZE,
        suite_name=official.suite,
        task_id=official.official_task_id,
    ).make_official_env()
    state_observation_enabled = official_env.obs_type == "pixels_agent_pos"
    observation_size_matches = (
        getattr(official_env, "observation_height", None)
        == OBSERVATION_SIZE
        and getattr(official_env, "observation_width", None)
        == OBSERVATION_SIZE
    )
    official_env.close()
    checkpoint_config = json.loads(
        (checkpoint / "config.json").read_text(encoding="utf-8")
    ) if (checkpoint / "config.json").is_file() else {}
    required = [
        checkpoint / "config.json",
        checkpoint / "model.safetensors",
        checkpoint / "policy_preprocessor.json",
        checkpoint / "policy_postprocessor.json",
        bddl_path,
        init_path,
    ]
    checks = {
        "all_required_files_present": all(path.is_file() for path in required),
        "official_problem_registered": bool(official.python_problem_impl),
        "query_source_is_authorized_for_control": compatibility.authorizes(
            retrieval.selected
        ),
        "taskgen_change_awaits_planner": change_contract.status == "pending",
        "batch23_parity_horizon_280": official.horizon_steps == HORIZON_STEPS,
        "batch23_parity_observation_360": observation_size_matches,
        "batch23_parity_action_chunk_10": N_ACTION_STEPS == 10,
        "relative_control": official.control_mode == "relative",
        "state_observation_enabled": state_observation_enabled,
        "checkpoint_requires_state": (
            "observation.state" in checkpoint_config.get("input_features", {})
        ),
    }
    result = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "checkpoint": {
            "path": str(checkpoint),
            "model_sha256": (
                _sha256_file(checkpoint / "model.safetensors")
                if (checkpoint / "model.safetensors").is_file()
                else None
            ),
        },
        "official_task_contract": official.to_dict(),
        "policy_task_compatibility": compatibility.to_dict(),
        "open_query_retrieval": retrieval.to_dict(),
        "controlled_change_contract": change_contract.to_dict(),
        "rollout_budget": 2,
        "horizon_steps_each": HORIZON_STEPS,
        "observation_size": OBSERVATION_SIZE,
        "n_action_steps": N_ACTION_STEPS,
    }
    if result["status"] != "passed":
        failed = [key for key, passed in checks.items() if not passed]
        raise RuntimeError("LIBERO Gate0 failed: " + ", ".join(failed))
    return result


def _capabilities(
    checkpoint: Path,
    official: TaskContract,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_card": {
            "policy": "SmolVLA",
            "checkpoint": str(checkpoint),
            "action_space": "7D relative end-effector",
            "observation": "two RGB views plus proprioception",
        },
        "simulator_card": {
            "benchmark": "LIBERO",
            "suite": official.suite,
            "official_control": (
                f"explicitly bound task {official.official_task_id} at the same "
                "initial simulator state"
            ),
            "phase_boundary": (
                "existing object identity may change; objects, regions, initial "
                "state, camera, workspace, action mode and horizon are fixed"
            ),
            "horizon_steps": HORIZON_STEPS,
        },
        "generation_card": {
            "taskgen_operations": [
                {
                    "operation": "state_compatible_bddl_goal_edit",
                    "controlled_axis": "existing_object_identity",
                    "generation_mode": "provider_written_bddl",
                    "allowed_change_roots": [
                        "language",
                        "obj_of_interest",
                        "goal",
                    ],
                }
            ],
            "toolgen": {
                "retrieve_first": True,
                "can_generate_rule_metric": True,
                "can_generate_vqa_question": False,
            },
        },
    }


def _persist_planner_bundle(
    root: Path,
    name: str,
    planner: ClaimFirstOpenQueryAgent,
    bundle: dict[str, Any],
) -> Path:
    target = root / "planner" / name
    target.mkdir(parents=True, exist_ok=True)
    if planner.last_prompt is not None:
        (target / "prompt.md").write_text(planner.last_prompt, encoding="utf-8")
    for index, response in enumerate(planner.last_responses, start=1):
        (target / f"response_attempt_{index}.txt").write_text(
            response, encoding="utf-8"
        )
    return _write_json(target / "proposal_bundle.json", bundle)


def _official_contract_artifact(root: Path, contract: TaskContract) -> Path:
    return _write_json(root / "round_01_official" / "task_contract.json", contract.to_dict())


def _method_chain_is_valid(
    *,
    official_success: bool,
    rollouts_executed: int,
    planner_taskgen_alignment: bool,
    compatibility_probe_passed: bool,
    aggregate_status: str,
    exact_reuse: bool,
    final_planner_bundle_present: bool,
    episode_protocol_matches: bool,
) -> bool:
    """Mechanism validity is independent of the custom policy outcome."""

    return bool(
        official_success
        and rollouts_executed == 2
        and planner_taskgen_alignment
        and compatibility_probe_passed
        and aggregate_status == "passed"
        and exact_reuse
        and final_planner_bundle_present
        and episode_protocol_matches
    )


def _planner_taskgen_misaligned_result(
    *,
    request: str,
    root: Path,
    official_success: bool,
    retrieval: BDDLRetrieval,
    compatibility: PolicyTaskCompatibility,
    change_contract: ControlledChangeContract,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "planner_taskgen_misaligned",
        "benchmark": "libero",
        "policy": "smolvla",
        "query": request,
        "rollouts_executed": 1,
        "rollout_budget": 2,
        "official_success": official_success,
        "custom_rollout_authorized": False,
        "stop_reason": "planner_change_not_expressible_by_taskgen",
        "method_chain_valid": False,
        "paper_performance_evidence": False,
        "scientific_evidence_eligible": False,
        "retrieval": retrieval.to_dict(),
        "policy_task_compatibility": compatibility.to_dict(),
        "controlled_change_contract": change_contract.to_dict(),
        "raw_run_dir": str(root),
    }


def run_libero_method_chain(
    *,
    repo_root: str | Path,
    request: str,
    evaluation_id: str,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    seed: int = DEFAULT_SEED,
    planner_model: str = "gpt-4o-2024-11-20",
    taskgen_model: str = "gpt-4o-2024-11-20",
    base_url: str | None = None,
    plan_only: bool = False,
    bound_suite: str | None = None,
    bound_task_id: int | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if bound_suite is None or bound_task_id is None:
        raise ValueError(
            "checkpoint task scope is unknown; an explicit LIBERO suite/task "
            "binding is required before retrieval or control"
        )
    root = repo / "mea" / "evaluation_runs" / evaluation_id
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    binding_runtime = MethodRuntime(
        LiberoMethodBackend(task_contract_factory=build_official_task_contract)
    )
    official_binding = binding_runtime.bind_task(
        BackendBindingRequest(
            task_reference={"suite": bound_suite, "task_id": bound_task_id}
        )
    )
    official_contract = official_binding.native_task
    _write_json(root / "runtime" / "task_binding.json", official_binding.to_dict())
    compatibility, query_retrieval, pending_change = _open_query_retrieval(
        request=request,
        checkpoint=checkpoint_path,
        official=official_contract,
    )
    gate = _gate0(
        checkpoint=checkpoint_path,
        official=official_contract,
        compatibility=compatibility,
        retrieval=query_retrieval,
        change_contract=pending_change,
    )
    _write_json(root / "gate0.json", gate)
    official_contract_path = _official_contract_artifact(root, official_contract)
    if plan_only:
        result = {
            "schema_version": 1,
            "status": "plan_only_passed",
            "method_chain_valid": False,
            "paper_performance_evidence": False,
            "scientific_evidence_eligible": False,
            "provider_required": False,
            "rollouts_executed": 0,
            "gate0": str(root / "gate0.json"),
            "query_concern": request,
            "retrieval": query_retrieval.to_dict(),
            "policy_task_compatibility": compatibility.to_dict(),
            "controlled_change_contract": pending_change.to_dict(),
        }
        _write_json(root / "compact_result.json", result)
        return result

    if not os.getenv("UIUI_API_KEY"):
        _write_json(
            root / "compact_result.json",
            {
                "schema_version": 1,
                "status": "startup_failed",
                "error_type": "MissingProviderCredential",
                "rollouts_executed": 0,
                "rollout_budget": 2,
                "method_chain_valid": False,
                "paper_performance_evidence": False,
                "scientific_evidence_eligible": False,
            },
        )
        raise RuntimeError(
            "live LIBERO evaluation requires UIUI_API_KEY in the current process"
        )
    provider = OpenAICompatibleProvider(
        base_url=base_url,
        text_model=planner_model,
        max_retries=1,
    )
    benchmark = LiberoBenchmarkAdapter(
        episode_length=HORIZON_STEPS,
        observation_size=OBSERVATION_SIZE,
        suite_name=official_contract.suite,
        task_id=official_contract.official_task_id,
    )
    policy = LeRobotPolicyAdapter(
        checkpoint=checkpoint_path,
        device="cuda",
        n_action_steps=N_ACTION_STEPS,
        horizon_steps=HORIZON_STEPS,
        observation_size=OBSERVATION_SIZE,
        suite_name=official_contract.suite,
        task_id=official_contract.official_task_id,
    )
    method_backend = LiberoMethodBackend(
        benchmark_adapter=benchmark,
        policy_adapter=policy,
        task_contract_factory=build_official_task_contract,
    )
    method_runtime = MethodRuntime(method_backend)
    rollouts_executed = 0
    try:
        _write_json(root / "policy_load.json", policy.load(seed=seed))
        official_candidate = method_backend.official_candidate(
            official_binding,
            source_query=request,
            task_contract_path=official_contract_path,
        )
        _write_json(
            root / "runtime" / "round_01_candidate.json",
            official_candidate.to_dict(),
        )
        official_rollout = method_runtime.rollout(
            official_candidate,
            RolloutRequest(
                round_id="round_01_official_control",
                seed=seed,
                output_dir=root / "round_01_official" / "episode",
                provenance={
                    "round": 1,
                    "route": "official_control",
                    "stock_task_ids": True,
                    "horizon_steps": HORIZON_STEPS,
                },
            ),
        )
        _write_json(
            root / "runtime" / "round_01_rollout.json",
            official_rollout.to_dict(),
        )
        official_record = official_rollout.native_episode
        rollouts_executed += 1
        control_evidence_record = method_runtime.evidence(
            official_rollout,
            EvidenceRequest(
                sub_aspect="official_control",
                hypothesis=(
                    "The local SmolVLA checkpoint can execute the unchanged task."
                ),
                perturbation="none",
                summary=(
                    f"Official {official_contract.suite}/task"
                    f"{official_contract.official_task_id} live rollout "
                    f"success={official_record.success}; "
                    f"reward_sum={official_record.reward_sum}; "
                    f"steps={official_record.executed_steps}."
                ),
                limitations=(
                    "N=1 fixed seed",
                    "batch23-parity 280-step feasibility horizon",
                ),
            ),
        )
        _write_json(
            root / "runtime" / "round_01_evidence.json",
            control_evidence_record.to_dict(),
        )
        control_evidence = control_evidence_record.to_planner_dict()
        _write_json(root / "round_01_official" / "evidence.json", control_evidence)
        if not official_record.success:
            result = {
                "schema_version": 1,
                "status": "control_failed",
                "benchmark": "libero",
                "policy": "smolvla",
                "query": request,
                "rollouts_executed": 1,
                "rollout_budget": 2,
                "official_success": False,
                "custom_rollout_authorized": False,
                "stop_reason": "official_control_failed",
                "method_chain_valid": False,
                "paper_performance_evidence": False,
                "scientific_evidence_eligible": False,
                "retrieval": query_retrieval.to_dict(),
                "policy_task_compatibility": compatibility.to_dict(),
                "controlled_change_contract": pending_change.to_dict(),
                "raw_run_dir": str(root),
            }
            _write_json(root / "compact_result.json", result)
            return result

        planner = ClaimFirstOpenQueryAgent(provider, model=planner_model)
        first_bundle = planner.propose(
            request,
            capabilities=_capabilities(checkpoint_path, official_contract),
            evidence_history=[control_evidence],
        )
        _persist_planner_bundle(root, "after_control", planner, first_bundle)
        if first_bundle["proposal"]["action"] != "continue":
            raise RuntimeError(
                "ClaimFirst stopped after control; no provider-authored custom task was authorized"
            )
        proposal = first_bundle["proposal"]
        planner_concern = " ".join(
            str(item)
            for item in (
                proposal.get("sub_aspect", ""),
                proposal.get("hypothesis", ""),
                proposal.get("requested_perturbation", {}).get("description", ""),
            )
            if item
        )
        planner_retrieval = BDDLTaskIndex.from_libero_suite(
            official_contract.suite
        ).retrieve_nearest(
            planner_concern or request,
            compatibility=compatibility,
        )
        controlled_change = authorize_controlled_change(
            planner_retrieval,
            first_bundle,
        )
        _write_json(
            root / "planner" / "after_control" / "taskgen_gate.json",
            {
                "retrieval": planner_retrieval.to_dict(),
                "policy_task_compatibility": compatibility.to_dict(),
                "controlled_change_contract": controlled_change.to_dict(),
            },
        )
        if not controlled_change.authorized:
            result = _planner_taskgen_misaligned_result(
                request=request,
                root=root,
                official_success=official_record.success,
                retrieval=planner_retrieval,
                compatibility=compatibility,
                change_contract=controlled_change,
            )
            _write_json(root / "compact_result.json", result)
            return result
        controlled_change.require_authorized()

        taskgen = LiberoTaskGenBackend(provider, model=taskgen_model)
        custom_backend = LiberoMethodBackend(
            benchmark_adapter=benchmark,
            policy_adapter=policy,
            taskgen_backend=taskgen,
            task_contract_factory=build_official_task_contract,
        )
        custom_runtime = MethodRuntime(custom_backend)
        custom_candidate = custom_runtime.materialize_candidate(
            official_binding,
            CandidateRequest(
                candidate_id="generated_custom_task",
                source_query=request,
                proposal_bundle=first_bundle,
                output_dir=root / "round_02_custom" / "taskgen",
                seed=seed,
                context={
                    "retrieval": planner_retrieval,
                    "change_contract": controlled_change,
                },
            ),
        )
        _write_json(
            root / "runtime" / "round_02_candidate.json",
            custom_candidate.to_dict(),
        )
        custom_contract = custom_candidate.native_task
        taskgen_result = custom_candidate.metadata["taskgen_result"]
        custom_contract_path = Path(taskgen_result["artifacts"]["task_contract"])
        probe = benchmark.render_and_init_probe(
            benchmark.make_custom_env(custom_contract),
            seed=seed,
            output_png=root / "round_02_custom" / "gate" / "first_frame.png",
        )
        _write_json(root / "round_02_custom" / "gate" / "compatibility_probe.json", probe)

        custom_rollout = custom_runtime.rollout(
            custom_candidate,
            RolloutRequest(
                round_id="round_02_custom_bddl",
                seed=seed,
                output_dir=root / "round_02_custom" / "episode",
                provenance={
                    "round": 2,
                    "route": "custom_offscreen_render_env_factory",
                    "stock_task_ids": False,
                    "official_init_state_reused_after_probe": True,
                    "horizon_steps": HORIZON_STEPS,
                },
            ),
        )
        _write_json(
            root / "runtime" / "round_02_rollout.json",
            custom_rollout.to_dict(),
        )
        custom_record = custom_rollout.native_episode
        rollouts_executed += 1

        tool_backend = LiberoPredicateToolBackend(registry_dir=root / "tool_registry")
        tool_result, tool_execution = tool_backend.compile_validate_register(
            output_dir=root / "round_02_custom" / "tool",
            episode_record=custom_record,
            goal_predicates=custom_contract.goal_predicates,
            source_query=first_bundle["proposal"]["tool_need"]["description"]
            or request,
        )
        aggregate = aggregate_tool_executions(
            [tool_execution],
            output_path=root / "round_02_custom" / "aggregate" / "aggregate_result.json",
        )
        reuse = tool_backend.exact_reuse(
            output_dir=root / "reuse_query",
            episode_record=custom_record,
            goal_predicates=custom_contract.goal_predicates,
            source_query=(
                "Using the already validated observable, did the generated "
                "LIBERO goal predicate succeed?"
            ),
        )
        tool_value = bool(
            tool_result["tool_execution"]["episodes"][0]["result"]["value"]
        )
        # The generated predicate Tool owns the experimental outcome even when
        # its value happens to agree with the benchmark termination signal.
        # Preserve the native episode while making that authority explicit in
        # the rich runtime evidence.
        evidence_rollout = replace(
            custom_rollout,
            success=tool_value,
            metadata={
                **custom_rollout.metadata,
                "outcome_authority": "generated_predicate_tool",
                "benchmark_success_agrees": (
                    tool_value == custom_rollout.success
                ),
            },
        )
        custom_evidence_record = custom_runtime.evidence(
            evidence_rollout,
            EvidenceRequest(
                sub_aspect=str(first_bundle["proposal"]["sub_aspect"]),
                hypothesis=str(first_bundle["proposal"]["hypothesis"]),
                perturbation=str(
                    first_bundle["proposal"]["requested_perturbation"][
                        "description"
                    ]
                ),
                summary=(
                    "Provider-written BDDL selected "
                    f"{taskgen_result['selected_object']}; live predicate="
                    f"{custom_record.goal_predicate_satisfied}; "
                    f"steps={custom_record.executed_steps}; compiled MetricSpec "
                    "adapter value entered Aggregate; outcome authority="
                    "generated_predicate_tool."
                ),
                limitations=(
                    "N=1 fixed seed",
                    "one state-compatible generated object-identity variation",
                ),
            ),
        )
        _write_json(
            root / "runtime" / "round_02_evidence.json",
            custom_evidence_record.to_dict(),
        )
        custom_evidence = custom_evidence_record.to_planner_dict()
        _write_json(root / "round_02_custom" / "evidence.json", custom_evidence)

        second_bundle: dict[str, Any] | None = None
        second_error: str | None = None
        try:
            second_bundle = planner.propose(
                request,
                capabilities=_capabilities(checkpoint_path, official_contract),
                evidence_history=[control_evidence, custom_evidence],
            )
            _persist_planner_bundle(root, "after_custom", planner, second_bundle)
        except Exception as exc:
            second_error = f"{type(exc).__name__}: {exc}"
            _write_json(
                root / "planner" / "after_custom" / "error.json",
                {"status": "failed", "error": second_error},
            )

        alternative_objects = sorted(
            item
            for values in custom_contract.objects.values()
            for item in values
            if item not in {"basket_1", "alphabet_soup_1", taskgen_result["selected_object"]}
        )
        if second_bundle is not None and second_bundle["proposal"]["action"] == "stop":
            sufficiency = {
                "evidence_sufficient": True,
                "should_stop": True,
                "stop_reason": "evidence_sufficient",
                "claim_verdict": str(second_bundle["proposal"]["hypothesis"]),
            }
        else:
            sufficiency = {
                "evidence_sufficient": False,
                "should_stop": True,
                "stop_reason": "budget_exhausted",
                "claim_verdict": (
                    "The official control and one generated variation are "
                    "insufficient to establish object-identity robustness."
                ),
            }
        sufficiency.update(
            {
                "observed_candidate_ids": [
                    "official_control",
                    custom_candidate.candidate_id,
                ],
                "untested_candidate_ids": alternative_objects,
                # A control/custom outcome difference is the tested effect, not
                # an evidence-source conflict. Reserve this field for genuine
                # Rule/VQA disagreement.
                "conflict_candidate_ids": [],
            }
        )
        evidence_packet = {
            "schema_version": 1,
            "request": request,
            "total_episodes": 2,
            "seeds": [seed],
            "rounds": [
                {"num_episodes": 1, "episodes": [official_record.to_dict()]},
                {"num_episodes": 1, "episodes": [custom_record.to_dict()]},
            ],
            "aggregate": aggregate,
            "query_sufficiency": sufficiency,
            "query_contract_sufficient": bool(
                sufficiency["evidence_sufficient"]
            ),
            "observations": {
                "pipeline_passed": True,
                "execution_vqa_conflict": False,
            },
            "rollout_budget": {"used": rollouts_executed, "maximum": 2},
            "tool_exact_reuse": reuse,
            "planner_after_custom_error": second_error,
        }
        _write_json(root / "evidence_packet.json", evidence_packet)
        answer_scope = build_answer_scope(evidence_packet)
        _write_json(root / "answer_scope.json", answer_scope)

        conclusion = sufficiency["claim_verdict"]
        compatibility_probe_passed = all(
            bool(probe.get(key))
            for key in (
                "reset",
                "robot_state_present",
                "official_init_state_applied",
                "render_nonempty",
            )
        )
        exact_reuse = reuse["route"] == "exact_registry_reuse"
        episode_protocol_matches = bool(
            official_record.seed == custom_record.seed == seed
            and official_record.horizon_steps
            == custom_record.horizon_steps
            == HORIZON_STEPS
        )
        method_chain_valid = _method_chain_is_valid(
            official_success=official_record.success,
            rollouts_executed=rollouts_executed,
            planner_taskgen_alignment=bool(
                taskgen_result["planner_taskgen_alignment"]
            ),
            compatibility_probe_passed=compatibility_probe_passed,
            aggregate_status=str(aggregate["status"]),
            exact_reuse=exact_reuse,
            final_planner_bundle_present=second_bundle is not None,
            episode_protocol_matches=episode_protocol_matches,
        )
        compact = {
            "schema_version": 1,
            "status": "completed",
            "benchmark": "libero",
            "policy": "smolvla",
            "query": request,
            "conclusion": conclusion,
            "rollouts_executed": rollouts_executed,
            "rollout_budget": 2,
            "horizon_steps_each": HORIZON_STEPS,
            "seed": seed,
            "official_success": official_record.success,
            "custom_success": custom_record.success,
            "custom_goal_object": taskgen_result["selected_object"],
            "provider_written_bddl": True,
            "custom_env_route": "OffScreenRenderEnv via explicit custom factory",
            "stock_task_id_faked_for_custom": False,
            "tool_live_value": custom_record.goal_predicate_satisfied,
            "tool_exact_reuse": exact_reuse,
            "reuse_additional_rollouts": reuse["additional_rollouts"],
            "aggregate_status": aggregate["status"],
            "planner_final_action": (
                second_bundle["proposal"]["action"] if second_bundle else None
            ),
            "planner_taskgen_alignment": bool(
                taskgen_result["planner_taskgen_alignment"]
            ),
            "query_contract_sufficient": bool(
                sufficiency["evidence_sufficient"]
            ),
            "method_chain_valid": method_chain_valid,
            "episode_protocol_matches": episode_protocol_matches,
            "scientific_evidence_eligible": False,
            "paper_performance_evidence": False,
            "query_sufficiency": sufficiency,
            "answer_scope": answer_scope,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "raw_run_dir": str(root),
            "artifacts": {
                "gate0": str(root / "gate0.json"),
                "official_episode": str(
                    root / "round_01_official" / "episode" / "episode.json"
                ),
                "planner_after_control": str(
                    root / "planner" / "after_control" / "proposal_bundle.json"
                ),
                "generated_bddl": custom_contract.bddl_path,
                "custom_task_contract": str(custom_contract_path),
                "custom_first_frame": probe["render_path"],
                "custom_episode": str(
                    root / "round_02_custom" / "episode" / "episode.json"
                ),
                "toolgen": str(
                    root / "round_02_custom" / "tool" / "toolgen_result.json"
                ),
                "aggregate": str(
                    root
                    / "round_02_custom"
                    / "aggregate"
                    / "aggregate_result.json"
                ),
                "reuse": str(root / "reuse_query" / "tool_reuse_result.json"),
                "evidence_packet": str(root / "evidence_packet.json"),
                "answer_scope": str(root / "answer_scope.json"),
                "runtime_task_binding": str(
                    root / "runtime" / "task_binding.json"
                ),
                "runtime_official_candidate": str(
                    root / "runtime" / "round_01_candidate.json"
                ),
                "runtime_official_rollout": str(
                    root / "runtime" / "round_01_rollout.json"
                ),
                "runtime_official_evidence": str(
                    root / "runtime" / "round_01_evidence.json"
                ),
                "runtime_custom_candidate": str(
                    root / "runtime" / "round_02_candidate.json"
                ),
                "runtime_custom_rollout": str(
                    root / "runtime" / "round_02_rollout.json"
                ),
                "runtime_custom_evidence": str(
                    root / "runtime" / "round_02_evidence.json"
                ),
            },
        }
        _write_json(root / "compact_result.json", compact)
        report = (
            f"# {evaluation_id}\n\n"
            f"- Query: {request}\n"
            f"- Official control: success={official_record.success}, "
            f"steps={official_record.executed_steps}\n"
            f"- Generated goal: {taskgen_result['selected_object']}, "
            f"success={custom_record.success}, steps={custom_record.executed_steps}\n"
            f"- Stop: {sufficiency['stop_reason']}\n"
            f"- Conclusion: {conclusion}\n"
            "- Scope: two N=1 rollouts at one seed; not a LIBERO benchmark result.\n"
        )
        (root / "README.md").write_text(report, encoding="utf-8")
        return compact
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "rollouts_executed": rollouts_executed,
            "rollout_budget": 2,
            "method_chain_valid": False,
            "paper_performance_evidence": False,
            "scientific_evidence_eligible": False,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "raw_run_dir": str(root),
        }
        _write_json(root / "compact_result.json", failure)
        raise
    finally:
        policy.unload()


def run_libero_agent_cli(args: Any) -> None:
    evaluation_id = args.evaluation_id or "eval_batch24_libero_method_chain_v1"
    bound_suite, bound_task_id = parse_bound_libero_task(
        getattr(args, "bound_task_name", None)
    )
    result = run_libero_method_chain(
        repo_root=args.repo_root,
        request=args.request,
        evaluation_id=evaluation_id,
        checkpoint=getattr(args, "libero_checkpoint", DEFAULT_CHECKPOINT),
        seed=(
            args.start_seed
            if args.start_seed is not None
            else getattr(args, "libero_seed", DEFAULT_SEED)
        ),
        planner_model=args.planner_model or "gpt-4o-2024-11-20",
        taskgen_model=args.taskgen_model or "gpt-4o-2024-11-20",
        base_url=args.base_url,
        plan_only=args.plan_only,
        bound_suite=bound_suite,
        bound_task_id=bound_task_id,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
