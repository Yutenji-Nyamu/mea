import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mea.method_runtime import (
    BackendBindingRequest,
    CandidateRequest,
    EvidenceRequest,
    MaterializedCandidate,
    RolloutObservation,
    RolloutRequest,
    build_round_evidence,
)
from mea.plan_agent_application import (
    refresh_plan_agent_capabilities_from_runtime_context,
)
from mea.planner.context import build_planning_context
from mea.planner.experiment_candidate import build_experiment_candidate
from mea.planner.open_task_resolver import policy_task_scope_from_card
from mea.planner.policy_task_binding import (
    PolicyTaskBindingError,
    build_policy_task_binding,
)
from mea.planner.runtime_task_binding import (
    build_hyvla_policy_spec,
    build_runtime_open_world_evaluation_target,
    build_runtime_policy_task_manifest,
    build_smolvla_policy_spec,
)
from mea.robotwin.native_agent_round import (
    NativeAgentRoundError,
    _build_native_run_id,
    _execute_robotwin_method_round,
    _project_trusted_checker_outcome,
    execute_act_method_round,
    execute_hyvla_method_round,
    execute_smolvla_method_round,
)
from mea.robotwin.hyvla_rollout import HyVLARobotwinRolloutRunner
from mea.taskgen.attempts import CandidateUnexecutableError
from mea.taskgen.generic_backend import GenericTaskGenError
from mea.taskgen.provider_scene_checker import validate_provider_run_id
from mea.robotwin.runtime import RoboTwinMethodBackend
from mea.robotwin_task_context import resolve_robotwin_task_context
from mea.robotwin.smolvla_rollout import (
    SmolVLARolloutError,
    SmolVLARobotwinRolloutRunner,
    _checker_outcome_snapshot,
    _persist_telemetry_outcome_semantics,
    _require_generated_task_simulator_source,
    run_smolvla_robotwin_episode,
)
from mea.round_evidence import aggregate_sources


def _write_official_task(root, task_name: str, description: str) -> None:
    env = root / "envs" / f"{task_name}.py"
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text(
        f"class {task_name}:\n"
        "    def load_actors(self):\n"
        "        self.target = create_actor(modelname='target')\n\n"
        "    def check_success(self):\n"
        "        return False\n",
        encoding="utf-8",
    )
    instruction = (
        root
        / "description"
        / "task_instruction"
        / f"{task_name}.json"
    )
    instruction.parent.mkdir(parents=True, exist_ok=True)
    instruction.write_text(
        json.dumps({"full_description": description}),
        encoding="utf-8",
    )


def _task_context_probe(root, task_name: str) -> dict:
    source = root / "envs" / f"{task_name}.py"
    return {
        "schema_version": 1,
        "task_name": task_name,
        "official_source": f"envs/{task_name}.py",
        "official_source_sha256": hashlib.sha256(
            source.read_bytes()
        ).hexdigest(),
        "setup_success": True,
        "official_check_success_callable": True,
        "physics_timestep_seconds": 0.004,
        "action_dimension": 14,
        "actors": [
            {
                "task_attribute": "target",
                "scene_name": "target",
            }
        ],
        "observables": {
            "simulation_clock": True,
            "policy_action": True,
            "robot_tcp": {"left": True, "right": True},
            "contact_events": True,
        },
    }


def test_native_generated_task_run_id_is_stable_and_importable():
    run_id = _build_native_run_id("eval_open_query", "round_1", "act")

    assert run_id.startswith("run_native_act_")
    assert validate_provider_run_id(run_id) == run_id
    assert run_id == _build_native_run_id(
        "eval_open_query", "round_1", "act"
    )
    assert run_id != _build_native_run_id(
        "eval_open_query", "round_2", "act"
    )


def test_legacy_act_binding_remains_readable_without_new_scope_fields():
    binding = build_policy_task_binding(
        task_name="adjust_bottle",
        task_family="object_manipulation",
        policy={"name": "ACT", "language_conditioned": False},
        checkpoint={
            "policy_name": "ACT",
            "checkpoint_id": "act-adjust_bottle/demo_clean-50",
            "ready": True,
        },
    )

    assert binding["hooks"]["rollout"]["kind"] == "act_eval_mea"
    assert "task_scope" not in binding["checkpoint"]


def test_smolvla_manifest_binds_discovered_tasks_to_one_checkpoint(tmp_path):
    _write_official_task(tmp_path, "alpha_task", "move the alpha object")
    _write_official_task(tmp_path, "beta_task", "move the beta object")
    checkpoint = tmp_path / "smolvla"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")

    manifest = build_runtime_policy_task_manifest(
        tmp_path,
        build_smolvla_policy_spec(checkpoint),
    )

    assert manifest["task_count"] == 2
    assert [item["task_name"] for item in manifest["tasks"]] == [
        "alpha_task",
        "beta_task",
    ]
    bindings = [item["policy_task_binding"] for item in manifest["tasks"]]
    assert {item["checkpoint"]["checkpoint_id"] for item in bindings} == {
        "lerobot/smolvla_robotwin"
    }
    assert all(
        item["checkpoint"]["task_scope"] == "robotwin_official_tasks"
        for item in bindings
    )
    assert all(
        item["hooks"]["rollout"]["kind"] == "smolvla_robotwin"
        for item in bindings
    )
    assert [item["policy"]["task_instruction"] for item in bindings] == [
        "alpha task",
        "beta task",
    ]
    assert all(
        item["policy"]["task_schema_available"] is False
        for item in bindings
    )
    assert all(
        item["task_schema"] == {
            "path": None,
            "task_family": "robotwin_official_task",
            "available": False,
        }
        for item in bindings
    )


def test_hyvla_manifest_is_multitask_and_records_external_runtime(tmp_path):
    _write_official_task(tmp_path, "alpha_task", "move the alpha object")
    checkpoint = tmp_path / "hyvla"
    checkpoint.mkdir()
    for name in ("config.json", "model.safetensors", "norm_stats.pkl"):
        (checkpoint / name).write_bytes(b"artifact")
    source = tmp_path / "hyvla-source"
    python_env = tmp_path / "hyvla-env"
    (source / "robotwin_eval").mkdir(parents=True)
    (source / "robotwin_eval" / "deploy_policy.py").write_bytes(b"source")
    (python_env / "bin").mkdir(parents=True)
    (python_env / "bin" / "python").write_bytes(b"python")

    manifest = build_runtime_policy_task_manifest(
        tmp_path,
        build_hyvla_policy_spec(
            checkpoint,
            source_dir=source,
            python_env=python_env,
        ),
    )

    binding = manifest["tasks"][0]["policy_task_binding"]
    assert manifest["policy_backend"] == "hyvla"
    assert binding["hooks"]["rollout"] == {
        "kind": "hyvla_robotwin_external",
        "entrypoint": "mea.robotwin.hyvla_rollout",
        "task_name": "alpha_task",
    }
    assert binding["policy"]["server_management"] == "external_only"
    assert binding["policy"]["action_chunk_size"] == 6
    assert binding["policy"]["physics_timestep_seconds"] == 0.004
    assert binding["policy"]["policy_source_path"] == str(source.resolve())
    assert binding["policy"]["policy_python"] == str(
        (python_env / "bin" / "python").resolve()
    )


def test_method_backend_allows_schema_less_official_control(tmp_path):
    _write_official_task(tmp_path, "alpha_task", "move the alpha object")
    probe_calls = []

    def probe_context(**kwargs):
        probe_calls.append(kwargs)
        return _task_context_probe(tmp_path, "alpha_task")

    backend = RoboTwinMethodBackend(
        repo_root=tmp_path,
        rollout_runner=lambda **_: {},
        task_context_probe_runner=probe_context,
    )

    binding = backend.bind_task(
        BackendBindingRequest(
            task_reference={
                "task_name": "alpha_task",
                "policy": {
                    "name": "SmolVLA",
                    "backend": "smolvla",
                    "action_dimension": 14,
                },
            }
        )
    )
    candidate = backend.official_candidate(
        binding,
        source_query="Can this policy solve the official task?",
        seed=7,
    )

    assert binding.task_contract["task_schema_available"] is False
    assert len(probe_calls) == 1
    assert probe_calls[0]["seed"] == 7
    assert candidate.validation == {
        "route": "official_control",
        "task_context": {
            "schema_origin": "runtime_probe",
            "runtime_probe_executed": True,
        },
    }
    assert candidate.task_contract["task_module"] == "envs.alpha_task"
    assert candidate.task_contract["task_schema_available"] is True
    assert (
        candidate.task_contract["task_context"]["schema_origin"]
        == "runtime_probe"
    )
    assert candidate.metadata["task_context_bound_before_rollout"] is True

    official_query = build_experiment_candidate(
        source_query="Can it solve only the official task?",
        base_task="alpha_task",
        semantic_concern="official task completion",
        rule_tool_need={
            "kind": "reuse",
            "description": "Reuse official check_success().",
            "reuse_first": True,
        },
    )
    materialized = backend.materialize_candidate(
        binding,
        CandidateRequest(
            candidate_id=official_query["candidate_id"],
            source_query=official_query["source_query"],
            proposal_bundle=official_query,
            seed=1,
            output_dir=tmp_path / "official_query",
        ),
    )
    assert materialized.validation["route"] == "official_task_tool_only"
    assert materialized.metadata["official_task_reused"] is True
    assert materialized.metadata["task_context_bound_before_rollout"] is True
    assert materialized.task_contract["task_schema_available"] is True


def test_runtime_task_context_refreshes_next_plan_agent_capabilities(
    tmp_path,
):
    _write_official_task(tmp_path, "alpha_task", "move the alpha object")
    context = resolve_robotwin_task_context(
        tmp_path,
        "alpha_task",
        runtime_probe=_task_context_probe(tmp_path, "alpha_task"),
    ).to_dict()
    capabilities = {
        "schema_version": 2,
        "policy_card": {
            "policy_name": "SmolVLA",
            "unknown_metadata": ["semantic_actor_schema"],
        },
        "simulator_card": {
            "task_name": "alpha_task",
            "tracked_actors": [],
            "success_contract": {
                "authority": "official_task_check_success_only",
                "semantic_telemetry_available": False,
            },
        },
        "generation_card": {
            "backend_primitives": {
                "scene": True,
                "checker": True,
                "telemetry": False,
                "rule": True,
                "vqa": True,
                "retrieve": True,
                "generate": True,
            }
        },
    }

    refreshed = refresh_plan_agent_capabilities_from_runtime_context(
        capabilities,
        {
            "generation_kind": "official_passthrough",
            "task_module": "envs.alpha_task",
            "runtime_task_context": context,
        },
    )

    assert refreshed["generation_card"]["backend_primitives"][
        "telemetry"
    ] is True
    assert refreshed["simulator_card"]["tracked_actors"][0]["id"] == (
        "target"
    )
    assert refreshed["simulator_card"]["success_contract"][
        "authority"
    ] == "official_check_success_runtime_callable"
    assert refreshed["simulator_card"]["task_context_authority"][
        "schema_origin"
    ] == "runtime_probe"
    assert refreshed["policy_card"]["unknown_metadata"] == []


def test_generated_round_does_not_promote_official_base_task_context(
    tmp_path,
):
    _write_official_task(tmp_path, "alpha_task", "move the alpha object")
    context = resolve_robotwin_task_context(
        tmp_path,
        "alpha_task",
        runtime_probe=_task_context_probe(tmp_path, "alpha_task"),
    ).to_dict()
    capabilities = {
        "schema_version": 2,
        "policy_card": {
            "policy_name": "SmolVLA",
            "unknown_metadata": ["semantic_actor_schema"],
        },
        "simulator_card": {
            "task_name": "alpha_task",
            "tracked_actors": [],
            "success_contract": {
                "authority": "official_task_check_success_only",
                "semantic_telemetry_available": False,
            },
        },
        "generation_card": {
            "backend_primitives": {
                "scene": True,
                "checker": True,
                "telemetry": False,
                "rule": True,
                "vqa": True,
                "retrieve": True,
                "generate": True,
            }
        },
    }

    retained = refresh_plan_agent_capabilities_from_runtime_context(
        capabilities,
        {
            "generation_kind": "generic_provider_scene_checker_codegen",
            "task_module": "mea.generated_tasks.run_alpha.task",
            "runtime_task_context": context,
        },
    )

    assert retained == capabilities


def test_native_schema_less_control_persists_runtime_task_context(tmp_path):
    _write_official_task(tmp_path, "alpha_task", "move the alpha object")
    checkpoint = tmp_path / "smolvla"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    target = build_runtime_open_world_evaluation_target(
        tmp_path,
        "alpha_task",
        max_rounds=2,
        policy_spec=build_smolvla_policy_spec(checkpoint),
    )

    def rollout_runner(*, candidate, **_kwargs):
        assert candidate.task_contract["task_schema_available"] is True
        return {
            "success": True,
            "episode": {"official_check_success": True},
            "artifacts": {},
            "metadata": {"semantic_telemetry_ready": True},
        }

    with patch(
        "mea.robotwin.runtime.probe_official_robotwin_task_context",
        return_value=_task_context_probe(tmp_path, "alpha_task"),
    ) as probe:
        result = _execute_robotwin_method_round(
            policy_backend="smolvla",
            policy_name="SmolVLA",
            rollout_runner=rollout_runner,
            repo_root=tmp_path,
            evaluation_dir=tmp_path / "evaluation",
            evaluation_id="eval_schema_less_control",
            round_plan={
                "round_id": "round_1",
                "template_id": "task_execution.official_baseline",
                "sub_aspect": "task_execution.official_baseline",
                "task_instruction": "Can the policy solve the official task?",
                "task_name": "alpha_task",
                "route": "official",
                "execution": {"seeds": [11]},
            },
            runtime_target=target,
            telemetry_profile="balanced_v1",
        )

    probe.assert_called_once()
    manifest = result["child_manifest"]
    assert manifest["runtime_task_context"]["schema_origin"] == (
        "runtime_probe"
    )
    assert manifest["method_runtime"]["candidate"]["task_contract"][
        "task_schema_available"
    ] is True
    assert manifest["method_runtime"]["candidate"]["metadata"][
        "runtime_task_context_probe_executed"
    ] is True


def test_policy_backend_rejects_a_mismatched_rollout_hook():
    with pytest.raises(PolicyTaskBindingError, match="policy backend"):
        build_policy_task_binding(
            task_name="alpha_task",
            task_family="robotwin_official_task",
            policy={"name": "SmolVLA", "backend": "smolvla"},
            checkpoint={
                "policy_name": "SmolVLA",
                "checkpoint_id": "lerobot/smolvla_robotwin",
                "task_scope": "robotwin_official_tasks",
                "ready": True,
            },
            rollout={
                "kind": "act_eval_mea",
                "entrypoint": "policy/ACT/eval_mea.sh",
                "task_name": "alpha_task",
            },
        )


def test_schema_less_smolvla_context_exposes_official_only_boundary(
    tmp_path,
):
    _write_official_task(tmp_path, "alpha_task", "move the alpha object")
    checkpoint = tmp_path / "smolvla"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    target = build_runtime_open_world_evaluation_target(
        tmp_path,
        "alpha_task",
        max_rounds=2,
        policy_spec=build_smolvla_policy_spec(checkpoint),
    )

    context = build_planning_context(tmp_path, target)

    assert context["policy_card"]["policy_name"] == "SmolVLA"
    assert context["policy_card"]["single_task_checkpoint"] is False
    assert policy_task_scope_from_card(context["policy_card"])[
        "training_tasks"
    ] == ["alpha_task"]
    assert context["policy_card"]["supports_unseen_tasks"] is False
    assert context["policy_card"]["expert_data_num"] is None
    assert "expert_data_num" in context["policy_card"]["unknown_metadata"]
    assert context["simulator_card"]["tracked_actors"] == []
    assert context["simulator_card"]["success_contract"] == {
        "authority": "official_task_check_success_only",
        "semantic_telemetry_available": False,
    }


def test_schema_less_smolvla_delegates_task_context_to_taskgen(
    tmp_path,
):
    _write_official_task(tmp_path, "alpha_task", "move the alpha object")
    checkpoint = tmp_path / "smolvla"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    target = build_runtime_open_world_evaluation_target(
        tmp_path,
        "alpha_task",
        max_rounds=2,
        policy_spec=build_smolvla_policy_spec(checkpoint),
    )
    query = "Measure the target trajectory."
    candidate = build_experiment_candidate(
        source_query=query,
        base_task="alpha_task",
        semantic_concern="trajectory",
        scene_need="Move the target to a new tested pose.",
    )

    with pytest.raises(
        NativeAgentRoundError,
        match="requires provider, text model, and vision model",
    ):
        execute_smolvla_method_round(
            repo_root=tmp_path,
            evaluation_dir=tmp_path / "evaluation",
            evaluation_id="eval",
            round_plan={
                "round_id": "round_1",
                "template_id": None,
                "candidate_id": candidate["candidate_id"],
                "proposal": candidate,
                "sub_aspect": "trajectory",
                "task_instruction": query,
                "task_name": "alpha_task",
                "route": "official",
                "execution": {
                    "backend": "act",
                    "seeds": [1],
                    "num_episodes": 1,
                },
            },
            runtime_target=target,
            telemetry_profile="balanced_v1",
            policy_server_port=18771,
            generated_task_materializer=lambda *args, **kwargs: {},
        )


def test_act_native_envelope_delegates_to_shared_method_round(tmp_path):
    expected = {"unsupported": False}
    with patch(
        "mea.robotwin.native_agent_round._execute_robotwin_method_round",
        return_value=expected,
    ) as execute_shared:
        result = execute_act_method_round(
            repo_root=tmp_path,
            evaluation_dir=tmp_path / "evaluation",
            evaluation_id="eval",
            round_plan={"round_id": "round_1"},
            runtime_target={},
            telemetry_profile="balanced_v1",
            policy_server_port=18771,
            gpu=2,
        )

    assert result is expected
    call = execute_shared.call_args.kwargs
    assert call["policy_backend"] == "act"
    assert call["policy_name"] == "ACT"
    assert call["telemetry_profile"] == "balanced_v1"
    assert call["rollout_runner"].gpu == 2
    assert call["rollout_runner"].repo_root == tmp_path.resolve()
    assert call["generated_task_materializer"] is None
    assert call["execution_vqa_connected"] is True


def test_smolvla_native_envelope_accepts_shared_taskgen_and_vqa(tmp_path):
    expected = {"unsupported": False}
    materializer = lambda *args, **kwargs: {}
    with patch(
        "mea.robotwin.native_agent_round._execute_robotwin_method_round",
        return_value=expected,
    ) as execute_shared:
        result = execute_smolvla_method_round(
            repo_root=tmp_path,
            evaluation_dir=tmp_path / "evaluation",
            evaluation_id="eval",
            round_plan={"round_id": "round_1"},
            runtime_target={},
            telemetry_profile="balanced_v1",
            policy_server_port=18771,
            generated_task_materializer=materializer,
        )

    assert result is expected
    call = execute_shared.call_args.kwargs
    assert call["policy_backend"] == "smolvla"
    assert call["generated_task_materializer"] is materializer
    assert call["execution_vqa_connected"] is True


def test_hyvla_native_envelope_reuses_shared_method_round(tmp_path):
    expected = {"unsupported": False}
    materializer = lambda *args, **kwargs: {}
    with patch(
        "mea.robotwin.native_agent_round._execute_robotwin_method_round",
        return_value=expected,
    ) as execute_shared:
        result = execute_hyvla_method_round(
            repo_root=tmp_path,
            evaluation_dir=tmp_path / "evaluation",
            evaluation_id="eval",
            round_plan={"round_id": "round_1"},
            runtime_target={},
            telemetry_profile="balanced_v1",
            policy_server_port=18781,
            generated_task_materializer=materializer,
        )

    assert result is expected
    call = execute_shared.call_args.kwargs
    assert call["policy_backend"] == "hyvla"
    assert call["policy_name"] == "Hy-VLA"
    assert isinstance(call["rollout_runner"], HyVLARobotwinRolloutRunner)
    assert call["rollout_runner"].port == 18781
    assert call["generated_task_materializer"] is materializer
    assert call["execution_vqa_connected"] is True


def test_native_taskgen_failure_leaves_compact_child_manifest(tmp_path):
    _write_official_task(
        tmp_path,
        "alpha_task",
        "move the alpha object",
    )
    candidate = build_experiment_candidate(
        source_query="Generate one shifted scene.",
        base_task="alpha_task",
        semantic_concern="shift robustness",
        scene_need="Shift the target laterally.",
    )
    contract = {
        "task_name": "alpha_task",
        "policy": {
            "name": "SmolVLA",
            "backend": "smolvla",
            "action_dimension": 14,
        },
        "checkpoint": {
            "checkpoint_id": "fixture-smolvla",
            "checkpoint_path": str(tmp_path / "smolvla"),
        },
        "task_schema": {"available": True},
    }

    def fail_taskgen(*_args, **_kwargs):
        raise GenericTaskGenError("provider repair exhausted")

    with patch(
        "mea.robotwin.native_agent_round.policy_task_binding_from_target",
        return_value=contract,
    ), pytest.raises(GenericTaskGenError, match="repair exhausted"):
        _execute_robotwin_method_round(
            policy_backend="smolvla",
            policy_name="SmolVLA",
            rollout_runner=lambda **_: {},
            repo_root=tmp_path,
            evaluation_dir=tmp_path / "evaluation",
            evaluation_id="eval_failure",
            round_plan={
                "round_id": "round_1",
                "candidate_id": candidate["candidate_id"],
                "proposal": candidate,
                "task_instruction": candidate["source_query"],
                "task_name": "alpha_task",
                "route": "generated",
                "execution": {"seeds": [1]},
            },
            runtime_target={},
            telemetry_profile="balanced_v1",
            provider=object(),
            text_model="fixture-model",
            vision_model="fixture-model",
            generated_task_materializer=fail_taskgen,
        )

    run_id = _build_native_run_id(
        "eval_failure",
        "round_1",
        "smolvla",
    )
    manifest = json.loads(
        (
            tmp_path / "mea" / "generated_tasks" / run_id / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["failure"]["message"] == "provider repair exhausted"


def test_candidate_unexecutable_returns_planning_evidence_without_rollout(
    tmp_path,
):
    _write_official_task(
        tmp_path,
        "alpha_task",
        "move the alpha object",
    )
    candidate = build_experiment_candidate(
        source_query="Find a feasible spatial stress test.",
        base_task="alpha_task",
        semantic_concern="target lateral displacement",
        scene_need="Shift the target laterally by 0.12 m.",
    )
    contract = {
        "task_name": "alpha_task",
        "policy": {
            "name": "SmolVLA",
            "backend": "smolvla",
            "action_dimension": 14,
        },
        "checkpoint": {
            "checkpoint_id": "fixture-smolvla",
            "checkpoint_path": str(tmp_path / "smolvla"),
        },
        "task_schema": {"available": True},
    }
    summary = {
        "status": "failed",
        "runtime": {
            "provider_calls": 2,
            "simulator_probes": 4,
            "expert_probes": 8,
            "act_rollouts_started": 0,
        },
        "attempts": [
            {
                "failure": {
                    "stage": "expert_gate",
                    "failure_kind": "candidate_unexecutable",
                    "message": (
                        "generated scene/expert failed official terminal-state "
                        "authority: target_pose cannot be None"
                    ),
                }
            }
        ],
    }

    def reject_candidate(*_args, **_kwargs):
        raise CandidateUnexecutableError(
            "TaskGen recovery failed after 2 attempt(s)",
            summary=summary,
        )

    rollout_calls = []

    def forbidden_rollout(**kwargs):
        rollout_calls.append(kwargs)
        raise AssertionError("policy rollout must not start")

    with patch(
        "mea.robotwin.native_agent_round.policy_task_binding_from_target",
        return_value=contract,
    ):
        result = _execute_robotwin_method_round(
            policy_backend="smolvla",
            policy_name="SmolVLA",
            rollout_runner=forbidden_rollout,
            repo_root=tmp_path,
            evaluation_dir=tmp_path / "evaluation",
            evaluation_id="eval_candidate_unexecutable",
            round_plan={
                "round_id": "round_1",
                "candidate_id": candidate["candidate_id"],
                "proposal": candidate,
                "sub_aspect": "target lateral displacement",
                "task_instruction": candidate["source_query"],
                "task_name": "alpha_task",
                "route": "generic_provider_scene_checker_codegen",
                "execution": {"seeds": [1]},
            },
            runtime_target={},
            telemetry_profile="balanced_v1",
            provider=object(),
            text_model="fixture-model",
            vision_model="fixture-model",
            generated_task_materializer=reject_candidate,
        )

    assert rollout_calls == []
    assert result["candidate_unexecutable"] is True
    assert result["semantic_telemetry_ready"] is False
    manifest = result["child_manifest"]
    assert manifest["status"] == "candidate_unexecutable"
    assert manifest["act_evaluation"]["actual_seeds"] == []
    assert manifest["policy_execution"] == {
        "started": False,
        "rollouts_started": 0,
        "sample_count": 0,
    }
    assert (
        manifest["candidate_unexecutable"]["policy_sample_count"] == 0
    )


def test_smolvla_runner_enables_telemetry_only_for_schema_backed_candidate(
    tmp_path,
):
    candidate = MaterializedCandidate(
        benchmark="robotwin",
        candidate_id="official_control",
        binding_id="alpha_task/SmolVLA",
        source_query="Can it solve the official task?",
        task_contract={
            "policy": {
                "name": "SmolVLA",
                "backend": "smolvla",
                "task_instruction": "alpha task",
            },
            "task_schema_available": True,
        },
        native_task=object(),
    )
    runner = SmolVLARobotwinRolloutRunner(
        repo_root=tmp_path,
        telemetry_profile="balanced_v1",
    )
    with patch(
        "mea.robotwin.smolvla_rollout.run_smolvla_robotwin_episode",
        return_value={"success": True, "episode": {}},
    ) as rollout:
        runner(
            candidate=candidate,
            request=RolloutRequest(
                round_id="round_1",
                seed=1,
                output_dir=tmp_path / "episode",
            ),
            manifest={
                "task_name": "alpha_task",
                "task_module": "envs.alpha_task",
            },
        )

    assert rollout.call_args.kwargs["repo_root"] == tmp_path.resolve()
    assert rollout.call_args.kwargs["telemetry_profile"] == "balanced_v1"
    assert (
        rollout.call_args.kwargs["outcome_metric"]
        == "official_check_success"
    )


def test_generated_smolvla_task_requires_taskgen_simulator_source(tmp_path):
    external_package = tmp_path / "external" / "envs" / "__init__.py"
    with patch(
        "mea.robotwin.smolvla_rollout.importlib.import_module",
        return_value=SimpleNamespace(__file__=str(external_package)),
    ):
        with pytest.raises(
            SmolVLARolloutError,
            match="TaskGen was validated against",
        ):
            _require_generated_task_simulator_source(
                task_module="mea.generated_tasks.run_open.task",
                repo_root=tmp_path / "mea",
            )


def test_smolvla_initializes_simulator_before_policy_connection(tmp_path):
    class FailingTask:
        def setup_demo(self, **_kwargs):
            raise RuntimeError("invalid generated simulator API")

        def close_env(self, *, clear_cache):
            assert clear_cache is True

    module = SimpleNamespace(alpha_task=FailingTask)
    with (
        patch(
            "mea.robotwin.smolvla_rollout._resolved_demo_clean_args",
            return_value={},
        ),
        patch("mea.robotwin.smolvla_rollout._PolicyClient") as client,
        patch(
            "mea.robotwin.smolvla_rollout.importlib.import_module",
            return_value=module,
        ),
        pytest.raises(RuntimeError, match="invalid generated simulator API"),
    ):
        run_smolvla_robotwin_episode(
            task_name="alpha_task",
            task_module="envs.alpha_task",
            seed=1,
            output_dir=tmp_path / "episode",
        )

    client.assert_not_called()


def test_smolvla_separates_generated_official_and_latched_outcomes(
    tmp_path,
):
    task = SimpleNamespace(
        eval_success=True,
        check_success=lambda: False,
        mea_official_check_success=lambda: True,
    )

    outcomes = _checker_outcome_snapshot(
        task,
        active_checker_metric="generated_check_success",
    )
    persisted = _persist_telemetry_outcome_semantics(
        tmp_path,
        {"success": True, "task_name": "alpha_task"},
        outcomes,
    )

    assert outcomes == {
        "active_checker_metric": "generated_check_success",
        "active_checker_success": True,
        "active_checker_final_predicate": False,
        "generated_checker_success": True,
        "official_check_success": True,
        "official_core_predicate_satisfied": True,
        "episode_latched_success": True,
    }
    assert persisted["generated_checker_success"] is True
    assert json.loads(
        (tmp_path / "episode.json").read_text(encoding="utf-8")
    ) == persisted
    scene_only = _checker_outcome_snapshot(
        task,
        active_checker_metric="official_check_success",
    )
    assert scene_only["generated_checker_success"] is None
    assert scene_only["active_checker_metric"] == "official_check_success"


def test_native_method_evidence_uses_trusted_generated_checker_result():
    rollout = RolloutObservation(
        benchmark="robotwin",
        round_id="round_1",
        candidate_id="generated_candidate",
        seed=17,
        success=False,
        episode={
            "active_checker_metric": "generated_check_success",
            "generated_checker_success": True,
            "official_check_success": False,
            "official_core_predicate_satisfied": False,
            "episode_latched_success": True,
        },
        native_episode={},
    )
    trusted = {
        "status": "passed",
        "outcome_metric": "generated_check_success",
        "outcome_authority": "llm_generated_python_ast_validated",
        "episodes": [
            {
                "role": "policy_under_evaluation",
                "tool_results": [
                    {
                        "tool": "generated_check_success",
                        "value": True,
                        "passed": True,
                        "details": {
                            "generated_checker_success": True,
                            "official_core_predicate_satisfied": False,
                        },
                    }
                ],
            }
        ],
    }

    projected, result = _project_trusted_checker_outcome(
        rollout,
        trusted,
        expected_metric="generated_check_success",
        policy_backend="smolvla",
    )

    assert projected.success is True
    assert result["value"] is True
    assert projected.episode["official_check_success"] is False
    assert projected.metadata["trusted_checker"] == {
        "metric": "generated_check_success",
        "authority": "llm_generated_python_ast_validated",
        "value": True,
    }
    evidence = build_round_evidence(
        projected,
        EvidenceRequest(
            sub_aspect="trajectory robustness",
            hypothesis="The generated checker is satisfied.",
            perturbation="generated scene",
            summary="Trusted generated checker passed.",
        ),
    )
    sources = aggregate_sources(
        {
            "round_id": "round_1",
            "sub_aspect": "trajectory robustness",
        },
        {"trusted_tool_evaluation": trusted},
        None,
    )
    assert evidence.outcome == "success"
    assert sources[0]["episodes"][0]["tool_results"][0] is not None
    assert (
        sources[0]["episodes"][0]["tool_results"][0]["value"]
        is projected.success
    )
