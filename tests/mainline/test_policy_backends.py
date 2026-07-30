import json
from unittest.mock import patch

import pytest

from mea.method_runtime import (
    BackendBindingRequest,
    CandidateRequest,
    MaterializedCandidate,
    RolloutRequest,
)
from mea.planner.context import build_planning_context
from mea.planner.experiment_candidate import build_experiment_candidate
from mea.planner.open_task_resolver import policy_task_scope_from_card
from mea.planner.policy_task_binding import (
    PolicyTaskBindingError,
    build_policy_task_binding,
)
from mea.planner.runtime_task_binding import (
    build_runtime_open_world_evaluation_target,
    build_runtime_policy_task_manifest,
    build_smolvla_policy_spec,
)
from mea.robotwin.native_agent_round import (
    _build_native_run_id,
    _execute_robotwin_method_round,
    execute_act_method_round,
    execute_smolvla_method_round,
)
from mea.taskgen.generic_backend import GenericTaskGenError
from mea.taskgen.provider_scene_checker import validate_provider_run_id
from mea.robotwin.runtime import RoboTwinMethodBackend
from mea.robotwin.smolvla_rollout import SmolVLARobotwinRolloutRunner


def _write_official_task(root, task_name: str, description: str) -> None:
    env = root / "envs" / f"{task_name}.py"
    env.parent.mkdir(parents=True, exist_ok=True)
    env.write_text(
        f"class {task_name}:\n"
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


def test_method_backend_allows_schema_less_official_control(tmp_path):
    _write_official_task(tmp_path, "alpha_task", "move the alpha object")
    backend = RoboTwinMethodBackend(
        repo_root=tmp_path,
        rollout_runner=lambda **_: {},
    )

    binding = backend.bind_task(
        BackendBindingRequest(
            task_reference={
                "task_name": "alpha_task",
                "policy": {"name": "SmolVLA", "backend": "smolvla"},
            }
        )
    )
    candidate = backend.official_candidate(
        binding,
        source_query="Can this policy solve the official task?",
    )

    assert binding.task_contract["task_schema_available"] is False
    assert candidate.validation == {"route": "official_control"}
    assert candidate.task_contract["task_module"] == "envs.alpha_task"

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


def test_schema_less_smolvla_records_scene_generation_as_unsupported(
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

    result = execute_smolvla_method_round(
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
            "execution": {"backend": "act", "seeds": [1], "num_episodes": 1},
        },
        runtime_target=target,
        telemetry_profile="balanced_v1",
        policy_server_port=18771,
        generated_task_materializer=lambda *args, **kwargs: {},
    )

    assert result["unsupported"] is True
    assert result["child_manifest"]["status"] == "unsupported"
    assert (
        result["child_manifest"]["unsupported_capability"]["reason_code"]
        == "task_context_insufficient_for_taskgen"
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


def test_native_taskgen_failure_leaves_compact_child_manifest(tmp_path):
    candidate = build_experiment_candidate(
        source_query="Generate one shifted scene.",
        base_task="alpha_task",
        semantic_concern="shift robustness",
        scene_need="Shift the target laterally.",
    )
    contract = {
        "task_name": "alpha_task",
        "policy": {"name": "SmolVLA", "backend": "smolvla"},
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
