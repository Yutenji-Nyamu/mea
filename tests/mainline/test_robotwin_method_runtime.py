from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from mea.method_runtime import (
    BackendBindingRequest,
    CandidateRequest,
    EvidenceRequest,
    MaterializedCandidate,
    MethodRuntime,
    RolloutRequest,
)
from mea.planner.experiment_candidate import build_experiment_candidate
from mea.robotwin import (
    ACTRobotwinRolloutRunner,
    RoboTwinMethodBackend,
)
from mea.robotwin_task_context import resolve_robotwin_task_context
from mea.taskgen.generic_backend import (
    GenericRoboTwinTaskAdapter,
    GenericRoboTwinTaskGenBackend,
    GenericTaskGenHooks,
    build_generic_task_subclass_module,
    validate_generic_task_methods,
)
from mea.taskgen.rollout_evidence import (
    evaluate_generic_task_rollout_telemetry,
)


class _Provider:
    def __init__(self) -> None:
        self.calls = 0
        self.last_metadata: dict[str, Any] = {}

    def text(self, _prompt: str, **_kwargs: Any) -> str:
        self.calls += 1
        self.last_metadata = {"call": self.calls}
        return json.dumps(
            {
                "load_actors": (
                    "def load_actors(self):\n"
                    '    self.target = "generated"\n'
                ),
                "check_success": (
                    "def check_success(self):\n"
                    '    return self.target == "generated"\n'
                ),
            }
        )


def _write_task_context(root: Path) -> Path:
    source = root / "envs/runtime_task.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class runtime_task:\n"
        "    def load_actors(self):\n"
        '        self.target = "official"\n\n'
        "    def check_success(self):\n"
        '        return self.target == "official"\n',
        encoding="utf-8",
    )
    readme = root / "mea/taskgen/README.Agent.md"
    readme.parent.mkdir(parents=True)
    readme.write_text("Generate the requested task methods.\n", encoding="utf-8")
    documentation = root / "description/runtime_task.md"
    documentation.parent.mkdir(parents=True)
    documentation.write_text("A fixture RoboTwin task.\n", encoding="utf-8")
    asset = root / "assets/runtime_target.json"
    asset.parent.mkdir(parents=True)
    asset.write_text("{}\n", encoding="utf-8")
    return source


def _adapter(root: Path) -> GenericRoboTwinTaskAdapter:
    source = _write_task_context(root)

    def validate(
        methods: Mapping[str, str],
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        report = validate_generic_task_methods(
            methods,
            official_source=source,
            official_class="runtime_task",
            required_method_changes={
                "load_actors": candidate["scene_need"] is not None,
                "check_success": candidate["checker_need"] is not None,
            },
        )
        report["checker_fixtures"] = [
            {"name": "positive", "passed": True},
            {"name": "negative", "passed": True},
        ]
        return report

    return GenericRoboTwinTaskAdapter(
        task_name="runtime_task",
        official_source="envs/runtime_task.py",
        official_class="runtime_task",
        task_schema={
            "tracked_actors": ["target"],
            "signals": ["target_position", "success"],
        },
        documentation_paths=("description/runtime_task.md",),
        asset_paths=("assets/runtime_target.json",),
        hooks=GenericTaskGenHooks(
            validate_methods=validate,
            build_module=lambda methods, _candidate: (
                build_generic_task_subclass_module(
                    methods,
                    official_module="envs.runtime_task",
                    official_class="runtime_task",
                )
            ),
            preflight_candidate=(
                lambda _path, _source, _candidate: {
                    "render_passed": True,
                    "expert_passed": True,
                    "scene_change_passed": True,
                    "scene_change": {
                        "passed": True,
                        "expected_state": "changed",
                        "authority": "fixture_simulator_state",
                    },
                }
            ),
            resolve_metric=lambda candidate: candidate["rule_tool_need"],
            resolve_checker_contract=lambda candidate: {
                "semantic_concern": candidate["semantic_concern"],
            },
        ),
    )


def test_robotwin_runtime_bind_materialize_rollout_evidence_contract(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    provider = _Provider()
    observed: dict[str, Any] = {}

    def rollout_runner(
        *,
        candidate,
        request,
        manifest,
    ) -> Mapping[str, Any]:
        observed["candidate"] = candidate
        observed["request"] = request
        observed["manifest"] = manifest
        return {
            "success": True,
            "episode": {
                "generated_checker_success": True,
                "official_success": False,
                "telemetry_episode_count": 1,
            },
            "artifacts": {"video": "evaluation/episode0.mp4"},
            "metadata": {"policy": "ACT"},
        }

    backend = RoboTwinMethodBackend(
        repo_root=tmp_path,
        task_adapter_factory=lambda task_name: (
            adapter
            if task_name == "runtime_task"
            else (_ for _ in ()).throw(KeyError(task_name))
        ),
        taskgen_backend=GenericRoboTwinTaskGenBackend(
            tmp_path,
            provider,
            model="fixture-model",
        ),
        rollout_runner=rollout_runner,
    )
    runtime = MethodRuntime(backend)
    query = "Does a shifted target expose a weakness?"
    experiment_candidate = build_experiment_candidate(
        source_query=query,
        base_task="runtime_task",
        semantic_concern="target pose robustness",
        scene_need="Shift the target to another valid pose.",
        checker_need="Require completion at the generated pose.",
        rule_tool_need="Measure generated-checker success.",
        candidate_id="dynamic.runtime.pose",
    )

    binding = runtime.bind_task(
        BackendBindingRequest(
            task_reference={
                "task_name": "runtime_task",
                "policy": {"name": "ACT", "checkpoint": "server-bound"},
            }
        )
    )
    materialized = runtime.materialize_candidate(
        binding,
        CandidateRequest(
            candidate_id=experiment_candidate["candidate_id"],
            source_query=query,
            proposal_bundle=experiment_candidate,
            output_dir=tmp_path / "run_runtime_contract",
            seed=11,
        ),
    )
    rollout = runtime.rollout(
        materialized,
        RolloutRequest(
            round_id="round_01",
            seed=11,
            output_dir=tmp_path / "episode",
        ),
    )
    evidence = runtime.evidence(
        rollout,
        EvidenceRequest(
            sub_aspect="target pose robustness",
            hypothesis="The shifted target remains solvable.",
            perturbation="shift target pose",
            summary="The generated checker passed in the ACT episode.",
            limitations=("N=1", "official/generated checker conflict"),
        ),
    )

    assert provider.calls == 1
    assert binding.binding_id == "runtime_task/ACT"
    assert materialized.validation["route"] == (
        "generic_provider_scene_checker_codegen"
    )
    assert observed["manifest"]["task_module"].startswith(
        "mea.generated_tasks.run_runtime_contract"
    )
    assert Path(observed["manifest"]["overlay"]).is_file()
    assert rollout.episode["generated_checker_success"] is True
    assert rollout.metadata["taskgen_route"] == (
        "generic_provider_scene_checker_codegen"
    )
    assert evidence.outcome == "success"
    assert evidence.to_planner_dict()["round_id"] == "round_01"


def test_robotwin_runtime_tool_only_candidate_reuses_official_task(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)

    class ForbiddenTaskGen:
        def materialize(self, *_args, **_kwargs):
            raise AssertionError("Tool-only candidate must bypass TaskGen")

    observed: dict[str, Any] = {}

    def rollout_runner(*, candidate, request, manifest):
        observed["manifest"] = manifest
        return {
            "success": False,
            "episode": {"precontact_jerk_peak": 1.25},
        }

    runtime = MethodRuntime(
        RoboTwinMethodBackend(
            repo_root=tmp_path,
            task_adapter_factory=lambda _task_name: adapter,
            taskgen_backend=ForbiddenTaskGen(),
            rollout_runner=rollout_runner,
        )
    )
    query = "Is there pre-contact jerk?"
    candidate_value = build_experiment_candidate(
        source_query=query,
        base_task="runtime_task",
        semantic_concern="motion.precontact_jerk",
        rule_tool_need="Measure peak jerk before first contact.",
        candidate_id="dynamic.runtime.jerk",
    )
    binding = runtime.bind_task(
        BackendBindingRequest(task_reference={"task_name": "runtime_task"})
    )
    candidate = runtime.materialize_candidate(
        binding,
        CandidateRequest(
            candidate_id=candidate_value["candidate_id"],
            source_query=query,
            proposal_bundle=candidate_value,
            output_dir=tmp_path / "run_runtime_tool_only",
            seed=13,
        ),
    )
    rollout = runtime.rollout(
        candidate,
        RolloutRequest(
            round_id="round_01",
            seed=13,
            output_dir=tmp_path / "episode",
        ),
    )

    assert candidate.validation["route"] == "official_task_tool_only"
    assert observed["manifest"]["task_module"] == "envs.runtime_task"
    assert rollout.episode["precontact_jerk_peak"] == 1.25


def test_schema_less_tool_only_candidate_binds_reset_task_context_before_rollout(
    tmp_path: Path,
) -> None:
    source = tmp_path / "envs/source_only_task.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class source_only_task:\n"
        "    def load_actors(self):\n"
        "        self.target = create_actor(modelname='target')\n\n"
        "    def check_success(self):\n"
        "        return False\n",
        encoding="utf-8",
    )
    order: list[str] = []

    def probe_runner(**kwargs: Any) -> Mapping[str, Any]:
        order.append("probe")
        context = resolve_robotwin_task_context(
            kwargs["repo_root"],
            kwargs["task_name"],
        )
        return {
            "schema_version": 1,
            "task_name": kwargs["task_name"],
            "official_source": context.official_source,
            "official_source_sha256": context.official_source_sha256,
            "setup_success": True,
            "official_check_success_callable": True,
            "physics_timestep_seconds": 0.004,
            "action_dimension": kwargs["action_dimension"],
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

    def rollout_runner(*, candidate, request, manifest):
        order.append("rollout")
        assert candidate.task_contract["task_schema_available"] is True
        assert candidate.task_contract["task_context"]["schema_origin"] == (
            "runtime_probe"
        )
        assert candidate.task_contract["task_schema"][
            "telemetry_observables"
        ]["policy_action"]["dimension"] == 14
        return {
            "success": False,
            "episode": {"trajectory_metric": 0.25},
        }

    runtime = MethodRuntime(
        RoboTwinMethodBackend(
            repo_root=tmp_path,
            rollout_runner=rollout_runner,
            task_context_probe_runner=probe_runner,
        )
    )
    query = "Does this policy move abruptly before contact?"
    proposal = build_experiment_candidate(
        source_query=query,
        base_task="source_only_task",
        semantic_concern="trajectory.precontact_motion",
        rule_tool_need="Measure pre-contact trajectory motion.",
        candidate_id="dynamic.source_only.motion",
    )
    binding = runtime.bind_task(
        BackendBindingRequest(
            task_reference={
                "task_name": "source_only_task",
                "policy": {
                    "name": "SmolVLA",
                    "backend": "smolvla",
                    "action_dimension": 14,
                },
            }
        )
    )
    materialized = runtime.materialize_candidate(
        binding,
        CandidateRequest(
            candidate_id=proposal["candidate_id"],
            source_query=query,
            proposal_bundle=proposal,
            output_dir=tmp_path / "source_only_round",
            seed=31,
        ),
    )
    runtime.rollout(
        materialized,
        RolloutRequest(
            round_id="round_01",
            seed=31,
            output_dir=tmp_path / "episode",
        ),
    )

    assert order == ["probe", "rollout"]
    assert materialized.validation["route"] == "official_task_tool_only"
    assert materialized.metadata[
        "task_context_bound_before_rollout"
    ] is True
    context_path = Path(materialized.artifacts["task_context"])
    assert context_path.is_file()
    context = json.loads(context_path.read_text(encoding="utf-8"))
    assert context["telemetry_observables"]["contact_events"][
        "scope"
    ] == "declared_contact_focus_actors"


def test_runtime_binds_accepted_scene_with_official_checker(
    tmp_path: Path,
) -> None:
    adapter = _adapter(tmp_path)
    backend = RoboTwinMethodBackend(
        repo_root=tmp_path,
        task_adapter_factory=lambda _task_name: adapter,
        rollout_runner=lambda **_kwargs: {},
    )
    binding = backend.bind_task(
        BackendBindingRequest(
            task_reference={
                "task_name": "runtime_task",
                "policy": {"name": "ACT", "backend": "act"},
            }
        )
    )
    query = "Does a new target appearance preserve task completion?"
    proposal = build_experiment_candidate(
        source_query=query,
        base_task="runtime_task",
        semantic_concern="target appearance robustness",
        scene_need="Change only the target appearance.",
        rule_tool_need="Reuse official check_success().",
        candidate_id="dynamic.runtime.appearance",
    )
    run_dir = tmp_path / "run_validated_scene"
    run_dir.mkdir()
    for name in (
        "task.py",
        "candidate_manifest.json",
        "manifest.json",
        "overlay.yml",
    ):
        (run_dir / name).write_text("{}\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "generated",
        "task_name": "runtime_task",
        "task_module": "mea.generated_tasks.run_validated_scene.task",
        "generation_kind": "generic_provider_scene_checker_codegen",
        "proposal": proposal,
        "provider": {"provider_call_count": 1},
        "task_generation_acceptance": {
            "status": "accepted",
            "act_rollouts_started_before_acceptance": 0,
            "visual_self_check_required": True,
        },
        "scene_validation": {
            "generic_preflight": {
                "render_passed": True,
                "expert_passed": True,
                "scene_change_passed": True,
                "checker_fixtures": [
                    {"name": "negative", "passed": True},
                    {"name": "positive", "passed": True},
                ],
            }
        },
        "vision_validation": {"status": "passed", "passed": True},
        "task_artifact_summary": {
            "success_origin": "official_method_reuse",
            "success_official_equivalent": True,
        },
    }

    materialized = backend.bind_validated_taskgen_candidate(
        binding,
        CandidateRequest(
            candidate_id=proposal["candidate_id"],
            source_query=query,
            proposal_bundle=proposal,
            output_dir=run_dir,
            seed=11,
        ),
        manifest,
    )

    assert materialized.metadata["generated_checker"] is False
    assert materialized.task_contract["task_module"] == (
        "mea.generated_tasks.run_validated_scene.task"
    )
    assert materialized.validation["route"] == (
        "validated_taskgen_artifact"
    )
    assert materialized.validation["taskgen"]["vision_validation"][
        "passed"
    ] is True


def test_native_act_runner_returns_one_aligned_method_observation(
    tmp_path: Path,
) -> None:
    task_name = "runtime_task"
    checkpoint = (
        tmp_path
        / "policy/ACT/act_ckpt"
        / f"act-{task_name}"
        / "demo_clean-50"
    )
    checkpoint.mkdir(parents=True)
    (checkpoint / "policy_last.ckpt").write_bytes(b"weights")
    (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
    run_dir = tmp_path / "mea/generated_tasks/native_act"
    run_dir.mkdir(parents=True)
    overlay = run_dir / "overlay.yml"
    overlay.write_text("{}\n", encoding="utf-8")

    def fake_command(
        command: list[str],
        *,
        cwd: Path,
        log_path: Path,
    ) -> int:
        assert cwd == tmp_path
        telemetry_root = Path(command[14])
        episode = telemetry_root / "episode_000"
        episode.mkdir(parents=True)
        (episode / "episode.json").write_text(
            json.dumps({"seed": 17, "task_name": task_name}) + "\n",
            encoding="utf-8",
        )
        (episode / "schema.json").write_text("{}\n", encoding="utf-8")
        (episode / "semantic_trace.npz").write_bytes(b"fixture-npz")
        evaluation = (
            tmp_path
            / "eval_result"
            / task_name
            / "ACT/demo_clean/demo_clean/run_001"
        )
        evaluation.mkdir(parents=True)
        (evaluation / "episode0.mp4").write_bytes(b"video")
        (evaluation / "_result.txt").write_text("1.0\n", encoding="utf-8")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("fixture\n", encoding="utf-8")
        return 0

    candidate = MaterializedCandidate(
        benchmark="robotwin",
        candidate_id="official_control",
        binding_id=f"{task_name}/ACT",
        source_query="Run the official control.",
        task_contract={
            "task_name": task_name,
            "task_module": f"envs.{task_name}",
            "policy": {
                "name": "ACT",
                "backend": "act",
                "checkpoint_setting": "demo_clean",
                "expert_data_num": 50,
            },
        },
        native_task=object(),
        artifacts={"overlay": str(overlay)},
    )
    result = ACTRobotwinRolloutRunner(
        repo_root=tmp_path,
        command_runner=fake_command,
    )(
        candidate=candidate,
        request=RolloutRequest(
            round_id="round_1",
            seed=17,
            output_dir=run_dir / "evaluation",
        ),
        manifest={
            "task_name": task_name,
            "task_module": f"envs.{task_name}",
            "overlay": str(overlay),
        },
    )

    assert result["success"] is True
    assert result["episode"]["seed"] == 17
    assert result["metadata"]["semantic_telemetry_ready"] is True
    assert result["artifacts"]["events"] == ""
    assert Path(result["artifacts"]["video"]).is_file()


def test_generic_rollout_bridge_preserves_generated_checker_authority(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "mea/generated_tasks/native_generated"
    manifest = {
        "generation_kind": "generic_provider_scene_checker_codegen",
        "user_request": "Test the generated success condition.",
        "task_name": "runtime_task",
        "task_module": "mea.generated_tasks.native_generated.task",
        "candidate_module_sha256": "a" * 64,
        "task_artifact_summary": {
            "success_outcome_label": "generated_check_success",
        },
    }
    toolkit_summary = {
        "episode_count": 1,
        "episodes": [
            {
                "episode_dir": "act/episode_000",
                "metadata": {
                    "policy_name": "ACT",
                    "seed": 17,
                    "success": True,
                },
                "tool_results": [
                    {
                        "tool": "generated_check_success",
                        "value": True,
                        "passed": True,
                    }
                ],
            }
        ],
    }

    with patch(
        "mea.taskgen.rollout_evidence.evaluate_telemetry_root",
        return_value=toolkit_summary,
    ) as evaluate:
        result = evaluate_generic_task_rollout_telemetry(
            tmp_path,
            run_dir,
            manifest,
        )

    assert evaluate.call_args.kwargs["outcome_binding"] == {
        "metric": "generated_check_success",
        "authority": "llm_generated_python_ast_validated",
        "module_sha256": "a" * 64,
        "task_module": "mea.generated_tasks.native_generated.task",
    }
    assert result["tool_retrieval"]["route"] == (
        "bound_llm_generated_checker"
    )
    assert result["episodes"][0]["role"] == "policy_under_evaluation"
    assert result["outcome_authority"] == (
        "llm_generated_python_ast_validated"
    )


def test_generic_rollout_bridge_reuses_official_checker_without_hash_binding(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "mea/generated_tasks/native_scene_only"
    manifest = {
        "generation_kind": "generic_provider_scene_checker_codegen",
        "user_request": "Test a scene change with official success.",
        "task_name": "runtime_task",
        "task_module": "mea.generated_tasks.native_scene_only.task",
        "candidate_module_sha256": "b" * 64,
        "task_artifact_summary": {
            "success_outcome_label": "official_check_success",
        },
    }
    toolkit_summary = {
        "episode_count": 1,
        "episodes": [
            {
                "episode_dir": "act/episode_000",
                "metadata": {
                    "policy_name": "ACT",
                    "seed": 19,
                    "success": False,
                },
                "tool_results": [
                    {
                        "tool": "official_check_success",
                        "value": False,
                        "passed": False,
                    }
                ],
            }
        ],
    }

    with patch(
        "mea.taskgen.rollout_evidence.evaluate_telemetry_root",
        return_value=toolkit_summary,
    ) as evaluate:
        result = evaluate_generic_task_rollout_telemetry(
            tmp_path,
            run_dir,
            manifest,
        )

    assert evaluate.call_args.kwargs["outcome_binding"] is None
    assert result["tool_retrieval"]["route"] == "official_checker_reuse"
    assert result["outcome_authority"] == "official_check_success_reused"
