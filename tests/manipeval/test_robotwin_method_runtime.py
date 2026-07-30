from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mea.method_runtime import (
    BackendBindingRequest,
    CandidateRequest,
    EvidenceRequest,
    MethodRuntime,
    RolloutRequest,
)
from mea.planner.experiment_candidate import build_experiment_candidate
from mea.robotwin import (
    RoboTwinMethodBackend,
    project_executed_round_through_method_runtime,
)
from mea.taskgen.generic_backend import (
    GenericRoboTwinTaskAdapter,
    GenericRoboTwinTaskGenBackend,
    GenericTaskGenHooks,
    build_generic_task_subclass_module,
    validate_generic_task_methods,
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


def test_executed_child_projection_uses_shared_runtime_without_rerun() -> None:
    query = "Does a shifted target expose a weakness?"
    candidate = build_experiment_candidate(
        source_query=query,
        base_task="runtime_task",
        semantic_concern="target pose robustness",
        scene_need="Shift the target to another valid pose.",
        checker_need="Require completion at the generated pose.",
        rule_tool_need="Measure generated-checker success.",
        candidate_id="dynamic.runtime.pose",
    )
    result = project_executed_round_through_method_runtime(
        task_name="runtime_task",
        round_plan={
            "round_id": "round_2",
            "candidate_id": candidate["candidate_id"],
            "proposal": candidate,
            "task_name": "runtime_task",
            "task_module": None,
            "sub_aspect": "target pose robustness",
            "task_instruction": query,
            "route": "generic_provider_scene_checker_codegen",
        },
        child_manifest={
            "run_id": "run_existing_child",
            "status": "completed",
            "task_module": "mea.generated_tasks.run_existing_child.task",
        },
        round_summary={
            "round_id": "round_2",
            "pipeline_passed": True,
            "observations": {
                "actual_seeds": [100000],
                "policy_success": 1.0,
                "policy_outcome": {
                    "metric": "generated_check_success",
                    "value": True,
                    "official_equivalent": False,
                },
                "planned_tool": {
                    "status": "passed",
                    "measurements": [{"value": 0.03}],
                },
                "aggregate": {"status": "passed_complete"},
                "execution_vqa": {"status": "passed"},
            },
        },
        artifacts={
            "child_manifest": (
                "mea/generated_tasks/run_existing_child/manifest.json"
            ),
        },
    )

    assert result["runtime"] == "MethodRuntime"
    assert result["execution_reused"] is True
    assert result["taskgen_reinvoked"] is False
    assert result["policy_rollout_reinvoked"] is False
    assert result["binding"]["binding_id"] == "runtime_task/ACT"
    assert result["candidate"]["candidate_id"] == candidate["candidate_id"]
    assert result["rollout"]["round_id"] == "round_2"
    assert result["evidence"]["outcome"] == "success"
    assert result["evidence"]["limitations"] == [
        "N=1 executed seed(s).",
        "The generated checker is not certified as official-equivalent.",
    ]
