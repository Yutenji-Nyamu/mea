import json
import inspect
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import mea.round_executor as round_executor_module
from mea.round_evidence import aggregate_sources
from mea.round_executor import RoundExecutionResult, RoundExecutor
from mea.round_executor import RoundExecutionRequest, RoundExecutionServices
from mea.round_summary import normalize_outcome_semantics
from mea.round_tools import (
    executed_policy_episode_dirs,
    executed_runtime_task_schema,
)
from mea.robotwin.production_round_executor import (
    build_production_round_executor,
)


class RoundExecutorBoundaryTests(unittest.TestCase):
    def test_execution_vqa_is_opt_in(self):
        required = round_executor_module._round_requests_execution_vqa

        self.assertFalse(required({}))
        self.assertFalse(
            required(
                {
                    "semantic_need_execution": {
                        "vqa_tool": {"requested": False}
                    }
                }
            )
        )
        self.assertTrue(
            required(
                {
                    "semantic_need_execution": {
                        "vqa_tool": {"requested": True}
                    }
                }
            )
        )

    def test_production_builder_returns_independent_executor(self):
        executor = build_production_round_executor()

        self.assertIsInstance(executor, RoundExecutor)
        self.assertNotIn(
            "scripts.manipeval_agent",
            inspect.getsource(round_executor_module),
        )
        self.assertEqual(
            set(executor._services.native_policy_rounds),
            {"act", "smolvla", "hyvla"},
        )

    def test_typed_result_exposes_named_round_outputs(self):
        result = RoundExecutionResult(
            child_manifest={"status": "completed"},
            child_dir=Path("/tmp/child"),
            round_summary={"pipeline_passed": True},
            tool_evaluation={"status": "passed"},
            returncode=0,
        )

        self.assertEqual(result.child_manifest["status"], "completed")
        self.assertEqual(result.child_dir, Path("/tmp/child"))
        self.assertTrue(result.round_summary["pipeline_passed"])
        self.assertEqual(result.tool_evaluation["status"], "passed")
        self.assertEqual(result.returncode, 0)

    def test_prepolicy_taskgen_outcomes_project_n_zero_planning_observation(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            scenarios = (
                {
                    "status": "candidate_unexecutable",
                    "reason_code": (
                        "taskgen_expert_gate_candidate_unexecutable"
                    ),
                    "diagnosis": "target_pose cannot be None",
                },
                {
                    "status": "taskgen_materialization_failed",
                    "failure_stage": "scene_codegen",
                    "reason_code": (
                        "taskgen_scene_codegen_invalid_candidate"
                    ),
                    "diagnosis": (
                        "checker did not preserve the official core"
                    ),
                },
            )
            for scenario in scenarios:
                status = scenario["status"]
                with self.subTest(status=status):
                    evaluation_id = f"eval_{status}"
                    evaluation_dir = (
                        repo_root / "mea/evaluation_runs" / evaluation_id
                    )
                    child_dir = (
                        repo_root / "mea/generated_tasks" / f"run_{status}"
                    )
                    manifest_path = child_dir / "manifest.json"
                    method_runtime_path = (
                        evaluation_dir
                        / "execution/round_1/method_runtime.json"
                    )
                    planning_observation = {
                        "schema_version": 1,
                        "kind": status,
                        "candidate_id": "dynamic.shifted_target",
                        "sub_aspect": "target displacement",
                        "reason_code": scenario["reason_code"],
                        "diagnosis": scenario["diagnosis"],
                        "policy_rollouts_started": 0,
                        "policy_sample_count": 0,
                        "taskgen_attempt_summary": (
                            f"mea/generated_tasks/run_{status}/validation/"
                            "task_generation_attempt_summary.json"
                        ),
                    }
                    if "failure_stage" in scenario:
                        planning_observation["failure_stage"] = scenario[
                            "failure_stage"
                        ]
                    child_manifest = {
                        "schema_version": 1,
                        "run_id": f"run_{status}",
                        "status": status,
                        "task_name": "alpha_task",
                        status: planning_observation,
                        "act_evaluation": {
                            "passed": False,
                            "actual_seeds": [],
                            "policy_name": "SmolVLA",
                        },
                        "policy_execution": {
                            "started": False,
                            "rollouts_started": 0,
                            "sample_count": 0,
                        },
                        "task_artifact_summary": {
                            "success_official_equivalent": None,
                            "success_execution_scope": "not_executed",
                        },
                        "trusted_tool_evaluation": {
                            "schema_version": 1,
                            "status": "skipped",
                            "outcome_metric": None,
                            "outcome_authority": None,
                            "episode_count": 0,
                            "episodes": [],
                        },
                    }
                    manifest_path.parent.mkdir(parents=True)
                    manifest_path.write_text(
                        json.dumps(child_manifest) + "\n", encoding="utf-8"
                    )
                    method_runtime_path.parent.mkdir(parents=True)
                    method_runtime_path.write_text("{}\n", encoding="utf-8")

                    def native_round(**_kwargs):
                        result = {
                            "child_manifest": child_manifest,
                            "child_dir": child_dir,
                            "manifest_path": manifest_path,
                            "method_runtime_path": method_runtime_path,
                            "semantic_telemetry_ready": False,
                            "candidate_id": "dynamic.shifted_target",
                            "evidence_outcome": status,
                            status: True,
                        }
                        if status == "taskgen_materialization_failed":
                            result["planning_observation"] = (
                                planning_observation
                            )
                        return result

                    executor = RoundExecutor(
                        RoundExecutionServices(
                            update_manifest=lambda *_args, **_kwargs: {},
                            native_policy_rounds={"smolvla": native_round},
                        )
                    )
                    result = executor.execute(
                        RoundExecutionRequest(
                            repo_root=repo_root,
                            evaluation_dir=evaluation_dir,
                            evaluation_id=evaluation_id,
                            round_plan={
                                "round_id": "round_1",
                                "template_id": None,
                                "candidate_id": "dynamic.shifted_target",
                                "sub_aspect": "target displacement",
                                "task_instruction": (
                                    "Find a feasible spatial test."
                                ),
                                "route": (
                                    "generic_provider_scene_checker_codegen"
                                ),
                                "execution": {
                                    "backend": "act",
                                    "seeds": [7],
                                    "num_episodes": 1,
                                },
                                "tool_request": {
                                    "metric": "official_check_success"
                                },
                            },
                            text_model="fixture-model",
                            vision_model="fixture-model",
                            gpu=0,
                            max_reflections=1,
                            provider=object(),
                            toolgen_model="fixture-model",
                            policy_backend="smolvla",
                            runtime_target={},
                        )
                    )

                    observations = result.round_summary["observations"]
                    self.assertEqual(observations["actual_seeds"], [])
                    self.assertIsNone(observations["policy_success"])
                    self.assertIsNone(observations["policy_outcome"]["value"])
                    self.assertEqual(
                        observations["planning_observation"],
                        planning_observation,
                    )
                    self.assertEqual(
                        observations["method_runtime"]["status"], status
                    )
                    self.assertEqual(
                        result.round_summary["failure_stage"],
                        scenario.get("failure_stage", "taskgen_expert_gate"),
                    )
                    self.assertEqual(
                        observations["outcome_semantics"]["reason_codes"],
                        [scenario["reason_code"]],
                    )
                    self.assertFalse(
                        observations["evidence_aggregate"]["policy"][
                            "reported"
                        ]
                    )
                    self.assertEqual(
                        observations["evidence_aggregate"][
                            "evidence_strength"
                        ],
                        "uncertain",
                    )
                    self.assertEqual(
                        observations["evidence_aggregate"]["reason_codes"],
                        [scenario["reason_code"]],
                    )
                    self.assertEqual(
                        observations["evidence_aggregate"]["rule"][
                            "observed_policy_episodes"
                        ],
                        0,
                    )
                    self.assertFalse(result.round_summary["pipeline_passed"])

    def test_executed_schema_discovery_is_policy_backend_neutral(self):
        with tempfile.TemporaryDirectory() as temporary:
            child_dir = Path(temporary) / "child"
            episode_dir = (
                child_dir
                / "evaluation"
                / "telemetry"
                / "smolvla"
                / "episode_000_seed_3"
            )
            episode_dir.mkdir(parents=True)
            schema = {
                "schema_version": 1,
                "task_name": "grab_roller",
                "semantic_fields": [
                    {
                        "name": "left_tcp_position",
                        "source": "robot_tcp_position",
                        "side": "left",
                    }
                ],
            }
            (episode_dir / "schema.json").write_text(
                json.dumps(schema) + "\n",
                encoding="utf-8",
            )
            (child_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "method_runtime": {
                            "rollout": {
                                "artifacts": {
                                    "telemetry_episode": str(episode_dir)
                                }
                            }
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            self.assertEqual(
                executed_policy_episode_dirs(child_dir),
                [episode_dir],
            )
            self.assertEqual(
                executed_runtime_task_schema(
                    child_dir,
                    task_name="grab_roller",
                ),
                schema,
            )

    def test_distinct_planned_tool_does_not_replace_checker_outcome(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            child_dir = repo_root / "mea/generated_tasks/run_native"
            episode_dir = (
                child_dir / "evaluation/telemetry/act/episode_000"
            )
            episode_dir.mkdir(parents=True)
            checker_result = {
                "tool": "generated_check_success",
                "value": False,
                "passed": False,
                "details": {
                    "generated_checker_success": False,
                    "official_core_predicate_satisfied": True,
                },
            }
            trusted_outcome = {
                "schema_version": 1,
                "status": "passed",
                "outcome_metric": "generated_check_success",
                "outcome_authority": "llm_generated_python_ast_validated",
                "episode_count": 1,
                "episodes": [
                    {
                        "episode_dir": "act/episode_000",
                        "role": "policy_under_evaluation",
                        "seed": 17,
                        "tool_results": [checker_result],
                    }
                ],
            }
            child_manifest = {
                "trusted_tool_evaluation": deepcopy(trusted_outcome)
            }
            planned = {
                "schema_version": 1,
                "status": "passed",
                "tool_request": {
                    "schema_version": 1,
                    "task_name": "grab_roller",
                    "metric": "terminal_roller_height",
                    "question": "What was the terminal roller height?",
                },
                "episodes": [
                    {
                        "episode_dir": str(episode_dir),
                        "role": "policy_under_evaluation",
                        "seed": 17,
                        "result": {
                            "tool": "terminal_roller_height",
                            "value": 0.81,
                            "unit": "m",
                            "passed": True,
                        },
                    }
                ],
                "artifacts": {
                    "tool_execution": (
                        "mea/evaluation_runs/eval/execution/round_1/"
                        "planned_tool/tool_execution.json"
                    )
                },
            }

            RoundExecutor._persist_native_planned_tool(
                repo_root=repo_root,
                child_dir=child_dir,
                child_manifest=child_manifest,
                tool_evaluation=planned,
            )

            self.assertEqual(
                child_manifest["trusted_tool_evaluation"],
                trusted_outcome,
            )
            self.assertEqual(
                child_manifest["planned_tool_evaluation"]["episodes"][0][
                    "episode_dir"
                ],
                "act/episode_000",
            )
            persisted = json.loads(
                (child_dir / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                persisted["trusted_tool_evaluation"]["outcome_metric"],
                "generated_check_success",
            )
            semantics = normalize_outcome_semantics(
                child_manifest["trusted_tool_evaluation"],
                {"success_official_equivalent": False},
            )
            self.assertEqual(
                semantics["status"],
                "expected_semantic_extension",
            )
            self.assertIs(
                semantics["episodes"][0]["official_success"],
                True,
            )
            sources = aggregate_sources(
                {
                    "round_id": "round_1",
                    "sub_aspect": "trajectory.quality",
                },
                child_manifest,
                planned,
            )
            self.assertEqual(len(sources), 2)
            self.assertEqual(
                sources[0]["episodes"][0]["tool_results"][0]["tool"],
                "generated_check_success",
            )
            self.assertEqual(
                sources[1]["tool_execution"]["episodes"][0]["result"]["tool"],
                "terminal_roller_height",
            )


if __name__ == "__main__":
    unittest.main()
