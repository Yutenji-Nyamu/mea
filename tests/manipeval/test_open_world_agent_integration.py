from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mea.planner.experiment_candidate import build_experiment_candidate
from scripts.manipeval_agent import (
    build_taskgen_command,
    compact_tool_evaluation,
    materialize_open_world_round,
    materialize_open_world_tool_request,
    summarize_round,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class _Provider:
    last_metadata = {"provider": "fixture"}

    def __init__(self):
        self.calls = 0

    def text(self, *_args, **_kwargs):
        self.calls += 1
        return json.dumps(
            {
                "schema_version": 2,
                "task_name": "adjust_bottle",
                "metric": "query_bottle_tcp_min_distance",
                "question": (
                    "What was the minimum distance from the left TCP to the bottle?"
                ),
                "metric_spec": {
                    "schema_version": 1,
                    "operation": "minimum_distance",
                    "left_signal": "left_tcp_position",
                    "right_signal": "bottle_position",
                    "dimensions": ["x", "y", "z"],
                    "unit": "m",
                    "null_semantics": "null_if_no_finite_sample",
                },
            }
        )


def _candidate():
    return build_experiment_candidate(
        source_query="Can the policy handle a heavier-looking bottle?",
        base_task="adjust_bottle",
        semantic_concern="object_physics.mass: mass may expose a weakness",
        scene_need="Change only the bottle's physical mass.",
        checker_need="Require the intended bottle adjustment.",
        tool_need="Measure minimum TCP-to-bottle distance.",
    )


class OpenWorldAgentIntegrationTest(unittest.TestCase):
    def test_materialized_round_has_no_catalog_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            evaluation = Path(temporary)
            provider = _Provider()
            round_plan, bundle = materialize_open_world_round(
                REPO_ROOT,
                evaluation,
                round_number=2,
                candidate=_candidate(),
                control_execution={
                    "backend": "act",
                    "seeds": [100000],
                    "num_episodes": 1,
                    "gates": [],
                },
            )
            self.assertIsNone(round_plan["template_id"])
            self.assertEqual(
                round_plan["candidate_id"],
                round_plan["experiment_candidate"]["candidate_id"],
            )
            self.assertEqual(
                round_plan["route"],
                "generic_provider_scene_checker_codegen",
            )
            self.assertEqual(
                bundle["source"],
                "deferred_until_executed_telemetry_schema",
            )
            self.assertTrue(round_plan["open_tool_request_deferred"])
            self.assertEqual(provider.calls, 0)

    def test_scene_only_round_keeps_official_success_authority(self):
        candidate = build_experiment_candidate(
            source_query="Which spatial change exposes a weakness?",
            base_task="click_bell",
            semantic_concern="bell spatial offset precision",
            scene_need="Shift the bell by a bounded horizontal offset.",
            checker_need=None,
            rule_tool_need="Measure terminal target contact error.",
        )
        with tempfile.TemporaryDirectory() as temporary:
            round_plan, _bundle = materialize_open_world_round(
                REPO_ROOT,
                Path(temporary),
                round_number=2,
                candidate=candidate,
                control_execution={
                    "backend": "act",
                    "seeds": [100000],
                    "num_episodes": 1,
                    "gates": [],
                },
            )

        self.assertEqual(
            round_plan["tool_request"]["metric"],
            "official_check_success",
        )
        self.assertIsNone(
            round_plan["experiment_candidate"]["checker_need"]
        )

    def test_vqa_only_round_keeps_rule_tool_unrequested(self):
        candidate = build_experiment_candidate(
            source_query="Does the bottle visibly wobble after release?",
            base_task="adjust_bottle",
            semantic_concern="motion.post_release_visible_wobble",
            vqa_tool_need={
                "kind": "vqa",
                "description": "Check whether the bottle visibly wobbles.",
                "reuse_first": True,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            round_plan, bundle = materialize_open_world_round(
                REPO_ROOT,
                Path(temporary),
                round_number=1,
                candidate=candidate,
                control_execution={
                    "backend": "act",
                    "seeds": [100000],
                    "num_episodes": 1,
                    "gates": [],
                },
            )

        self.assertFalse(
            round_plan["semantic_need_execution"]["rule_tool"]["requested"]
        )
        self.assertTrue(
            round_plan["semantic_need_execution"]["vqa_tool"]["requested"]
        )
        self.assertNotIn("planned_tool", round_plan["observations"])
        self.assertNotIn("open_vqa_request_deferred", round_plan)
        self.assertEqual(bundle["source"], "vqa_only_no_rule_tool_requested")

    def test_open_tool_request_uses_executed_episode_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "child"
            episode = (
                child / "evaluation/telemetry/act/episode_000_seed_100000"
            )
            episode.mkdir(parents=True)
            runtime_schema = json.loads(
                (
                    REPO_ROOT
                    / "mea/toolkit/schemas/adjust_bottle.json"
                ).read_text(encoding="utf-8")
            )
            (episode / "schema.json").write_text(
                json.dumps(runtime_schema),
                encoding="utf-8",
            )
            (child / "manifest.json").write_text(
                json.dumps(
                    {
                        "trusted_tool_evaluation": {
                            "outcome_metric": "official_check_success",
                            "episodes": [
                                {
                                    "tool_results": [
                                        {
                                            "tool": "official_check_success",
                                            "value": True,
                                        },
                                        {
                                            "tool": "time_to_success",
                                            "value": 1.0,
                                        },
                                    ]
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            round_plan, _bundle = materialize_open_world_round(
                REPO_ROOT,
                root / "evaluation",
                round_number=2,
                candidate=_candidate(),
                control_execution={
                    "backend": "act",
                    "seeds": [100000],
                    "num_episodes": 1,
                    "gates": [],
                },
            )
            provider = _Provider()
            result = materialize_open_world_tool_request(
                REPO_ROOT,
                root / "execution",
                round_plan=round_plan,
                child_dir=child,
                provider=provider,
                toolgen_model="fixture",
            )

            self.assertEqual(
                result["tool_request"]["metric"],
                "query_bottle_tcp_min_distance",
            )
            self.assertEqual(
                result["context"]["telemetry_schema_source"],
                "executed_episode_schema",
            )
            self.assertEqual(
                result["context"]["forbidden_metric_ids"],
                ["official_check_success", "time_to_success"],
            )
            self.assertEqual(provider.calls, 1)

    def test_taskgen_command_carries_candidate_not_template(self):
        candidate = _candidate()
        round_plan = {
            "round_id": "round_2",
            "template_id": None,
            "candidate_id": candidate["candidate_id"],
            "experiment_candidate": candidate,
            "sub_aspect": "object_physics.mass",
            "task_instruction": candidate["source_query"],
            "task_name": "adjust_bottle",
            "task_module": None,
            "route": "generic_provider_scene_checker_codegen",
            "variant_hint": {},
            "tool_request": {
                "schema_version": 1,
                "task_name": "adjust_bottle",
                "metric": "generated_check_success",
                "question": "Did the generated checker pass?",
            },
            "vqa_phenomenon_ids": [],
            "execution": {
                "backend": "act",
                "seeds": [100000],
                "num_episodes": 1,
                "gates": [],
            },
        }
        command, run_id = build_taskgen_command(
            REPO_ROOT,
            "eval_open_world_fixture",
            round_plan,
            text_model="fixture",
            vision_model="fixture",
            base_url=None,
            gpu=0,
            max_reflections=1,
        )
        self.assertEqual(run_id, "run_open_world_fixture_round_2")
        self.assertIn("--experiment-candidate-json", command)
        self.assertIn("generic_provider_scene_checker_codegen", command)
        self.assertNotIn("--variant-id", command)
        self.assertIn("--run-act", command)
        self.assertNotIn("--vision-check", command)

    def test_generic_round_requires_generic_visual_diagnosis_not_legacy_gate(self):
        candidate = _candidate()
        round_plan = {
            "round_id": "round_2",
            "template_id": None,
            "candidate_id": candidate["candidate_id"],
            "experiment_candidate": candidate,
            "sub_aspect": "object_physics.mass",
            "task_instruction": candidate["source_query"],
            "task_name": "adjust_bottle",
            "route": "generic_provider_scene_checker_codegen",
            "execution": {
                "backend": "act",
                "seeds": [100000],
                "num_episodes": 1,
                "gates": [],
            },
        }
        child_manifest = {
            "run_id": "run_fixture",
            "status": "completed",
            "generation_kind": "generic_provider_scene_checker_codegen",
            "static_validation": {
                "provider_scene_checker": {
                    "valid": True,
                    "model_written_python": True,
                    "restricted_success_spec_compiler_used": False,
                    "ast_policy": "generic_official_api_ast_v1:fixture",
                }
            },
            "scene_validation": {
                "render_success": True,
                "rule_check": {"passed": True},
                "expert": {"passed": True},
                "generic_preflight": {
                    "render_passed": True,
                    "expert_passed": True,
                    "scene_change_passed": True,
                    "checker_fixtures": [{"passed": True}, {"passed": True}],
                },
            },
            "vision_validation": {
                "status": "passed",
                "passed": True,
                "render_usable": True,
                "key_task_actors_visible": True,
                "requested_change_assessment": "consistent",
                "visual_physical_plausibility": "plausible",
                "unexpected_changes": [],
                "diagnosis": "The generated scene is visible and plausible.",
                "repair_instructions": [],
                "confidence": 0.9,
            },
            "task_generation_acceptance": {
                "visual_self_check_required": True,
            },
            "position_samples": {"passed": True, "samples": [], "metrics": {}},
            "act_evaluation": {"passed": True, "actual_seeds": [100000]},
            "trusted_tool_evaluation": {
                "episode_count": 1,
                "outcome_metric": "generated_check_success",
                "outcome_authority": "llm_generated_python_ast_validated",
                "episodes": [],
            },
            "task_artifact_summary": {
                "success_official_equivalent": False,
                "success_execution_scope": "provider_generated_checker",
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            child_dir = Path(temporary)
            (child_dir / "evaluation").mkdir()
            (child_dir / "evaluation/_result.txt").write_text(
                "0\n",
                encoding="utf-8",
            )
            result = summarize_round(
                round_plan,
                child_manifest,
                child_dir,
                tool_evaluation={"status": "passed", "episodes": []},
                aggregate_result={"status": "passed"},
                execution_vqa={"status": "skipped"},
            )

        self.assertTrue(result["pipeline_passed"])
        self.assertNotIn("gate_status", result)
        self.assertTrue(result["required_gate_status"]["passed"])

    def test_bound_checker_tool_results_keep_value_when_compacted(self):
        compact = compact_tool_evaluation(
            {
                "status": "passed",
                "route": "bound_child_trusted_checker",
                "reference_tool": "generated_check_success",
                "episodes": [
                    {
                        "policy_name": "ACT",
                        "seed": 100000,
                        "role": "policy_under_evaluation",
                        "tool_results": [
                            {
                                "tool": "generated_check_success",
                                "value": False,
                                "unit": None,
                                "passed": False,
                                "details": {
                                    "authority": (
                                        "llm_generated_python_ast_validated"
                                    )
                                },
                            }
                        ],
                    }
                ],
            }
        )

        self.assertEqual(
            compact["episodes"][0]["metric"],
            "generated_check_success",
        )
        self.assertIs(compact["episodes"][0]["value"], False)


if __name__ == "__main__":
    unittest.main()
