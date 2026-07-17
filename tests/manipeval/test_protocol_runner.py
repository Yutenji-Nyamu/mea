import json
import tempfile
import unittest
from pathlib import Path

from mea.protocol import (
    ProtocolError,
    build_expected_sample_identities,
    build_repetition_schedule,
    collect_evaluation_measurement,
    evaluation_id_for_attempt,
    summarize_protocol,
    validate_budget,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ProtocolRunnerTests(unittest.TestCase):
    def test_agile_budgets_and_non_overlapping_schedule(self):
        self.assertEqual(validate_budget(3, name="repetitions"), 3)
        with self.assertRaises(ProtocolError):
            validate_budget(2, name="repetitions")
        schedule = build_repetition_schedule(
            repetitions=3, episodes=3, start_seed=100
        )
        self.assertEqual([item["start_seed"] for item in schedule], [100, 103, 106])
        self.assertEqual(
            build_expected_sample_identities(
                variant_ids=["left", "right"], episodes=3, start_seed=100
            ),
            [
                {"variant_id": "left", "seed": 100},
                {"variant_id": "left", "seed": 101},
                {"variant_id": "left", "seed": 102},
                {"variant_id": "right", "seed": 100},
                {"variant_id": "right", "seed": 101},
                {"variant_id": "right", "seed": 102},
            ],
        )

    def test_evaluation_id_is_append_only_by_attempt(self):
        self.assertEqual(
            evaluation_id_for_attempt("protocol_bell", 1, 2),
            "eval_protocol_bell_rep_001_attempt_02",
        )

    def test_collects_only_act_episode_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_id = "eval_protocol_test_rep_001_attempt_01"
            write_json(
                root / f"mea/evaluation_runs/{evaluation_id}/manifest.json",
                {
                    "status": "completed",
                    "lifecycle_status": "completed",
                    "task_name": "click_bell",
                    "child_run_ids": ["run_child"],
                },
            )
            write_json(
                root / "mea/generated_tasks/run_child/manifest.json",
                {
                    "status": "completed",
                    "task_name": "click_bell",
                    "trusted_tool_evaluation": {
                        "episodes": [
                            {
                                "policy_name": "ACT",
                                "episode_dir": "act/episode_0",
                                "seed": 7,
                                "success": True,
                            },
                            {"policy_name": "expert", "episode_dir": "expert/episode_0"},
                        ]
                    },
                },
            )
            write_json(
                root
                / "mea/generated_tasks/run_child/evaluation/telemetry/act/episode_0/episode.json",
                {
                    "task_name": "click_bell",
                    "policy_name": "ACT",
                    "seed": 7,
                    "success": True,
                    "policy_steps": 11,
                    "physics_steps": 101,
                    "simulation_duration_seconds": 0.4,
                    "wall_duration_seconds": 0.8,
                    "error": None,
                },
            )
            measurement = collect_evaluation_measurement(
                root,
                evaluation_id=evaluation_id,
                requested_episodes=1,
                returncode=0,
                agent_wall_duration_seconds=1.5,
            )
            self.assertTrue(measurement["completed"])
            self.assertEqual(measurement["samples"]["observed_policy_episodes"], 1)
            self.assertEqual(measurement["samples"]["policy_steps"], 11)
            self.assertEqual(measurement["samples"]["success_rate"], 1.0)

    def test_missing_episode_and_pipeline_failure_are_not_completed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_id = "eval_protocol_broken_rep_001_attempt_01"
            write_json(
                root / f"mea/evaluation_runs/{evaluation_id}/manifest.json",
                {
                    "status": "completed_with_pipeline_failure",
                    "lifecycle_status": "completed",
                    "task_name": "click_bell",
                    "child_run_ids": [],
                },
            )
            measurement = collect_evaluation_measurement(
                root,
                evaluation_id=evaluation_id,
                requested_episodes=1,
                returncode=0,
                agent_wall_duration_seconds=1.0,
            )
            self.assertFalse(measurement["completed"])
            self.assertEqual(measurement["failure_stage"], "agent_or_feedback")
            self.assertTrue(measurement["artifact_issues"])

    def test_generated_samples_use_variant_and_seed_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_id = "eval_protocol_generated_rep_001_attempt_01"
            children = ["run_left", "run_right"]
            write_json(
                root / f"mea/evaluation_runs/{evaluation_id}/manifest.json",
                {
                    "status": "completed",
                    "lifecycle_status": "completed",
                    "task_name": "click_bell",
                    "child_run_ids": children,
                },
            )
            write_json(
                root / f"mea/evaluation_runs/{evaluation_id}/summary/summary.json",
                {
                    "rounds": [
                        {"taskgen_run_id": "run_right", "variant_id": "right"},
                        {"taskgen_run_id": "run_left", "variant_id": "left"},
                    ]
                },
            )
            for child in children:
                variant = child.removeprefix("run_")
                write_json(
                    root / f"mea/generated_tasks/{child}/manifest.json",
                    {
                        "status": "completed",
                        "task_name": "click_bell",
                        "variant_id": variant,
                        "trusted_tool_evaluation": {
                            "episodes": [
                                {
                                    "policy_name": "ACT",
                                    "episode_dir": "act/episode_0",
                                    "seed": 7,
                                    "success": variant == "right",
                                }
                            ]
                        },
                    },
                )
                write_json(
                    root
                    / f"mea/generated_tasks/{child}/evaluation/telemetry/act/episode_0/episode.json",
                    {
                        "task_name": "click_bell",
                        "policy_name": "ACT",
                        "seed": 7,
                        "success": variant == "right",
                        "policy_steps": 10,
                        "physics_steps": 100,
                        "simulation_duration_seconds": 0.4,
                        "wall_duration_seconds": 0.8,
                        "error": None,
                    },
                )
            expected = [
                {"variant_id": "left", "seed": 7},
                {"variant_id": "right", "seed": 7},
            ]
            measurement = collect_evaluation_measurement(
                root,
                evaluation_id=evaluation_id,
                requested_episodes=1,
                expected_sample_identities=expected,
                returncode=0,
                agent_wall_duration_seconds=2.0,
            )
            self.assertTrue(measurement["completed"])
            self.assertEqual(measurement["samples"]["observed_policy_episodes"], 2)
            self.assertEqual(measurement["samples"]["duplicate_sample_identities"], [])
            self.assertEqual(
                measurement["samples"]["by_variant"]["right"]["success_rate"],
                1.0,
            )

            child_manifest_path = (
                root / "mea/generated_tasks/run_left/manifest.json"
            )
            child_manifest = json.loads(
                child_manifest_path.read_text(encoding="utf-8")
            )
            child_manifest["variant_id"] = "right"
            write_json(child_manifest_path, child_manifest)
            mismatched = collect_evaluation_measurement(
                root,
                evaluation_id=evaluation_id,
                requested_episodes=1,
                expected_sample_identities=expected,
                returncode=0,
                agent_wall_duration_seconds=2.0,
            )
            self.assertFalse(mismatched["completed"])
            self.assertTrue(
                any(
                    "generated-round variant mismatch" in issue
                    for issue in mismatched["artifact_issues"]
                )
            )
            child_manifest["variant_id"] = "left"
            write_json(child_manifest_path, child_manifest)

            duplicate = collect_evaluation_measurement(
                root,
                evaluation_id=evaluation_id,
                requested_episodes=1,
                expected_sample_identities=[
                    {"variant_id": "left", "seed": 7},
                    {"variant_id": "left", "seed": 7},
                ],
                returncode=0,
                agent_wall_duration_seconds=2.0,
            )
            self.assertFalse(duplicate["completed"])
            self.assertTrue(duplicate["samples"]["missing_sample_identities"])
            self.assertTrue(duplicate["samples"]["unexpected_sample_identities"])

    def test_budget_validation_rejects_bool_and_float(self):
        for value in (True, 1.0, 1.2):
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                validate_budget(value, name="budget")

    def test_summary_marks_one_by_one_as_smoke(self):
        manifest = {
            "run_id": "protocol_test",
            "config": {"repetitions": 1, "episodes": 1},
            "repetitions": [
                {
                    "status": "completed",
                    "attempts": [
                        {
                            "status": "completed",
                            "measurement": {
                                "agent_wall_duration_seconds": 2.0,
                                "samples": {
                                    "requested_policy_episodes": 1,
                                    "observed_policy_episodes": 1,
                                    "successes": 1,
                                    "policy_steps": 20,
                                    "physics_steps": 200,
                                    "rollout_wall_duration_seconds": 1.0,
                                },
                            },
                        }
                    ],
                }
            ],
        }
        summary = summarize_protocol(manifest)
        self.assertEqual(summary["status"], "completed")
        self.assertTrue(summary["smoke_only"])
        self.assertEqual(summary["pooled_success_rate"], 1.0)

    def test_summary_rejects_duplicate_actual_seeds_across_repetitions(self):
        def repetition():
            return {
                "status": "completed",
                "attempts": [
                    {
                        "status": "completed",
                        "measurement": {
                            "agent_wall_duration_seconds": 1.0,
                            "samples": {
                                "requested_policy_episodes": 1,
                                "observed_policy_episodes": 1,
                                "successes": 1,
                                "actual_seeds": [7],
                            },
                        },
                    }
                ],
            }

        summary = summarize_protocol(
            {
                "run_id": "protocol_duplicate",
                "config": {"repetitions": 3, "episodes": 1},
                "repetitions": [repetition(), repetition(), repetition()],
            }
        )
        self.assertEqual(summary["status"], "completed_with_protocol_violation")
        self.assertFalse(summary["valid_for_comparison"])
        self.assertEqual(summary["duplicate_actual_seeds"], [7])

    def test_generated_summary_allows_raw_seed_reuse_across_variants(self):
        measurement = {
            "agent_wall_duration_seconds": 2.0,
            "samples": {
                "requested_policy_episodes": 2,
                "observed_policy_episodes": 2,
                "successes": 1,
                "actual_seeds": [7, 7],
                "actual_sample_identities": [
                    {"variant_id": "left", "seed": 7},
                    {"variant_id": "right", "seed": 7},
                ],
                "by_variant": {
                    "left": {
                        "observed_policy_episodes": 1,
                        "successes": 0,
                    },
                    "right": {
                        "observed_policy_episodes": 1,
                        "successes": 1,
                    },
                },
            },
        }
        summary = summarize_protocol(
            {
                "run_id": "protocol_generated",
                "config": {
                    "repetitions": 1,
                    "episodes": 1,
                    "expected_variant_ids": ["left", "right"],
                },
                "repetitions": [
                    {
                        "status": "completed",
                        "attempts": [
                            {"status": "completed", "measurement": measurement}
                        ],
                    }
                ],
            }
        )
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["requested_policy_episodes"], 2)
        self.assertEqual(summary["duplicate_actual_seeds"], [7])
        self.assertEqual(summary["duplicate_sample_identities"], [])
        self.assertEqual(summary["variants"]["right"]["success_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
