import json
import tempfile
import unittest
from pathlib import Path

from mea.protocol import (
    ProtocolError,
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


if __name__ == "__main__":
    unittest.main()
