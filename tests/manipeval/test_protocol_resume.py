import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.manipeval_protocol import run_protocol


def measurement(*, completed: bool) -> dict:
    return {
        "completed": completed,
        "failure_stage": None if completed else "interrupted_agent",
        "agent_wall_duration_seconds": 0.0,
        "samples": {
            "requested_policy_episodes": 1,
            "observed_policy_episodes": 1 if completed else 0,
            "successes": 1 if completed else 0,
            "actual_seeds": [7] if completed else [],
            "actual_sample_identities": [],
            "policy_steps": 10 if completed else 0,
            "physics_steps": 100 if completed else 0,
            "rollout_wall_duration_seconds": 1.0 if completed else 0.0,
        },
        "artifact_issues": [] if completed else ["incomplete evaluation"],
    }


def running_manifest() -> dict:
    return {
        "schema_version": 1,
        "protocol": "agent_act_agile_v1",
        "run_id": "protocol_resume_test",
        "status": "running",
        "config": {
            "request": "evaluate click bell",
            "task_name": "click_bell",
            "task_module": "envs.click_bell",
            "policy": "ACT",
            "repetitions": 1,
            "episodes": 1,
            "start_seed": 7,
            "model_profile": "economy",
            "telemetry_profile": "balanced_v1",
            "gpu": 0,
            "max_reflections": 0,
            "base_url": None,
            "history": "disabled_for_repetition_comparability",
            "task_profile": "official",
            "generated_rounds": None,
            "expected_variant_ids": [],
            "sample_identity_fields": ["seed"],
        },
        "repetitions": [
            {
                "index": 1,
                "start_seed": 7,
                "requested_episodes": 1,
                "status": "running",
                "attempts": [
                    {
                        "attempt_index": 1,
                        "evaluation_id": (
                            "eval_protocol_resume_test_rep_001_attempt_01"
                        ),
                        "status": "running",
                        "child_pid": 999999,
                        "measurement": None,
                    }
                ],
            }
        ],
    }


def args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        repo_root=root,
        resume_run="protocol_resume_test",
        run_id=None,
        chunk_size=1,
        retry_failed=False,
    )


class ProtocolResumeTests(unittest.TestCase):
    def test_resume_adopts_completed_stale_child_without_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/protocol_runs/protocol_resume_test"
            run_dir.mkdir(parents=True)
            manifest = running_manifest()
            with (
                patch("scripts.manipeval_protocol._validate_robotwin_runtime"),
                patch(
                    "scripts.manipeval_protocol._load_manifest",
                    return_value=(run_dir, manifest),
                ),
                patch(
                    "scripts.manipeval_protocol._pid_matches_attempt",
                    return_value=False,
                ),
                patch(
                    "scripts.manipeval_protocol.collect_evaluation_measurement",
                    return_value=measurement(completed=True),
                ),
                patch("scripts.manipeval_protocol._run_logged") as run_logged,
            ):
                summary = run_protocol(args(root))

            self.assertEqual(summary["status"], "completed")
            run_logged.assert_not_called()
            persisted = json.loads(
                (run_dir / "protocol_manifest.json").read_text(encoding="utf-8")
            )
            attempt = persisted["repetitions"][0]["attempts"][0]
            self.assertEqual(attempt["status"], "completed")
            self.assertTrue(attempt["measurement"]["completed"])

    def test_resume_preserves_interrupted_attempt_and_appends_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/protocol_runs/protocol_resume_test"
            run_dir.mkdir(parents=True)
            manifest = running_manifest()
            with (
                patch("scripts.manipeval_protocol._validate_robotwin_runtime"),
                patch(
                    "scripts.manipeval_protocol._load_manifest",
                    return_value=(run_dir, manifest),
                ),
                patch(
                    "scripts.manipeval_protocol._pid_matches_attempt",
                    return_value=False,
                ),
                patch(
                    "scripts.manipeval_protocol.collect_evaluation_measurement",
                    side_effect=[
                        measurement(completed=False),
                        measurement(completed=True),
                    ],
                ),
                patch(
                    "scripts.manipeval_protocol._run_logged", return_value=0
                ) as run_logged,
            ):
                summary = run_protocol(args(root))

            self.assertEqual(summary["status"], "completed")
            run_logged.assert_called_once()
            persisted = json.loads(
                (run_dir / "protocol_manifest.json").read_text(encoding="utf-8")
            )
            attempts = persisted["repetitions"][0]["attempts"]
            self.assertEqual([item["attempt_index"] for item in attempts], [1, 2])
            self.assertEqual(attempts[0]["status"], "interrupted")
            self.assertEqual(attempts[1]["status"], "completed")
            self.assertEqual(
                attempts[1]["evaluation_id"],
                "eval_protocol_resume_test_rep_001_attempt_02",
            )


if __name__ == "__main__":
    unittest.main()
