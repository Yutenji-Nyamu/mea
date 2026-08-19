import tempfile
import unittest
from pathlib import Path

from mea.taskgen.attempts import (
    REGENERATE_CANDIDATE,
    REPAIR_SCENE,
    REPAIR_SUCCESS_SPEC,
    TERMINAL,
    TaskGenerationRecoveryError,
    TaskGenerationStageError,
    run_task_generation,
    task_generation_recovery_action,
)


class TaskGenerationAttemptTests(unittest.TestCase):
    def test_stage_table_separates_local_repair_from_policy_outcomes(self):
        self.assertEqual(
            task_generation_recovery_action("success_spec", "invalid_spec"),
            REPAIR_SUCCESS_SPEC,
        )
        self.assertEqual(
            task_generation_recovery_action("vision_validation", "failed"),
            REPAIR_SCENE,
        )
        self.assertEqual(
            task_generation_recovery_action("expert_gate", "unsolvable"),
            REGENERATE_CANDIDATE,
        )
        self.assertEqual(
            task_generation_recovery_action("policy_execution", "policy_failure"),
            TERMINAL,
        )

    def test_typed_visual_failure_calls_repair_then_launches_act_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            actions = []
            launched = []

            def attempt(path: Path, index: int, action: str | None):
                actions.append(action)
                if index == 1:
                    raise TaskGenerationStageError(
                        "vision_validation",
                        "failed",
                        "wrong rendered color",
                        runtime={"simulator_probes": 1},
                        diagnosis={"field": "block.color"},
                    )
                self.assertEqual(action, REPAIR_SCENE)
                (path / "repaired.txt").write_text("ok", encoding="utf-8")
                return {
                    "status": "accepted",
                    "candidate_id": "candidate_02",
                    "runtime": {"simulator_probes": 1},
                }

            def launch(candidate):
                launched.append(candidate["candidate_id"])
                return {"act_rollouts_started": 1, "status": "policy_failure"}

            root = Path(temporary) / "taskgen_attempts"
            result = run_task_generation(
                root,
                execute=attempt,
                execute_after_acceptance=launch,
            )
            self.assertEqual(actions, [None, REPAIR_SCENE])
            self.assertEqual(launched, ["candidate_02"])
            self.assertEqual(result["status"], "accepted")
            self.assertEqual(result["repair"]["action"], REPAIR_SCENE)
            self.assertEqual(result["runtime"]["act_rollouts_started"], 0)
            self.assertEqual(
                result["post_acceptance_execution"]["act_rollouts_started"], 1
            )
            self.assertTrue((root / "generation").is_dir())
            self.assertTrue((root / "repair/repaired.txt").is_file())
            self.assertTrue((root / "task_generation_result.json").is_file())
            self.assertFalse((root / "attempt_01").exists())
            self.assertFalse((root / "attempt_02").exists())

    def test_policy_failure_is_terminal_and_never_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = {"count": 0}

            def attempt(_path: Path, _index: int, _action: str | None):
                calls["count"] += 1
                raise TaskGenerationStageError(
                    "policy_execution", "policy_failure", "ACT failed"
                )

            with self.assertRaises(TaskGenerationRecoveryError) as raised:
                run_task_generation(
                    Path(temporary) / "taskgen_attempts",
                    execute=attempt,
                )
            self.assertEqual(calls["count"], 1)
            self.assertEqual(
                raised.exception.result["generation"]["failure"]["stage"],
                "policy_execution",
            )

    def test_one_repair_then_reports_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "taskgen_attempts"

            def attempt(_path: Path, _index: int, _action: str | None):
                raise TaskGenerationStageError(
                    "static_validation", "failed", "bad AST"
                )

            with self.assertRaises(TaskGenerationRecoveryError) as raised:
                run_task_generation(
                    root,
                    execute=attempt,
                )
            self.assertEqual(raised.exception.result["repair"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
