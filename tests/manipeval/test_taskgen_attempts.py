import json
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
    run_bounded_task_generation,
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
            summary = run_bounded_task_generation(
                root,
                proposal_identity={"proposal_id": "proposal_1", "seed": 100401},
                execute_attempt=attempt,
                execute_after_acceptance=launch,
            )
            self.assertEqual(actions, [None, REPAIR_SCENE])
            self.assertEqual(launched, ["candidate_02"])
            self.assertEqual(summary["status"], "accepted")
            self.assertEqual(summary["regenerations_used"], 1)
            self.assertEqual(summary["runtime"]["act_rollouts_started"], 0)
            self.assertEqual(
                summary["post_acceptance_execution"]["act_rollouts_started"], 1
            )
            persisted = json.loads(
                (root / "task_generation_attempt_summary.json").read_text()
            )
            self.assertFalse(persisted["policy_retry_allowed"])

    def test_policy_failure_is_terminal_and_never_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = {"count": 0}

            def attempt(_path: Path, _index: int, _action: str | None):
                calls["count"] += 1
                raise TaskGenerationStageError(
                    "policy_execution", "policy_failure", "ACT failed"
                )

            with self.assertRaises(TaskGenerationRecoveryError) as raised:
                run_bounded_task_generation(
                    Path(temporary) / "taskgen_attempts",
                    proposal_identity={"proposal_id": "proposal_1"},
                    execute_attempt=attempt,
                )
            self.assertEqual(calls["count"], 1)
            self.assertEqual(raised.exception.summary["attempt_count"], 1)
            self.assertEqual(
                raised.exception.summary["attempts"][0]["recovery_action"], TERMINAL
            )

    def test_budget_and_append_only_root_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "taskgen_attempts"

            def attempt(_path: Path, _index: int, _action: str | None):
                raise TaskGenerationStageError(
                    "static_validation", "failed", "bad AST"
                )

            with self.assertRaises(TaskGenerationRecoveryError) as raised:
                run_bounded_task_generation(
                    root,
                    proposal_identity={"proposal_id": "proposal_1"},
                    execute_attempt=attempt,
                )
            self.assertEqual(raised.exception.summary["attempt_count"], 2)
            with self.assertRaisesRegex(
                TaskGenerationRecoveryError, "already exists"
            ):
                run_bounded_task_generation(
                    root,
                    proposal_identity={"proposal_id": "proposal_1"},
                    execute_attempt=attempt,
                )


if __name__ == "__main__":
    unittest.main()
