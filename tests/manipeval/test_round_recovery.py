import json
import tempfile
import unittest
from pathlib import Path

from mea.round_recovery import (
    FAIL_EVALUATION,
    RECORD_POLICY_FAILURE,
    REGENERATE_TASK,
    REGENERATE_TOOL,
    RESTART_PLANNING,
    RESTART_WHOLE_ROUND,
    StageFailure,
    WholeRoundRecoveryError,
    recovery_action,
    run_stage_aware_round_recovery,
)


def runtime(*, act: int = 0, simulator: bool = False):
    return {
        "provider_called": False,
        "simulator_called": simulator,
        "act_rollouts_started": act,
    }


class RoundRecoveryTests(unittest.TestCase):
    def test_central_stage_table_allows_only_tool_exception_to_restart_round(self):
        self.assertEqual(
            recovery_action("planning", "ground_truth_disagreement"),
            RESTART_PLANNING,
        )
        self.assertEqual(
            recovery_action("task_generation", "visual_self_check_failed"),
            REGENERATE_TASK,
        )
        self.assertEqual(
            recovery_action("tool_generation", "unit_test_failed"),
            REGENERATE_TOOL,
        )
        self.assertEqual(
            recovery_action("tool_execution", "unexpected_exception"),
            RESTART_WHOLE_ROUND,
        )
        self.assertEqual(
            recovery_action("policy_execution", "policy_failure"),
            RECORD_POLICY_FAILURE,
        )
        self.assertEqual(
            recovery_action("simulation", "engine_failure"),
            RECORD_POLICY_FAILURE,
        )
        self.assertEqual(
            recovery_action("tool_execution", "contract_failure"),
            FAIL_EVALUATION,
        )

    def test_unexpected_tool_exception_restarts_the_entire_round_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "round_recovery"
            calls = []

            def execute(attempt_dir: Path, attempt: int):
                calls.append((attempt_dir.name, attempt))
                if attempt == 1:
                    raise StageFailure(
                        "tool_execution",
                        "unexpected_exception",
                        "injected Tool runtime error",
                        runtime=runtime(act=1, simulator=True),
                    )
                return {
                    "status": "passed",
                    "child_run_id": "run_eval_round_attempt_02",
                    "runtime": runtime(act=1, simulator=True),
                }

            summary = run_stage_aware_round_recovery(
                root,
                logical_round_id="round_1",
                round_identity={
                    "round_plan_sha256": "a" * 64,
                    "seed": 100401,
                    "checkpoint_sha256": "b" * 64,
                },
                execute_round=execute,
            )
            self.assertEqual(calls, [("attempt_01", 1), ("attempt_02", 2)])
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["restarts_used"], 1)
            self.assertTrue(summary["whole_round_restarted"])
            self.assertTrue(summary["policy_or_simulator_restarted"])
            self.assertEqual(summary["runtime"]["act_rollouts_started"], 2)
            self.assertEqual(
                summary["additional_act_rollouts_started_by_recovery"], 1
            )
            self.assertEqual(
                summary["attempts"][0]["recovery_action"], RESTART_WHOLE_ROUND
            )
            self.assertTrue((root / "attempt_01/attempt_result.json").is_file())
            self.assertTrue((root / "attempt_02/attempt_result.json").is_file())
            self.assertTrue((root / "recovery_summary.json").is_file())

    def test_policy_and_simulator_failures_are_terminal_policy_outcomes(self):
        for stage, kind in (
            ("policy_execution", "policy_failure"),
            ("simulation", "engine_failure"),
        ):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                calls = {"count": 0}

                def execute(_attempt_dir: Path, _attempt: int):
                    calls["count"] += 1
                    raise StageFailure(
                        stage,
                        kind,
                        "policy could not complete",
                        runtime=runtime(act=1, simulator=True),
                    )

                summary = run_stage_aware_round_recovery(
                    Path(temporary) / "round_recovery",
                    logical_round_id="round_1",
                    round_identity={"seed": 7},
                    execute_round=execute,
                )
                self.assertEqual(calls["count"], 1)
                self.assertEqual(summary["status"], "completed_with_policy_failure")
                self.assertEqual(summary["restarts_used"], 0)
                self.assertFalse(summary["whole_round_restarted"])
                self.assertEqual(
                    summary["attempts"][0]["recovery_action"],
                    RECORD_POLICY_FAILURE,
                )

    def test_local_regeneration_actions_do_not_become_round_restarts(self):
        for stage, kind, action in (
            ("task_generation", "visual_self_check_failed", REGENERATE_TASK),
            ("tool_generation", "unit_test_failed", REGENERATE_TOOL),
        ):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "round_recovery"
                calls = {"count": 0}

                def execute(_attempt_dir: Path, _attempt: int):
                    calls["count"] += 1
                    raise StageFailure(stage, kind, "local stage budget exhausted")

                with self.assertRaises(WholeRoundRecoveryError) as raised:
                    run_stage_aware_round_recovery(
                        root,
                        logical_round_id="round_1",
                        round_identity={"seed": 7},
                        execute_round=execute,
                    )
                self.assertEqual(calls["count"], 1)
                self.assertEqual(raised.exception.summary["restarts_used"], 0)
                self.assertEqual(
                    raised.exception.summary["attempts"][0]["recovery_action"],
                    action,
                )

    def test_retry_exhaustion_and_append_only_root_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "round_recovery"

            def execute(_attempt_dir: Path, attempt: int):
                raise StageFailure(
                    "tool_execution",
                    "unexpected_exception",
                    f"failure {attempt}",
                    runtime=runtime(act=1, simulator=True),
                )

            with self.assertRaises(WholeRoundRecoveryError) as raised:
                run_stage_aware_round_recovery(
                    root,
                    logical_round_id="round_1",
                    round_identity={"seed": 7},
                    execute_round=execute,
                )
            self.assertEqual(raised.exception.summary["attempt_count"], 2)
            self.assertEqual(raised.exception.summary["restarts_used"], 1)
            persisted = json.loads((root / "recovery_summary.json").read_text())
            self.assertEqual(persisted["status"], "failed")
            with self.assertRaisesRegex(WholeRoundRecoveryError, "already exists"):
                run_stage_aware_round_recovery(
                    root,
                    logical_round_id="round_1",
                    round_identity={"seed": 7},
                    execute_round=execute,
                )


if __name__ == "__main__":
    unittest.main()
