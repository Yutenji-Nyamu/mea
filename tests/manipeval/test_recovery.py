import json
import tempfile
import unittest
from pathlib import Path

from mea.recovery import (
    BoundedRecoveryError,
    UnexpectedToolExecutionError,
    run_bounded_tool_recovery,
)


class RecoveryTests(unittest.TestCase):
    def test_unexpected_exception_restarts_once_without_act(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = {"calls": 0}

            def execute(_attempt_dir: Path, _attempt: int):
                state["calls"] += 1
                if state["calls"] == 1:
                    raise UnexpectedToolExecutionError("injected runtime failure")
                return {"status": "passed", "value": 1.0}

            result = run_bounded_tool_recovery(
                Path(temporary) / "attempts",
                logical_round_id="round_1",
                execute=execute,
                telemetry_sha256=lambda: "immutable-telemetry",
            )
            self.assertEqual(result["attempt_count"], 2)
            self.assertEqual(result["restarts_used"], 1)
            self.assertTrue(result["same_telemetry_reused"])
            self.assertEqual(
                result["additional_act_rollouts_started_by_recovery"], 0
            )
            self.assertFalse(result["policy_or_simulator_restarted"])
            self.assertTrue(result["provider_or_registry_work_may_repeat"])
            self.assertTrue(
                (
                    Path(temporary)
                    / "attempts/attempt_01/attempt_started.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    Path(temporary)
                    / "attempts/attempt_01/attempt_result.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    Path(temporary)
                    / "attempts/attempt_02/attempt_started.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    Path(temporary)
                    / "attempts/attempt_02/attempt_result.json"
                ).is_file()
            )

    def test_retry_budget_exhaustion_preserves_both_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempts"

            def execute(_attempt_dir: Path, attempt: int):
                raise UnexpectedToolExecutionError(f"failure {attempt}")

            with self.assertRaises(BoundedRecoveryError):
                run_bounded_tool_recovery(
                    root,
                    logical_round_id="round_1",
                    execute=execute,
                    telemetry_sha256=lambda: "same",
                )
            summary = json.loads((root / "recovery_summary.json").read_text())
            self.assertEqual(summary["attempt_count"], 2)
            self.assertEqual(summary["restarts_used"], 1)
            self.assertEqual(summary["status"], "failed")

    def test_unclassified_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempts"
            calls = {"count": 0}

            def execute(_attempt_dir: Path, _attempt: int):
                calls["count"] += 1
                raise ValueError("semantic validation failure")

            with self.assertRaises(BoundedRecoveryError):
                run_bounded_tool_recovery(
                    root,
                    logical_round_id="round_1",
                    execute=execute,
                    telemetry_sha256=lambda: "same",
                )
            self.assertEqual(calls["count"], 1)

    def test_changed_telemetry_fails_integrity_before_second_execute(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempts"
            hashes = iter(("first", "first", "changed"))
            calls = {"count": 0}

            def execute(_attempt_dir: Path, _attempt: int):
                calls["count"] += 1
                raise UnexpectedToolExecutionError("retry me")

            with self.assertRaises(BoundedRecoveryError):
                run_bounded_tool_recovery(
                    root,
                    logical_round_id="round_1",
                    execute=execute,
                    telemetry_sha256=lambda: next(hashes),
                )
            self.assertEqual(calls["count"], 1)
            summary = json.loads((root / "recovery_summary.json").read_text())
            self.assertFalse(summary["same_telemetry_reused"])


if __name__ == "__main__":
    unittest.main()
