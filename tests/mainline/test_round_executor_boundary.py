import inspect
import unittest
from pathlib import Path

import mea.round_executor as round_executor_module
from mea.round_executor import RoundExecutionResult, RoundExecutor
from scripts.manipeval_agent import build_production_round_executor


class RoundExecutorBoundaryTests(unittest.TestCase):
    def test_production_builder_returns_independent_executor(self):
        executor = build_production_round_executor()

        self.assertIsInstance(executor, RoundExecutor)
        self.assertNotIn(
            "scripts.manipeval_agent",
            inspect.getsource(round_executor_module),
        )
        self.assertEqual(
            set(executor._services.native_policy_rounds),
            {"act", "smolvla"},
        )

    def test_typed_result_preserves_legacy_tuple_order(self):
        result = RoundExecutionResult(
            child_manifest={"status": "completed"},
            child_dir=Path("/tmp/child"),
            round_summary={"pipeline_passed": True},
            tool_evaluation={"status": "passed"},
            returncode=0,
        )

        self.assertEqual(
            result.as_legacy_tuple(),
            (
                result.child_manifest,
                result.child_dir,
                result.round_summary,
                result.tool_evaluation,
                result.returncode,
            ),
        )


if __name__ == "__main__":
    unittest.main()
