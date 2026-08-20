from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


class ProductionCliBoundaryTests(unittest.TestCase):
    def test_round_allowance_is_positive_not_a_small_method_cap(self) -> None:
        from unittest.mock import patch

        from mea.agent_cli import parse_args

        with patch(
            "sys.argv",
            ["manipeval_agent.py", "--request", "q", "--generated-rounds", "10"],
        ):
            self.assertEqual(parse_args().generated_rounds, 10)
        with patch(
            "sys.argv",
            ["manipeval_agent.py", "--request", "q", "--generated-rounds", "0"],
        ), self.assertRaises(SystemExit):
            parse_args()

    def test_production_help_exposes_current_entry_options(self) -> None:
        process = subprocess.run(
            [sys.executable, "scripts/manipeval_agent.py", "--help"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        for option in (
            "--request",
            "--auto-route",
            "--policy-backend",
            "--generated-rounds",
        ):
            self.assertIn(option, process.stdout)

    def test_candidate_budget_matches_control_requirement(self) -> None:
        from mea.agent_cli import (
            resolve_plan_agent_candidate_budget,
            resolve_plan_agent_control_required,
        )

        self.assertTrue(resolve_plan_agent_control_required())
        self.assertEqual(
            resolve_plan_agent_candidate_budget(
                1,
                candidate_resolution=None,
            ),
            0,
        )

        resolution = {
            "resolution": "official_execution_from_typed_needs",
            "execution_authorized": True,
        }
        self.assertFalse(
            resolve_plan_agent_control_required(
                candidate_resolution=resolution,
            )
        )
        self.assertEqual(
            resolve_plan_agent_candidate_budget(
                1,
                candidate_resolution=resolution,
            ),
            1,
        )

        broad_resolution = {"resolution": "proposal_reuse_or_generate"}
        self.assertTrue(
            resolve_plan_agent_control_required(
                candidate_resolution=broad_resolution,
            )
        )


if __name__ == "__main__":
    unittest.main()
