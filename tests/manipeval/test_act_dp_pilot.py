import tempfile
import unittest
from pathlib import Path

from mea.act_dp_pilot import (
    ACT_CHECKPOINT_REF,
    ACT_STATS_REF,
    DP_CHECKPOINT_REF,
    ActDpPilotError,
    build_act_dp_readiness,
)


def write(root: Path, ref: str, payload: bytes = b"x") -> Path:
    path = root / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


class ActDpPilotTests(unittest.TestCase):
    def test_missing_dp_is_reported_without_substitution_or_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, ACT_CHECKPOINT_REF)
            write(root, ACT_STATS_REF)
            write(root, "policy/DP/deploy_policy.yml", b"policy_name: DP\n")
            write(root, "script/eval_policy.py")
            act_python = write(root, "envs/act/python")

            report = build_act_dp_readiness(
                root,
                act_python=str(act_python),
                dp_python=str(root / "envs/dp/python"),
            )

            self.assertEqual(report["status"], "blocked_missing_prerequisites")
            self.assertFalse(report["live_execution_authorized"])
            self.assertIn("dp_checkpoint", report["missing_requirements"])
            self.assertIn("dp_environment", report["missing_requirements"])
            self.assertIsNone(report["command_templates"])
            self.assertIn("dp3", report["substitution_forbidden"])
            self.assertEqual(sum(report["calls_started"].values()), 0)

    def test_ready_report_freezes_exact_act_and_dp_commands(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(root, ACT_CHECKPOINT_REF, b"act")
            write(root, ACT_STATS_REF, b"stats")
            write(root, DP_CHECKPOINT_REF, b"dp")
            write(root, "policy/DP/deploy_policy.yml", b"policy_name: DP\n")
            write(root, "script/eval_policy.py")
            act_python = write(root, "envs/act/python")
            dp_python = write(root, "envs/dp/python")

            report = build_act_dp_readiness(
                root,
                act_python=str(act_python),
                dp_python=str(dp_python),
            )

            self.assertEqual(report["status"], "ready")
            self.assertTrue(report["live_execution_authorized"])
            self.assertEqual(report["missing_requirements"], [])
            self.assertEqual(len(report["command_templates"]["act"]), 3)
            self.assertEqual(len(report["command_templates"]["dp"]), 3)
            dp_argv = report["command_templates"]["dp"][0]["argv"]
            self.assertIn("DP", dp_argv)
            self.assertIn("600", dp_argv)
            self.assertNotIn("DP3", dp_argv)

    def test_seed_contract_rejects_non_n3_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ActDpPilotError, "exactly three"):
                build_act_dp_readiness(temporary, seeds=[1, 2])


if __name__ == "__main__":
    unittest.main()
