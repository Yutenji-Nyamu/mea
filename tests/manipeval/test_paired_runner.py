import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.paired import build_seed_manifest
from scripts.manipeval_paired import (
    act_command,
    classify_probe,
    merge_condition_measurements,
    probe_command,
    run_paired,
)


class PairedRunnerTests(unittest.TestCase):
    def test_probe_status_is_explicit_and_never_implies_replacement(self):
        passed = {
            "setup_success": True,
            "render_success": True,
            "rule_check": {"passed": True},
            "expert": {"passed": True},
        }
        self.assertEqual(classify_probe(0, passed), "passed")
        self.assertEqual(
            classify_probe(1, {"error": {"type": "UnStableError"}}),
            "unstable",
        )
        self.assertEqual(
            classify_probe(2, {"expert": {"passed": False}}),
            "expert_failed",
        )
        self.assertEqual(classify_probe(1, {}), "error")

    def test_paired_probe_uses_the_act_evaluation_distribution(self):
        command = probe_command(
            repo_root=Path("repo"),
            task_name="click_bell",
            task_module="envs.click_bell",
            task_config="demo_randomized",
            checkpoint_setting="demo_clean",
            seed=7,
            episode_index=0,
            image_path=Path("image.png"),
            result_path=Path("probe.json"),
        )
        self.assertIn("--eval-mode", command)

    def test_act_command_passes_exact_manifest_as_thirteenth_argument(self):
        manifest = build_seed_manifest(task_name="click_bell", seeds=[7, 11])
        condition = manifest["conditions"][1]
        command = act_command(
            manifest=manifest,
            condition=condition,
            task_module="envs.click_bell",
            gpu=3,
            telemetry_profile="balanced_v1",
            selected_manifest_path=Path("selected.json"),
            result_path=Path("result.json"),
            telemetry_root=Path("telemetry"),
            output_dir=Path("output"),
        )
        self.assertEqual(command[3], "demo_randomized")
        self.assertEqual(command[8], "2")
        self.assertEqual(command[10], "")
        self.assertEqual(command[11], "7")
        self.assertEqual(command[14:], ["selected.json", "result.json", "output"])

    def test_merge_rejects_recheck_drift_as_protocol_violation(self):
        rows = merge_condition_measurements(
            candidate_seeds=[7],
            selected_seeds=[7],
            probe_rows=[
                {
                    "seed": 7,
                    "eligibility_status": "passed",
                    "returncode": 0,
                    "artifacts": {},
                }
            ],
            exact_result={
                "seed_measurements": [
                    {
                        "seed": 7,
                        "eligibility_status": "unstable",
                        "policy_executed": False,
                        "policy_success": None,
                    }
                ]
            },
            tools_by_seed={},
        )
        self.assertEqual(rows[0]["eligibility_status"], "protocol_violation")
        self.assertFalse(rows[0]["policy_executed"])

    def test_dry_run_needs_no_uiui_key_and_writes_no_run_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = (
                root
                / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "policy_last.ckpt").write_bytes(b"test")
            (checkpoint / "dataset_stats.pkl").write_bytes(b"test")
            arguments = argparse.Namespace(
                repo_root=root,
                manifest=None,
                task_name="click_bell",
                task_module=None,
                seeds=[7],
                run_id="run_dry",
                gpu=0,
                telemetry_profile="balanced_v1",
                dry_run=True,
                allow_protocol_violations=False,
            )
            with patch.dict(os.environ, {}, clear=True):
                result = run_paired(arguments)
            self.assertEqual(result["status"], "dry_run")
            self.assertFalse(result["requires_uiui"])
            self.assertFalse((root / "mea/paired_runs/run_dry").exists())


if __name__ == "__main__":
    unittest.main()
