import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_act_checkpoint import checkpoint_patterns, main


class DownloadActCheckpointTests(unittest.TestCase):
    def test_builds_two_paths_per_unique_task(self):
        self.assertEqual(
            checkpoint_patterns(["click_bell", "grab_roller", "click_bell"]),
            [
                "act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt",
                "act_ckpt/act-click_bell/demo_clean-50/dataset_stats.pkl",
                "act_ckpt/act-grab_roller/demo_clean-50/policy_last.ckpt",
                "act_ckpt/act-grab_roller/demo_clean-50/dataset_stats.pkl",
            ],
        )

    def test_rejects_path_traversal_and_empty_input(self):
        for task_name in ("../click_bell", "click-bell", ""):
            with self.subTest(task_name=task_name):
                with self.assertRaises(ValueError):
                    checkpoint_patterns([task_name])
        with self.assertRaises(ValueError):
            checkpoint_patterns([])

    def test_download_failure_points_to_server_side_acceleration(self):
        fake_hub = types.ModuleType("huggingface_hub")

        def fail_download(**kwargs):
            raise OSError("network unreachable")

        fake_hub.snapshot_download = fail_download
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            sys.modules, {"huggingface_hub": fake_hub}
        ), patch(
            "sys.argv",
            [
                "download_act_checkpoint.py",
                "--local-dir",
                str(Path(temporary) / "act"),
                "click_bell",
            ],
        ):
            with self.assertRaises(SystemExit) as raised:
                main()
        message = str(raised.exception)
        self.assertIn("AutoDL", message)
        self.assertIn("HF_HUB_DOWNLOAD_TIMEOUT=300", message)
        self.assertIn("server-side HF_ENDPOINT", message)
        self.assertIn("Do not relay", message)


if __name__ == "__main__":
    unittest.main()
