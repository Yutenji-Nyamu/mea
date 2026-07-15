import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.manipeval_agent import build_taskgen_command
from scripts.manipeval_taskgen import collect_position_samples, newest_eval_dir


ROUND_2 = {
    "round_id": "round_2",
    "task_instruction": "position variation",
    "route": "reuse",
    "execution": {
        "seeds": [100002, 100003],
        "num_episodes": 2,
    },
}


class MultiRoundRuntimeTests(unittest.TestCase):
    def test_newest_eval_dir_never_reuses_stale_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_root = (
                root
                / "eval_result/beat_block_hammer/ACT/demo_clean/demo_clean"
            )
            old = eval_root / "old"
            old.mkdir(parents=True)
            before = {old}
            self.assertIsNone(newest_eval_dir(root, before))

            new = eval_root / "new"
            new.mkdir()
            self.assertEqual(newest_eval_dir(root, before), new)

    def test_taskgen_command_forwards_two_episodes(self):
        command, run_id = build_taskgen_command(
            Path("/repo"),
            "eval_test",
            ROUND_2,
            text_model="text",
            vision_model="vision",
            base_url=None,
            gpu=0,
            max_reflections=2,
        )
        index = command.index("--num-episodes")
        self.assertEqual(command[index + 1], "2")
        self.assertTrue(run_id.endswith("_round_2"))

    def test_collects_exact_simulator_positions(self):
        first_scene = {
            "block_pose": {
                "position": [0.12, 0.02, 0.76],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            },
            "rule_check": {"passed": True},
            "expert": {"passed": True},
            "image": "first.png",
        }
        second_scene = {
            "block_pose": {
                "position": [-0.16, 0.11, 0.76],
                "quaternion": [1.0, 0.0, 0.0, 0.0],
            },
            "rule_check": {"passed": True},
            "expert": {"passed": True},
            "image": "second.png",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            run_dir.mkdir()
            with patch(
                "scripts.manipeval_taskgen.run_probe",
                return_value=second_scene,
            ) as mocked_probe:
                result = collect_position_samples(
                    root,
                    run_dir,
                    {"task_name": "beat_block_hammer", "task_module": "fake"},
                    start_seed=100002,
                    num_episodes=2,
                    first_scene=first_scene,
                )
            self.assertEqual(mocked_probe.call_count, 1)
            self.assertTrue(result["passed"])
            self.assertTrue(result["metrics"]["position_varied"])
            self.assertEqual(result["metrics"]["unique_xy_count"], 2)
            self.assertAlmostEqual(result["metrics"]["x_span"], 0.28)
            self.assertTrue(
                (run_dir / "validation/position_samples.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
