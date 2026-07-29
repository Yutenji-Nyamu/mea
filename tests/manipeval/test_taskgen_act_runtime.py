import tempfile
import unittest
from pathlib import Path

from mea.taskgen.act_runtime import build_act_command, newest_eval_dir


class TaskGenActRuntimeTests(unittest.TestCase):
    def test_command_builder_preserves_eval_mea_positional_contract(self):
        receipt = Path("/run/act_execution_receipt.json")
        command = build_act_command(
            python_executable="/venv/bin/python",
            task_name="click_bell",
            task_config="demo_clean",
            checkpoint_setting="demo_clean",
            expert_data_num=50,
            policy_seed=0,
            gpu=1,
            num_episodes=1,
            task_module="envs.click_bell",
            overlay_path=Path("/run/overlay.yml"),
            seed=7,
            telemetry_root=Path("/run/telemetry/act"),
            telemetry_profile="balanced_v1",
            execution_receipt=receipt,
        )

        self.assertEqual(
            command[:4],
            [
                "env",
                "PYTHON_BIN=/venv/bin/python",
                "bash",
                "policy/ACT/eval_mea.sh",
            ],
        )
        self.assertEqual(
            command[4:16],
            [
                "click_bell",
                "demo_clean",
                "demo_clean",
                "50",
                "0",
                "1",
                "1",
                "envs.click_bell",
                "/run/overlay.yml",
                "7",
                "/run/telemetry/act",
                "balanced_v1",
            ],
        )
        self.assertEqual(command[16:], ["", "", "", str(receipt)])

    def test_execution_receipt_rejects_multi_episode_invocation(self):
        with self.assertRaisesRegex(
            RuntimeError,
            "execution receipt ACT runs require num_episodes=1",
        ):
            build_act_command(
                python_executable="/venv/bin/python",
                task_name="beat_block_hammer",
                task_config="demo_clean",
                checkpoint_setting="demo_clean",
                expert_data_num=50,
                policy_seed=0,
                gpu=0,
                num_episodes=2,
                task_module="envs.beat_block_hammer",
                overlay_path=Path("/run/overlay.yml"),
                seed=7,
                telemetry_root=Path("/run/telemetry/act"),
                telemetry_profile="balanced_v1",
                execution_receipt=Path("/run/receipt.json"),
            )

    def test_newest_eval_dir_ignores_preexisting_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            eval_root = (
                root
                / "eval_result/click_bell/ACT/demo_clean/demo_clean"
            )
            old = eval_root / "old"
            old.mkdir(parents=True)
            before = {old}

            self.assertIsNone(
                newest_eval_dir(root, before, task_name="click_bell")
            )
            new = eval_root / "new"
            new.mkdir()
            self.assertEqual(
                newest_eval_dir(root, before, task_name="click_bell"),
                new,
            )


if __name__ == "__main__":
    unittest.main()
