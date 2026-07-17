import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.toolgen import contact_tool_request
from scripts.manipeval_agent import build_taskgen_command, execute_round
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

    def test_execute_round_routes_planned_tool_before_summary(self):
        tool_evaluation = {
            "schema_version": 1,
            "status": "passed",
            "route": "reuse",
            "reference_tool": "hammer_block_contact_ever",
            "source": {
                "scope": "trusted_catalog",
                "tool": "hammer_block_contact_ever",
            },
            "episodes": [
                {
                    "policy_name": "ACT",
                    "seed": 100002,
                    "role": "policy_under_evaluation",
                    "result": {
                        "value": False,
                        "passed": False,
                        "evidence_steps": [],
                        "details": {},
                    },
                }
            ],
            "validation": {"provider_called": False},
            "artifacts": {"tool_execution": "tool_execution.json"},
        }
        round_plan = {
            **ROUND_2,
            "sub_aspect": "object_position",
            "tool_request": contact_tool_request(),
        }
        child_manifest = {
            "run_id": "run_test_round_2",
            "status": "completed",
            "scene_validation": {
                "rule_check": {"passed": True},
                "expert": {"passed": True},
            },
            "vision_validation": {"passed": True, "observed_color": "blue"},
            "act_evaluation": {"passed": True},
            "position_samples": {"passed": True, "samples": [], "metrics": {}},
            "trusted_tool_evaluation": {"episodes": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            evaluation_dir = repo_root / "mea/evaluation_runs/eval_test"
            evaluation_dir.mkdir(parents=True)
            (evaluation_dir / "manifest.json").write_text(
                json.dumps({"status": "planned"}), encoding="utf-8"
            )
            child_dir = repo_root / "mea/generated_tasks/run_test_round_2"
            child_dir.mkdir(parents=True)
            (child_dir / "manifest.json").write_text(
                json.dumps(child_manifest), encoding="utf-8"
            )
            telemetry_dir = child_dir / "evaluation/telemetry"
            telemetry_dir.mkdir(parents=True)
            (telemetry_dir / "fixture.json").write_text(
                json.dumps({"seed": 1001, "source": "unit_test"}),
                encoding="utf-8",
            )
            provider = object()
            with (
                patch("scripts.manipeval_agent.run_logged", return_value=0),
                patch(
                    "scripts.manipeval_agent.execute_tool_request",
                    return_value=tool_evaluation,
                ) as routed_tool,
                patch(
                    "scripts.manipeval_agent.run_round_execution_vqa",
                    return_value={
                        "status": "passed",
                        "evidence_conflict": False,
                    },
                ),
            ):
                (
                    returned_manifest,
                    returned_child,
                    summary,
                    returned_tool,
                    returncode,
                ) = execute_round(
                    repo_root,
                    evaluation_dir,
                    "eval_test",
                    round_plan,
                    text_model="text",
                    vision_model="vision",
                    base_url=None,
                    gpu=0,
                    max_reflections=1,
                    provider=provider,
                    toolgen_model="tool-model",
                )
            self.assertEqual(returncode, 0)
            self.assertEqual(returned_manifest, child_manifest)
            self.assertEqual(returned_child, child_dir)
            self.assertEqual(returned_tool, tool_evaluation)
            self.assertTrue(summary["pipeline_passed"])
            self.assertEqual(
                summary["observations"]["planned_tool"]["route"], "reuse"
            )
            routed_tool.assert_called_once_with(
                repo_root,
                child_dir,
                evaluation_dir / "execution/round_2/planned_tool",
                round_plan["tool_request"],
                provider=provider,
                model="tool-model",
            )

            failed_evaluation_dir = (
                repo_root / "mea/evaluation_runs/eval_fail"
            )
            failed_evaluation_dir.mkdir(parents=True)
            (failed_evaluation_dir / "manifest.json").write_text(
                json.dumps({"status": "planned"}), encoding="utf-8"
            )
            failed_child_dir = (
                repo_root / "mea/generated_tasks/run_fail_round_2"
            )
            failed_child_dir.mkdir(parents=True)
            failed_manifest = {
                **child_manifest,
                "run_id": "run_fail_round_2",
            }
            (failed_child_dir / "manifest.json").write_text(
                json.dumps(failed_manifest), encoding="utf-8"
            )
            with (
                patch("scripts.manipeval_agent.run_logged", return_value=7),
                patch(
                    "scripts.manipeval_agent.execute_tool_request"
                ) as skipped_tool,
            ):
                (
                    _,
                    _,
                    failed_summary,
                    skipped_evaluation,
                    failed_returncode,
                ) = execute_round(
                    repo_root,
                    failed_evaluation_dir,
                    "eval_fail",
                    round_plan,
                    text_model="text",
                    vision_model="vision",
                    base_url=None,
                    gpu=0,
                    max_reflections=1,
                    provider=provider,
                    toolgen_model="tool-model",
                )
            self.assertEqual(failed_returncode, 7)
            self.assertEqual(skipped_evaluation["status"], "skipped")
            self.assertEqual(skipped_evaluation["requested_route"], "auto")
            self.assertIsNone(skipped_evaluation["route"])
            self.assertIn(
                "code 7", skipped_evaluation["validation"]["reason"]
            )
            self.assertFalse(failed_summary["pipeline_passed"])
            self.assertEqual(failed_summary["taskgen_returncode"], 7)
            skipped_tool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
