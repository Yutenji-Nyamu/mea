import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.recovery import BoundedRecoveryError
from scripts.manipeval_agent import execute_round_stage_aware


class AgentWholeRoundRecoveryIntegrationTests(unittest.TestCase):
    def test_unexpected_tool_exception_gets_new_child_and_whole_round_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_dir = root / "mea/evaluation_runs/eval_recovery"
            (evaluation_dir / "summary").mkdir(parents=True)
            generated = root / "mea/generated_tasks"
            generated.mkdir(parents=True)
            calls = []

            def fake_execute(*args, **kwargs):
                attempt = kwargs["round_attempt_index"]
                calls.append(attempt)
                suffix = "" if attempt == 1 else f"_attempt_{attempt:02d}"
                run_id = f"run_recovery_round_1{suffix}"
                child_dir = generated / run_id
                child_dir.mkdir()
                child = {
                    "run_id": run_id,
                    "act_evaluation": {"actual_seeds": [7]},
                    "scene_validation": {"render_success": True},
                    "provider": {
                        "calls": {"proposal": {"request_id": "test"}}
                    },
                }
                (child_dir / "manifest.json").write_text(
                    json.dumps(child), encoding="utf-8"
                )
                if attempt == 1:
                    recovery = (
                        evaluation_dir
                        / "execution/round_1/tool_recovery/recovery_summary.json"
                    )
                    recovery.parent.mkdir(parents=True)
                    recovery.write_text(
                        json.dumps(
                            {
                                "failure_class": (
                                    "unexpected_tool_execution_exception"
                                )
                            }
                        ),
                        encoding="utf-8",
                    )
                    raise BoundedRecoveryError("injected unexpected Tool failure")
                summary = {
                    "round_id": "round_1",
                    "pipeline_passed": True,
                    "execution_artifact_dir": (
                        "mea/evaluation_runs/eval_recovery/execution/round_1/"
                        "round_attempt_02"
                    ),
                    "observations": {
                        "execution_vqa": {
                            "status": "passed",
                            "model_requested": "vision",
                        }
                    },
                }
                tool = {"route_decision": {"provider_called": False}}
                return child, child_dir, summary, tool, 0

            round_plan = {
                "round_id": "round_1",
                "execution": {"seeds": [7], "num_episodes": 1},
            }
            with patch("scripts.manipeval_agent.execute_round", side_effect=fake_execute):
                result = execute_round_stage_aware(
                    root,
                    evaluation_dir,
                    "eval_recovery",
                    round_plan,
                    text_model="text",
                    vision_model="vision",
                    base_url=None,
                    gpu=0,
                    max_reflections=1,
                    provider=object(),
                    toolgen_model="tool",
                    round_recovery_max_restarts=1,
                    inject_tool_exception_once=True,
                )
            self.assertEqual(calls, [1, 2])
            self.assertEqual(result[0]["run_id"], "run_recovery_round_1_attempt_02")
            recovery = result[2]["observations"]["whole_round_recovery"]
            self.assertTrue(recovery["whole_round_restarted"])
            self.assertEqual(recovery["attempt_count"], 2)
            self.assertEqual(recovery["additional_act_rollouts_started_by_recovery"], 1)
            self.assertTrue(recovery["runtime"]["provider_called"])
            self.assertTrue(recovery["runtime"]["simulator_called"])
            self.assertEqual(
                recovery["attempts"][0]["recovery_action"], "restart_whole_round"
            )
            self.assertTrue(
                (
                    evaluation_dir
                    / "execution/round_1/whole_round_recovery/recovery_summary.json"
                ).is_file()
            )


if __name__ == "__main__":
    unittest.main()
