import json
import tempfile
import unittest
from pathlib import Path

from mea.feedback import FeedbackAgent, render_evaluation_report


FEEDBACK = {
    "answer": "The requested blue-block scene was generated and evaluated.",
    "evaluation_scope": "One blue-block episode at seed 100000.",
    "findings": [
        "The render and vision checks observed a blue block.",
        "The evaluation pipeline completed, while policy success was 0/1.",
    ],
    "limitations": ["A single episode cannot establish generalization."],
    "recommended_next_step": "Evaluate multiple positions and seeds.",
}


class FakeProvider:
    last_metadata = {"model": "fake-feedback"}

    def text(self, prompt, **kwargs):
        return json.dumps(FEEDBACK)


class ContradictoryThenCorrectProvider:
    last_metadata = {"model": "fake-feedback"}

    def __init__(self):
        self.calls = 0

    def text(self, prompt, **kwargs):
        self.calls += 1
        value = dict(FEEDBACK)
        if self.calls == 1:
            value["findings"] = ["ACT 管道正常，任务成功完成。"]
        else:
            value["answer"] = "Pipeline completed, but the policy did not complete the task."
        return json.dumps(value)


class AlwaysContradictoryProvider:
    last_metadata = {"model": "fake-feedback"}

    def text(self, prompt, **kwargs):
        value = dict(FEEDBACK)
        value["answer"] = "ACT 表现符合任务要求，任务成功完成。"
        value["findings"] = ["ACT 管道正常，任务成功完成。"]
        return json.dumps(value)


class FeedbackAgentTests(unittest.TestCase):
    def test_generates_feedback_and_unified_report(self):
        repo_root = Path(__file__).resolve().parents[2]
        evidence = {
            "evaluation_id": "eval_test",
            "child_run_id": "run_test",
            "user_request": "Evaluate a blue block.",
            "sub_aspect": "object_appearance",
            "task_instruction": "Make only the block blue.",
            "route": "force_codegen",
            "seed": 100000,
            "num_episodes": 1,
            "task_retrieval": {
                "catalog_size": 50,
                "selected_tasks": [
                    "beat_block_hammer",
                    "blocks_ranking_rgb",
                ],
                "reasoning": "Use behavior and RGB examples.",
            },
            "observations": {
                "scene_alignment": True,
                "observed_color": "blue",
                "expert_solvable": True,
                "act_pipeline_status": True,
                "policy_success": 0.0,
                "pipeline_passed": True,
            },
            "visual_self_reflection": {
                "passed": True,
                "repairs_used": 1,
                "attempt_count": 2,
            },
            "artifacts": {"scene_image": "evidence/initial_head.png"},
        }
        with tempfile.TemporaryDirectory() as temp:
            feedback = FeedbackAgent(
                repo_root,
                FakeProvider(),
                model="fake-feedback",
            ).generate(evidence, output_dir=Path(temp))
            self.assertEqual(feedback["answer"], FEEDBACK["answer"])
            self.assertTrue((Path(temp) / "feedback.json").is_file())
            report = render_evaluation_report(evidence, feedback)
            self.assertIn("`beat_block_hammer`", report)
            self.assertIn("policy success: `0.0`", report)
            self.assertIn("visual repairs used: `1`", report)
            self.assertIn(FEEDBACK["recommended_next_step"], report)

    def test_retries_feedback_that_contradicts_policy_result(self):
        repo_root = Path(__file__).resolve().parents[2]
        evidence = {
            "observations": {
                "pipeline_passed": True,
                "policy_success": 0.0,
            }
        }
        provider = ContradictoryThenCorrectProvider()
        with tempfile.TemporaryDirectory() as temp:
            feedback = FeedbackAgent(
                repo_root,
                provider,
                model="fake-feedback",
            ).generate(evidence, output_dir=Path(temp))
            self.assertEqual(provider.calls, 2)
            self.assertEqual(
                feedback["consistency_validation"]["rejected_responses"],
                1,
            )
            self.assertTrue((Path(temp) / "retry_response.txt").is_file())

    def test_applies_deterministic_guard_after_two_contradictions(self):
        repo_root = Path(__file__).resolve().parents[2]
        evidence = {
            "observations": {
                "pipeline_passed": True,
                "policy_success": 0.0,
            }
        }
        with tempfile.TemporaryDirectory() as temp:
            feedback = FeedbackAgent(
                repo_root,
                AlwaysContradictoryProvider(),
                model="fake-feedback",
            ).generate(evidence, output_dir=Path(temp))
            self.assertIn("未完成任务", feedback["answer"])
            self.assertTrue(
                feedback["consistency_validation"]["deterministic_correction"]
            )
            self.assertEqual(
                feedback["consistency_validation"]["rejected_responses"],
                2,
            )


if __name__ == "__main__":
    unittest.main()
