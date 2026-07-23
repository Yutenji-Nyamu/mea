import json
import tempfile
import unittest
from pathlib import Path

from mea.taskgen import (
    execute_reflection_loop,
    validate_vision_observation,
)
from mea.taskgen.reflection import _repair_prompt
from scripts.manipeval_taskgen import run_vision_check


SPEC = {
    "task_name": "beat_block_hammer",
    "changes": {
        "block": {
            "color": [0.0, 0.2, 1.0],
            "scale": 1.0,
            "position_mode": "official_random",
            "yaw_mode": "official_random",
        }
    }
}


class CaptureVisionProvider:
    def __init__(self):
        self.prompt = None
        self.last_metadata = {"model": "fake-vision"}

    def vision(self, prompt, image_path, **kwargs):
        self.prompt = prompt
        return json.dumps(
            {
                "aligned": True,
                "target_actor": "block",
                "observed_color": "blue",
                "unexpected_changes": [],
                "diagnosis": "The visible scene is plausible.",
                "suggestions": [],
                "confidence": 0.9,
            }
        )


class VisualReflectionTests(unittest.TestCase):
    def test_failed_observation_triggers_one_repair_then_passes(self):
        state = {"repaired": False}

        def observe(attempt_index):
            if not state["repaired"]:
                return {
                    "passed": False,
                    "vision": {
                        "diagnosis": "The block is visibly oversized.",
                        "suggestions": ["Restore half_size to 0.025."],
                    },
                }
            return {"passed": True, "vision": {"diagnosis": "Aligned."}}

        def repair(repair_index, observation):
            self.assertEqual(repair_index, 1)
            self.assertIn("oversized", observation["vision"]["diagnosis"])
            state["repaired"] = True
            return {"installed": True}

        result = execute_reflection_loop(
            max_repairs=2,
            observe=observe,
            repair=repair,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["repairs_used"], 1)
        self.assertEqual(result["final_attempt"], 1)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertTrue(result["attempts"][0]["repair"]["installed"])

    def test_exhausted_budget_is_reported(self):
        result = execute_reflection_loop(
            max_repairs=1,
            observe=lambda attempt: {"passed": False},
            repair=lambda index, observation: {"installed": True},
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["repairs_used"], 1)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertIn("exhausted", result["failure_reason"])

    def test_vision_contract_combines_alignment_and_expected_color(self):
        mismatch = validate_vision_observation(
            {
                "aligned": False,
                "observed_color": "blue",
                "unexpected_changes": ["block is oversized"],
                "diagnosis": "Scale mismatch.",
                "suggestions": ["Use half_size 0.025."],
                "confidence": 0.9,
            },
            SPEC,
        )
        self.assertFalse(mismatch["passed"])
        self.assertTrue(mismatch["color_matches"])

        aligned = validate_vision_observation(
            {
                "aligned": True,
                "observed_color": "蓝色",
                "unexpected_changes": [],
                "confidence": 1.0,
            },
            SPEC,
        )
        self.assertTrue(aligned["passed"])

        contradictory = validate_vision_observation(
            {
                "aligned": True,
                "observed_color": "blue",
                "unexpected_changes": ["an extra object appeared"],
                "confidence": 0.8,
            },
            SPEC,
        )
        self.assertFalse(contradictory["passed"])

    def test_bbh_vision_prompt_defers_random_pose_authority(self):
        provider = CaptureVisionProvider()
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            result = run_vision_check(
                provider,
                run_dir,
                SPEC,
                model="fake-vision",
            )
        self.assertTrue(result["passed"])
        self.assertIn(
            "one sampled RGB frame cannot",
            provider.prompt,
        )
        self.assertIn("the expert gate own that judgment", provider.prompt)

    def test_repair_prompt_requires_literal_scale_and_preserves_sampling(self):
        prompt = _repair_prompt(
            "bounded scale test",
            SPEC,
            "def load_actors(self):\n    pass\n",
            {
                "passed": False,
                "diagnosis": "The block appears too large.",
            },
        )
        self.assertIn("literal numeric triple", prompt)
        self.assertIn("arithmetic expressions", prompt)
        self.assertIn("never change its sampling bounds", prompt)


if __name__ == "__main__":
    unittest.main()
