import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from mea.taskgen.generic_visual import (
    GenericVisualDiagnosisError,
    build_generic_visual_prompt,
    diagnose_generic_scene_render,
    validate_generic_visual_response,
)


def _response(**overrides):
    value = {
        "schema_version": 1,
        "render_usable": True,
        "key_task_actors_visible": True,
        "requested_change_assessment": "consistent",
        "visual_physical_plausibility": "plausible",
        "unexpected_changes": [],
        "diagnosis": "The generated scene is visibly plausible.",
        "repair_instructions": [],
        "confidence": 0.8,
    }
    value.update(overrides)
    return value


class _Provider:
    def __init__(self, value):
        self.value = value
        self.calls = 0
        self.last_metadata = {"model": "fixture"}

    def vision(self, _prompt, _image, **_kwargs):
        self.calls += 1
        return json.dumps(self.value)


class GenericVisualDiagnosisTests(unittest.TestCase):
    def test_prompt_names_preserved_conditions_without_claiming_nonvisual_facts(
        self,
    ):
        prompt = build_generic_visual_prompt(
            {
                "semantic_concern": "appearance robustness",
                "scene_need": {
                    "kind": "adapt",
                    "description": "Change the target color.",
                    "reuse_first": True,
                },
                "checker_need": None,
                "evaluation_intent": {
                    "preserved_conditions": [
                        "background appearance",
                        "target mass",
                    ]
                },
            }
        )

        self.assertIn("- background appearance", prompt)
        self.assertIn("- target mass", prompt)
        self.assertIn("visible preservation violation", prompt)
        self.assertIn("mass, friction, identity", prompt)
        self.assertIn(
            "exactly one of: consistent, contradicted", prompt
        )

    def test_valid_response_passes(self):
        result = validate_generic_visual_response(
            _response(), scene_change_passed=True
        )
        self.assertTrue(result["passed"])

    def test_zero_confidence_fails(self):
        result = validate_generic_visual_response(
            _response(confidence=0.0), scene_change_passed=True
        )
        self.assertFalse(result["passed"])

    def test_contradicted_change_fails(self):
        result = validate_generic_visual_response(
            _response(requested_change_assessment="contradicted"),
            scene_change_passed=True,
        )
        self.assertFalse(result["passed"])

    def test_nonvisual_change_uses_simulator_authority(self):
        result = validate_generic_visual_response(
            _response(requested_change_assessment="not_visually_decidable"),
            scene_change_passed=True,
        )
        self.assertTrue(result["passed"])
        without_state = validate_generic_visual_response(
            _response(requested_change_assessment="not_visually_decidable"),
            scene_change_passed=False,
        )
        self.assertFalse(without_state["passed"])

    def test_invalid_enum_and_extra_field_are_rejected(self):
        with self.assertRaises(GenericVisualDiagnosisError):
            validate_generic_visual_response(
                _response(visual_physical_plausibility="good"),
                scene_change_passed=True,
            )
        with self.assertRaises(GenericVisualDiagnosisError):
            validate_generic_visual_response(
                {**_response(), "aligned": True},
                scene_change_passed=True,
            )

    def test_diagnosis_persists_prompt_response_result_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official = root / "official.png"
            generated = root / "generated.png"
            Image.new("RGB", (20, 16), "white").save(official)
            Image.new("RGB", (20, 16), "blue").save(generated)
            provider = _Provider(_response())
            result = diagnose_generic_scene_render(
                provider,
                {
                    "semantic_concern": "appearance robustness",
                    "scene_need": {
                        "kind": "adapt",
                        "description": "Change the target color.",
                        "reuse_first": True,
                    },
                    "checker_need": None,
                },
                official_image=official,
                generated_image=generated,
                output_dir=root / "visual",
                model="fixture",
                scene_change_passed=True,
            )
            self.assertTrue(result["passed"])
            self.assertEqual(provider.calls, 1)
            for name in (
                "official_vs_generated.png",
                "vision_prompt.md",
                "vision_response.txt",
                "vision.json",
            ):
                self.assertTrue((root / "visual" / name).is_file())


if __name__ == "__main__":
    unittest.main()
