import unittest

from mea.providers import (
    ModelProfileError,
    available_model_profiles,
    resolve_model_profile,
)


class ModelProfileTests(unittest.TestCase):
    def test_balanced_uses_terra_for_codegen_and_luna_elsewhere(self):
        resolved = resolve_model_profile("balanced")
        self.assertEqual(resolved["taskgen"], "gpt-5.6-terra")
        self.assertEqual(resolved["toolgen"], "gpt-5.6-terra")
        self.assertEqual(resolved["planner"], "gpt-5.6-luna")
        self.assertEqual(resolved["vision"], "gpt-5.6-luna")
        self.assertEqual(resolved["feedback"], "gpt-5.6-luna")

    def test_explicit_stage_override_wins(self):
        resolved = resolve_model_profile(
            "economy", {"toolgen": "custom-code-model", "vision": None}
        )
        self.assertEqual(resolved["toolgen"], "custom-code-model")
        self.assertEqual(resolved["vision"], "gpt-5.6-luna")

    def test_legacy_preserves_previous_defaults(self):
        resolved = resolve_model_profile("legacy")
        self.assertEqual(set(resolved.values()), {"gpt-4o-2024-11-20"})
        self.assertIn("balanced", available_model_profiles())
        self.assertEqual(
            set(resolve_model_profile("quality").values()),
            {"gpt-5.6-sol"},
        )

    def test_rejects_unknown_profile_or_stage(self):
        with self.assertRaises(ModelProfileError):
            resolve_model_profile("missing")
        with self.assertRaises(ModelProfileError):
            resolve_model_profile("legacy", {"unknown": "model"})


if __name__ == "__main__":
    unittest.main()
