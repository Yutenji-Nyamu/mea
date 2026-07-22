import unittest

from scripts.manipeval_agent import should_enable_adaptive_plan_step


class RegisteredAdaptiveRuntimeTests(unittest.TestCase):
    def test_registered_dynamic_uses_adaptive_step(self):
        self.assertTrue(
            should_enable_adaptive_plan_step(
                fixed_click_bell=False,
                legacy_click_bell=False,
                registered_strategy="dynamic_evidence_v1",
            )
        )

    def test_fixed_and_legacy_keep_existing_planners(self):
        self.assertFalse(
            should_enable_adaptive_plan_step(
                fixed_click_bell=True,
                legacy_click_bell=False,
                registered_strategy="fixed_predeclared_v1",
            )
        )
        self.assertFalse(
            should_enable_adaptive_plan_step(
                fixed_click_bell=False,
                legacy_click_bell=True,
                registered_strategy=None,
            )
        )


if __name__ == "__main__":
    unittest.main()
