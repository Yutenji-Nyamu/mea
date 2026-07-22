import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.manipeval_agent import (
    initialize_registered_dynamic_runtime,
    should_enable_adaptive_plan_step,
)


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

    def test_registered_dynamic_initializes_catalog_and_provider_as_a_pair(self):
        catalog = {"schema_version": 1, "tasks": []}
        provider = object()
        with (
            patch(
                "scripts.manipeval_agent.build_act_catalog", return_value=catalog
            ) as build,
            patch(
                "scripts.manipeval_agent.OpenAICompatibleProvider",
                return_value=provider,
            ) as provider_class,
        ):
            actual_catalog, actual_provider = initialize_registered_dynamic_runtime(
                Path("."),
                None,
                None,
                registered_strategy="dynamic_evidence_v1",
                base_url="https://provider.invalid/v1",
                text_model="planner-model",
                vision_model="vision-model",
            )
            self.assertIs(actual_catalog, catalog)
            self.assertIs(actual_provider, provider)
            build.assert_called_once()
            provider_class.assert_called_once_with(
                base_url="https://provider.invalid/v1",
                text_model="planner-model",
                vision_model="vision-model",
                timeout=180.0,
            )

        with (
            patch("scripts.manipeval_agent.build_act_catalog") as build,
            patch("scripts.manipeval_agent.OpenAICompatibleProvider") as provider_class,
        ):
            actual_catalog, actual_provider = initialize_registered_dynamic_runtime(
                Path("."),
                None,
                None,
                registered_strategy="fixed_predeclared_v1",
                base_url="https://provider.invalid/v1",
                text_model="planner-model",
                vision_model="vision-model",
            )
            self.assertIsNone(actual_catalog)
            self.assertIsNone(actual_provider)
            build.assert_not_called()
            provider_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
