import unittest

from mea.taskgen import (
    CapabilityError,
    build_variant_spec,
    capability_card,
    load_legacy_variant_spec,
    validate_variant_spec_envelope,
)


class TaskCapabilityTests(unittest.TestCase):
    def test_click_position_contract_injects_trusted_fields(self):
        spec = build_variant_spec(
            task_name="click_bell",
            variant_id="object_position.left_fixed",
            capability_id="object_position.fixed_xy",
            intent="evaluate_bell_object_position_generalization",
            changes={
                "bell": {"position_mode": "fixed", "xy": [-0.2, -0.08]}
            },
        )
        self.assertEqual(spec["schema_version"], 2)
        self.assertEqual(spec["controlled_axis"], "object_position")
        self.assertEqual(spec["generation_mode"], "bounded_variant_overlay")
        self.assertIn("official_bell_assets", spec["preserve"])
        self.assertEqual(validate_variant_spec_envelope(spec), spec)

    def test_unknown_capability_and_tampered_axis_are_rejected(self):
        with self.assertRaises(CapabilityError):
            build_variant_spec(
                task_name="click_bell",
                variant_id="x",
                capability_id="object_scale.freeform",
                intent="x",
                changes={"bell": {"scale": 2}},
            )
        spec = build_variant_spec(
            task_name="click_bell",
            variant_id="object_instance.base0",
            capability_id="object_instance.official_id",
            intent="evaluate_bell_object_instance_generalization",
            changes={
                "bell": {
                    "position_mode": "official_random",
                    "instance_mode": "fixed",
                    "bell_id": 0,
                }
            },
        )
        spec["controlled_axis"] = "object_position"
        with self.assertRaises(CapabilityError):
            validate_variant_spec_envelope(spec)

    def test_legacy_click_spec_upgrades_and_card_is_separate(self):
        upgraded = load_legacy_variant_spec(
            {
                "schema_version": 1,
                "task_name": "click_bell",
                "intent": "legacy",
                "controlled_axis": "object_instance",
                "changes": {
                    "bell": {
                        "position_mode": "official_random",
                        "instance_mode": "fixed",
                        "bell_id": 1,
                    }
                },
            }
        )
        self.assertEqual(upgraded["capability_id"], "object_instance.official_id")
        card = capability_card("click_bell")
        self.assertEqual(card["schema_version"], 1)
        self.assertEqual(
            {item["capability_id"] for item in card["capabilities"]},
            {
                "object_position.fixed_xy",
                "object_instance.official_id",
                "robustness.scene_clutter",
            },
        )


if __name__ == "__main__":
    unittest.main()
