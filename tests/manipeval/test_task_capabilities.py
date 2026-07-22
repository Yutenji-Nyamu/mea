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

    def test_object_and_scene_capabilities_reject_cross_scope_changes(self):
        with self.assertRaisesRegex(CapabilityError, "must stay within roots"):
            build_variant_spec(
                task_name="click_bell",
                variant_id="object_position.left_fixed",
                capability_id="object_position.fixed_xy",
                intent="invalid scene mutation through object capability",
                changes={
                    "bell": {"position_mode": "fixed", "xy": [-0.2, -0.08]},
                    "domain_randomization": {"random_light": True},
                },
            )
        with self.assertRaisesRegex(CapabilityError, "must stay within roots"):
            build_variant_spec(
                task_name="click_bell",
                variant_id="scene_lighting.static_random",
                capability_id="scene_lighting",
                intent="invalid object mutation through scene capability",
                changes={"bell": {"bell_id": 0}},
            )

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
                "scene_background_texture",
                "scene_lighting",
            },
        )

    def test_scene_capabilities_preserve_upstream_simulator_contracts(self):
        background = build_variant_spec(
            task_name="click_bell",
            variant_id="scene_background_texture.unseen",
            capability_id="scene_background_texture",
            intent="evaluate_scene_background_texture",
            changes={
                "domain_randomization": {
                    "random_background": True,
                    "clean_background_rate": 0.0,
                }
            },
        )
        self.assertEqual(background["controlled_axis"], "scene_background_texture")
        self.assertIn("eval_mode_unseen_texture_split", background["preserve"])
        lighting = build_variant_spec(
            task_name="click_bell",
            variant_id="scene_lighting.static_random",
            capability_id="scene_lighting",
            intent="evaluate_scene_lighting",
            changes={
                "domain_randomization": {
                    "random_light": True,
                    "crazy_random_light_rate": 0.0,
                }
            },
        )
        self.assertEqual(lighting["controlled_axis"], "scene_lighting")
        self.assertIn("static_per_episode_lighting", lighting["preserve"])

    def test_bbh_scale_capability_is_bounded_and_single_axis(self):
        spec = build_variant_spec(
            task_name="beat_block_hammer",
            variant_id="object_scale.run_local_1_2",
            capability_id="object_scale.bounded",
            intent="evaluate bounded scale generalization",
            changes={
                "block": {
                    "position_mode": "official_random",
                    "yaw_mode": "official_random",
                    "scale": 1.2,
                    "color": [1.0, 0.0, 0.0],
                }
            },
        )
        self.assertEqual(spec["controlled_axis"], "object_scale")
        self.assertIn("official_block_color", spec["preserve"])

        for invalid_scale in (True, 0.5, 1.5, float("inf")):
            with self.subTest(scale=invalid_scale), self.assertRaises(CapabilityError):
                build_variant_spec(
                    task_name="beat_block_hammer",
                    variant_id="object_scale.invalid",
                    capability_id="object_scale.bounded",
                    intent="invalid bounded scale",
                    changes={
                        "block": {
                            "position_mode": "official_random",
                            "yaw_mode": "official_random",
                            "scale": invalid_scale,
                            "color": [1.0, 0.0, 0.0],
                        }
                    },
                )

    def test_bbh_scene_numeric_contract_rejects_non_finite_and_bool(self):
        for invalid in (True, float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=invalid), self.assertRaises(CapabilityError):
                build_variant_spec(
                    task_name="beat_block_hammer",
                    variant_id="object_appearance.invalid",
                    capability_id="object_appearance.color",
                    intent="invalid scene numeric",
                    changes={
                        "block": {
                            "position_mode": "official_random",
                            "yaw_mode": "official_random",
                            "scale": 1.0,
                            "color": [invalid, 0.0, 0.0],
                        }
                    },
                )


if __name__ == "__main__":
    unittest.main()
