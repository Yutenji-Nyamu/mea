import unittest
from copy import deepcopy

from mea.aspects import (
    AspectError,
    aspect_semantics,
    canonicalize_aspect_id,
    canonicalize_aspect_ids,
    public_aspect_ontology,
)
from mea.capability_adapter import (
    CapabilityAdapterError,
    registered_templates,
    resolve_capability_contract,
    validate_capability_contract,
    validate_contract_changes,
)


class AspectOntologyTests(unittest.TestCase):
    def test_explicit_aliases_resolve_to_existing_canonical_ids(self):
        cases = {
            " object.position ": "object_position",
            "object.instance": "object_instance",
            "object.color": "object_appearance.color",
            "scene.clutter": "robustness.scene_clutter",
            "scene.background_texture": "scene_background_texture",
            "scene.lighting": "scene_lighting",
        }
        for alias, expected in cases.items():
            with self.subTest(alias=alias):
                self.assertEqual(canonicalize_aspect_id(alias), expected)

    def test_unknown_is_rejected_unless_reporting_an_unsupported_gap(self):
        with self.assertRaisesRegex(AspectError, "unknown aspect_id"):
            canonicalize_aspect_id("robot.embodiment")
        self.assertEqual(
            canonicalize_aspect_id("robot.embodiment", allow_unknown=True),
            "robot.embodiment",
        )

    def test_duplicate_aliases_are_rejected_after_canonicalization(self):
        with self.assertRaisesRegex(AspectError, "after canonicalization"):
            canonicalize_aspect_ids(["object_position", "object.position"])

    def test_object_scene_and_performance_scopes_are_explicit(self):
        self.assertEqual(
            aspect_semantics("object_position")["semantic_scope"], "object"
        )
        self.assertEqual(
            aspect_semantics("scene.background_texture")["semantic_scope"],
            "scene",
        )
        self.assertEqual(
            aspect_semantics("performance.completion_time")["semantic_scope"],
            "performance",
        )

    def test_query_gold_unsupported_axes_have_one_canonical_vocabulary(self):
        expected = {
            "camera_viewpoint",
            "conclusion.multi_task_consistency",
            "language.paraphrase_consistency",
            "object_appearance.material_gloss",
            "object_appearance.texture",
            "object_physics.mass",
            "object_scale",
            "occlusion.target_contact",
            "performance.motion_smoothness",
            "performance.path_efficiency",
            "safety.boundary_clearance",
            "safety.unintended_contact",
        }
        public_ids = {item["aspect_id"] for item in public_aspect_ontology()}
        self.assertTrue(expected.issubset(public_ids))
        self.assertEqual(canonicalize_aspect_id("camera.viewpoint"), "camera_viewpoint")
        self.assertEqual(
            canonicalize_aspect_id("clutter.target_selection"),
            "robustness.scene_clutter",
        )
        self.assertEqual(aspect_semantics("physics.mass")["semantic_scope"], "physics")


class CapabilityAdapterTests(unittest.TestCase):
    def test_registry_covers_every_current_bbh_and_click_bell_template(self):
        self.assertEqual(
            registered_templates("beat_block_hammer"),
            [
                "object_appearance.color_blue",
                "object_position.official_random",
                "performance.pickup_to_contact_timing",
            ],
        )
        self.assertEqual(
            registered_templates("click_bell"),
            [
                "object_instance.base0",
                "object_instance.base1",
                "object_position.left_fixed",
                "object_position.right_fixed",
                "performance.completion_time_stability.official",
                "robustness.scene_clutter.official_table",
                "scene_background_texture.unseen",
                "scene_lighting.static_random",
                "task_execution.official_baseline",
            ],
        )

    def test_bbh_template_identity_is_separate_from_reused_task_variant(self):
        appearance = resolve_capability_contract(
            "beat_block_hammer", "object_appearance.color_blue"
        )
        position = resolve_capability_contract(
            "beat_block_hammer", "object_position.official_random"
        )
        timing = resolve_capability_contract(
            "beat_block_hammer", "performance.pickup_to_contact_timing"
        )
        self.assertEqual(appearance["taskgen"]["operation"], "force_codegen")
        for contract in (appearance, position, timing):
            self.assertEqual(
                contract["taskgen"]["task_variant_id"],
                "object_appearance.color_blue",
            )
            self.assertEqual(
                contract["taskgen"]["capability_id"], "object_appearance.color"
            )
        self.assertEqual(position["template_id"], "object_position.official_random")
        self.assertEqual(position["taskgen"]["operation"], "reuse_variant")
        self.assertEqual(timing["aspect"]["semantic_scope"], "performance")
        self.assertEqual(timing["taskgen"]["change_scope"], "object")

    def test_same_contract_shape_drives_taskgen_tool_vqa_and_gates(self):
        bbh = resolve_capability_contract(
            "beat_block_hammer", "object_appearance.color_blue"
        )
        bell = resolve_capability_contract("click_bell", "object_position.left_fixed")
        self.assertEqual(set(bbh), set(bell))
        self.assertEqual(set(bbh["taskgen"]), set(bell["taskgen"]))
        self.assertEqual(bell["tool"]["metric"], "bell_active_tcp_min_xy_error")
        self.assertEqual(bell["vqa"]["phenomenon_ids"], ["bell_visibly_pressed"])
        for gate in ("variant_spec", "expert", "act", "planned_tool", "execution_vqa"):
            self.assertIn(gate, bell["required_gates"])

    def test_scene_contracts_use_only_domain_randomization_and_scene_vqa(self):
        expected = {
            "robustness.scene_clutter.official_table": (
                "robustness.scene_clutter",
                "bell_target_selected_among_clutter",
            ),
            "scene_background_texture.unseen": (
                "scene_background_texture",
                "bell_visible_with_unseen_background_texture",
            ),
            "scene_lighting.static_random": (
                "scene_lighting",
                "bell_visible_under_random_lighting",
            ),
        }
        for template_id, (aspect_id, phenomenon_id) in expected.items():
            with self.subTest(template_id=template_id):
                contract = resolve_capability_contract("click_bell", template_id)
                self.assertEqual(contract["aspect"]["aspect_id"], aspect_id)
                self.assertEqual(contract["aspect"]["semantic_scope"], "scene")
                self.assertEqual(contract["taskgen"]["change_scope"], "scene")
                self.assertEqual(
                    contract["taskgen"]["allowed_change_roots"],
                    ["domain_randomization"],
                )
                self.assertIn(phenomenon_id, contract["vqa"]["phenomenon_ids"])

    def test_object_and_scene_change_roots_cannot_cross(self):
        object_contract = resolve_capability_contract(
            "click_bell", "object_position.left_fixed"
        )
        scene_contract = resolve_capability_contract(
            "click_bell", "scene_lighting.static_random"
        )
        with self.assertRaisesRegex(CapabilityAdapterError, "exceed allowed roots"):
            validate_contract_changes(
                object_contract,
                {"domain_randomization": {"random_light": True}},
            )
        with self.assertRaisesRegex(CapabilityAdapterError, "exceed allowed roots"):
            validate_contract_changes(
                scene_contract,
                {"bell": {"position_mode": "fixed"}},
            )
        self.assertEqual(
            validate_contract_changes(
                object_contract,
                {"bell": {"position_mode": "fixed", "xy": [0.1, -0.1]}},
            )["bell"]["position_mode"],
            "fixed",
        )

    def test_official_passthrough_has_no_fake_variant_or_changes(self):
        for template_id in (
            "performance.completion_time_stability.official",
            "task_execution.official_baseline",
        ):
            with self.subTest(template_id=template_id):
                contract = resolve_capability_contract("click_bell", template_id)
                taskgen = contract["taskgen"]
                self.assertEqual(taskgen["operation"], "official_passthrough")
                self.assertIsNone(taskgen["task_variant_id"])
                self.assertIsNone(taskgen["change_scope"])
                self.assertEqual(taskgen["allowed_change_roots"], [])
                self.assertEqual(taskgen["changes"], {})
                self.assertNotIn("variant_spec", contract["required_gates"])

    def test_exact_registry_validation_rejects_contract_tampering(self):
        contract = resolve_capability_contract("click_bell", "object_instance.base0")
        tampered = deepcopy(contract)
        tampered["tool"]["metric"] = "time_to_success"
        with self.assertRaisesRegex(CapabilityAdapterError, "contract changed"):
            validate_capability_contract(tampered)

        wrong_scope = deepcopy(contract)
        wrong_scope["aspect"]["semantic_scope"] = "scene"
        with self.assertRaisesRegex(CapabilityAdapterError, "does not match"):
            validate_capability_contract(wrong_scope)

        wrong_mode = deepcopy(contract)
        wrong_mode["taskgen"]["generation_mode"] = "reuse"
        with self.assertRaisesRegex(CapabilityAdapterError, "operation and"):
            validate_capability_contract(wrong_mode)

        unknown_aspect = deepcopy(contract)
        unknown_aspect["aspect"]["aspect_id"] = "object.unknown"
        with self.assertRaisesRegex(CapabilityAdapterError, "unknown aspect_id"):
            validate_capability_contract(unknown_aspect)

    def test_unknown_task_or_template_fails_closed(self):
        with self.assertRaisesRegex(CapabilityAdapterError, "unknown capability"):
            resolve_capability_contract("click_bell", "object_scale.freeform")
        with self.assertRaisesRegex(CapabilityAdapterError, "unknown capability"):
            resolve_capability_contract("future_task", "object_position.left_fixed")


if __name__ == "__main__":
    unittest.main()
