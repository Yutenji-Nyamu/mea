import json
import shutil
import unittest
from pathlib import Path

from mea.taskgen import (
    TaskGenError,
    TaskGenPrototype,
    build_variant_spec,
    default_bbh_success_spec,
    validate_load_actors,
    validate_variant_spec,
)
from mea.taskgen.success_spec import experimental_bbh_success_spec_v2
from mea.proposal_agent import BoundedProposalAgent
from mea.taskgen.production_acceptance import (
    record_production_task_acceptance,
)


BLUE_METHOD = '''
def load_actors(self):
    self.hammer = create_actor(
        scene=self,
        pose=sapien.Pose([0, -0.06, 0.783], [0, 0, 0.995, 0.105]),
        modelname="020_hammer",
        convex=True,
        model_id=0,
    )
    block_pose = rand_pose(
        xlim=[-0.25, 0.25],
        ylim=[-0.05, 0.15],
        zlim=[0.76],
        qpos=[1, 0, 0, 0],
        rotate_rand=True,
        rotate_lim=[0, 0, 0.5],
    )
    while abs(block_pose.p[0]) < 0.05 or np.sum(pow(block_pose.p[:2], 2)) < 0.001:
        block_pose = rand_pose(
            xlim=[-0.25, 0.25],
            ylim=[-0.05, 0.15],
            zlim=[0.76],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 0.5],
        )
    self.block = create_box(
        scene=self,
        pose=block_pose,
        half_size=(0.025, 0.025, 0.025),
        color=(0.0, 0.2, 1.0),
        name="box",
        is_static=True,
    )
    self.hammer.set_mass(0.001)
    self.add_prohibit_area(self.hammer, padding=0.10)
    self.prohibited_area.append([
        block_pose.p[0] - 0.05,
        block_pose.p[1] - 0.05,
        block_pose.p[0] + 0.05,
        block_pose.p[1] + 0.05,
    ])
'''


SPEC = {
    "task_name": "beat_block_hammer",
    "intent": "change_object_appearance",
    "generation_mode": "force_codegen",
    "changes": {
        "block": {
            "position_mode": "official_random",
            "yaw_mode": "official_random",
            "scale": 1.0,
            "color": [0.0, 0.2, 1.0],
        }
    },
    "preserve": ["play_once", "check_success"],
}


class FakeProvider:
    def __init__(self):
        self.responses = [
            json.dumps(SPEC, ensure_ascii=False),
            json.dumps(
                {
                    "selected_tasks": [
                        "beat_block_hammer",
                        "blocks_ranking_rgb",
                    ],
                    "reasoning": (
                        "Use the canonical task for behavior and the RGB task "
                        "as an appearance-construction reference."
                    ),
                }
            ),
            f"```python\n{BLUE_METHOD}\n```",
        ]
        self.last_metadata = {"model": "fake"}

    def text(self, prompt, **kwargs):
        return self.responses.pop(0)


class RedProposalProvider(FakeProvider):
    def __init__(self):
        super().__init__()
        red = json.loads(json.dumps(SPEC))
        red["changes"]["block"]["color"] = [1.0, 0.0, 0.0]
        self.responses[0] = json.dumps(red, ensure_ascii=False)


class PublicExperimentalProposalProvider:
    """Deterministic public Proposal -> TaskGen provider fixture."""

    def __init__(self):
        changes = {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.0,
                "color": [0.25, 0.25, 0.75],
            }
        }
        proposal = {
            "schema_version": 1,
            "task_proposal": {
                "schema_version": 2,
                "proposal_id": "object_appearance.public_v2_fixture",
                "task_name": "beat_block_hammer",
                "aspect_id": "object_appearance.color",
                "intent": (
                    "generate a purple scene and a separately labeled bounded "
                    "experimental success predicate"
                ),
                "capability_id": "object_appearance.color",
                "reuse_first": True,
                "changes": changes,
                "preserve_success_semantics": False,
                "success_spec": experimental_bbh_success_spec_v2(
                    thresholds_m=(0.025, 0.025)
                ),
            },
            "tool_proposal": {
                "schema_version": 1,
                "proposal_id": "object_appearance.public_v2_fixture.tool",
                "task_name": "beat_block_hammer",
                "aspect_id": "object_appearance.color",
                "evaluation_goal": (
                    "measure contact without treating experimental success as "
                    "official policy success"
                ),
                "metric": "hammer_block_contact_ever",
                "question": "Did strict hammer-block contact occur?",
                "vqa_phenomenon_ids": ["block_visibly_displaced"],
                "reuse_first": True,
            },
        }
        variant = json.loads(json.dumps(SPEC))
        variant["changes"] = changes
        method = BLUE_METHOD.replace(
            "color=(0.0, 0.2, 1.0)",
            "color=(0.25, 0.25, 0.75)",
        )
        self.responses = [
            json.dumps(proposal, ensure_ascii=False),
            json.dumps(variant, ensure_ascii=False),
            json.dumps(
                {
                    "selected_tasks": [
                        "beat_block_hammer",
                        "blocks_ranking_rgb",
                    ],
                    "reasoning": (
                        "Use the canonical behavior and bounded RGB construction."
                    ),
                }
            ),
            f"```python\n{method}\n```",
        ]
        self.last_metadata = {"model": "deterministic-fixture"}

    def text(self, prompt, **kwargs):
        return self.responses.pop(0)


class TaskGenPrototypeTests(unittest.TestCase):
    def test_public_proposal_v2_materializes_and_reaches_zero_act_acceptance(self):
        repo_root = Path(__file__).resolve().parents[2]
        run_id = "run_unittest_public_proposal_success_v2"
        run_dir = repo_root / "mea/generated_tasks" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        provider = PublicExperimentalProposalProvider()
        target = {
            "task_name": "beat_block_hammer",
            "aspects": [{"aspect_id": "object_appearance.color"}],
        }
        try:
            bundle = BoundedProposalAgent(
                provider, model="deterministic-fixture"
            ).propose(
                (
                    "How does this policy behave on a novel block appearance "
                    "under a bounded experimental 2.5 cm criterion?"
                ),
                target=target,
                aspect_id="object_appearance.color",
                base_template_id="object_appearance.color_blue",
                capability_mode="experimental_success_bounded",
            )
            manifest = TaskGenPrototype(
                repo_root, provider, model="deterministic-fixture"
            ).generate(
                "materialize the public bounded experimental proposal fixture",
                run_id=run_id,
                task_proposal=bundle["task_proposal"],
            )
            acceptance = record_production_task_acceptance(
                run_dir,
                manifest,
                scene={
                    "setup_success": True,
                    "render_success": True,
                    "rule_check": {"passed": True},
                    "expert": {"passed": True},
                },
                position_samples={"passed": True},
                require_expert=True,
            )

            self.assertEqual(acceptance["status"], "accepted")
            self.assertEqual(
                acceptance["runtime"]["act_rollouts_started"], 0
            )
            self.assertFalse(
                manifest["task_artifact_summary"][
                    "success_official_equivalent"
                ]
            )
            self.assertEqual(
                manifest["task_artifact_summary"]["success_execution_scope"],
                "experimental_bounded_probe_only",
            )
            comparison = bundle["success_semantics_comparison"]
            self.assertEqual(
                comparison["official"]["result_status"],
                "not_measured_by_proposal",
            )
            self.assertEqual(
                comparison["experimental"]["result_status"],
                "pending_materialization_or_probe",
            )
            persisted = json.loads(
                (
                    run_dir / "generation/task_proposal.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["schema_version"], 2)
            self.assertFalse(persisted["preserve_success_semantics"])
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir)

    def test_v1_proposal_cannot_smuggle_experimental_success_candidate(self):
        proposal = {
            "schema_version": 1,
            "proposal_id": "object_appearance.preserved_success",
            "task_name": "beat_block_hammer",
            "aspect_id": "object_appearance.color",
            "intent": "preserve official success",
            "capability_id": "object_appearance.color",
            "reuse_first": True,
            "changes": SPEC["changes"],
            "preserve_success_semantics": True,
        }
        with self.assertRaisesRegex(
            TaskGenError, "cannot be combined with a legacy SuccessSpec"
        ):
            TaskGenPrototype(
                Path(__file__).resolve().parents[2], FakeProvider(), model="fake"
            ).generate(
                "reject contradictory success authority",
                task_proposal=proposal,
                success_spec_candidate=experimental_bbh_success_spec_v2(),
            )

    def test_task_proposal_v2_compiles_bounded_success_with_provenance(self):
        repo_root = Path(__file__).resolve().parents[2]
        run_id = "run_unittest_proposal_success_spec_v2"
        run_dir = repo_root / "mea/generated_tasks" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        proposal = {
            "schema_version": 2,
            "proposal_id": "object_appearance.experimental_success",
            "task_name": "beat_block_hammer",
            "aspect_id": "object_appearance.color",
            "intent": "blue block with a bounded experimental success threshold",
            "capability_id": "object_appearance.color",
            "reuse_first": True,
            "changes": SPEC["changes"],
            "preserve_success_semantics": False,
            "success_spec": experimental_bbh_success_spec_v2(
                thresholds_m=(0.015, 0.03)
            ),
        }
        trusted = build_variant_spec(
            task_name="beat_block_hammer",
            variant_id=proposal["proposal_id"],
            capability_id=proposal["capability_id"],
            intent=proposal["intent"],
            changes=proposal["changes"],
            generation_mode="force_codegen",
            preserve_success_semantics=False,
        )
        try:
            manifest = TaskGenPrototype(
                repo_root, FakeProvider(), model="fake"
            ).generate(
                "evaluate a proposal-derived success predicate",
                run_id=run_id,
                trusted_variant_spec=trusted,
                task_proposal=proposal,
            )

            task_source = (run_dir / "task.py").read_text(encoding="utf-8")
            self.assertIn("np.array([0.015, 0.03])", task_source)
            self.assertIn("def check_success(self):", task_source)
            code_response = (
                run_dir / "generation/code_response.txt"
            ).read_text(encoding="utf-8")
            self.assertNotIn("check_success", code_response)

            provenance = json.loads(
                (run_dir / "generation/success_spec_provenance.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(provenance["source"], "task_proposal_v2")
            self.assertFalse(provenance["preserve_success_semantics"])
            self.assertFalse(provenance["official_equivalent"])
            self.assertTrue(provenance["compiler_eligible"])
            self.assertFalse(provenance["act_runtime_eligible"])
            self.assertTrue(provenance["experimental_bounded"])
            self.assertEqual(
                provenance["execution_scope"],
                "experimental_bounded_probe_only",
            )
            self.assertFalse(provenance["generated_by_model"])

            bundle = json.loads(
                (run_dir / "generation/task_artifact_bundle.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIsNotNone(bundle["task_proposal_sha256"])
            self.assertIsNotNone(
                bundle["success_semantics"]["success_spec_sha256"]
            )
            scene_check = json.loads(
                (run_dir / "generation/scene_check_spec.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                scene_check["success_semantics"],
                "experimental_bounded_success_spec",
            )
            self.assertIn(
                "compiled_success_spec", scene_check["simulator_authorities"]
            )
            self.assertFalse(
                manifest["task_artifact_summary"]["success_semantics_preserved"]
            )
            self.assertNotIn("check_success", trusted["preserve"])
            self.assertIn(
                "compiled_experimental_success_spec", trusted["preserve"]
            )
            self.assertEqual(
                manifest["task_artifact_summary"]["success_execution_scope"],
                "experimental_bounded_probe_only",
            )
            self.assertFalse(
                manifest["task_artifact_summary"]["success_act_eligible"]
            )
            self.assertTrue(
                manifest["task_artifact_summary"]["success_compiler_eligible"]
            )
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir)

    def test_invalid_success_spec_is_diagnosed_and_repaired_once(self):
        repo_root = Path(__file__).resolve().parents[2]
        run_id = "run_unittest_success_spec_repair"
        run_dir = repo_root / "mea/generated_tasks" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        invalid = default_bbh_success_spec()
        invalid["predicates"][0]["thresholds_m"] = [0.2, 0.2]
        try:
            TaskGenPrototype(repo_root, FakeProvider(), model="fake").generate(
                "generate a blue block after repairing the success checker",
                run_id=run_id,
                success_spec_candidate=invalid,
                success_spec_max_repairs=1,
            )
            repair = json.loads(
                (run_dir / "generation/success_spec_repair.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(repair["repaired"])
            self.assertEqual(repair["final_source"], "trusted_default")
            self.assertFalse(repair["attempts"][0]["valid"])
            self.assertTrue(repair["attempts"][1]["valid"])
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir)

    def test_complete_load_actors_generation(self):
        repo_root = Path(__file__).resolve().parents[2]
        run_id = "run_unittest_complete_method"
        run_dir = repo_root / "mea/generated_tasks" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        try:
            manifest = TaskGenPrototype(
                repo_root,
                FakeProvider(),
                model="fake",
            ).generate(
                "把红色方块改为蓝色",
                run_id=run_id,
            )
            self.assertEqual(manifest["status"], "generated")
            self.assertIn(run_id, manifest["task_module"])
            generated = (run_dir / "task.py").read_text(encoding="utf-8")
            self.assertIn("def load_actors(self):", generated)
            self.assertIn("def check_success(self):", generated)
            self.assertIn("color=(0.0, 0.2, 1.0)", generated)
            bundle = json.loads(
                (run_dir / "generation/task_artifact_bundle.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(bundle["scene_method"]["origin"], "generated_code")
            self.assertEqual(
                bundle["success_method"]["origin"], "compiled_success_spec"
            )
            self.assertTrue(bundle["success_semantics"]["preserved"])
            self.assertTrue(bundle["success_semantics"]["generated_from_spec"])
            self.assertFalse(bundle["success_semantics"]["generated_by_model"])
            static = json.loads(
                (run_dir / "validation/static.json").read_text(encoding="utf-8")
            )
            self.assertTrue(static["load_actors_ast"]["complete_method_generated"])
            self.assertFalse(static["load_actors_ast"]["calls_super"])
            self.assertEqual(
                static["success_spec"]["predicates"],
                ["planar_axis_distance", "physical_contact"],
            )
            self.assertTrue((run_dir / "generation/success_spec.json").is_file())
            retrieval = json.loads(
                (run_dir / "generation/retrieval.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(retrieval["catalog_size"], 50)
            self.assertEqual(
                retrieval["selected_tasks"],
                ["beat_block_hammer", "blocks_ranking_rgb"],
            )
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir)

    def test_planner_variant_spec_is_authoritative_over_model_proposal(self):
        repo_root = Path(__file__).resolve().parents[2]
        run_id = "run_unittest_planner_authority"
        run_dir = repo_root / "mea/generated_tasks" / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
        trusted = build_variant_spec(
            task_name="beat_block_hammer",
            variant_id="object_appearance.color_blue",
            capability_id="object_appearance.color",
            intent="planner_capability:object_appearance.color_blue",
            changes=SPEC["changes"],
            generation_mode="force_codegen",
        )
        try:
            manifest = TaskGenPrototype(
                repo_root,
                RedProposalProvider(),
                model="fake",
            ).generate(
                "evaluate the trusted blue variant",
                run_id=run_id,
                variant_id=trusted["variant_id"],
                trusted_variant_spec=trusted,
            )
            materialized = json.loads(
                (run_dir / "variant_spec.json").read_text(encoding="utf-8")
            )
            self.assertEqual(materialized, trusted)
            self.assertEqual(
                manifest["variant_spec_authority"],
                "planner_capability_contract",
            )
            self.assertEqual(
                materialized["changes"]["block"]["color"],
                [0.0, 0.2, 1.0],
            )
        finally:
            if run_dir.exists():
                shutil.rmtree(run_dir)

    def test_rejects_file_access(self):
        malicious = BLUE_METHOD.replace(
            "def load_actors(self):",
            "def load_actors(self):\n    open('/tmp/unwanted', 'w')",
        )
        with self.assertRaises(TaskGenError):
            validate_load_actors(malicious, SPEC)

    def test_rejects_super_delegation(self):
        delegated = "def load_actors(self):\n    return super().load_actors()\n"
        with self.assertRaises(TaskGenError):
            validate_load_actors(delegated, SPEC)

    def test_bounded_non_template_scale_is_checked_against_geometry(self):
        scaled = build_variant_spec(
            task_name="beat_block_hammer",
            variant_id="object_scale.run_local_1_2",
            capability_id="object_scale.bounded",
            intent="evaluate a query-generated bounded block scale",
            changes={
                "block": {
                    "position_mode": "official_random",
                    "yaw_mode": "official_random",
                    "scale": 1.2,
                    "color": [1.0, 0.0, 0.0],
                }
            },
        )
        method = BLUE_METHOD.replace(
            "half_size=(0.025, 0.025, 0.025)",
            "half_size=(0.03, 0.03, 0.03)",
        ).replace("color=(0.0, 0.2, 1.0)", "color=(1.0, 0.0, 0.0)")
        result = validate_load_actors(method, scaled)
        self.assertEqual(result["generated_half_size"], [0.03, 0.03, 0.03])

        wrong_geometry = method.replace(
            "half_size=(0.03, 0.03, 0.03)",
            "half_size=(0.025, 0.025, 0.025)",
        )
        with self.assertRaisesRegex(TaskGenError, "does not match VariantSpec scale"):
            validate_load_actors(wrong_geometry, scaled)

    def test_variant_scene_numbers_reject_bool_nan_and_infinity(self):
        for invalid in (True, float("nan"), float("inf"), float("-inf")):
            with self.subTest(field="scale", value=invalid), self.assertRaises(
                TaskGenError
            ):
                validate_variant_spec(
                    {
                        "task_name": "beat_block_hammer",
                        "changes": {
                            "block": {
                                "color": [0.0, 0.2, 1.0],
                                "scale": invalid,
                            }
                        },
                    },
                    "beat_block_hammer",
                )
            with self.subTest(field="color", value=invalid), self.assertRaises(
                TaskGenError
            ):
                validate_variant_spec(
                    {
                        "task_name": "beat_block_hammer",
                        "changes": {
                            "block": {
                                "color": [invalid, 0.2, 1.0],
                                "scale": 1.0,
                            }
                        },
                    },
                    "beat_block_hammer",
                )


if __name__ == "__main__":
    unittest.main()
