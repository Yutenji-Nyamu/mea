import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from mea.taskgen import (
    TaskArtifactBundleError,
    build_scene_check_spec,
    compile_success_spec,
    default_bbh_success_spec,
    validate_task_artifact_bundle,
    write_task_artifact_bundle,
)
from mea.taskgen.success_spec import experimental_bbh_success_spec_v2


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TaskArtifactBundleTests(unittest.TestCase):
    def test_experimental_success_requires_task_proposal_v2(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "mea/generated_tasks/run_unbound_experimental"
            success_spec = experimental_bbh_success_spec_v2()
            success_method, _ = compile_success_spec(success_spec)
            write(
                run / "task.py",
                "class beat_block_hammer:\n"
                "    def load_actors(self):\n        pass\n"
                + textwrap.indent(success_method, "    "),
            )
            write(run / "generation/success_spec.json", json.dumps(success_spec))
            write(
                run / "variant_spec.json",
                json.dumps(
                    {
                        "task_name": "beat_block_hammer",
                        "controlled_axis": "object_appearance",
                        "changes": {"block": {"color": [0.0, 0.2, 1.0]}},
                    }
                ),
            )
            with self.assertRaisesRegex(
                TaskArtifactBundleError, "requires TaskProposal v2 provenance"
            ):
                write_task_artifact_bundle(
                    root,
                    run,
                    {
                        "task_name": "beat_block_hammer",
                        "task_module": (
                            "mea.generated_tasks.run_unbound_experimental.task"
                        ),
                        "mode": "force_codegen",
                    },
                )

    def test_generated_route_binds_generated_scene_and_official_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "mea/generated_tasks/run_generated"
            success_method, _ = compile_success_spec(default_bbh_success_spec())
            write(
                run / "task.py",
                "class beat_block_hammer:\n"
                "    def load_actors(self):\n        self.block = object()\n"
                + textwrap.indent(success_method, "    "),
            )
            write(
                run / "generation/success_spec.json",
                json.dumps({
                    "schema_version": 1,
                    "task_name": "beat_block_hammer",
                    "logic": "all",
                    "predicates": [
                        {
                            "predicate": "planar_axis_distance",
                            "left": {"actor": "hammer", "functional_point_id": 0},
                            "right": {"actor": "block", "functional_point_id": 1},
                            "axes": [0, 1],
                            "thresholds_m": [0.02, 0.02],
                            "comparison": "strict_lt",
                        },
                        {
                            "predicate": "physical_contact",
                            "actors": ["hammer", "block"],
                        },
                    ],
                }),
            )
            write(
                root / "envs/beat_block_hammer.py",
                "class beat_block_hammer:\n"
                "    def check_success(self):\n        return True\n",
            )
            spec = {
                "task_name": "beat_block_hammer",
                "controlled_axis": "object_appearance",
                "changes": {"block": {"color": [0.0, 0.2, 1.0]}},
            }
            write(run / "variant_spec.json", json.dumps(spec))
            manifest = {
                "task_name": "beat_block_hammer",
                "task_module": "mea.generated_tasks.run_generated.task",
                "mode": "force_codegen",
            }

            bundle = write_task_artifact_bundle(root, run, manifest)

            self.assertEqual(bundle["scene_method"]["origin"], "generated_code")
            self.assertTrue(bundle["scene_method"]["symbol_declared"])
            self.assertEqual(
                bundle["success_method"]["origin"], "compiled_success_spec"
            )
            self.assertTrue(bundle["success_method"]["symbol_declared"])
            self.assertFalse(
                bundle["success_semantics"]["generated_by_model"]
            )
            self.assertTrue(bundle["success_semantics"]["generated_from_spec"])
            validate_task_artifact_bundle(bundle)

    def test_generated_success_must_match_success_spec_compiler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "mea/generated_tasks/run_wrong_success"
            write(
                run / "task.py",
                "class beat_block_hammer:\n"
                "    def load_actors(self):\n        pass\n"
                "    def check_success(self):\n        return True\n",
            )
            write(
                run / "generation/success_spec.json",
                json.dumps(default_bbh_success_spec()),
            )
            write(
                run / "variant_spec.json",
                json.dumps(
                    {
                        "task_name": "beat_block_hammer",
                        "controlled_axis": "object_appearance",
                        "changes": {"block": {"color": [0.0, 0.2, 1.0]}},
                    }
                ),
            )
            with self.assertRaisesRegex(
                TaskArtifactBundleError, "does not match SuccessSpec"
            ):
                write_task_artifact_bundle(
                    root,
                    run,
                    {
                        "task_name": "beat_block_hammer",
                        "task_module": "mea.generated_tasks.run_wrong_success.task",
                        "mode": "force_codegen",
                    },
                )

    def test_generated_success_requires_success_spec_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "mea/generated_tasks/run_unbound_success"
            write(
                run / "task.py",
                "class beat_block_hammer:\n"
                "    def load_actors(self):\n        pass\n"
                "    def check_success(self):\n        return True\n",
            )
            write(
                root / "envs/beat_block_hammer.py",
                "class beat_block_hammer:\n"
                "    def check_success(self):\n        return True\n",
            )
            write(
                run / "variant_spec.json",
                json.dumps({
                    "task_name": "beat_block_hammer",
                    "controlled_axis": "object_appearance",
                    "changes": {"block": {"color": [0.0, 0.2, 1.0]}},
                }),
            )
            with self.assertRaisesRegex(
                TaskArtifactBundleError, "no SuccessSpec provenance"
            ):
                write_task_artifact_bundle(
                    root,
                    run,
                    {
                        "task_name": "beat_block_hammer",
                        "task_module": "mea.generated_tasks.run_unbound_success.task",
                        "mode": "force_codegen",
                    },
                )

    def test_overlay_and_official_routes_have_honest_origins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write(
                root / "mea/tasks/click_bell.py",
                "class click_bell:\n"
                "    def load_actors(self):\n        self.bell = object()\n",
            )
            write(
                root / "envs/click_bell.py",
                "class click_bell:\n"
                "    def load_actors(self):\n        self.bell = object()\n"
                "    def check_success(self):\n        return True\n",
            )
            spec = {
                "task_name": "click_bell",
                "controlled_axis": "object_position",
                "changes": {
                    "bell": {"position_mode": "fixed", "xy": [-0.14, -0.12]}
                },
            }

            overlay_run = root / "mea/generated_tasks/run_overlay"
            write(overlay_run / "variant_spec.json", json.dumps(spec))
            proposal = {
                "proposal_id": "object_position.query_1",
                "task_name": "click_bell",
                "aspect_id": "object_position",
                "changes": spec["changes"],
            }
            overlay = write_task_artifact_bundle(
                root,
                overlay_run,
                {
                    "task_name": "click_bell",
                    "task_module": "mea.tasks.click_bell",
                    "mode": "reuse",
                    "generation_kind": "bounded_variant_overlay",
                },
                task_proposal=proposal,
            )
            self.assertEqual(
                overlay["scene_method"]["origin"], "bounded_overlay_wrapper"
            )
            self.assertEqual(overlay["success_method"]["origin"], "official_reuse")
            self.assertEqual(overlay["scene_check_spec"]["source"], "task_proposal")
            self.assertEqual(overlay["scene_check_spec"]["repair_mode"], "validate_only")

            official_run = root / "mea/generated_tasks/run_official"
            write(official_run / "variant_spec.json", json.dumps({
                "task_name": "click_bell", "changes": {}
            }))
            official = write_task_artifact_bundle(
                root,
                official_run,
                {
                    "task_name": "click_bell",
                    "task_module": "envs.click_bell",
                    "mode": "official",
                    "generation_kind": "official_passthrough",
                },
            )
            self.assertEqual(official["scene_method"]["origin"], "official_reuse")
            self.assertEqual(official["success_method"]["origin"], "official_reuse")
            self.assertEqual(
                official["scene_check_spec"]["source"], "variant_spec"
            )
            scene_check = json.loads(
                (official_run / "generation/scene_check_spec.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(scene_check["aspect_id"], "task_execution")

    def test_scene_check_repairs_only_the_codegen_family(self):
        bbh = build_scene_check_spec(
            {
                "task_name": "beat_block_hammer",
                "controlled_axis": "object_appearance",
                "changes": {"block": {"color": [0.0, 0.2, 1.0]}},
            }
        )
        self.assertEqual(
            bbh["repair_policy"]["handler"], "regenerate_load_actors"
        )

        click = build_scene_check_spec(
            {
                "task_name": "click_bell",
                "controlled_axis": "object_position",
                "changes": {
                    "bell": {"position_mode": "fixed", "xy": [-0.14, -0.12]}
                },
            },
            task_proposal={
                "proposal_id": "query.position.1",
                "task_name": "click_bell",
                "aspect_id": "object_position",
                "changes": {
                    "bell": {"position_mode": "fixed", "xy": [-0.14, -0.12]}
                },
            },
        )
        self.assertEqual(click["source"], "task_proposal")
        self.assertEqual(click["repair_policy"]["mode"], "validate_only")
        self.assertIn(
            "simulator_tracked_actor.bell_xy", click["simulator_authorities"]
        )


if __name__ == "__main__":
    unittest.main()
