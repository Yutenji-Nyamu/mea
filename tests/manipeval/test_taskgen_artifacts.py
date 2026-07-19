import json
import tempfile
import unittest
from pathlib import Path

from mea.taskgen import (
    build_scene_check_spec,
    validate_task_artifact_bundle,
    write_task_artifact_bundle,
)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TaskArtifactBundleTests(unittest.TestCase):
    def test_generated_route_binds_generated_scene_and_official_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "mea/generated_tasks/run_generated"
            write(
                run / "task.py",
                "class beat_block_hammer:\n"
                "    def load_actors(self):\n        self.block = object()\n",
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
            self.assertEqual(bundle["success_method"]["origin"], "official_reuse")
            self.assertTrue(bundle["success_method"]["symbol_declared"])
            self.assertFalse(
                bundle["success_semantics"]["generated_by_model"]
            )
            validate_task_artifact_bundle(bundle)

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
