import json
import shutil
import unittest
from pathlib import Path

from mea.taskgen import (
    TaskGenError,
    TaskGenPrototype,
    build_variant_spec,
    validate_load_actors,
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


class TaskGenPrototypeTests(unittest.TestCase):
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
            self.assertIn("color=(0.0, 0.2, 1.0)", generated)
            bundle = json.loads(
                (run_dir / "generation/task_artifact_bundle.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(bundle["scene_method"]["origin"], "generated_code")
            self.assertEqual(bundle["success_method"]["origin"], "official_reuse")
            self.assertTrue(bundle["success_semantics"]["preserved"])
            static = json.loads(
                (run_dir / "validation/static.json").read_text(encoding="utf-8")
            )
            self.assertTrue(static["load_actors_ast"]["complete_method_generated"])
            self.assertFalse(static["load_actors_ast"]["calls_super"])
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


if __name__ == "__main__":
    unittest.main()
