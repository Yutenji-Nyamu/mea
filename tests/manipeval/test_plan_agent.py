import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mea.planner import (
    BLUE_TASK_INSTRUCTION,
    PlanAgentError,
    PlanAgentPrototype,
    validate_evaluation_plan,
)


PLAN = {
    "schema_version": 1,
    "task_name": "beat_block_hammer",
    "policy": {
        "name": "ACT",
        "checkpoint_setting": "demo_clean",
        "expert_data_num": 50,
        "language_conditioned": False,
    },
    "evaluation_goal": "evaluate_act_with_blue_block",
    "rounds": [
        {
            "round_id": "round_1",
            "sub_aspect": "object_appearance.color",
            "rationale": "用户要求评估蓝色方块。",
            "task_instruction": BLUE_TASK_INSTRUCTION,
            "route": "force_codegen",
            "variant_hint": {
                "block": {
                    "position_mode": "official_random",
                    "yaw_mode": "official_random",
                    "scale": 1.0,
                    "color": [0.0, 0.2, 1.0],
                }
            },
            "execution": {
                "seeds": [100000],
                "num_episodes": 1,
                "gates": ["ast", "render", "rule", "vision", "expert", "act"],
            },
            "observations": [
                "scene_alignment",
                "observed_color",
                "expert_solvable",
                "act_pipeline_status",
                "policy_success",
            ],
        }
    ],
    "stop_after_round": 1,
}


class FakeProvider:
    last_metadata = {"model": "fake-planner"}

    def text(self, prompt, **kwargs):
        return json.dumps(PLAN, ensure_ascii=False)


class PlanAgentPrototypeTests(unittest.TestCase):
    def test_generates_single_round_blue_plan(self):
        source_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            readme = repo_root / "mea/planner/README.Agent.md"
            readme.parent.mkdir(parents=True)
            shutil.copy2(source_root / "mea/planner/README.Agent.md", readme)

            manifest = PlanAgentPrototype(
                repo_root,
                FakeProvider(),
                model="fake-planner",
            ).plan(
                "评估 ACT 在蓝色方块场景中的表现。",
                evaluation_id="eval_unittest_blue",
            )
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(
                manifest["plan"]["rounds"][0]["task_instruction"],
                BLUE_TASK_INSTRUCTION,
            )
            self.assertEqual(
                manifest["plan"]["rounds"][0]["execution"]["num_episodes"],
                1,
            )
            self.assertTrue(
                (
                    repo_root
                    / "mea/evaluation_runs/eval_unittest_blue/plan/evaluation_plan.json"
                ).is_file()
            )

    def test_rejects_multiple_rounds(self):
        invalid = json.loads(json.dumps(PLAN, ensure_ascii=False))
        invalid["rounds"].append(dict(invalid["rounds"][0]))
        with self.assertRaises(PlanAgentError):
            validate_evaluation_plan(invalid)

    def test_rejects_unvalidated_color(self):
        invalid = json.loads(json.dumps(PLAN, ensure_ascii=False))
        invalid["rounds"][0]["variant_hint"]["block"]["color"] = [0.0, 1.0, 0.0]
        with self.assertRaises(PlanAgentError):
            validate_evaluation_plan(invalid)


if __name__ == "__main__":
    unittest.main()
