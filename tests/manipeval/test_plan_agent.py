import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mea.planner import (
    BLUE_TASK_INSTRUCTION,
    POSITION_TASK_INSTRUCTION,
    PlanAgentError,
    PlanAgentPrototype,
    validate_evaluation_plan,
    validate_next_round_decision,
)
from mea.toolgen import contact_tool_spec, pickup_to_contact_tool_spec


ROUND_1 = {
    "round_id": "round_1",
    "sub_aspect": "object_appearance.color",
    "rationale": "用户要求评估蓝色方块。",
    "task_instruction": BLUE_TASK_INSTRUCTION,
    "route": "force_codegen",
    "tool_spec": pickup_to_contact_tool_spec("force_codegen"),
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

PLAN = {
    "schema_version": 4,
    "task_name": "beat_block_hammer",
    "policy": {
        "name": "ACT",
        "checkpoint_setting": "demo_clean",
        "expert_data_num": 50,
        "language_conditioned": False,
    },
    "evaluation_goal": "evaluate_blue_block_and_position_variation",
    "rounds": [ROUND_1],
    "max_rounds": 2,
}

NEXT_DECISION = {
    "schema_version": 1,
    "action": "continue",
    "observation_summary": "Round 1 pipeline passed; policy success was recorded.",
    "decision_reason": "The user also requested position variation.",
    "next_round": {
        "round_id": "round_2",
        "sub_aspect": "object_position",
        "rationale": "Collect two simulator-native position samples.",
        "task_instruction": POSITION_TASK_INSTRUCTION,
        "route": "reuse",
        "tool_spec": contact_tool_spec("reuse"),
        "variant_hint": {
            "block": {
                "position_mode": "official_random",
                "yaw_mode": "official_random",
                "scale": 1.0,
                "color": [0.0, 0.2, 1.0],
            }
        },
        "execution": {
            "seeds": [100002, 100003],
            "num_episodes": 2,
            "gates": ["ast", "render", "rule", "vision", "expert", "act"],
        },
        "observations": [
            "scene_alignment",
            "observed_color",
            "expert_solvable",
            "act_pipeline_status",
            "policy_success",
        ],
    },
}


class FakeProvider:
    last_metadata = {"model": "fake-planner"}

    def text(self, prompt, **kwargs):
        if "ROUND 1 OBSERVATION" in prompt:
            return json.dumps(NEXT_DECISION, ensure_ascii=False)
        return json.dumps(PLAN, ensure_ascii=False)


class PlanAgentPrototypeTests(unittest.TestCase):
    def test_plans_round_1_then_adapts_to_round_2(self):
        source_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            readme = repo_root / "mea/planner/README.Agent.md"
            readme.parent.mkdir(parents=True)
            shutil.copy2(source_root / "mea/planner/README.Agent.md", readme)

            agent = PlanAgentPrototype(repo_root, FakeProvider(), model="fake-planner")
            manifest = agent.plan(
                "评估 ACT 在蓝色方块和位置变化下的表现。",
                evaluation_id="eval_unittest_multi_round",
            )
            self.assertEqual(manifest["status"], "planned_round_1")
            self.assertEqual(len(manifest["plan"]["rounds"]), 1)

            updated, decision = agent.decide_next_round(
                evaluation_id=manifest["evaluation_id"],
                user_request=manifest["user_request"],
                current_plan=manifest["plan"],
                round_1_observation={
                    "round_id": "round_1",
                    "pipeline_passed": True,
                    "observations": {"policy_success": 0.0},
                },
            )
            self.assertEqual(decision["action"], "continue")
            self.assertEqual(len(updated["rounds"]), 2)
            self.assertEqual(updated["rounds"][1]["execution"]["num_episodes"], 2)
            self.assertEqual(updated["rounds"][1]["route"], "reuse")
            self.assertEqual(
                manifest["plan"]["rounds"][0]["tool_spec"],
                pickup_to_contact_tool_spec("force_codegen"),
            )
            self.assertEqual(
                updated["rounds"][1]["tool_spec"],
                contact_tool_spec("reuse"),
            )
            self.assertTrue(
                (
                    repo_root
                    / "mea/evaluation_runs/eval_unittest_multi_round/plan/round_2_decision.json"
                ).is_file()
            )

    def test_initial_plan_rejects_premature_second_round(self):
        invalid = json.loads(json.dumps(PLAN, ensure_ascii=False))
        invalid["rounds"].append(dict(invalid["rounds"][0]))
        with self.assertRaises(PlanAgentError):
            validate_evaluation_plan(invalid)

    def test_rejects_unvalidated_color(self):
        invalid = json.loads(json.dumps(PLAN, ensure_ascii=False))
        invalid["rounds"][0]["variant_hint"]["block"]["color"] = [0.0, 1.0, 0.0]
        with self.assertRaises(PlanAgentError):
            validate_evaluation_plan(invalid)

    def test_rejects_tool_spec_route_mismatch(self):
        invalid = json.loads(json.dumps(PLAN, ensure_ascii=False))
        invalid["rounds"][0]["tool_spec"] = contact_tool_spec("reuse")
        with self.assertRaises(PlanAgentError):
            validate_evaluation_plan(invalid)

    def test_failed_pipeline_requires_stop(self):
        with self.assertRaises(PlanAgentError):
            validate_next_round_decision(
                NEXT_DECISION,
                {"pipeline_passed": False},
            )


if __name__ == "__main__":
    unittest.main()
