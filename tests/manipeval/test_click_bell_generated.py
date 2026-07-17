import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mea.planner import ClickBellPositionPlanAgent
from mea.taskgen import (
    ClickBellTaskGenError,
    compile_click_bell_overlay,
    create_click_bell_variant_run,
    validate_click_bell_variant_hint,
    validate_click_bell_vision_observation,
)
from scripts.manipeval_agent import build_taskgen_command
from scripts.manipeval_taskgen import validate_click_bell_scene_position


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_repo(root: Path) -> None:
    schema_dir = root / "mea/toolkit/schemas"
    schema_dir.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "mea/toolkit/schemas/click_bell.json",
        schema_dir / "click_bell.json",
    )
    for relative in (
        "envs/click_bell.py",
        "policy/ACT/eval.sh",
        "script/eval_policy.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# protected fixture\n", encoding="utf-8")


class ClickBellGeneratedTests(unittest.TestCase):
    def test_variant_hint_is_strict_and_compiles_overlay(self):
        hint = {"bell": {"position_mode": "fixed", "xy": [-0.2, -0.08]}}
        self.assertEqual(validate_click_bell_variant_hint(hint), hint)
        overlay = compile_click_bell_overlay(hint)
        self.assertEqual(overlay["mea"]["bell"]["xy"], [-0.2, -0.08])
        with self.assertRaises(ClickBellTaskGenError):
            validate_click_bell_variant_hint(
                {"bell": {"position_mode": "fixed", "xy": [0.01, -0.08]}}
            )

    def test_bounded_run_records_overlay_not_codegen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root)
            manifest = create_click_bell_variant_run(
                root,
                "evaluate bell position",
                variant_hint={
                    "bell": {"position_mode": "fixed", "xy": [-0.2, -0.08]}
                },
                run_id="run_click_bell_position_test",
            )
            self.assertEqual(manifest["mode"], "reuse")
            self.assertEqual(manifest["generation_kind"], "bounded_variant_overlay")
            self.assertEqual(manifest["task_module"], "mea.tasks.click_bell")
            self.assertFalse(
                manifest["static_validation"]["code_generation"]["performed"]
            )

    def test_planner_continues_left_to_right_with_same_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root)
            planner = ClickBellPositionPlanAgent(
                root, start_seed=10, num_episodes=1, max_rounds=2
            )
            manifest = planner.plan(
                "evaluate bell position", evaluation_id="eval_bell_position_test"
            )
            plan = manifest["plan"]
            first = plan["rounds"][0]
            self.assertEqual(first["template_id"], "object_position.left_fixed")
            updated, decision = planner.decide_next_round(
                evaluation_id="eval_bell_position_test",
                user_request="evaluate bell position",
                current_plan=plan,
                observation_history=[{"pipeline_passed": True}],
            )
            self.assertEqual(decision["action"], "continue")
            second = updated["rounds"][1]
            self.assertEqual(second["template_id"], "object_position.right_fixed")
            self.assertEqual(first["execution"]["seeds"], second["execution"]["seeds"])
            stopped, final = planner.decide_next_round(
                evaluation_id="eval_bell_position_test",
                user_request="evaluate bell position",
                current_plan=updated,
                observation_history=[
                    {"pipeline_passed": True},
                    {"pipeline_passed": True},
                ],
            )
            self.assertEqual(final["action"], "stop")
            self.assertEqual(stopped["planning_state"], "stopped_after_round_2")

    def test_agent_command_carries_trusted_variant_json(self):
        round_plan = {
            "round_id": "round_1",
            "task_instruction": "evaluate",
            "task_name": "click_bell",
            "task_module": "mea.tasks.click_bell",
            "route": "reuse",
            "variant_hint": {
                "bell": {"position_mode": "fixed", "xy": [-0.2, -0.08]}
            },
            "execution": {"backend": "act", "seeds": [7], "num_episodes": 1},
        }
        command, _ = build_taskgen_command(
            Path("/repo"),
            "eval_click_generated",
            round_plan,
            text_model="text",
            vision_model="vision",
            base_url=None,
            gpu=0,
            max_reflections=0,
        )
        index = command.index("--variant-hint-json")
        self.assertEqual(json.loads(command[index + 1]), round_plan["variant_hint"])
        self.assertIn("--vision-check", command)
        self.assertIn("--run-act", command)

    def test_scene_xy_is_numeric_authority_and_vqa_is_plausibility_only(self):
        spec = {
            "task_name": "click_bell",
            "changes": {"bell": {"xy": [-0.2, -0.08]}},
        }
        scene = {
            "tracked_actors": [
                {"id": "bell", "position": [-0.2, -0.08, 0.741]}
            ]
        }
        self.assertTrue(validate_click_bell_scene_position(scene, spec)["passed"])
        vision = validate_click_bell_vision_observation(
            {
                "aligned": True,
                "target_actor": "bell",
                "bell_visible": True,
                "unexpected_changes": [],
                "diagnosis": "ok",
                "suggestions": [],
                "confidence": 0.9,
            }
        )
        self.assertTrue(vision["passed"])
        self.assertEqual(vision["position_authority"], "simulator_tracked_actor_xy")

    def test_click_bell_vqa_rejects_string_booleans_and_wrong_actor(self):
        base = {
            "aligned": True,
            "target_actor": "bell",
            "bell_visible": True,
            "unexpected_changes": [],
            "diagnosis": "ok",
            "suggestions": [],
            "confidence": 0.9,
        }
        for update in (
            {"aligned": "false"},
            {"bell_visible": "false"},
            {"target_actor": "block"},
        ):
            with self.subTest(update=update), self.assertRaises(Exception):
                validate_click_bell_vision_observation({**base, **update})

    def test_failed_first_round_preserves_unexecuted_right_template(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_repo(root)
            planner = ClickBellPositionPlanAgent(root, max_rounds=2)
            manifest = planner.plan("evaluate", evaluation_id="eval_bell_failed")
            _, decision = planner.decide_next_round(
                evaluation_id="eval_bell_failed",
                user_request="evaluate",
                current_plan=manifest["plan"],
                observation_history=[{"pipeline_passed": False}],
            )
            self.assertEqual(decision["action"], "stop")
            self.assertEqual(
                decision["remaining_template_ids_before_decision"],
                ["object_position.right_fixed"],
            )


if __name__ == "__main__":
    unittest.main()
