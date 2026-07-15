import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mea.planner import (
    MAX_ROUNDS,
    SUB_ASPECT_CATALOG,
    PlanAgentError,
    PlanAgentPrototype,
    validate_evaluation_plan,
    validate_next_round_decision,
)
from mea.toolgen import contact_tool_request, pickup_to_contact_tool_request


REQUESTED = list(SUB_ASPECT_CATALOG)
PROPOSAL = {
    "schema_version": 5,
    "task_name": "beat_block_hammer",
    "policy": {
        "name": "ACT",
        "checkpoint_setting": "demo_clean",
        "expert_data_num": 50,
        "language_conditioned": False,
    },
    "evaluation_goal": "evaluate_blue_position_and_timing",
    "requested_template_ids": REQUESTED,
    "first_template_id": "object_appearance.color_blue",
    "max_rounds": 3,
}


def decision(action, next_template_id, *, summary="observation received"):
    return {
        "schema_version": 2,
        "action": action,
        "observation_summary": summary,
        "decision_reason": "bounded adaptive decision",
        "next_template_id": next_template_id,
    }


def observation(round_id, *, pipeline_passed=True, policy_success=0.0):
    return {
        "round_id": round_id,
        "pipeline_passed": pipeline_passed,
        "observations": {"policy_success": policy_success},
    }


class FakeProvider:
    last_metadata = {"model": "fake-planner"}

    def __init__(self):
        self.decisions = [
            decision("continue", "object_position.official_random"),
            decision("continue", "performance.pickup_to_contact_timing"),
            decision("stop", None),
        ]

    def text(self, prompt, **kwargs):
        if "OBSERVATION HISTORY" in prompt:
            return json.dumps(self.decisions.pop(0), ensure_ascii=False)
        return json.dumps(PROPOSAL, ensure_ascii=False)


class RetryInitialProvider:
    last_metadata = {"model": "fake-planner"}

    def __init__(self):
        self.calls = 0

    def text(self, prompt, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return json.dumps({"schema_version": 5})
        return json.dumps(PROPOSAL, ensure_ascii=False)


class PlanAgentPrototypeTests(unittest.TestCase):
    def test_initial_proposal_retries_once_after_schema_error(self):
        source_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            readme = repo_root / "mea/planner/README.Agent.md"
            readme.parent.mkdir(parents=True)
            shutil.copy2(source_root / "mea/planner/README.Agent.md", readme)
            provider = RetryInitialProvider()
            manifest = PlanAgentPrototype(
                repo_root, provider, model="fake"
            ).plan(
                "评估蓝色方块、位置变化和接触时间。",
                evaluation_id="eval_unittest_initial_retry",
            )
            self.assertEqual(provider.calls, 2)
            self.assertEqual(len(manifest["plan"]["rounds"]), 1)
            self.assertEqual(
                len(manifest["planner"]["round_1_validation_errors"]), 1
            )
            self.assertTrue(
                (
                    repo_root
                    / "mea/evaluation_runs/eval_unittest_initial_retry/plan/round_1_response_retry_1.txt"
                ).is_file()
            )

    def test_materializes_three_unique_rounds_then_stops(self):
        source_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            readme = repo_root / "mea/planner/README.Agent.md"
            readme.parent.mkdir(parents=True)
            shutil.copy2(source_root / "mea/planner/README.Agent.md", readme)

            agent = PlanAgentPrototype(repo_root, FakeProvider(), model="fake")
            manifest = agent.plan(
                "评估蓝色方块、位置变化和拿起锤子到接触的时间。",
                evaluation_id="eval_unittest_adaptive",
            )
            plan = manifest["plan"]
            self.assertEqual(plan["max_rounds"], MAX_ROUNDS)
            self.assertEqual(len(plan["rounds"]), 1)
            first = plan["rounds"][0]
            self.assertEqual(first["template_id"], REQUESTED[0])
            self.assertEqual(first["tool_request"], contact_tool_request())
            self.assertNotIn("route", first["tool_request"])

            history = [observation("round_1")]
            plan, first_decision = agent.decide_next_round(
                evaluation_id=manifest["evaluation_id"],
                user_request=manifest["user_request"],
                current_plan=plan,
                observation_history=history,
            )
            self.assertEqual(first_decision["action"], "continue")
            self.assertEqual(plan["rounds"][1]["route"], "reuse")
            self.assertEqual(plan["rounds"][1]["execution"]["seeds"], [100002, 100003])
            self.assertEqual(plan["rounds"][1]["tool_request"], contact_tool_request())

            history.append(observation("round_2", policy_success=0.5))
            plan, second_decision = agent.decide_next_round(
                evaluation_id=manifest["evaluation_id"],
                user_request=manifest["user_request"],
                current_plan=plan,
                observation_history=history,
            )
            self.assertEqual(second_decision["action"], "continue")
            self.assertEqual(
                plan["rounds"][2]["tool_request"],
                pickup_to_contact_tool_request(),
            )
            self.assertEqual(plan["rounds"][2]["execution"]["seeds"], [100000])

            history.append(observation("round_3", policy_success=1.0))
            plan, final_decision = agent.decide_next_round(
                evaluation_id=manifest["evaluation_id"],
                user_request=manifest["user_request"],
                current_plan=plan,
                observation_history=history,
            )
            self.assertEqual(final_decision["action"], "stop")
            self.assertEqual(plan["planning_state"], "stopped_after_round_3")
            self.assertEqual(
                [item["template_id"] for item in plan["rounds"]], REQUESTED
            )

            plan_dir = repo_root / "mea/evaluation_runs/eval_unittest_adaptive/plan"
            for number in (1, 2, 3):
                self.assertTrue(
                    (plan_dir / f"decision_after_round_{number}.json").is_file()
                )

    def test_initial_proposal_cannot_supply_execution_fields(self):
        invalid = json.loads(json.dumps(PROPOSAL, ensure_ascii=False))
        invalid["rounds"] = []
        with self.assertRaises(PlanAgentError):
            validate_evaluation_plan(invalid)

    def test_initial_proposal_rejects_unknown_or_duplicate_templates(self):
        unknown = json.loads(json.dumps(PROPOSAL, ensure_ascii=False))
        unknown["requested_template_ids"] = ["untrusted.template"]
        unknown["first_template_id"] = "untrusted.template"
        with self.assertRaises(PlanAgentError):
            validate_evaluation_plan(unknown)

        duplicate = json.loads(json.dumps(PROPOSAL, ensure_ascii=False))
        duplicate["requested_template_ids"] = [REQUESTED[0], REQUESTED[0]]
        with self.assertRaises(PlanAgentError):
            validate_evaluation_plan(duplicate)

    def test_continue_can_only_select_remaining_requested_template(self):
        plan = validate_evaluation_plan(PROPOSAL)
        history = [observation("round_1")]
        with self.assertRaises(PlanAgentError):
            validate_next_round_decision(
                decision("continue", REQUESTED[0]), plan, history
            )

    def test_requested_remaining_template_cannot_be_skipped(self):
        plan = validate_evaluation_plan(PROPOSAL)
        history = [observation("round_1")]
        with self.assertRaisesRegex(PlanAgentError, "必须继续"):
            validate_next_round_decision(
                decision("stop", None), plan, history
            )

    def test_pipeline_failure_forces_stop(self):
        plan = validate_evaluation_plan(PROPOSAL)
        history = [observation("round_1", pipeline_passed=False)]
        with self.assertRaises(PlanAgentError):
            validate_next_round_decision(
                decision("continue", REQUESTED[1]), plan, history
            )
        stopped = validate_next_round_decision(
            decision("stop", None), plan, history
        )
        self.assertEqual(stopped["action"], "stop")
        self.assertIsNone(stopped["next_round"])

    def test_no_remaining_template_forces_stop(self):
        proposal = json.loads(json.dumps(PROPOSAL, ensure_ascii=False))
        proposal["requested_template_ids"] = [REQUESTED[0]]
        proposal["first_template_id"] = REQUESTED[0]
        plan = validate_evaluation_plan(proposal)
        history = [observation("round_1")]
        with self.assertRaises(PlanAgentError):
            validate_next_round_decision(
                decision("continue", REQUESTED[1]), plan, history
            )

    def test_history_must_cover_each_planned_round(self):
        plan = validate_evaluation_plan(PROPOSAL)
        with self.assertRaises(PlanAgentError):
            validate_next_round_decision(decision("stop", None), plan, [])


if __name__ == "__main__":
    unittest.main()
