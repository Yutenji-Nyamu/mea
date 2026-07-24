import json
import shutil
import tempfile
import unittest
from copy import deepcopy
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


REQUESTED = [
    "object_appearance.color_blue",
    "object_position.official_random",
    "performance.pickup_to_contact_timing",
]
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
    episodes = 2 if round_id == "round_2" else 1
    metric = (
        "pickup_to_first_contact_time"
        if round_id == "round_3"
        else "hammer_block_contact_ever"
    )
    return {
        "round_id": round_id,
        "pipeline_passed": pipeline_passed,
        "observations": {
            "policy_success": policy_success,
            "aggregate": {
                "status": "passed",
                "input_issues": [],
                "metrics": [
                    {
                        "metric": metric,
                        "cohorts": [
                            {
                                "role": "policy_under_evaluation",
                                "summary": {
                                    "quality": {
                                        "valid": episodes,
                                        "missing": 0,
                                        "invalid": 0,
                                    }
                                },
                            }
                        ],
                    }
                ],
            },
            "planned_tool": {
                "route_decision": {"metric": metric},
                "episodes": [],
            },
            "execution_vqa": {
                "status": "passed",
                "evidence_conflict": False,
            },
        },
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


class CaptureInitialProvider:
    last_metadata = {"model": "fake-planner"}

    def __init__(self):
        self.prompt = None

    def text(self, prompt, **kwargs):
        self.prompt = prompt
        return json.dumps(PROPOSAL, ensure_ascii=False)


class VerifyProvider:
    last_metadata = {"model": "fake-planner"}

    def text(self, prompt, **kwargs):
        if "OBSERVATION HISTORY" in prompt:
            return json.dumps(
                decision("verify", "object_appearance.color_blue"),
                ensure_ascii=False,
            )
        proposal = json.loads(json.dumps(PROPOSAL, ensure_ascii=False))
        proposal["requested_template_ids"] = [
            "object_appearance.color_blue"
        ]
        proposal["first_template_id"] = "object_appearance.color_blue"
        return json.dumps(proposal, ensure_ascii=False)


class NeverCalledProvider:
    last_metadata = {}

    def text(self, *_args, **_kwargs):
        raise AssertionError("task-specific model must not run after global routing")


class PlanAgentPrototypeTests(unittest.TestCase):
    def test_runtime_seed_override_is_bound_across_base_rounds(self):
        plan = validate_evaluation_plan(
            PROPOSAL,
            execution_seeds=[100600],
        )
        self.assertEqual(
            plan["execution_seed_override"],
            [100600],
        )
        self.assertEqual(
            plan["rounds"][0]["execution"]["seeds"],
            [100600],
        )

        accepted = validate_next_round_decision(
            decision("continue", REQUESTED[1]),
            plan,
            [observation("round_1")],
        )
        self.assertEqual(
            accepted["next_round"]["execution"]["seeds"],
            [100600],
        )

    def test_open_query_scale_template_materializes_bounded_codegen_contract(self):
        proposal = deepcopy(PROPOSAL)
        proposal["evaluation_goal"] = "evaluate bounded target-object scale"
        proposal["requested_template_ids"] = ["object_scale.bounded_1_2"]
        proposal["first_template_id"] = "object_scale.bounded_1_2"
        plan = validate_evaluation_plan(proposal)
        first = plan["rounds"][0]
        self.assertEqual(first["sub_aspect"], "object_scale")
        self.assertEqual(first["capability_id"], "object_scale.bounded")
        self.assertEqual(first["variant_hint"]["block"]["scale"], 1.2)
        self.assertEqual(first["route"], "force_codegen")

    def test_validated_global_proposal_skips_task_specific_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            manifest = PlanAgentPrototype(
                repo_root, NeverCalledProvider(), model="fake"
            ).plan(
                "evaluate object generalization",
                evaluation_id="eval_global_route_bypass",
                validated_proposal=PROPOSAL,
            )
            self.assertFalse(manifest["planner"]["provider_called"])
            self.assertEqual(
                manifest["planner"]["initial_proposal_source"],
                "global_query_route",
            )
            self.assertTrue(
                (
                    repo_root
                    / "mea/evaluation_runs/eval_global_route_bypass/plan/global_route_proposal.json"
                ).is_file()
            )

    def test_similar_history_is_compact_planning_context_only(self):
        source_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            readme = repo_root / "mea/planner/README.Agent.md"
            readme.parent.mkdir(parents=True)
            shutil.copy2(source_root / "mea/planner/README.Agent.md", readme)
            provider = CaptureInitialProvider()
            candidate = {
                "evaluation_id": "eval_previous_blue",
                "similarity": 0.97,
                "user_request": "把红色方块改成蓝色",
                "task_name": "beat_block_hammer",
                "policy": {
                    "name": "ACT",
                    "checkpoint_setting": "demo_clean",
                },
                "planning": {
                    "requested_template_ids": [REQUESTED[0]],
                    "executed_rounds": [
                        {"template_id": REQUESTED[0]}
                    ],
                    "planning_state": "stopped_after_round_1",
                },
                "outcome": {
                    "status": "completed",
                    "pipeline_passed": True,
                    "evidence_conflict": False,
                    "secret_metric_that_must_not_be_in_prompt": 0.25,
                },
                "compatibility": {
                    "same_policy": True,
                    "same_checkpoint": True,
                    "base_commit": "abc",
                },
                "artifacts": {
                    "plan": "past/plan.json",
                    "evidence": "past/evidence.json",
                    "report": "past/report.md",
                },
            }
            manifest = PlanAgentPrototype(
                repo_root, provider, model="fake"
            ).plan(
                "评估蓝色方块、位置变化和接触时间。",
                evaluation_id="eval_unittest_history",
                history_context=[candidate],
                history_metadata={"selection_policy": {"task_filter": "exact"}},
            )
            retrieval = json.loads(
                (
                    repo_root
                    / "mea/evaluation_runs/eval_unittest_history/plan/history_retrieval.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(retrieval["match_count"], 1)
            self.assertEqual(
                retrieval["matches"][0]["evaluation_id"],
                "eval_previous_blue",
            )
            self.assertIn("eval_previous_blue", provider.prompt)
            self.assertNotIn(
                "secret_metric_that_must_not_be_in_prompt", provider.prompt
            )
            self.assertEqual(
                manifest["history_retrieval_path"],
                "plan/history_retrieval.json",
            )

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
            self.assertEqual(plan["rounds"][1]["execution"]["seeds"], [100002])
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

    def test_outer_session_can_cap_adaptive_plan_to_two_rounds(self):
        plan = validate_evaluation_plan(PROPOSAL)
        plan["max_rounds"] = 2
        plan["rounds"][0]["task_name"] = "beat_block_hammer"
        history = [observation("round_1")]
        accepted = validate_next_round_decision(
            decision("continue", REQUESTED[1]), plan, history
        )
        self.assertEqual(accepted["action"], "continue")
        self.assertEqual(accepted["round_budget_before_decision"], 1)

    def test_requested_remaining_template_cannot_be_skipped(self):
        plan = validate_evaluation_plan(PROPOSAL)
        history = [observation("round_1")]
        with self.assertRaisesRegex(PlanAgentError, "action=continue"):
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

    def test_conflict_materializes_one_same_template_verification(self):
        proposal = json.loads(json.dumps(PROPOSAL, ensure_ascii=False))
        proposal["requested_template_ids"] = [REQUESTED[0]]
        proposal["first_template_id"] = REQUESTED[0]
        plan = validate_evaluation_plan(proposal)
        conflicted = observation("round_1")
        conflicted["observations"]["execution_vqa"][
            "evidence_conflict"
        ] = True
        with self.assertRaisesRegex(PlanAgentError, "action=verify"):
            validate_next_round_decision(
                decision("stop", None), plan, [conflicted]
            )
        verified = validate_next_round_decision(
            decision("verify", REQUESTED[0]), plan, [conflicted]
        )
        next_round = verified["next_round"]
        self.assertEqual(next_round["route"], "reuse")
        self.assertEqual(next_round["execution"]["seeds"], [100001])
        self.assertEqual(next_round["execution"]["num_episodes"], 1)
        self.assertEqual(next_round["verification_of"], REQUESTED[0])
        self.assertEqual(
            next_round["verification_trigger"], "evidence_conflict"
        )

        plan["rounds"].append(next_round)
        still_conflicted = observation("round_2")
        still_conflicted["observations"]["aggregate"]["metrics"][0][
            "cohorts"
        ][0]["summary"]["quality"]["valid"] = 1
        still_conflicted["observations"]["execution_vqa"][
            "evidence_conflict"
        ] = True
        stopped = validate_next_round_decision(
            decision("stop", None),
            plan,
            [conflicted, still_conflicted],
        )
        self.assertTrue(stopped["evidence_assessment"]["unresolved"])
        self.assertEqual(
            stopped["evidence_assessment"]["verification_attempts_used"],
            1,
        )

    def test_decide_next_round_persists_verification_round(self):
        source_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            repo_root = Path(temporary)
            readme = repo_root / "mea/planner/README.Agent.md"
            readme.parent.mkdir(parents=True)
            shutil.copy2(source_root / "mea/planner/README.Agent.md", readme)
            evaluation_id = "eval_unittest_verification_persistence"
            agent = PlanAgentPrototype(repo_root, VerifyProvider(), model="fake")
            manifest = agent.plan(
                "评估蓝色方块，并在证据冲突时复核。",
                evaluation_id=evaluation_id,
            )
            conflicted = observation("round_1")
            conflicted["observations"]["execution_vqa"][
                "evidence_conflict"
            ] = True

            updated, verified = agent.decide_next_round(
                evaluation_id=evaluation_id,
                user_request=manifest["user_request"],
                current_plan=manifest["plan"],
                observation_history=[conflicted],
            )

            self.assertEqual(verified["action"], "verify")
            self.assertEqual(len(updated["rounds"]), 2)
            self.assertEqual(
                updated["planning_state"], "awaiting_round_2_observation"
            )
            self.assertEqual(
                updated["rounds"][1]["verification_of"],
                "object_appearance.color_blue",
            )
            persisted = json.loads(
                (
                    repo_root
                    / f"mea/evaluation_runs/{evaluation_id}/plan/evaluation_plan.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(len(persisted["rounds"]), 2)
            self.assertEqual(
                persisted["planning_state"],
                "awaiting_round_2_observation",
            )

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
