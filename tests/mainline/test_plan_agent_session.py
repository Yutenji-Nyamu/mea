"""High-information tests for the paper-aligned Plan Agent session."""

from __future__ import annotations

import unittest

from mea.planner.experiment_candidate import build_experiment_candidate
from mea.planner.plan_agent_errors import PlanAgentSessionError
from mea.planner.plan_agent_session import PlanAgentSession
from mea.planner.policy_task_binding import build_policy_task_binding
from mea.planner.semantic_coverage import build_evaluation_intent


def target(*, max_rounds: int = 3) -> dict:
    return {
        "task_name": "click_bell",
        "max_rounds": max_rounds,
        "policy": {"policy_name": "ACT"},
        "aspects": [
            {
                "aspect_id": "task_execution.official_baseline",
                "description": "Unchanged official task.",
                "template_ids": ["task_execution.official_baseline"],
            },
            {
                "aspect_id": "object_position",
                "description": "Position variants available for retrieval.",
                "template_ids": ["object_position.left_fixed"],
            },
        ],
    }


def capabilities() -> dict:
    return {
        "schema_version": 2,
        "policy_card": {
            "policy_name": "ACT",
            "task_name": "click_bell",
        },
        "simulator_card": {
            "simulator_name": "RoboTwin",
            "task_name": "click_bell",
            "tracked_actors": ["bell", "robot"],
        },
        "generation_card": {
            "backend_primitives": {
                "scene": True,
                "checker": True,
                "telemetry": True,
                "rule": True,
                "vqa": True,
                "retrieve": True,
                "generate": True,
            }
        },
    }


def continue_bundle(sub_aspect: str = "object_position.left_fixed") -> dict:
    return {
        "schema_version": 2,
        "source": "fixture_plan_agent",
        "proposal": {
            "schema_version": 2,
            "action": "continue",
            "sub_aspect": sub_aspect,
            "hypothesis": "A left placement may expose a weakness.",
            "requested_perturbation": {
                "description": "Place the bell at a safe left position.",
                "controlled_changes": [
                    {
                        "actor": "bell",
                        "property": "position",
                        "axis": "y",
                        "signed_delta": -0.03,
                        "unit": "m",
                        "reference": "same_seed_official_reset",
                    }
                ],
                "preserve": [
                    {
                        "actor": "bell",
                        "property": "position",
                        "axis": "x",
                        "relation": "preserve",
                    },
                    {
                        "actor": "bell",
                        "property": "position",
                        "axis": "z",
                        "relation": "preserve",
                    },
                    {
                        "actor": "bell",
                        "property": "orientation",
                        "axis": None,
                        "relation": "preserve",
                    },
                    {
                        "actor": None,
                        "property": "official_goal",
                        "axis": None,
                        "relation": "preserve",
                    },
                ],
            },
            "scene_need": {
                "required": True,
                "description": "Move only the bell and preserve task identity.",
            },
            "checker_need": {
                "required": False,
                "description": None,
            },
            "rule_tool_need": {
                "required": True,
                "description": "Measure official success and completion time.",
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "rationale": "This is the next high-information experiment.",
            "answer": None,
            "claim_verdict": None,
            "evidence_sufficient": False,
        },
    }


def stop_bundle(*, sufficient: bool = True) -> dict:
    return {
        "schema_version": 2,
        "source": "fixture_plan_agent",
        "proposal": {
            "schema_version": 2,
            "action": "stop",
            "sub_aspect": None,
            "hypothesis": "The completed evidence answers the bounded Query.",
            "requested_perturbation": None,
            "scene_need": {"required": False, "description": None},
            "checker_need": {"required": False, "description": None},
            "rule_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "vqa_tool_need": {
                "required": False,
                "description": None,
                "reuse_first": True,
            },
            "rationale": (
                "The observed failure directly answers the bounded Query."
                if sufficient
                else "The available evidence remains inconclusive."
            ),
            "answer": (
                "The tested bounded variation exposes a policy failure."
                if sufficient
                else None
            ),
            "claim_verdict": "supported" if sufficient else "inconclusive",
            "evidence_sufficient": sufficient,
        },
    }


def evidence(round_id: str, outcome: str) -> dict:
    return {
        "schema_version": 1,
        "round_id": round_id,
        "tested_sub_aspect": "object_position.left_fixed",
        "tested_hypothesis": "A left placement may expose a weakness.",
        "tested_perturbation": "Place the bell at a safe left position.",
        "outcome": outcome,
        "evidence_summary": "Simulator and Tool evidence completed.",
        "limitations": ["N=1"],
    }


def observation(
    *,
    outcome: str = "failure",
    candidate_outcome: str = "fail",
    round_budget: int = 2,
) -> dict:
    should_stop = round_budget <= 1
    return {
        "schema_version": 1,
        "control_template_id": "task_execution.official_baseline",
        "control_required": False,
        "control_passed": None,
        "runtime_limits": {
            "schema_version": 4,
            "round_budget": round_budget,
            "control_requirement": "not_required",
        },
        "assessment": {
            "schema_version": 2,
            "should_stop": should_stop,
            "stop_reason": "budget_exhausted" if should_stop else "continue",
            "claim_verdict": "inconclusive",
            "evidence_sufficient": False,
            "completed_rounds": 1,
            "round_budget": round_budget,
            "budget_remaining": max(round_budget - 1, 0),
            "observed_candidate_ids": ["dynamic.click_bell.left"],
            "decisive_candidate_ids": ["dynamic.click_bell.left"],
            "conflict_candidate_ids": [],
            "unknown_candidate_ids": [],
            "untested_required_candidate_ids": [],
            "untested_candidate_ids": [],
            "recommended_candidate_ids": [],
            "rationale": "The Plan Agent must judge semantic sufficiency.",
            "statistics": {},
            "limitations": ["N=1"],
        },
        "records": [{"round_id": "round_1", "evidence_refs": []}],
        "open_query_evidence_history": [evidence("round_1", outcome)],
        "query_answer": None,
        "candidate_evidence": [
            {
                "candidate_id": "dynamic.click_bell.left",
                "outcome": candidate_outcome,
            }
        ],
    }


class CapturingPlanner:
    def __init__(self, bundle: dict) -> None:
        self.bundle = bundle
        self.histories: list[list[dict]] = []

    def propose(
        self,
        user_query,
        *,
        capabilities,
        evidence_history,
        evaluation_intent=None,
        decision_context=None,
    ):
        self.histories.append(list(evidence_history))
        return self.bundle


class PlanAgentRuntimeTests(unittest.TestCase):
    def test_apply_plan_step_after_required_control_uses_candidate_budget(self):
        query = "Where does this policy first expose a weakness?"
        binding = build_policy_task_binding(
            task_name="click_bell",
            task_family="manipulation",
            policy={"name": "ACT"},
            checkpoint={
                "ready": True,
                "checkpoint_id": "act-click_bell/demo_clean-50",
            },
        )
        frozen_target = {
            "schema_version": 3,
            "binding_mode": "single_task_single_checkpoint_open_world",
            "policy_task_binding": binding,
            "max_rounds": 3,
        }
        control_round = {
            "round_id": "round_1",
            "template_id": "task_execution.official_baseline",
        }
        session = PlanAgentSession(
            query,
            frozen_target,
            require_control_anchor=True,
            control_round=control_round,
        )
        plan = session.normalize_plan(
            {
                "rounds": [control_round],
                "round_decisions": [],
            }
        )
        candidate = build_experiment_candidate(
            source_query=query,
            base_task="click_bell",
            semantic_concern="object_position.left_fixed",
            scene_need={
                "kind": "adapt",
                "description": "Move only the bell to a safe left position.",
                "reuse_first": True,
            },
            candidate_id="dynamic.click_bell.left",
        )
        control_summary = {
            "candidate_evidence": {
                "candidate_id": "task_execution.official_baseline",
                "outcome": "pass",
            }
        }

        updated, decision, _ = session.apply_plan_step(
            plan,
            [control_summary],
            {
                "action": "propose",
                "proposal": candidate,
                "rationale": "The control passed; test the first variation.",
                "answered_query": False,
            },
            materialized_round={
                "round_id": "round_2",
                "proposal": candidate,
            },
        )

        assessment = decision["query_assessment"]
        self.assertEqual(assessment["completed_rounds"], 0)
        self.assertEqual(assessment["budget_remaining"], 2)
        self.assertEqual(assessment["observed_candidate_ids"], [])
        self.assertEqual(len(updated["rounds"]), 2)

    def test_runtime_limits_keep_only_budget_and_control_choice(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(),
            require_control_anchor=False,
        )

        self.assertEqual(
            session.runtime_limits,
            {
                "schema_version": 4,
                "round_budget": 3,
                "control_requirement": "not_required",
            },
        )

    def test_evidence_is_seen_before_the_next_concern_is_authored(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(),
            require_control_anchor=False,
        )
        planner = CapturingPlanner(continue_bundle("object_instance.base0"))
        state = observation(round_budget=3)

        authored = session.propose_semantic_step(
            planner,
            state,
            capabilities=capabilities(),
        )

        self.assertEqual(
            planner.histories[0][0]["round_id"],
            "round_1",
        )
        self.assertEqual(
            authored["proposal"]["sub_aspect"],
            "object_instance.base0",
        )

    def test_continue_becomes_a_generic_executable_candidate(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(),
            require_control_anchor=False,
        )

        bound = session.bind_evidence_conditioned_semantic_step(
            continue_bundle(),
            observation(round_budget=3),
            capabilities=capabilities(),
            executed_candidate_ids=[],
        )

        self.assertEqual(bound["plan_step"]["action"], "propose")
        self.assertEqual(
            bound["plan_step"]["execution_mode"],
            "reuse_or_generate",
        )
        self.assertEqual(
            bound["plan_step"]["proposal"]["base_task"],
            "click_bell",
        )
        self.assertEqual(
            set(bound),
            {
                "schema_version",
                "semantic_proposal_bundle",
                "semantic_needs",
                "resolution",
                "runtime_limits",
                "plan_step",
            },
        )

    def test_agent_owned_stop_answers_from_decisive_evidence(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(),
            require_control_anchor=False,
        )

        stopped = session.bind_semantic_step(
            stop_bundle(),
            observation(round_budget=3),
            executed_template_ids=[],
        )

        self.assertEqual(stopped["plan_step"]["action"], "stop")
        self.assertTrue(stopped["plan_step"]["answered_query"])
        self.assertEqual(stopped["query_assessment"]["stop_reason"], "agent_stop")
        self.assertEqual(
            stopped["query_answer"]["answer"],
            "The tested bounded variation exposes a policy failure.",
        )

    def test_answered_stop_rejects_missing_decisive_evidence(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(),
            require_control_anchor=False,
        )
        state = observation(round_budget=3)
        state["assessment"]["decisive_candidate_ids"] = []

        with self.assertRaisesRegex(
            PlanAgentSessionError,
            "requires decisive completed evidence",
        ):
            session.bind_semantic_step(
                stop_bundle(),
                state,
                executed_template_ids=[],
            )

    def test_external_cap_rejects_an_agent_continue(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(max_rounds=1),
            require_control_anchor=False,
        )

        with self.assertRaisesRegex(
            PlanAgentSessionError,
            "external round cap",
        ):
            session.bind_semantic_step(
                continue_bundle(),
                observation(round_budget=1),
                executed_template_ids=[],
            )

    def test_plan_preservation_prose_is_rejected(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(),
            require_control_anchor=False,
        )
        bundle = continue_bundle()
        bundle["proposal"]["requested_perturbation"]["preserve"] = [
            "task identity"
        ]

        with self.assertRaisesRegex(PlanAgentSessionError, "typed object"):
            session.bind_semantic_step(
                bundle,
                observation(round_budget=3),
                executed_template_ids=[],
            )

    def test_preservation_is_carried_as_typed_taskgen_input(self):
        intent = build_evaluation_intent(
            source_query="Where does this policy first expose a weakness?",
            original_concern="object_position.left_fixed",
            hypothesis="A left placement may expose a weakness.",
            requested_change="Place the bell at a safe left position.",
            preserved_conditions=continue_bundle()["proposal"][
                "requested_perturbation"
            ]["preserve"],
            required_observation="Measure official success and completion time.",
        )
        session = PlanAgentSession(
            intent["source_query"],
            target(),
            require_control_anchor=False,
        )

        bound = session.bind_evidence_conditioned_semantic_step(
            continue_bundle(),
            observation(round_budget=3),
            capabilities=capabilities(),
            executed_candidate_ids=[],
            evaluation_intent=intent,
        )

        candidate = bound["plan_step"]["proposal"]
        self.assertEqual(
            candidate["evaluation_intent"]["preserved_conditions"],
            [
                {
                    "actor": "bell",
                    "property": "position",
                    "axis": "x",
                    "relation": "preserve",
                },
                {
                    "actor": "bell",
                    "property": "position",
                    "axis": "z",
                    "relation": "preserve",
                },
                {
                    "actor": "bell",
                    "property": "orientation",
                    "axis": None,
                    "relation": "preserve",
                },
                {
                    "actor": None,
                    "property": "official_goal",
                    "axis": None,
                    "relation": "preserve",
                },
            ],
        )
        self.assertEqual(
            candidate["intent_alignment"]["relationship"],
            "direct",
        )
        self.assertEqual(
            candidate["scene_need"]["controlled_changes"],
            [
                {
                    "actor": "bell",
                    "property": "position",
                    "axis": "y",
                    "signed_delta": -0.03,
                    "unit": "m",
                    "reference": "same_seed_official_reset",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
