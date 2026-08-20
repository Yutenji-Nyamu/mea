"""High-information tests for the paper-aligned Plan Agent session."""

from __future__ import annotations

import unittest
from unittest.mock import patch

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


def official_retry_bundle() -> dict:
    return {
        "schema_version": 2,
        "source": "fixture_plan_agent",
        "proposal": {
            "schema_version": 2,
            "action": "continue",
            "sub_aspect": "task_execution.official_retry",
            "hypothesis": (
                "A second unchanged official attempt may establish the missing "
                "baseline."
            ),
            "requested_perturbation": {
                "description": "Retry the unchanged official task.",
                "controlled_changes": [],
                "preserve": [],
            },
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
                "The first official attempt was unsuccessful, so retry the "
                "same frozen task and seed before making an attributed claim."
            ),
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


def non_comparable_record() -> dict:
    return {
        "round_id": "round_1",
        "template_id": "",
        "candidate_id": "dynamic.click_bell.generated_checker",
        "candidate_evidence": {
            "candidate_id": "dynamic.click_bell.generated_checker",
            "outcome": "unknown",
        },
        "round_evidence": {
            "pipeline": {"passed": True},
            "policy": {"success_rate": None},
        },
        "evaluation_outcome": {
            "metric": "generated_check_success",
            "official_equivalent": None,
        },
        "outcome_semantics": {
            "status": "non_comparable",
            "evidence_conflict": False,
        },
        "planning_observation": None,
        "open_query_evidence": evidence("round_1", "ambiguous"),
        "evidence_refs": [],
    }


def control_record(
    *,
    round_id: str = "round_1",
    candidate_id: str = "task_execution.official_baseline",
    template_id: str | None = "task_execution.official_baseline",
    success_rate: float | None = 1.0,
    pipeline_passed: bool = True,
    planning_observation: dict | None = None,
    authority: str | None = "official_check_success",
    official_equivalent: bool | None = True,
    semantics_status: str = "official_only",
) -> dict:
    official_identity = bool(
        authority
        in {"official_check_success", "official_check_success_reused"}
        and official_equivalent is True
        and semantics_status == "official_only"
    )
    candidate_outcome = (
        "unknown"
        if not pipeline_passed
        or success_rate is None
        or not official_identity
        else "pass"
        if success_rate >= 1.0
        else "fail"
    )
    semantic_outcome = {
        "pass": "success",
        "fail": "failure",
        "unknown": "ambiguous",
    }[candidate_outcome]
    return {
        "round_id": round_id,
        "template_id": template_id,
        "candidate_id": candidate_id,
        "candidate_evidence": {
            "candidate_id": candidate_id,
            "outcome": candidate_outcome,
        },
        "round_evidence": {
            "pipeline": {"passed": pipeline_passed},
            "policy": {
                "success_rate": success_rate,
                "metric": "official_check_success",
                "authority": authority,
                "official_equivalent": official_equivalent,
            },
        },
        "evaluation_outcome": {
            "metric": "official_check_success",
            "authority": authority,
            "official_equivalent": official_equivalent,
        },
        "outcome_semantics": {
            "status": semantics_status,
            "evidence_conflict": False,
        },
        "planning_observation": planning_observation,
        "open_query_evidence": evidence(round_id, semantic_outcome),
        "evidence_refs": [],
    }


def observe_records(
    session: PlanAgentSession,
    round_plans: list[dict],
    records: list[dict],
) -> dict:
    with patch(
        "mea.planner.plan_agent_evidence_session."
        "build_plan_agent_evidence_record",
        side_effect=records,
    ):
        return session.observe(
            round_plans,
            [{"round_id": plan["round_id"]} for plan in round_plans],
        )


def observe_record(session: PlanAgentSession, record: dict) -> dict:
    return observe_records(
        session,
        [{"round_id": record["round_id"]}],
        [record],
    )


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

    def test_answered_stop_round_trips_through_execution_transport(self):
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
        session = PlanAgentSession(
            query,
            {
                "schema_version": 3,
                "binding_mode": "single_task_single_checkpoint_open_world",
                "policy_task_binding": binding,
                "max_rounds": 3,
            },
            require_control_anchor=False,
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
        plan = session.normalize_plan(
            {
                "rounds": [
                    {
                        "round_id": "round_1",
                        "proposal": candidate,
                    }
                ],
                "round_decisions": [],
            }
        )
        bound = session.bind_evidence_conditioned_semantic_step(
            stop_bundle(),
            observation(round_budget=3),
            capabilities=capabilities(),
            executed_candidate_ids=[candidate["candidate_id"]],
        )

        updated, decision, directive = session.apply_plan_step(
            plan,
            [
                {
                    "candidate_evidence": {
                        "candidate_id": candidate["candidate_id"],
                        "outcome": "fail",
                    }
                }
            ],
            bound["plan_step"],
            runtime_limits=bound["runtime_limits"],
        )

        self.assertEqual(decision["action"], "stop")
        self.assertTrue(decision["answered_query"])
        self.assertEqual(
            decision["query_assessment"]["agent_answer"],
            "The tested bounded variation exposes a policy failure.",
        )
        self.assertEqual(
            updated["planning_state"], "stopped_after_round_1"
        )
        self.assertTrue(directive["query_assessment"]["should_stop"])

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
            bound["plan_step"]["proposal"]["candidate_id"],
            "dynamic.click_bell.round_1",
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

    def test_repeated_concern_uses_readable_round_scoped_candidate_ids(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(),
            require_control_anchor=False,
        )
        first = session.bind_evidence_conditioned_semantic_step(
            continue_bundle(),
            observation(round_budget=3),
            capabilities=capabilities(),
            executed_candidate_ids=[],
        )
        second_bundle = continue_bundle()
        second_bundle["proposal"]["requested_perturbation"][
            "controlled_changes"
        ][0]["signed_delta"] = -0.04
        with self.assertRaisesRegex(
            PlanAgentSessionError,
            "dynamic candidate id collision",
        ):
            session.bind_evidence_conditioned_semantic_step(
                second_bundle,
                observation(round_budget=3),
                capabilities=capabilities(),
                executed_candidate_ids=[],
            )
        second = session.bind_evidence_conditioned_semantic_step(
            second_bundle,
            observation(round_budget=3),
            capabilities=capabilities(),
            executed_candidate_ids=[
                first["plan_step"]["proposal"]["candidate_id"]
            ],
        )

        self.assertEqual(
            first["plan_step"]["proposal"]["candidate_id"],
            "dynamic.click_bell.round_1",
        )
        self.assertEqual(
            second["plan_step"]["proposal"]["candidate_id"],
            "dynamic.click_bell.round_2",
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

    def test_conflict_with_budget_allows_a_disambiguating_candidate(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(),
            require_control_anchor=False,
        )
        conflict_record = {
            "round_id": "round_1",
            "candidate_id": "dynamic.click_bell.left",
            "candidate_evidence": {
                "candidate_id": "dynamic.click_bell.left",
                "outcome": "conflict",
            },
            "round_evidence": {
                "pipeline": {"passed": True},
                "policy": {"success_rate": None},
            },
            "evaluation_outcome": {"metric": "official_check_success"},
            "outcome_semantics": {"status": "conflict"},
            "planning_observation": None,
            "open_query_evidence": evidence("round_1", "ambiguous"),
        }
        with patch(
            "mea.planner.plan_agent_evidence_session."
            "build_plan_agent_evidence_record",
            return_value=conflict_record,
        ):
            state = session.observe(
                [{"round_id": "round_1"}],
                [{"round_id": "round_1"}],
            )

        self.assertFalse(state["assessment"]["should_stop"])
        self.assertEqual(
            state["assessment"]["conflict_candidate_ids"],
            ["dynamic.click_bell.left"],
        )

        bound = session.bind_semantic_step(
            continue_bundle("object_position.disambiguation"),
            state,
            executed_template_ids=[],
        )

        self.assertEqual(bound["plan_step"]["action"], "propose")
        self.assertEqual(
            bound["plan_step"]["proposal"]["evaluation_intent"][
                "original_concern"
            ],
            "object_position.disambiguation",
        )

    def test_non_comparable_candidate_with_budget_can_continue(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(max_rounds=3),
            require_control_anchor=False,
        )
        record = non_comparable_record()
        state = observe_record(session, record)

        self.assertFalse(state["assessment"]["should_stop"])
        self.assertEqual(state["assessment"]["stop_reason"], "continue")
        self.assertEqual(
            state["assessment"]["unknown_candidate_ids"],
            [record["candidate_id"]],
        )
        self.assertEqual(
            state["records"][0]["outcome_semantics"]["status"],
            "non_comparable",
        )

        bound = session.bind_semantic_step(
            continue_bundle("checker_semantics.disambiguation"),
            state,
            executed_template_ids=[record["candidate_id"]],
        )
        self.assertEqual(bound["plan_step"]["action"], "propose")

    def test_control_identity_is_tracked_without_forcing_budgeted_stop(self):
        cases = (
            (control_record(), True),
            (
                control_record(authority="official_check_success_reused"),
                True,
            ),
            (control_record(authority=None), False),
            (control_record(official_equivalent=None), False),
            (control_record(semantics_status="non_comparable"), False),
        )

        for record, expected_valid in cases:
            with self.subTest(
                authority=record["evaluation_outcome"]["authority"],
                official_equivalent=record["evaluation_outcome"][
                    "official_equivalent"
                ],
                status=record["outcome_semantics"]["status"],
            ):
                session = PlanAgentSession(
                    "Where does this policy first expose a weakness?",
                    target(max_rounds=3),
                    require_control_anchor=True,
                )
                state = observe_record(session, record)

                self.assertIs(state["control_passed"], expected_valid)
                self.assertFalse(state["assessment"]["should_stop"])
                self.assertEqual(state["assessment"]["stop_reason"], "continue")
                self.assertEqual(state["assessment"]["budget_remaining"], 2)
                self.assertEqual(state["records"], [record])
                if not expected_valid:
                    self.assertTrue(
                        any(
                            "requires a successful unchanged official baseline"
                            in limitation
                            for limitation in state["assessment"]["limitations"]
                        )
                    )

    def test_unsuccessful_control_can_schedule_an_unchanged_retry(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(max_rounds=3),
            require_control_anchor=True,
        )
        failed_control = control_record(success_rate=0.0)
        state = observe_record(session, failed_control)

        bound = session.bind_semantic_step(
            official_retry_bundle(),
            state,
            executed_template_ids=[failed_control["candidate_id"]],
        )

        retry = bound["plan_step"]["proposal"]
        self.assertEqual(bound["plan_step"]["action"], "propose")
        self.assertEqual(retry["candidate_id"], "dynamic.click_bell.round_2")
        for field in (
            "scene_need",
            "checker_need",
            "rule_tool_need",
            "vqa_tool_need",
            "tool_need",
        ):
            self.assertIsNone(retry[field])
        self.assertEqual(
            bound["semantic_proposal_bundle"]["proposal"][
                "requested_perturbation"
            ]["controlled_changes"],
            [],
        )

    def test_generated_scene_official_metric_does_not_restore_baseline(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(max_rounds=3),
            require_control_anchor=True,
        )
        generated = build_experiment_candidate(
            source_query=session.user_query,
            base_task="click_bell",
            semantic_concern="object_position.left_fixed",
            scene_need="Move the bell left.",
            candidate_id="dynamic.click_bell.generated_scene",
        )
        plans = [
            {
                "round_id": "round_1",
                "template_id": "task_execution.official_baseline",
                "route": "official",
            },
            {
                "round_id": "round_2",
                "route": "generic_provider_scene_checker_codegen",
                "proposal": generated,
            },
        ]
        records = [
            control_record(success_rate=0.0),
            control_record(
                round_id="round_2",
                candidate_id=generated["candidate_id"],
                template_id=None,
            ),
        ]

        state = observe_records(session, plans, records)

        self.assertFalse(state["control_passed"])
        self.assertEqual(
            state["assessment"]["observed_candidate_ids"],
            [generated["candidate_id"]],
        )
        self.assertEqual(
            state["assessment"]["decisive_candidate_ids"],
            [generated["candidate_id"]],
        )
        for verdict in ("supported", "refuted"):
            with self.subTest(verdict=verdict):
                proposal = stop_bundle()
                proposal["proposal"]["claim_verdict"] = verdict
                with self.assertRaisesRegex(
                    PlanAgentSessionError,
                    "requires a valid unchanged official baseline",
                ):
                    session.bind_semantic_step(
                        proposal,
                        state,
                        executed_template_ids=[
                            item["candidate_id"] for item in records
                        ],
                    )

        stopped = session.bind_semantic_step(
            stop_bundle(sufficient=False),
            state,
            executed_template_ids=[item["candidate_id"] for item in records],
        )
        self.assertFalse(stopped["plan_step"]["answered_query"])
        self.assertEqual(
            stopped["plan_step"]["claim_verdict"], "inconclusive"
        )

    def test_official_tool_round_is_candidate_evidence_not_a_baseline_retry(self):
        cases = {
            "rule": {
                "rule_tool_need": {
                    "kind": "measure",
                    "description": "Measure completion time.",
                    "reuse_first": True,
                }
            },
            "vqa": {
                "vqa_tool_need": {
                    "kind": "vqa",
                    "description": "Inspect visible hesitation.",
                    "reuse_first": True,
                }
            },
        }

        for label, requested_need in cases.items():
            with self.subTest(need=label):
                session = PlanAgentSession(
                    "Where does this policy first expose a weakness?",
                    target(max_rounds=3),
                    require_control_anchor=True,
                )
                candidate = build_experiment_candidate(
                    source_query=session.user_query,
                    base_task="click_bell",
                    semantic_concern=f"observation.{label}",
                    candidate_id=f"dynamic.click_bell.{label}",
                    **requested_need,
                )
                plans = [
                    {
                        "round_id": "round_1",
                        "template_id": "task_execution.official_baseline",
                        "route": "official",
                    },
                    {
                        "round_id": "round_2",
                        "route": "official",
                        "proposal": candidate,
                    },
                ]
                records = [
                    control_record(success_rate=0.0),
                    control_record(
                        round_id="round_2",
                        candidate_id=candidate["candidate_id"],
                        template_id=None,
                    ),
                ]

                state = observe_records(session, plans, records)

                self.assertFalse(state["control_passed"])
                self.assertEqual(
                    state["assessment"]["observed_candidate_ids"],
                    [candidate["candidate_id"]],
                )

    def test_later_unchanged_retry_establishes_baseline_without_candidate_evidence(
        self,
    ):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(max_rounds=4),
            require_control_anchor=True,
        )
        generated = build_experiment_candidate(
            source_query=session.user_query,
            base_task="click_bell",
            semantic_concern="object_position.left_fixed",
            scene_need="Move the bell left.",
            candidate_id="dynamic.click_bell.generated_scene",
        )
        retry = build_experiment_candidate(
            source_query=session.user_query,
            base_task="click_bell",
            semantic_concern="task_execution.official_retry",
            candidate_id="dynamic.click_bell.official_retry",
        )
        plans = [
            {
                "round_id": "round_1",
                "template_id": "task_execution.official_baseline",
                "route": "official",
            },
            {
                "round_id": "round_2",
                "route": "generic_provider_scene_checker_codegen",
                "proposal": generated,
            },
            {
                "round_id": "round_3",
                "route": "official",
                "proposal": retry,
            },
        ]
        records = [
            control_record(success_rate=0.0),
            control_record(
                round_id="round_2",
                candidate_id=generated["candidate_id"],
                template_id=None,
                success_rate=0.0,
            ),
            control_record(
                round_id="round_3",
                candidate_id=retry["candidate_id"],
                template_id=None,
                authority="official_check_success_reused",
            ),
        ]

        state = observe_records(session, plans, records)

        self.assertTrue(state["control_passed"])
        self.assertEqual(state["assessment"]["completed_rounds"], 2)
        self.assertEqual(state["assessment"]["budget_remaining"], 1)
        self.assertEqual(
            state["assessment"]["observed_candidate_ids"],
            [generated["candidate_id"]],
        )
        self.assertEqual(
            state["assessment"]["decisive_candidate_ids"],
            [generated["candidate_id"]],
        )
        self.assertEqual(len(state["records"]), 3)

    def test_unchanged_retries_consume_budget_until_external_cap(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(max_rounds=3),
            require_control_anchor=True,
        )
        retries = [
            build_experiment_candidate(
                source_query=session.user_query,
                base_task="click_bell",
                semantic_concern=f"task_execution.official_retry_{index}",
                candidate_id=f"dynamic.click_bell.official_retry_{index}",
            )
            for index in (1, 2)
        ]
        plans = [
            {
                "round_id": "round_1",
                "template_id": "task_execution.official_baseline",
                "route": "official",
            },
            *[
                {
                    "round_id": f"round_{index + 2}",
                    "route": "official",
                    "proposal": retry,
                }
                for index, retry in enumerate(retries)
            ],
        ]
        records = [
            control_record(success_rate=0.0),
            control_record(
                round_id="round_2",
                candidate_id=retries[0]["candidate_id"],
                template_id=None,
                success_rate=None,
                pipeline_passed=False,
                planning_observation={
                    "kind": "official_baseline_unavailable",
                    "status": "inconclusive",
                },
            ),
            control_record(
                round_id="round_3",
                candidate_id=retries[1]["candidate_id"],
                template_id=None,
                success_rate=0.0,
            ),
        ]

        state = observe_records(session, plans, records)

        self.assertFalse(state["control_passed"])
        self.assertTrue(state["assessment"]["should_stop"])
        self.assertEqual(
            state["assessment"]["stop_reason"], "budget_exhausted"
        )
        self.assertEqual(state["assessment"]["completed_rounds"], 2)
        self.assertEqual(state["assessment"]["budget_remaining"], 0)
        self.assertEqual(state["assessment"]["observed_candidate_ids"], [])
        self.assertEqual(len(state["records"]), 3)

    def test_non_comparable_candidate_stops_only_at_external_cap(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(max_rounds=1),
            require_control_anchor=False,
        )
        record = non_comparable_record()
        state = observe_record(session, record)

        self.assertTrue(state["assessment"]["should_stop"])
        self.assertEqual(
            state["assessment"]["stop_reason"], "budget_exhausted"
        )
        with self.assertRaisesRegex(
            PlanAgentSessionError,
            "external round cap",
        ):
            session.bind_semantic_step(
                continue_bundle("checker_semantics.disambiguation"),
                state,
                executed_template_ids=[record["candidate_id"]],
            )

    def test_non_comparable_candidate_cannot_support_answered_stop(self):
        session = PlanAgentSession(
            "Where does this policy first expose a weakness?",
            target(max_rounds=3),
            require_control_anchor=False,
        )
        record = non_comparable_record()
        state = observe_record(session, record)

        for verdict in ("supported", "refuted"):
            with self.subTest(verdict=verdict):
                proposal = stop_bundle()
                proposal["proposal"]["claim_verdict"] = verdict
                with self.assertRaisesRegex(
                    PlanAgentSessionError,
                    "requires decisive completed evidence",
                ):
                    session.bind_semantic_step(
                        proposal,
                        state,
                        executed_template_ids=[record["candidate_id"]],
                    )

        stopped = session.bind_semantic_step(
            stop_bundle(sufficient=False),
            state,
            executed_template_ids=[record["candidate_id"]],
        )
        self.assertFalse(stopped["plan_step"]["answered_query"])
        self.assertEqual(
            stopped["plan_step"]["claim_verdict"], "inconclusive"
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
        self.assertNotIn("intent_alignment", candidate)
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
