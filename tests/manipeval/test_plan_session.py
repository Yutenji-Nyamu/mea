import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from mea.capability_adapter import (
    build_contract_tool_request,
    resolve_capability_contract,
    taskgen_route,
)
from mea.planner import (
    AdaptivePlanStepAgent,
    BoundTaskPlanSession,
    GlobalRouteError,
    PlanAgentPrototype,
    PlanSessionError,
    build_act_catalog,
    validate_route_selection,
)
from mea.proposals import (
    ProposalError,
    attach_round_proposals,
    validate_task_proposal,
)


def _ready_catalog(root: Path) -> dict:
    for task in ("beat_block_hammer", "click_bell"):
        schema = root / "mea/toolkit/schemas" / f"{task}.json"
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text(
            json.dumps({"task_name": task, "task_family": "manipulation"}),
            encoding="utf-8",
        )
        checkpoint = root / "policy/ACT/act_ckpt" / f"act-{task}" / "demo_clean-50"
        checkpoint.mkdir(parents=True)
        (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
        (checkpoint / "policy_last.ckpt").write_bytes(b"weights")
    return build_act_catalog(root)


def _round(template_id: str, round_id: str = "round_1") -> dict:
    contract = resolve_capability_contract("click_bell", template_id)
    return attach_round_proposals(
        {
            "round_id": round_id,
            "template_id": template_id,
            "capability_id": contract["taskgen"]["capability_id"],
            "task_variant_id": contract["taskgen"]["task_variant_id"],
            "capability_contract": contract,
            "sub_aspect": contract["aspect"]["aspect_id"],
            "aspect_id": contract["aspect"]["aspect_id"],
            "rationale": "bounded test round",
            "task_instruction": "evaluate one bounded click_bell variant",
            "task_name": "click_bell",
            "task_module": "mea.tasks.click_bell",
            "route": taskgen_route(contract),
            "variant_hint": contract["taskgen"]["changes"],
            "execution": {
                "backend": "act",
                "seeds": [100401],
                "num_episodes": 1,
                "gates": contract["required_gates"],
            },
            "observations": ["policy_success", "aggregate", "execution_vqa"],
            "tool_request": build_contract_tool_request(contract),
            "vqa_phenomenon_ids": contract["vqa"]["phenomenon_ids"],
        }
    )


def _observation(*, success: float) -> dict:
    return {
        "round_id": "round_1",
        "pipeline_passed": True,
        "observations": {
            "policy_success": success,
            "aggregate": {
                "status": "passed",
                "input_issues": [],
                "metrics": [
                    {
                        "metric": "bell_active_tcp_min_xy_error",
                        "cohorts": [
                            {
                                "role": "policy_under_evaluation",
                                "summary": {
                                    "quality": {"valid": 1, "missing": 0, "invalid": 0}
                                },
                            }
                        ],
                    }
                ],
            },
            "planned_tool": {"episodes": []},
            "execution_vqa": {
                "status": "passed",
                "evidence_conflict": False,
            },
        },
    }


class _StepProvider:
    last_metadata = {"model": "fake-step"}

    def __init__(self, proposal: dict):
        self.proposal = proposal
        self.prompts = []

    def text(self, prompt, **_kwargs):
        self.prompts.append(prompt)
        return json.dumps(self.proposal)


class PlanSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = _ready_catalog(self.root)
        self.session = BoundTaskPlanSession.from_catalog(
            self.catalog, "click_bell", max_rounds=2
        )
        self.plan = {
            "schema_version": 6,
            "task_name": "click_bell",
            "policy": self.catalog["policy"],
            "evaluation_goal": "position and instance robustness",
            "requested_aspect_ids": ["object_position", "object_instance"],
            "requested_template_ids": [
                "object_position.left_fixed",
                "object_position.right_fixed",
                "object_instance.base0",
            ],
            "rounds": [_round("object_position.left_fixed")],
            "round_decisions": [],
            "max_rounds": 2,
            "planning_state": "awaiting_round_1_observation",
        }

    def _candidate(
        self,
        observation: dict,
        *,
        include_adaptive_fields: bool = True,
        next_template_id: str | None = None,
    ) -> tuple[dict, dict, dict]:
        directive = self.session.directive(self.plan, [observation])
        selected_template = (
            directive["next_template_id"]
            if next_template_id is None
            else next_template_id
        )
        next_round = (
            _round(selected_template, "round_2")
            if directive["action"] == "continue"
            else None
        )
        decision = {
            "schema_version": 1,
            "action": directive["action"],
            "observation_summary": "adapter summary",
            "decision_reason": "adapter materialized trusted template",
            "next_template_id": selected_template,
            "evidence_assessment": {"source": "legacy_adapter"},
            "next_round": next_round,
        }
        if include_adaptive_fields:
            decision.update(
                {
                    "transition": directive["transition"],
                    "next_aspect_id": directive["next_aspect_id"],
                }
            )
        updated = deepcopy(self.plan)
        updated["round_decisions"] = [deepcopy(decision)]
        if next_round is not None:
            updated["rounds"].append(deepcopy(next_round))
            updated["planning_state"] = "awaiting_round_2_observation"
        else:
            updated["planning_state"] = "stopped_after_round_1"
        return updated, decision, directive

    def tearDown(self):
        self.temp.cleanup()

    def test_snapshot_freezes_one_task_and_checkpoint(self):
        snapshot = self.session.snapshot("How robust is this policy?", self.plan)
        self.assertEqual(snapshot["target"]["task_name"], "click_bell")
        self.assertEqual(
            snapshot["target"]["checkpoint"]["checkpoint_id"],
            "act-click_bell/demo_clean-50",
        )
        self.assertEqual(
            snapshot["selected_aspect_ids"], ["object_position", "object_instance"]
        )
        self.assertIn("task_proposal", snapshot["rounds"][0])
        self.assertIn("tool_proposal", snapshot["rounds"][0])

    def test_dynamic_navigation_discovers_an_aspect_not_frozen_before_round_one(self):
        plan = deepcopy(self.plan)
        plan["requested_aspect_ids"] = ["object_position"]
        plan["requested_template_ids"] = [
            "object_position.left_fixed",
            "object_position.right_fixed",
        ]
        success = _observation(success=1.0)
        options = self.session.navigation_options(plan, [success])
        proposed = {
            item["aspect_id"]: item["template_ids"]
            for item in options["available_steps"]["propose"]
        }
        self.assertIn("object_instance", proposed)
        self.assertNotIn("object_instance", plan["requested_aspect_ids"])

        step = {
            "schema_version": 1,
            "action": "propose",
            "aspect_id": "object_instance",
            "template_id": "object_instance.base0",
            "rationale": "Position succeeds, so test a new instance boundary.",
            "answered_query": False,
        }
        updated, decision, _ = self.session.apply_plan_step(
            plan,
            [success],
            step,
            materialized_round=_round("object_instance.base0", "round_2"),
        )
        self.assertIn("object_instance", updated["requested_aspect_ids"])
        self.assertEqual(decision["transition"], "switch_aspect")
        coverage = self.session.coverage(updated, [success])
        self.assertEqual(coverage["covered_aspect_ids"], ["object_position"])
        self.assertEqual(coverage["discovered_aspect_ids"], ["object_instance"])

    def test_failure_options_refine_while_success_can_propose_new_aspect(self):
        plan = deepcopy(self.plan)
        plan["requested_aspect_ids"] = ["object_position"]
        plan["requested_template_ids"] = [
            "object_position.left_fixed",
            "object_position.right_fixed",
        ]
        failed = self.session.navigation_options(plan, [_observation(success=0.0)])
        succeeded = self.session.navigation_options(plan, [_observation(success=1.0)])
        self.assertEqual(
            failed["available_steps"]["refine"][0]["template_ids"],
            ["object_position.right_fixed"],
        )
        self.assertEqual(failed["available_steps"]["propose"], [])
        self.assertFalse(failed["available_steps"]["stop"])
        self.assertTrue(succeeded["available_steps"]["propose"])
        self.assertTrue(succeeded["available_steps"]["stop"])

    def test_registered_navigation_stays_inside_candidate_universe(self):
        plan = deepcopy(self.plan)
        plan["evaluation_goal"] = "instance robustness"
        plan["requested_aspect_ids"] = ["object_instance"]
        plan["initial_requested_aspect_ids"] = ["object_instance"]
        plan["requested_template_ids"] = [
            "object_instance.base0",
            "object_instance.base1",
        ]
        plan["rounds"] = [_round("object_instance.base0")]
        observation = _observation(success=1.0)
        observation["observations"]["aggregate"]["metrics"][0]["metric"] = (
            plan["rounds"][0]["tool_request"]["metric"]
        )
        options = self.session.navigation_options(
            plan,
            [observation],
            allowed_template_ids=(
                "object_instance.base0",
                "object_instance.base1",
            ),
        )
        self.assertEqual(options["available_steps"]["propose"], [])
        self.assertEqual(options["discoverable_aspect_ids"], [])
        self.assertEqual(
            options["available_steps"]["refine"],
            [
                {
                    "aspect_id": "object_instance",
                    "template_ids": ["object_instance.base1"],
                }
            ],
        )
        self.assertTrue(options["available_steps"]["stop"])

        with self.assertRaisesRegex(PlanSessionError, "unknown templates"):
            self.session.navigation_options(
                plan,
                [observation],
                allowed_template_ids=("object_instance.base0", "unknown.template"),
            )

    def test_required_scope_blocks_early_stop_but_fallback_avoids_scope_creep(self):
        success = _observation(success=1.0)
        two_required = self.session.navigation_options(self.plan, [success])
        self.assertFalse(two_required["available_steps"]["stop"])
        self.assertEqual(
            two_required["uncovered_initial_required_aspect_ids"],
            ["object_instance"],
        )
        self.assertEqual(
            two_required["fallback_step"]["aspect_id"], "object_instance"
        )

        one_required = deepcopy(self.plan)
        one_required["requested_aspect_ids"] = ["object_position"]
        one_required["requested_template_ids"] = [
            "object_position.left_fixed",
            "object_position.right_fixed",
        ]
        finished = self.session.navigation_options(one_required, [success])
        self.assertTrue(finished["available_steps"]["stop"])
        self.assertEqual(finished["fallback_step"]["action"], "stop")
        self.assertTrue(finished["fallback_step"]["answered_query"])
        self.assertIn("object_instance", finished["discoverable_aspect_ids"])

    def test_provider_failure_uses_honest_fallback_decision_reason(self):
        plan = deepcopy(self.plan)
        plan["requested_aspect_ids"] = ["object_position"]
        plan["requested_template_ids"] = [
            "object_position.left_fixed",
            "object_position.right_fixed",
        ]
        success = _observation(success=1.0)
        options = self.session.navigation_options(plan, [success])
        invalid_provider = _StepProvider({"not": "a PlanStepProposal"})
        bundle = AdaptivePlanStepAgent(
            invalid_provider, model="fake-step"
        ).propose(
            "Is the requested position condition handled?",
            navigation_options=options,
            planning_context={"target": "click_bell"},
        )
        self.assertEqual(
            bundle["source"], "deterministic_fallback_after_provider_failure"
        )
        updated, decision, _ = self.session.apply_plan_step(
            plan,
            [success],
            bundle["proposal"],
            source=bundle["source"],
        )
        self.assertEqual(updated["planning_state"], "stopped_after_round_1")
        self.assertEqual(
            decision["decision_reason"],
            "deterministic_fallback_after_provider_failure",
        )

    def test_provider_authored_step_reads_rule_vqa_and_query(self):
        options = self.session.navigation_options(
            self.plan, [_observation(success=1.0)]
        )
        proposal = {
            "schema_version": 1,
            "action": "propose",
            "aspect_id": "robustness.scene_clutter",
            "template_id": "robustness.scene_clutter.official_table",
            "rationale": "The first condition passes; inspect clutter next.",
            "answered_query": False,
        }
        provider = _StepProvider(proposal)
        bundle = AdaptivePlanStepAgent(provider, model="fake-step").propose(
            "Can it tolerate target and scene changes?",
            navigation_options=options,
            planning_context={"target": "click_bell"},
        )
        self.assertEqual(bundle["source"], "provider")
        self.assertEqual(bundle["proposal"]["aspect_id"], "robustness.scene_clutter")
        self.assertIn("Can it tolerate", provider.prompts[0])
        self.assertIn('"rule"', provider.prompts[0])
        self.assertIn('"vqa"', provider.prompts[0])
        self.assertIn("aggregate_status only says", provider.prompts[0])

    def test_claim_first_provider_source_keeps_provider_provenance(self):
        plan = deepcopy(self.plan)
        success = _observation(success=1.0)
        proposal = {
            "schema_version": 1,
            "action": "propose",
            "aspect_id": "object_instance",
            "template_id": "object_instance.base0",
            "rationale": "Probe a supported alternate object instance.",
            "answered_query": False,
        }
        materialized = _round("object_instance.base0", "round_2")

        _, decision, _ = self.session.apply_plan_step(
            plan,
            [success],
            proposal,
            materialized_round=materialized,
            source="provider_claim_first_open_query",
        )

        self.assertEqual(
            decision["decision_reason"], "provider_authored_plan_step"
        )
        self.assertEqual(
            decision["plan_step_source"],
            "provider_claim_first_open_query",
        )

    def test_bbh_uses_the_same_dynamic_discovery_api(self):
        session = BoundTaskPlanSession.from_catalog(
            self.catalog, "beat_block_hammer", max_rounds=2
        )
        adapter = PlanAgentPrototype(self.root, object(), model="unused")
        first = adapter.materialize_plan_step(
            "object_appearance.color_blue", 1, "test object generalization"
        )
        plan = {
            "schema_version": 5,
            "task_name": "beat_block_hammer",
            "policy": self.catalog["policy"],
            "evaluation_goal": "object generalization",
            "requested_aspect_ids": ["object_appearance.color"],
            "requested_template_ids": ["object_appearance.color_blue"],
            "rounds": [first],
            "round_decisions": [],
            "max_rounds": 2,
            "planning_state": "awaiting_round_1_observation",
        }
        observation = _observation(success=1.0)
        options = session.navigation_options(plan, [observation])
        proposed_aspects = {
            item["aspect_id"] for item in options["available_steps"]["propose"]
        }
        self.assertIn("object_scale", proposed_aspects)
        self.assertIn("safety.hammer_left_camera_contact", proposed_aspects)
        scale_round = adapter.materialize_plan_step(
            "object_scale.bounded_1_2", 2, "test object generalization"
        )
        updated, decision, _ = session.apply_plan_step(
            plan,
            [observation],
            {
                "schema_version": 1,
                "action": "propose",
                "aspect_id": "object_scale",
                "template_id": "object_scale.bounded_1_2",
                "rationale": "Appearance passes, so discover bounded scale.",
                "answered_query": False,
            },
            materialized_round=scale_round,
        )
        self.assertEqual(decision["transition"], "switch_aspect")
        self.assertEqual(updated["rounds"][-1]["capability_id"], "object_scale.bounded")

    def test_normalize_plan_enriches_legacy_round_for_execution(self):
        legacy_plan = dict(self.plan)
        legacy_round = dict(self.plan["rounds"][0])
        legacy_round.pop("task_proposal")
        legacy_round.pop("tool_proposal")
        legacy_plan["rounds"] = [legacy_round]
        normalized = self.session.normalize_plan(legacy_plan)
        self.assertIn("task_proposal", normalized["rounds"][0])
        self.assertIn("tool_proposal", normalized["rounds"][0])
        self.assertEqual(
            normalized["rounds"][0]["task_proposal"]["task_name"],
            "click_bell",
        )

    def test_cached_failure_drills_down_and_success_switches_aspect(self):
        failed = self.session.assess(self.plan, [_observation(success=0.0)])
        self.assertEqual(failed["required_transition"], "drill_down")
        self.assertEqual(failed["required_next_aspect_id"], "object_position")
        succeeded = self.session.assess(self.plan, [_observation(success=1.0)])
        self.assertEqual(succeeded["required_transition"], "switch_aspect")
        self.assertEqual(succeeded["required_next_aspect_id"], "object_instance")

    def test_adjudicate_failure_drills_down_and_enriches_legacy_decision(self):
        observation = _observation(success=0.0)
        candidate, decision, directive = self._candidate(
            observation, include_adaptive_fields=False
        )
        updated, canonical = self.session.adjudicate(
            self.plan,
            [observation],
            candidate_plan=candidate,
            candidate_decision=decision,
        )
        self.assertEqual(directive["action"], "continue")
        self.assertEqual(canonical["transition"], "drill_down")
        self.assertEqual(canonical["next_aspect_id"], "object_position")
        self.assertEqual(canonical["next_template_id"], "object_position.right_fixed")
        self.assertEqual(updated["rounds"][-1], canonical["next_round"])
        self.assertEqual(updated["round_decisions"][-1], canonical)

    def test_adjudicate_success_switches_aspect(self):
        observation = _observation(success=1.0)
        candidate, decision, directive = self._candidate(observation)
        updated, canonical = self.session.adjudicate(
            self.plan,
            [observation],
            candidate_plan=candidate,
            candidate_decision=decision,
        )
        self.assertEqual(directive["transition"], "switch_aspect")
        self.assertEqual(canonical["next_aspect_id"], "object_instance")
        self.assertEqual(canonical["next_template_id"], "object_instance.base0")
        self.assertEqual(updated["planning_state"], "awaiting_round_2_observation")

    def test_adjudicate_accepts_a_non_first_allowed_template(self):
        observation = _observation(success=1.0)
        next_round = _round("object_instance.base1", "round_2")
        decision = {
            "schema_version": 1,
            "action": "continue",
            "transition": "switch_aspect",
            "observation_summary": "choose another bounded instance",
            "decision_reason": "model selected a legal candidate",
            "next_aspect_id": "object_instance",
            "next_template_id": "object_instance.base1",
            "next_round": next_round,
        }
        candidate = deepcopy(self.plan)
        candidate["rounds"].append(deepcopy(next_round))
        candidate["round_decisions"] = [deepcopy(decision)]
        candidate["planning_state"] = "awaiting_round_2_observation"

        updated, canonical = self.session.adjudicate(
            self.plan,
            [observation],
            candidate_plan=candidate,
            candidate_decision=decision,
        )
        self.assertEqual(canonical["next_template_id"], "object_instance.base1")
        self.assertEqual(updated["rounds"][-1]["template_id"], "object_instance.base1")

    def test_adjudicate_accepts_any_allowed_unseen_aspect(self):
        plan = deepcopy(self.plan)
        plan["requested_aspect_ids"].append("robustness.scene_clutter")
        plan["requested_template_ids"].append(
            "robustness.scene_clutter.official_table"
        )
        observation = _observation(success=1.0)
        assessment = self.session.assess(plan, [observation])
        self.assertEqual(
            assessment["available_transitions"]["switch_aspect"],
            ["object_instance", "robustness.scene_clutter"],
        )
        next_round = _round("robustness.scene_clutter.official_table", "round_2")
        decision = {
            "schema_version": 1,
            "action": "continue",
            "transition": "switch_aspect",
            "observation_summary": "select an alternative uncovered aspect",
            "decision_reason": "both aspects are evidence-compatible",
            "next_aspect_id": "robustness.scene_clutter",
            "next_template_id": "robustness.scene_clutter.official_table",
            "next_round": next_round,
        }
        candidate = deepcopy(plan)
        candidate["rounds"].append(deepcopy(next_round))
        candidate["round_decisions"] = [deepcopy(decision)]
        candidate["planning_state"] = "awaiting_round_2_observation"
        _, canonical = self.session.adjudicate(
            plan,
            [observation],
            candidate_plan=candidate,
            candidate_decision=decision,
        )
        self.assertEqual(canonical["next_aspect_id"], "robustness.scene_clutter")

        rejected = deepcopy(decision)
        rejected["next_aspect_id"] = "scene_lighting"
        rejected["next_template_id"] = "scene_lighting.static_random"
        with self.assertRaisesRegex(PlanSessionError, "outside allowed"):
            self.session.directive(
                plan,
                [observation],
                candidate_decision=rejected,
            )

    def test_adjudicate_stops_at_bound_budget(self):
        session = BoundTaskPlanSession.from_catalog(
            self.catalog, "click_bell", max_rounds=1
        )
        plan = deepcopy(self.plan)
        plan["max_rounds"] = 1
        observation = _observation(success=0.0)
        directive = session.directive(plan, [observation])
        decision = {
            "schema_version": 1,
            "action": "stop",
            "transition": "stop",
            "observation_summary": "budget exhausted",
            "decision_reason": "bound budget forces stop",
            "next_aspect_id": None,
            "next_template_id": None,
            "next_round": None,
        }
        candidate = deepcopy(plan)
        candidate["round_decisions"] = [deepcopy(decision)]
        candidate["planning_state"] = "stopped_after_round_1"
        updated, canonical = session.adjudicate(
            plan,
            [observation],
            candidate_plan=candidate,
            candidate_decision=decision,
        )
        self.assertEqual(directive["action"], "stop")
        self.assertEqual(canonical["transition"], "stop")
        self.assertIsNone(canonical["next_template_id"])
        self.assertEqual(len(updated["rounds"]), 1)

    def test_adjudicate_rejects_candidate_scope_changes(self):
        observation = _observation(success=0.0)
        candidate, decision, _ = self._candidate(observation)

        cases = []
        changed_task = deepcopy(candidate)
        changed_task["task_name"] = "beat_block_hammer"
        cases.append(("task", changed_task, decision))

        changed_checkpoint = deepcopy(candidate)
        changed_checkpoint["checkpoint_id"] = "act-click_bell/other"
        cases.append(("checkpoint", changed_checkpoint, decision))

        changed_aspects = deepcopy(candidate)
        changed_aspects["requested_aspect_ids"] = ["object_position"]
        cases.append(("aspect", changed_aspects, decision))

        changed_budget = deepcopy(candidate)
        changed_budget["max_rounds"] = 1
        cases.append(("budget", changed_budget, decision))

        changed_template, template_decision, _ = self._candidate(
            observation, next_template_id="object_instance.base0"
        )
        cases.append(("template", changed_template, template_decision))

        for name, changed_plan, changed_decision in cases:
            with self.subTest(name=name):
                with self.assertRaises(PlanSessionError):
                    self.session.adjudicate(
                        self.plan,
                        [observation],
                        candidate_plan=changed_plan,
                        candidate_decision=changed_decision,
                    )

    def test_plan_and_proposal_cannot_switch_task(self):
        changed = dict(self.plan)
        changed["task_name"] = "beat_block_hammer"
        with self.assertRaisesRegex(ValueError, "cannot switch bound task"):
            self.session.snapshot("query", changed)
        proposal = dict(self.plan["rounds"][0]["task_proposal"])
        proposal["task_name"] = "beat_block_hammer"
        with self.assertRaises(ProposalError):
            validate_task_proposal(proposal, expected_task_name="click_bell")

    def test_round_budget_fails_closed(self):
        for invalid in (0, False, "2", 3):
            with self.subTest(invalid=invalid):
                changed = dict(self.plan)
                changed["max_rounds"] = invalid
                with self.assertRaisesRegex(ValueError, "max_rounds|round budget"):
                    self.session.normalize_plan(changed)

    def test_bound_global_route_rejects_another_ready_task(self):
        selection = {
            "schema_version": 2,
            "decision": "route",
            "task_name": "beat_block_hammer",
            "task_profile": "generated",
            "evaluation_goal": "appearance",
            "requested_aspect_ids": ["object_appearance.color"],
            "first_aspect_id": "object_appearance.color",
            "unsupported_capabilities": [],
        }
        with self.assertRaisesRegex(GlobalRouteError, "bound to task"):
            validate_route_selection(
                selection,
                self.catalog,
                expected_task_name="click_bell",
            )


if __name__ == "__main__":
    unittest.main()
