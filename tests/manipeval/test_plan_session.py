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
    BoundTaskPlanSession,
    GlobalRouteError,
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
            "execution_vqa": {"evidence_conflict": False},
        },
    }


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
