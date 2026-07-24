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
from mea.planner import BoundTaskPlanSession, PlanSessionError, build_act_catalog
from mea.proposals import (
    attach_round_proposals,
    task_proposal_from_contract,
    tool_proposal_from_contract,
)
from mea.proposal_agent import ProposalAgentError
from scripts.manipeval_agent import (
    apply_bounded_round_proposal,
    persist_adaptive_step_selection,
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


def _round(template_id: str, round_id: str) -> dict:
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
            "rationale": "bounded integration test",
            "task_instruction": "evaluate one trusted click_bell variation",
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


def _successful_observation() -> dict:
    return {
        "round_id": "round_1",
        "pipeline_passed": True,
        "observations": {
            "policy_success": 1.0,
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


class FakeProposalAgent:
    def __init__(self) -> None:
        self.last_prompt = None
        self.last_responses = []
        self.calls = []

    def propose(
        self,
        user_query,
        *,
        target,
        aspect_id,
        base_template_id,
        capability_mode,
        planning_context,
        require_novel_changes,
        require_new_tool,
    ):
        call_number = len(self.calls) + 1
        self.calls.append(
            {
                "user_query": user_query,
                "target": deepcopy(target),
                "aspect_id": aspect_id,
                "base_template_id": base_template_id,
                "capability_mode": capability_mode,
                "planning_context": deepcopy(planning_context),
                "require_novel_changes": require_novel_changes,
                "require_new_tool": require_new_tool,
            }
        )
        self.last_prompt = f"prompt for round {call_number}"
        self.last_responses = [f"response for round {call_number}"]
        contract = resolve_capability_contract("click_bell", base_template_id)
        task_proposal = task_proposal_from_contract(
            contract, intent=f"model-authored intent for round {call_number}"
        )
        task_proposal["proposal_id"] = f"{base_template_id}.authored_{call_number}"
        tool_proposal = tool_proposal_from_contract(
            contract, evaluation_goal=f"model-authored goal for round {call_number}"
        )
        tool_proposal["proposal_id"] = f"{base_template_id}.tool_authored_{call_number}"
        return {
            "schema_version": 1,
            "task_proposal": task_proposal,
            "tool_proposal": tool_proposal,
            "tool_route_preview": {"resolved_route": "unit_test"},
        }


class FailingProposalAgent:
    def __init__(self) -> None:
        self.last_prompt = "prompt retained after provider failure"
        self.last_responses = ["malformed first response"]
        self.last_errors = ["ProposalError: malformed first response"]
        self.last_repairs = []

    def propose(self, *args, **kwargs):
        raise ProposalAgentError("proposal failed twice")


class AgentBoundedProposalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = _ready_catalog(self.root)
        self.session = BoundTaskPlanSession.from_catalog(
            self.catalog, "click_bell", max_rounds=2
        )
        self.evaluation_dir = self.root / "mea/evaluation_runs/eval_test"
        self.context = {"schema_version": 1, "source": "unit_test"}

    def tearDown(self):
        self.temp.cleanup()

    def _apply(self, agent, round_plan, round_number):
        return apply_bounded_round_proposal(
            proposal_agent=agent,
            user_query="How robust is this policy?",
            target=self.session.target,
            planning_context=self.context,
            round_plan=round_plan,
            evaluation_dir=self.evaluation_dir,
            round_number=round_number,
        )

    def test_each_round_has_an_independent_proposal_artifact_directory(self):
        agent = FakeProposalAgent()
        _, first = self._apply(
            agent, _round("object_position.left_fixed", "round_1"), 1
        )
        _, second = self._apply(
            agent, _round("object_instance.base1", "round_2"), 2
        )

        proposal_root = self.evaluation_dir / "plan/bounded_proposal"
        for number, expected in ((1, first), (2, second)):
            round_dir = proposal_root / f"round_{number:02d}"
            self.assertTrue((round_dir / "prompt.md").is_file())
            self.assertTrue((round_dir / "response_1.txt").is_file())
            self.assertEqual(
                json.loads((round_dir / "proposal_bundle.json").read_text()),
                expected,
            )
        self.assertEqual(first["round_number"], 1)
        self.assertEqual(second["round_number"], 2)
        self.assertEqual(first["attempt_count"], 1)
        self.assertEqual(first["provider_or_validation_errors"], [])
        self.assertEqual(agent.calls[0]["capability_mode"], "novel_bounded")
        self.assertEqual(agent.calls[1]["capability_mode"], "registered_reuse")

    def test_adaptive_step_selection_is_durable_before_task_materialization(self):
        bundle = {
            "schema_version": 1,
            "source": "unit_test",
            "proposal": {
                "action": "continue",
                "template_id": "object_instance.base1",
            },
        }
        navigation = {
            "schema_version": 1,
            "completed_template_ids": ["object_position.left_fixed"],
        }

        relative = persist_adaptive_step_selection(
            self.evaluation_dir,
            after_round=1,
            prompt="choose the next evidence-bearing test",
            responses=["continue with object_instance.base1"],
            step_bundle=bundle,
            navigation_options=navigation,
        )

        step_dir = self.evaluation_dir / relative
        self.assertEqual(relative, "plan/adaptive_steps/after_round_01")
        self.assertEqual(
            json.loads((step_dir / "plan_step_bundle.json").read_text()), bundle
        )
        self.assertEqual(
            json.loads(
                (self.evaluation_dir / "plan/evidence_after_round_1.json").read_text()
            ),
            navigation,
        )
        self.assertEqual(
            (step_dir / "response_1.txt").read_text(encoding="utf-8"),
            "continue with object_instance.base1\n",
        )

    def test_failed_proposal_retains_prompt_response_and_typed_failure(self):
        with self.assertRaisesRegex(ProposalAgentError, "failed twice"):
            self._apply(
                FailingProposalAgent(),
                _round("object_position.left_fixed", "round_1"),
                1,
            )

        proposal_dir = self.evaluation_dir / "plan/bounded_proposal/round_01"
        failure = json.loads(
            (proposal_dir / "proposal_failure.json").read_text(encoding="utf-8")
        )
        self.assertEqual(failure["status"], "failed")
        self.assertEqual(failure["failure"]["type"], "ProposalAgentError")
        self.assertEqual(
            failure["provider_or_validation_errors"],
            ["ProposalError: malformed first response"],
        )
        self.assertEqual(
            (proposal_dir / "prompt.md").read_text(encoding="utf-8"),
            "prompt retained after provider failure",
        )
        self.assertEqual(
            (proposal_dir / "response_1.txt").read_text(encoding="utf-8"),
            "malformed first response\n",
        )

    def test_round_one_keeps_novel_first_round_compatibility_artifacts(self):
        agent = FakeProposalAgent()
        _, first = self._apply(
            agent, _round("object_position.left_fixed", "round_1"), 1
        )
        proposal_root = self.evaluation_dir / "plan/bounded_proposal"
        compatibility_bundle = json.loads(
            (proposal_root / "proposal_bundle.json").read_text(encoding="utf-8")
        )
        self.assertEqual(compatibility_bundle, first)
        self.assertEqual(
            (proposal_root / "prompt.md").read_text(encoding="utf-8"),
            "prompt for round 1",
        )

        self._apply(agent, _round("object_instance.base1", "round_2"), 2)
        self.assertEqual(
            json.loads(
                (proposal_root / "proposal_bundle.json").read_text(encoding="utf-8")
            ),
            first,
        )
        self.assertEqual(
            (proposal_root / "prompt.md").read_text(encoding="utf-8"),
            "prompt for round 1",
        )

    def test_replaced_candidate_decision_preserves_frozen_session_bindings(self):
        first_round = _round("object_position.left_fixed", "round_1")
        plan = {
            "schema_version": 6,
            "task_name": "click_bell",
            "policy": deepcopy(self.session.target["policy"]),
            "checkpoint": deepcopy(self.session.target["checkpoint"]),
            "checkpoint_id": self.session.target["checkpoint"]["checkpoint_id"],
            "evaluation_goal": "position and instance robustness",
            "requested_aspect_ids": ["object_position", "object_instance"],
            "requested_template_ids": [
                "object_position.left_fixed",
                "object_instance.base1",
            ],
            "rounds": [first_round],
            "round_decisions": [],
            "max_rounds": 2,
            "planning_state": "awaiting_round_1_observation",
        }
        observation = _successful_observation()
        next_round = _round("object_instance.base1", "round_2")
        candidate_decision = {
            "schema_version": 1,
            "action": "continue",
            "transition": "switch_aspect",
            "observation_summary": "switch after successful position evidence",
            "decision_reason": "model selected an allowed uncovered aspect",
            "next_aspect_id": "object_instance",
            "next_template_id": "object_instance.base1",
            "next_round": deepcopy(next_round),
        }
        candidate_plan = deepcopy(plan)
        candidate_plan["rounds"].append(deepcopy(next_round))
        candidate_plan["round_decisions"].append(deepcopy(candidate_decision))
        candidate_plan["planning_state"] = "awaiting_round_2_observation"

        replacement, _ = self._apply(FakeProposalAgent(), next_round, 2)
        candidate_plan["rounds"][-1] = replacement
        candidate_decision["next_round"] = replacement
        candidate_plan["round_decisions"][-1] = candidate_decision
        updated, canonical = self.session.adjudicate(
            plan,
            [observation],
            candidate_plan=candidate_plan,
            candidate_decision=candidate_decision,
        )

        self.assertEqual(
            canonical["next_round"]["task_proposal"]["proposal_id"],
            "object_instance.base1.authored_1",
        )
        self.assertEqual(updated["task_name"], plan["task_name"])
        self.assertEqual(updated["max_rounds"], plan["max_rounds"])
        self.assertEqual(updated["checkpoint"], plan["checkpoint"])
        self.assertEqual(updated["checkpoint_id"], plan["checkpoint_id"])

        tampered_values = {
            "task_name": "beat_block_hammer",
            "max_rounds": 1,
            "checkpoint_id": "act-click_bell/other",
        }
        for field, value in tampered_values.items():
            with self.subTest(field=field):
                tampered = deepcopy(candidate_plan)
                tampered[field] = value
                with self.assertRaises(PlanSessionError):
                    self.session.adjudicate(
                        plan,
                        [observation],
                        candidate_plan=tampered,
                        candidate_decision=candidate_decision,
                    )


if __name__ == "__main__":
    unittest.main()
