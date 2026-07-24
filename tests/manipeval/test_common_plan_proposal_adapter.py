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
from mea.planner import BoundTaskPlanSession, build_act_catalog
from mea.planner.prototype import (
    EXPECTED_POLICY,
    PlanAgentPrototype,
    _validate_current_plan,
    validate_evaluation_plan,
)
from mea.proposals import (
    attach_round_proposals,
    task_proposal_from_contract,
    tool_proposal_from_contract,
)
from scripts.manipeval_agent import (
    adjudicate_bounded_transition,
    apply_bounded_round_proposal,
)


def _ready_catalog(root: Path) -> dict:
    for task_name in ("beat_block_hammer", "click_bell"):
        schema = root / "mea/toolkit/schemas" / f"{task_name}.json"
        schema.parent.mkdir(parents=True, exist_ok=True)
        schema.write_text(
            json.dumps({"task_name": task_name, "task_family": "manipulation"}),
            encoding="utf-8",
        )
        checkpoint = (
            root
            / "policy/ACT/act_ckpt"
            / f"act-{task_name}"
            / "demo_clean-50"
        )
        checkpoint.mkdir(parents=True)
        (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")
        (checkpoint / "policy_last.ckpt").write_bytes(b"weights")
    return build_act_catalog(root)


def _round(task_name: str, template_id: str, round_number: int) -> dict:
    contract = resolve_capability_contract(task_name, template_id)
    return attach_round_proposals(
        {
            "round_id": f"round_{round_number}",
            "template_id": template_id,
            "capability_id": contract["taskgen"]["capability_id"],
            "task_variant_id": contract["taskgen"]["task_variant_id"],
            "capability_contract": contract,
            "sub_aspect": contract["aspect"]["aspect_id"],
            "aspect_id": contract["aspect"]["aspect_id"],
            "rationale": "common adapter replay",
            "task_instruction": "evaluate one trusted bounded variation",
            "task_name": task_name,
            "task_module": f"mea.tasks.{task_name}",
            "route": taskgen_route(contract),
            "variant_hint": deepcopy(contract["taskgen"]["changes"]),
            "execution": {
                "backend": "act",
                "seeds": [100000 + round_number],
                "num_episodes": 1,
                "gates": list(contract["required_gates"]),
            },
            "observations": ["policy_success", "aggregate", "execution_vqa"],
            "tool_request": build_contract_tool_request(contract),
            "vqa_phenomenon_ids": list(contract["vqa"]["phenomenon_ids"]),
        }
    )


def _observation(round_plan: dict, *, success: float, pipeline_passed=True) -> dict:
    metric = round_plan["tool_request"]["metric"]
    return {
        "round_id": round_plan["round_id"],
        "pipeline_passed": pipeline_passed,
        "failure_stage": None if pipeline_passed else "execution",
        "observations": {
            "policy_success": success,
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
                                    "quality": {"valid": 1, "missing": 0, "invalid": 0}
                                },
                            }
                        ],
                    }
                ],
            },
            "planned_tool": {"episodes": []},
            "execution_vqa": {"status": "passed", "evidence_conflict": False},
        },
    }


def _plan(session: BoundTaskPlanSession, templates: list[str]) -> dict:
    first = _round(session.target["task_name"], templates[0], 1)
    return session.normalize_plan(
        {
            "schema_version": 6,
            "task_name": session.target["task_name"],
            "policy": deepcopy(session.target["policy"]),
            "checkpoint": deepcopy(session.target["checkpoint"]),
            "checkpoint_id": session.target["checkpoint"]["checkpoint_id"],
            "evaluation_goal": "common evidence-driven replay",
            "requested_template_ids": list(templates),
            "rounds": [first],
            "round_decisions": [],
            "max_rounds": 2,
            "planning_state": "awaiting_round_1_observation",
        }
    )


def _candidate(
    session: BoundTaskPlanSession,
    plan: dict,
    observation: dict,
) -> tuple[dict, dict]:
    directive = session.directive(plan, [observation])
    next_round = None
    if directive["action"] == "continue":
        next_round = _round(
            session.target["task_name"], directive["next_template_id"], 2
        )
    decision = {
        "schema_version": 1,
        "action": directive["action"],
        "observation_summary": "replayed typed evidence",
        "decision_reason": "choose one transition inside the trusted boundary",
        "next_template_id": directive["next_template_id"],
        "next_round": next_round,
    }
    candidate = deepcopy(plan)
    candidate["round_decisions"].append(deepcopy(decision))
    if next_round is None:
        candidate["planning_state"] = "stopped_after_round_1"
    else:
        candidate["rounds"].append(next_round)
        candidate["planning_state"] = "awaiting_round_2_observation"
    return candidate, decision


class FakeRegisteredProposalAgent:
    def __init__(self):
        self.last_prompt = "registered proposal prompt"
        self.last_responses = ["registered proposal response"]

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
        del user_query, aspect_id, planning_context
        self.last_prompt = f"registered proposal for {target['task_name']}"
        contract = resolve_capability_contract(target["task_name"], base_template_id)
        task = task_proposal_from_contract(
            contract, intent="model-authored but registered BBH intent"
        )
        task["proposal_id"] = f"{base_template_id}.replay"
        tool = tool_proposal_from_contract(
            contract, evaluation_goal="model-authored registered Tool assignment"
        )
        tool["proposal_id"] = f"{base_template_id}.tool_replay"
        self.last_responses = [json.dumps({"task_proposal": task})]
        self.capability_mode = capability_mode
        self.require_novel_changes = require_novel_changes
        self.require_new_tool = require_new_tool
        return {
            "schema_version": 1,
            "task_proposal": task,
            "tool_proposal": tool,
            "tool_route_preview": {"resolved_route": "registered_reuse"},
        }


class FakeDecisionProvider:
    last_metadata = {"provider": "fake"}

    def text(self, _prompt, **_kwargs):
        return json.dumps(
            {
                "schema_version": 2,
                "action": "continue",
                "observation_summary": "first BBH round has sufficient evidence",
                "decision_reason": "continue to the remaining requested aspect",
                "next_template_id": "object_position.official_random",
            }
        )


class CommonPlanProposalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.catalog = _ready_catalog(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def _replay(self, task_name, templates, observation):
        session = BoundTaskPlanSession.from_catalog(
            self.catalog, task_name, max_rounds=2
        )
        plan = _plan(session, templates)
        candidate, decision = _candidate(session, plan, observation(plan["rounds"][0]))
        updated, canonical, directive = adjudicate_bounded_transition(
            plan_session=session,
            user_query="How robust is this ACT checkpoint?",
            observation_history=[observation(plan["rounds"][0])],
            current_plan=plan,
            candidate_plan=candidate,
            candidate_decision=decision,
            proposal_mode="catalog",
            proposal_agent=None,
            planning_context=None,
            evaluation_dir=self.root / "mea/evaluation_runs/eval_replay",
        )
        return session, updated, canonical, directive

    def test_click_failure_drills_down_but_success_switches_aspect(self):
        templates = [
            "object_position.left_fixed",
            "object_position.right_fixed",
            "object_instance.base0",
        ]
        _, _, failed, _ = self._replay(
            "click_bell",
            templates,
            lambda round_plan: _observation(round_plan, success=0.0),
        )
        _, _, succeeded, _ = self._replay(
            "click_bell",
            templates,
            lambda round_plan: _observation(round_plan, success=1.0),
        )
        self.assertEqual(failed["transition"], "drill_down")
        self.assertEqual(failed["next_template_id"], "object_position.right_fixed")
        self.assertEqual(succeeded["transition"], "switch_aspect")
        self.assertEqual(succeeded["next_template_id"], "object_instance.base0")

    def test_bbh_success_switches_aspect_but_pipeline_failure_stops(self):
        templates = [
            "object_appearance.color_blue",
            "object_position.official_random",
        ]
        _, _, succeeded, _ = self._replay(
            "beat_block_hammer",
            templates,
            lambda round_plan: _observation(round_plan, success=1.0),
        )
        _, stopped, failed, directive = self._replay(
            "beat_block_hammer",
            templates,
            lambda round_plan: _observation(
                round_plan, success=0.0, pipeline_passed=False
            ),
        )
        self.assertEqual(succeeded["transition"], "switch_aspect")
        self.assertEqual(
            succeeded["next_template_id"], "object_position.official_random"
        )
        self.assertEqual(failed["transition"], "stop")
        self.assertEqual(directive["action"], "stop")
        self.assertEqual(len(stopped["rounds"]), 1)

    def test_bbh_registered_proposal_round_remains_valid_for_legacy_adapter(self):
        session = BoundTaskPlanSession.from_catalog(
            self.catalog, "beat_block_hammer", max_rounds=3
        )
        plan = session.normalize_plan(
            validate_evaluation_plan(
                {
                    "schema_version": 5,
                    "task_name": "beat_block_hammer",
                    "policy": deepcopy(EXPECTED_POLICY),
                    "evaluation_goal": "proposal compatibility replay",
                    "requested_template_ids": [
                        "object_appearance.color_blue",
                        "object_position.official_random",
                    ],
                    "first_template_id": "object_appearance.color_blue",
                    "max_rounds": 3,
                }
            )
        )
        agent = FakeRegisteredProposalAgent()
        proposed, artifact = apply_bounded_round_proposal(
            proposal_agent=agent,
            user_query="How robust is ACT?",
            target=session.target,
            planning_context={"schema_version": 1},
            round_plan=plan["rounds"][0],
            evaluation_dir=self.root / "mea/evaluation_runs/eval_bbh_proposal",
            round_number=1,
        )
        plan["rounds"][0] = proposed
        self.assertIs(_validate_current_plan(plan), plan)
        self.assertEqual(artifact["proposal_capability_mode"], "registered_reuse")
        self.assertFalse(agent.require_novel_changes)

    def test_bbh_legacy_planner_and_common_adjudicator_share_proposed_rounds(self):
        session = BoundTaskPlanSession.from_catalog(
            self.catalog, "beat_block_hammer", max_rounds=3
        )
        raw_proposal = {
            "schema_version": 5,
            "task_name": "beat_block_hammer",
            "policy": deepcopy(EXPECTED_POLICY),
            "evaluation_goal": "cross-aspect bounded replay",
            "requested_template_ids": [
                "object_appearance.color_blue",
                "object_position.official_random",
            ],
            "first_template_id": "object_appearance.color_blue",
            "max_rounds": 3,
        }
        evaluation_id = "eval_bbh_common_adapter"
        planner = PlanAgentPrototype(
            self.root, FakeDecisionProvider(), model="fake-model"
        )
        manifest = planner.plan(
            "How robust is this ACT checkpoint?",
            evaluation_id=evaluation_id,
            validated_proposal=raw_proposal,
        )
        plan = session.normalize_plan(manifest["plan"])
        proposal_agent = FakeRegisteredProposalAgent()
        plan["rounds"][0], _ = apply_bounded_round_proposal(
            proposal_agent=proposal_agent,
            user_query="How robust is this ACT checkpoint?",
            target=session.target,
            planning_context={"schema_version": 1},
            round_plan=plan["rounds"][0],
            evaluation_dir=self.root / "mea/evaluation_runs" / evaluation_id,
            round_number=1,
        )
        observation = _observation(plan["rounds"][0], success=1.0)
        candidate, decision = planner.decide_next_round(
            evaluation_id=evaluation_id,
            user_request="How robust is this ACT checkpoint?",
            current_plan=plan,
            observation_history=[observation],
        )
        updated, canonical, _ = adjudicate_bounded_transition(
            plan_session=session,
            user_query="How robust is this ACT checkpoint?",
            observation_history=[observation],
            current_plan=plan,
            candidate_plan=candidate,
            candidate_decision=decision,
            proposal_mode="bounded_each_round",
            proposal_agent=proposal_agent,
            planning_context={"schema_version": 1},
            evaluation_dir=self.root / "mea/evaluation_runs" / evaluation_id,
        )
        self.assertEqual(canonical["transition"], "switch_aspect")
        self.assertEqual(updated["rounds"][-1]["template_id"], "object_position.official_random")
        self.assertEqual(
            updated["rounds"][-1]["task_proposal"]["intent"],
            "model-authored but registered BBH intent",
        )
        self.assertTrue(
            (
                self.root
                / "mea/evaluation_runs"
                / evaluation_id
                / "plan/bounded_proposal/round_02/proposal_bundle.json"
            ).is_file()
        )


if __name__ == "__main__":
    unittest.main()
