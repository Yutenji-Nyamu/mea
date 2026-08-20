import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from mea.feedback import (
    AnswerScopeError,
    build_answer_scope,
    build_scoped_plan_agent_answer,
)
from mea.plan_agent_finalization import PlanAgentFinalizationMixin


def round_evidence(
    candidate_id: str,
    seed: int,
    *,
    success_rate: float = 1.0,
    vqa_conflict: bool = False,
) -> dict:
    return {
        "schema_version": 1,
        "round_id": f"round_{seed}",
        "candidate_id": candidate_id,
        "planning_observation": None,
        "policy": {
            "success_rate": success_rate,
            "metric": "official_check_success",
            "authority": "official_check_success",
            "official_equivalent": True,
            "execution_scope": "official_check_success",
            "seeds": [seed],
        },
        "rule": {
            "requested": False,
            "status": None,
            "metric": None,
            "route": None,
            "source": None,
            "results": [],
        },
        "vqa": {
            "required": vqa_conflict,
            "status": "passed" if vqa_conflict else "missing",
            "evidence_conflict": vqa_conflict,
            "observation": None,
        },
        "outcome_semantics": {
            "status": "official_only",
            "evidence_conflict": False,
        },
        "scene_change": None,
    }


def evidence(stop_reason: str = "agent_stop") -> dict:
    should_stop = stop_reason != "continue"
    return {
        "schema_version": 3,
        "evaluation_id": "eval_scope",
        "query": "Where is the bounded weakness?",
        "plan": {
            "max_rounds": 3,
            "executed_rounds": 2,
            "planning_state": "stopped_after_round_2",
            "round_budget_remaining": 1,
        },
        "rounds": [
            round_evidence("candidate.a", 7),
            round_evidence(
                "candidate.b",
                8,
                success_rate=0.0,
                vqa_conflict=True,
            ),
        ],
        "total_policy_episodes": 2,
        "plan_agent_session": {
            "assessment": {
                "should_stop": should_stop,
                "stop_reason": stop_reason,
                "claim_verdict": "inconclusive",
                "evidence_sufficient": False,
                "observed_candidate_ids": ["candidate.a", "candidate.b"],
                "untested_candidate_ids": ["candidate.c"],
                "conflict_candidate_ids": ["candidate.b"],
            }
        },
    }


class AnswerScopeTests(unittest.TestCase):
    def test_scope_reads_policy_seeds_vqa_and_plan_assessment(self):
        scope = build_answer_scope(evidence())

        self.assertEqual(scope["sample_count"], 2)
        self.assertEqual(scope["seeds"], [7, 8])
        self.assertEqual(
            scope["tested_candidate_ids"],
            ["candidate.a", "candidate.b"],
        )
        self.assertEqual(scope["untested_candidate_ids"], ["candidate.c"])
        self.assertTrue(scope["evidence_conflict"])
        self.assertEqual(scope["termination"], "agent_stop")
        self.assertNotIn("unsupported_capabilities", scope)

    def test_scope_does_not_count_n_zero_planning_round(self):
        value = evidence("continue")
        value["rounds"].append(
            {
                **round_evidence("candidate.rejected", 9),
                "round_id": "round_3",
                "planning_observation": {
                    "kind": "candidate_unexecutable",
                    "policy_rollouts_started": 0,
                    "policy_sample_count": 0,
                },
                "policy": {
                    "success_rate": None,
                    "metric": None,
                    "authority": None,
                    "official_equivalent": None,
                    "execution_scope": "not_executed",
                    "seeds": [],
                },
            }
        )
        value["plan"]["executed_rounds"] = 3

        scope = build_answer_scope(value)

        self.assertEqual(scope["sample_count"], 2)
        self.assertEqual(scope["seeds"], [7, 8])
        self.assertEqual(scope["termination"], "continue")

    def test_episode_count_remains_distinct_from_unique_seeds(self):
        value = evidence()
        value["rounds"][1]["policy"]["seeds"] = [7]

        scope = build_answer_scope(value)

        self.assertEqual(scope["sample_count"], 2)
        self.assertEqual(scope["seeds"], [7])

    def test_scoped_answer_keeps_plan_text_and_structured_scope(self):
        value = evidence()
        query_answer = {
            "answer": "The bounded evidence remains inconclusive.",
            "claim_verdict": "inconclusive",
            "tested_candidate_ids": ["candidate.a", "candidate.b"],
            "untested_candidate_ids": ["candidate.c"],
            "evaluation_outcomes": [],
            "limitations": ["The visual conflict remains unresolved."],
        }

        feedback = build_scoped_plan_agent_answer(value, query_answer)

        self.assertEqual(feedback["answer_scope"], build_answer_scope(value))
        self.assertEqual(feedback["answer"], query_answer["answer"])
        self.assertEqual(feedback["limitations"], query_answer["limitations"])
        self.assertTrue(feedback["answer_scope"]["evidence_conflict"])
        self.assertEqual(
            set(feedback),
            {
                "answer",
                "evaluation_scope",
                "findings",
                "limitations",
                "recommended_next_step",
                "answer_scope",
            },
        )
        self.assertNotIn(
            "required_limitations", feedback["answer_scope"]
        )

    def test_final_projection_never_rewrites_agent_answer(self):
        value = evidence()
        for item in value["rounds"]:
            item["policy"]["success_rate"] = 0.0
        query_answer = {
            "answer": "证据反驳了‘策略执行成功’这一命题。",
            "claim_verdict": "inconclusive",
            "tested_candidate_ids": ["candidate.a", "candidate.b"],
            "untested_candidate_ids": ["candidate.c"],
            "evaluation_outcomes": [],
            "limitations": ["The claim wording is quoted, not asserted."],
        }

        feedback = build_scoped_plan_agent_answer(value, query_answer)

        self.assertEqual(feedback["answer"], query_answer["answer"])
        self.assertEqual(feedback["limitations"], query_answer["limitations"])

    def test_hard_cap_uses_one_budget_exhausted_assessment_everywhere(self):
        bundle = evidence("continue")
        bundle["rounds"] = [round_evidence("candidate.a", 7)]
        bundle["total_policy_episodes"] = 1
        bundle["plan"] = {
            "max_rounds": 1,
            "executed_rounds": 1,
            "planning_state": "stopped_after_round_1_by_hard_cap",
            "round_budget_remaining": 0,
        }
        bundle.pop("plan_agent_session")
        round_runs = [
            {
                "round_summary": {
                    "observations": {"planning_observation": None},
                },
                "child_manifest": {"run_id": "child_1"},
            }
        ]
        runtime_state = {
            "runtime_limits": {"round_budget": 1},
            "assessment": {
                "should_stop": False,
                "stop_reason": "continue",
                "evidence_sufficient": False,
                "claim_verdict": "inconclusive",
                "rationale": "The Agent requested another round.",
                "observed_candidate_ids": ["candidate.a"],
                "untested_candidate_ids": ["candidate.b"],
                "conflict_candidate_ids": [],
                "limitations": [],
            },
            "records": [],
            "control_passed": True,
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_dir = (
                root / "mea" / "evaluation_runs" / "eval_hard_cap"
            )
            evaluation_dir.mkdir(parents=True)
            (evaluation_dir / "manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            finalizer = PlanAgentFinalizationMixin()
            finalizer.repo_root = root
            finalizer.evaluation_dir = evaluation_dir
            finalizer.evaluation_id = "eval_hard_cap"
            finalizer.user_request = "Where is the bounded weakness?"
            finalizer.history_disabled = True
            finalizer.history_database = None
            finalizer.history_retrieval = {}
            finalizer.history_context_count = 0
            with (
                patch(
                    "mea.plan_agent_finalization."
                    "aggregate_evaluation_results",
                    return_value=None,
                ),
                patch(
                    "mea.plan_agent_finalization.build_evidence_bundle",
                    return_value=deepcopy(bundle),
                ),
                patch(
                    "mea.plan_agent_finalization.write_evidence_report",
                    return_value={},
                ),
            ):
                result = finalizer._finalize(
                    plan={
                        "max_rounds": 1,
                        "planning_state": (
                            "stopped_after_round_1_by_hard_cap"
                        ),
                    },
                    round_runs=round_runs,
                    runtime_state=runtime_state,
                    query_answer=None,
                    executed_rounds=1,
                )

            query_answer = json.loads(
                (
                    evaluation_dir
                    / "plan/plan_agent_session/query_answer.json"
                ).read_text(encoding="utf-8")
            )
            persisted_evidence = json.loads(
                (
                    evaluation_dir / "summary/evidence_bundle.json"
                ).read_text(encoding="utf-8")
            )
            answer = json.loads(
                (evaluation_dir / "answer/answer.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (evaluation_dir / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(query_answer["stop_reason"], "budget_exhausted")
        self.assertEqual(
            persisted_evidence["plan_agent_session"]["assessment"][
                "stop_reason"
            ],
            "budget_exhausted",
        )
        self.assertEqual(
            answer["answer_scope"]["termination"], "budget_exhausted"
        )
        self.assertEqual(
            {
                query_answer["answer"],
                answer["answer"],
                manifest["answer"]["answer"],
                result["answer"]["answer"],
            },
            {query_answer["answer"]},
        )

    def test_scope_rejects_inconsistent_assessment(self):
        value = deepcopy(evidence())
        value["plan_agent_session"]["assessment"]["evidence_sufficient"] = True

        with self.assertRaisesRegex(AnswerScopeError, "inconsistent"):
            build_answer_scope(value)


if __name__ == "__main__":
    unittest.main()
