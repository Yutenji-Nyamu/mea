from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path

from mea.feedback import (
    AnswerScopeError,
    FeedbackAgent,
    build_answer_scope,
    project_answer_scope,
    validate_answer_scope_projection,
)


BASE_FEEDBACK = {
    "answer": "The tested candidate failed.",
    "evaluation_scope": "A bounded two-candidate evaluation.",
    "findings": ["The numeric rule recorded a failure."],
    "limitations": ["This is bounded evidence."],
    "recommended_next_step": "Test the remaining candidate.",
}


class Provider:
    last_metadata = {"model": "fake"}

    def text(self, prompt, **kwargs):
        return json.dumps(BASE_FEEDBACK)


def evidence(stop_reason="budget_exhausted"):
    return {
        "total_episodes": 1,
        "rounds": [
            {
                "seeds": [1001],
                "num_episodes": 1,
                "round_plan": {"template_id": "position.left"},
                "execution_vqa": {"evidence_conflict": True},
            }
        ],
        "observations": {
            "pipeline_passed": True,
            "execution_vqa_conflict": True,
            "policy_success": 0.0,
        },
        "plan": {
            "completed_template_ids": ["position.left"],
            "remaining_template_ids": ["position.right"],
        },
        "global_query_route": {
            "selection": {
                "unsupported_capabilities": [
                    {"task_name": "click_bell", "aspect_id": "object_mass"}
                ]
            }
        },
        "query_sufficiency": {
            "stop_reason": stop_reason,
            "should_stop": True,
            "evidence_sufficient": stop_reason == "evidence_sufficient",
            "claim_verdict": (
                "inconclusive"
                if stop_reason != "evidence_sufficient"
                else "refuted"
            ),
            "observed_candidate_ids": ["position.left"],
            "untested_candidate_ids": ["position.right"],
        },
    }


class AnswerScopeTests(unittest.TestCase):
    def test_projects_all_evidence_required_limitations(self):
        scope = build_answer_scope(evidence())
        self.assertEqual(scope["sample_count"], 1)
        self.assertEqual(scope["seeds"], [1001])
        self.assertEqual(scope["tested_candidate_ids"], ["position.left"])
        self.assertEqual(scope["untested_candidate_ids"], ["position.right"])
        self.assertEqual(
            scope["unsupported_capabilities"], ["click_bell:object_mass"]
        )
        self.assertTrue(scope["evidence_conflict"])
        self.assertEqual(scope["termination"], "budget_exhausted")
        codes = [item["code"] for item in scope["required_limitations"]]
        self.assertEqual(
            codes,
            [
                "sample_count_and_seeds",
                "untested_candidates",
                "unsupported_capabilities",
                "evidence_conflict",
                "termination_budget_exhausted",
            ],
        )
        projected = project_answer_scope(BASE_FEEDBACK, scope)
        validate_answer_scope_projection(projected, scope)
        for limitation in scope["required_limitations"]:
            self.assertIn(limitation["text"], projected["limitations"])

    def test_evidence_sufficient_and_budget_exhausted_are_distinct(self):
        sufficient = build_answer_scope(evidence("evidence_sufficient"))
        exhausted = build_answer_scope(evidence("budget_exhausted"))
        self.assertEqual(sufficient["termination"], "evidence_sufficient")
        self.assertEqual(exhausted["termination"], "budget_exhausted")
        self.assertNotEqual(
            sufficient["required_limitations"][-1]["text"],
            exhausted["required_limitations"][-1]["text"],
        )
        self.assertIn(
            "not a statistical generalization guarantee",
            sufficient["required_limitations"][-1]["text"],
        )
        self.assertIn(
            "before the query-sufficiency contract was satisfied",
            exhausted["required_limitations"][-1]["text"],
        )

    def test_claim_first_runtime_assessment_reaches_final_scope(self):
        value = {
            "total_episodes": 2,
            "rounds": [
                {
                    "seeds": [102000],
                    "num_episodes": 1,
                    "round_plan": {
                        "template_id": (
                            "performance.completion_time_stability.official"
                        )
                    },
                },
                {
                    "seeds": [102000],
                    "num_episodes": 1,
                    "round_plan": {
                        "template_id": "object_instance.base0"
                    },
                },
            ],
            "claim_first_runtime": {
                "assessment": {
                    "stop_reason": "evidence_sufficient",
                    "should_stop": True,
                    "evidence_sufficient": True,
                    "claim_verdict": "supported",
                    "observed_candidate_ids": ["object_instance.base0"],
                    "untested_candidate_ids": [
                        "object_position.left_fixed",
                        "object_position.right_fixed",
                        "object_instance.base1",
                    ],
                    "conflict_candidate_ids": [],
                }
            },
        }

        scope = build_answer_scope(value)

        self.assertEqual(scope["termination"], "evidence_sufficient")
        self.assertEqual(scope["claim_verdict"], "supported")
        self.assertEqual(
            scope["tested_candidate_ids"], ["object_instance.base0"]
        )
        self.assertEqual(
            scope["untested_candidate_ids"],
            [
                "object_position.left_fixed",
                "object_position.right_fixed",
                "object_instance.base1",
            ],
        )

    def test_adversarial_omissions_fail_closed(self):
        scope = build_answer_scope(evidence())
        projected = project_answer_scope(BASE_FEEDBACK, scope)

        missing_text = deepcopy(projected)
        missing_text["limitations"].remove(
            scope["required_limitations"][0]["text"]
        )
        with self.assertRaisesRegex(
            AnswerScopeError, "omitted evidence-required"
        ):
            validate_answer_scope_projection(missing_text, scope)

        missing_code = deepcopy(projected)
        missing_code["limitation_codes"].pop()
        with self.assertRaisesRegex(AnswerScopeError, "limitation_codes"):
            validate_answer_scope_projection(missing_code, scope)

        altered_scope = deepcopy(projected)
        altered_scope["answer_scope"]["termination"] = "evidence_sufficient"
        with self.assertRaisesRegex(
            AnswerScopeError, "required_limitations|differs from evidence"
        ):
            validate_answer_scope_projection(altered_scope, scope)

        no_scope = deepcopy(projected)
        del no_scope["answer_scope"]
        with self.assertRaisesRegex(AnswerScopeError, "missing structured"):
            validate_answer_scope_projection(no_scope)

        false_sufficiency = deepcopy(projected)
        false_sufficiency["answer"] = (
            "The evidence is sufficient to establish generalization."
        )
        with self.assertRaisesRegex(AnswerScopeError, "contradicts"):
            validate_answer_scope_projection(false_sufficiency, scope)

        false_coverage = deepcopy(projected)
        false_coverage["findings"] = ["All variants were tested."]
        with self.assertRaisesRegex(AnswerScopeError, "complete testing"):
            validate_answer_scope_projection(false_coverage, scope)

        false_consensus = deepcopy(projected)
        false_consensus["findings"] = ["The evidence sources agree."]
        with self.assertRaisesRegex(AnswerScopeError, "recorded evidence conflict"):
            validate_answer_scope_projection(false_consensus, scope)

        chinese_false_sufficiency = deepcopy(projected)
        chinese_false_sufficiency["answer"] = "这些证据已经充分，足以证明广泛泛化。"
        with self.assertRaisesRegex(AnswerScopeError, "contradicts"):
            validate_answer_scope_projection(chinese_false_sufficiency, scope)

        chinese_false_coverage = deepcopy(projected)
        chinese_false_coverage["findings"] = ["所有候选条件均已测试完。"]
        with self.assertRaisesRegex(AnswerScopeError, "complete testing"):
            validate_answer_scope_projection(chinese_false_coverage, scope)

    def test_feedback_agent_attaches_scope_deterministically(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            feedback = FeedbackAgent(
                repo_root, Provider(), model="fake"
            ).generate(evidence(), output_dir=Path(temp))
        self.assertEqual(
            feedback["answer_scope"]["termination"], "budget_exhausted"
        )
        self.assertIn(
            "termination_budget_exhausted", feedback["limitation_codes"]
        )
        validate_answer_scope_projection(
            feedback, build_answer_scope(evidence())
        )

    def test_unknown_scope_is_explicit_for_legacy_evidence(self):
        scope = build_answer_scope({"observations": {"pipeline_passed": True}})
        self.assertIsNone(scope["sample_count"])
        self.assertEqual(scope["seeds"], [])
        self.assertEqual(scope["termination"], "unknown")
        self.assertEqual(
            [item["code"] for item in scope["required_limitations"]],
            ["sample_count_and_seeds", "termination_unknown"],
        )

    def test_legacy_hard_cap_is_never_called_evidence_sufficient(self):
        scope = build_answer_scope(
            {
                "observations": {"pipeline_passed": True},
                "plan": {
                    "planning_state": "stopped_after_round_2",
                    "round_budget_remaining": 0,
                },
            }
        )
        self.assertEqual(scope["termination"], "budget_exhausted")
        self.assertIsNone(scope["claim_verdict"])

    def test_policy_execution_count_precedes_cross_cohort_aggregate_count(self):
        scope = build_answer_scope(
            {
                "seed": 1001,
                "num_episodes": 1,
                "observations": {
                    "pipeline_passed": True,
                    "aggregate": {"unique_episode_count": 3},
                },
            }
        )
        self.assertEqual(scope["sample_count"], 1)

    def test_pipeline_invalid_precedes_stale_sufficiency_assessment(self):
        value = evidence("evidence_sufficient")
        value["observations"]["pipeline_passed"] = False
        scope = build_answer_scope(value)
        self.assertEqual(scope["termination"], "pipeline_invalid")
        self.assertIsNone(scope["claim_verdict"])

    def test_query_candidate_conflict_is_projected_without_vqa_flag(self):
        value = evidence()
        value["observations"]["execution_vqa_conflict"] = False
        value["rounds"][0]["execution_vqa"]["evidence_conflict"] = False
        value["query_sufficiency"]["conflict_candidate_ids"] = [
            "position.left"
        ]
        scope = build_answer_scope(value)
        self.assertTrue(scope["evidence_conflict"])
        self.assertIn(
            "evidence_conflict",
            [item["code"] for item in scope["required_limitations"]],
        )


if __name__ == "__main__":
    unittest.main()
