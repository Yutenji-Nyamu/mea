from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path

from mea.feedback import (
    AnswerScopeError,
    PlanAgentFinalSummary,
    build_scoped_plan_agent_answer,
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
    decisive_agent_stop = stop_reason in {
        "agent_stop",
        # Historical fixture spelling retained for the reader-compatibility
        # test below; current evidence writes ``agent_stop``.
        "evidence_sufficient",
    }
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
        "plan_agent_session": {
            "assessment": {
                "stop_reason": stop_reason,
                "should_stop": True,
                "evidence_sufficient": decisive_agent_stop,
                "claim_verdict": (
                    "inconclusive"
                    if not decisive_agent_stop
                    else "refuted"
                ),
                "observed_candidate_ids": ["position.left"],
                "untested_candidate_ids": ["position.right"],
            }
        },
    }


class AnswerScopeTests(unittest.TestCase):
    def test_final_summary_prompt_does_not_invent_stricter_statistics(self):
        prompt = (
            Path(__file__).resolve().parents[2]
            / "mea/feedback/README.Agent.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "Do not upgrade “one",
            prompt,
        )
        self.assertIn(
            "scalar computed from rollout telemetry” into a missing trajectory",
            prompt,
        )
        self.assertIn(
            "explicitly asks for a peak",
            prompt,
        )

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

    def test_agent_stop_and_budget_exhausted_are_distinct(self):
        sufficient = build_answer_scope(evidence("agent_stop"))
        exhausted = build_answer_scope(evidence("budget_exhausted"))
        self.assertEqual(sufficient["termination"], "agent_stop")
        self.assertEqual(exhausted["termination"], "budget_exhausted")
        self.assertNotEqual(
            sufficient["required_limitations"][-1]["text"],
            exhausted["required_limitations"][-1]["text"],
        )
        self.assertIn(
            "limited to the recorded task",
            sufficient["required_limitations"][-1]["text"],
        )
        self.assertIn(
            "round budget was exhausted",
            exhausted["required_limitations"][-1]["text"],
        )

    def test_historical_evidence_sufficient_stop_reads_as_agent_stop(self):
        scope = build_answer_scope(evidence("evidence_sufficient"))

        self.assertEqual(scope["termination"], "agent_stop")
        self.assertEqual(scope["claim_verdict"], "refuted")

    def test_control_stop_is_inconclusive_but_valid_answer_scope(self):
        scope = build_answer_scope(evidence("control_baseline_policy_failed"))
        self.assertEqual(scope["termination"], "control_not_passed")
        self.assertEqual(scope["claim_verdict"], "inconclusive")
        self.assertIn(
            "no property attribution",
            scope["required_limitations"][-1]["text"],
        )

    def test_agent_inconclusive_stop_keeps_untested_work_explicit(self):
        value = evidence("agent_inconclusive_stop")
        scope = build_answer_scope(value)

        self.assertEqual(scope["termination"], "agent_inconclusive_stop")
        self.assertEqual(scope["claim_verdict"], "inconclusive")
        self.assertEqual(scope["untested_candidate_ids"], ["position.right"])
        self.assertTrue(scope["required_limitations"][-1]["text"])

    def test_plan_agent_session_assessment_reaches_final_scope(self):
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
            "plan_agent_session": {
                "assessment": {
                    "stop_reason": "agent_stop",
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

        self.assertEqual(scope["termination"], "agent_stop")
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
        altered_scope["answer_scope"]["termination"] = "agent_stop"
        with self.assertRaisesRegex(
            AnswerScopeError, "required_limitations|differs from evidence"
        ):
            validate_answer_scope_projection(altered_scope, scope)

        no_scope = deepcopy(projected)
        del no_scope["answer_scope"]
        with self.assertRaisesRegex(AnswerScopeError, "missing structured"):
            validate_answer_scope_projection(no_scope)

    def test_plan_agent_summary_attaches_scope_deterministically(self):
        repo_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temp:
            feedback = PlanAgentFinalSummary(
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

    def test_session_query_answer_uses_the_same_scope_without_provider(self):
        query_answer = {
            "answer": (
                "The bounded evidence does not yet satisfy the truth conditions "
                "needed to answer the original Query."
            ),
            "claim_verdict": "inconclusive",
            "tested_candidate_ids": ["position.left"],
            "untested_candidate_ids": ["position.right"],
            "limitations": ["This is a bounded Plan Agent session answer."],
            "evaluation_outcomes": [{"authority": "official_check_success"}],
        }

        feedback = build_scoped_plan_agent_answer(evidence(), query_answer)

        self.assertEqual(feedback["answer"], query_answer["answer"])
        self.assertEqual(feedback["answer_scope"]["sample_count"], 1)
        self.assertEqual(
            feedback["answer_scope"]["termination"], "budget_exhausted"
        )
        self.assertEqual(feedback["provider_metadata"]["called"], False)
        self.assertEqual(feedback["consistency_validation"]["attempts_used"], 0)
        validate_answer_scope_projection(
            feedback, build_answer_scope(evidence())
        )

    def test_decisive_agent_stop_recommends_replication_without_provider(self):
        value = evidence("agent_stop")
        value["plan"]["remaining_template_ids"] = []
        value["rounds"][0]["execution_vqa"]["evidence_conflict"] = False
        value["observations"]["execution_vqa_conflict"] = False
        value["plan_agent_session"]["assessment"]["untested_candidate_ids"] = []
        value["plan_agent_session"]["assessment"]["conflict_candidate_ids"] = []
        query_answer = {
            "answer": "The bounded candidate is supported.",
            "claim_verdict": "supported",
            "tested_candidate_ids": ["position.left"],
            "untested_candidate_ids": [],
            "limitations": ["This is one bounded evaluation."],
            "evaluation_outcomes": [{"authority": "official_check_success"}],
        }

        feedback = build_scoped_plan_agent_answer(value, query_answer)

        self.assertEqual(feedback["answer_scope"]["termination"], "agent_stop")
        self.assertIn("additional seeds", feedback["recommended_next_step"])
        self.assertFalse(feedback["provider_metadata"]["called"])

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
        value = evidence("agent_stop")
        value["observations"]["pipeline_passed"] = False
        scope = build_answer_scope(value)
        self.assertEqual(scope["termination"], "pipeline_invalid")
        self.assertIsNone(scope["claim_verdict"])

    def test_query_candidate_conflict_is_projected_without_vqa_flag(self):
        value = evidence()
        value["observations"]["execution_vqa_conflict"] = False
        value["rounds"][0]["execution_vqa"]["evidence_conflict"] = False
        value["plan_agent_session"]["assessment"]["conflict_candidate_ids"] = [
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
