import unittest

from mea.planner.runtime_limits import (
    PlanRuntimeError,
    build_plan_runtime_limits,
    summarize_plan_evidence,
    validate_agent_stop,
)


def evidence(candidate_id, outcome):
    return {"candidate_id": candidate_id, "outcome": outcome}


class PlanAgentRuntimeLimitsTests(unittest.TestCase):
    def test_limits_contain_only_budget_and_control_choice(self):
        limits = build_plan_runtime_limits(
            "How robust is this policy?",
            round_budget=3,
            control_requirement="not_required",
        )

        self.assertEqual(
            limits,
            {
                "schema_version": 4,
                "round_budget": 3,
                "control_requirement": "not_required",
            },
        )

    def test_evidence_does_not_precompute_semantic_sufficiency(self):
        assessment = summarize_plan_evidence(
            build_plan_runtime_limits("Broad Query", round_budget=3),
            [evidence("candidate.a", "pass")],
        )

        self.assertFalse(assessment["should_stop"])
        self.assertFalse(assessment["evidence_sufficient"])
        self.assertEqual(assessment["claim_verdict"], "inconclusive")
        self.assertEqual(assessment["decisive_candidate_ids"], ["candidate.a"])

    def test_conflict_can_continue_until_the_external_cap(self):
        limits = build_plan_runtime_limits("Broad Query", round_budget=1)
        capped = summarize_plan_evidence(
            limits,
            [evidence("candidate.a", "unknown")],
        )
        conflicted = summarize_plan_evidence(
            build_plan_runtime_limits("Broad Query", round_budget=3),
            [evidence("candidate.a", "conflict")],
        )

        self.assertTrue(capped["should_stop"])
        self.assertEqual(capped["stop_reason"], "budget_exhausted")
        self.assertFalse(conflicted["should_stop"])
        self.assertEqual(conflicted["stop_reason"], "continue")
        self.assertEqual(
            conflicted["conflict_candidate_ids"], ["candidate.a"]
        )

    def test_agent_can_answer_from_decisive_completed_evidence(self):
        assessment = summarize_plan_evidence(
            build_plan_runtime_limits("Broad Query", round_budget=3),
            [evidence("candidate.a", "fail")],
        )

        stopped = validate_agent_stop(
            assessment,
            rationale="The completed failure directly answers the bounded Query.",
            answer="The policy fails for the tested bounded variation.",
            claim_verdict="supported",
            evidence_sufficient=True,
        )

        self.assertTrue(stopped["should_stop"])
        self.assertTrue(stopped["evidence_sufficient"])
        self.assertEqual(stopped["stop_reason"], "agent_stop")

    def test_answered_stop_rejects_missing_or_conflicting_evidence(self):
        no_evidence = summarize_plan_evidence(
            build_plan_runtime_limits("Broad Query", round_budget=3),
            [],
        )
        conflict = summarize_plan_evidence(
            build_plan_runtime_limits("Broad Query", round_budget=3),
            [evidence("candidate.a", "conflict")],
        )

        for assessment in (no_evidence, conflict):
            with self.assertRaises(PlanRuntimeError):
                validate_agent_stop(
                    assessment,
                    rationale="Stop.",
                    answer="Answered.",
                    claim_verdict="supported",
                    evidence_sufficient=True,
                )

    def test_agent_may_stop_inconclusively(self):
        assessment = summarize_plan_evidence(
            build_plan_runtime_limits("Broad Query", round_budget=3),
            [evidence("candidate.a", "unknown")],
        )

        stopped = validate_agent_stop(
            assessment,
            rationale="The available evidence is inconclusive.",
            answer=None,
            claim_verdict="inconclusive",
            evidence_sufficient=False,
        )

        self.assertTrue(stopped["should_stop"])
        self.assertFalse(stopped["evidence_sufficient"])


if __name__ == "__main__":
    unittest.main()
