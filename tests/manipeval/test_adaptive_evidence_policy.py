import unittest

from mea.planner.evidence_policy import (
    assess_conditional_transition,
    assess_evidence,
)


def round_plan(
    *,
    round_id="round_1",
    template_id="object_appearance.color_blue",
    metric="hammer_block_contact_ever",
    episodes=3,
    verification_of=None,
):
    value = {
        "round_id": round_id,
        "template_id": template_id,
        "execution": {"num_episodes": episodes, "seeds": list(range(episodes))},
        "tool_request": {"metric": metric},
    }
    if verification_of:
        value["verification_of"] = verification_of
    return value


def aggregate(metric, *, valid, missing=0, invalid=0, issues=None):
    return {
        "status": "passed" if not issues else "passed_with_input_issues",
        "input_issues": list(issues or []),
        "metrics": [
            {
                "metric": metric,
                "cohorts": [
                    {
                        "role": "policy_under_evaluation",
                        "summary": {
                            "quality": {
                                "valid": valid,
                                "missing": missing,
                                "invalid": invalid,
                            }
                        },
                    }
                ],
            }
        ],
    }


def observation(
    plan,
    *,
    valid=3,
    missing=0,
    invalid=0,
    conflict=False,
    pipeline_passed=True,
    planned_episodes=None,
    issues=None,
):
    metric = plan["tool_request"]["metric"]
    return {
        "round_id": plan["round_id"],
        "pipeline_passed": pipeline_passed,
        "observations": {
            "aggregate": aggregate(
                metric,
                valid=valid,
                missing=missing,
                invalid=invalid,
                issues=issues,
            ),
            "planned_tool": {
                "route_decision": {"metric": metric},
                "episodes": list(planned_episodes or []),
            },
            "execution_vqa": {"evidence_conflict": conflict},
        },
    }


def current_plan(rounds, requested=None, max_rounds=3):
    return {
        "rounds": rounds,
        "requested_template_ids": requested
        or ["object_appearance.color_blue"],
        "max_rounds": max_rounds,
    }


class AdaptiveEvidencePolicyTests(unittest.TestCase):
    def test_clear_negative_boolean_evidence_is_sufficient(self):
        planned = round_plan()
        result = assess_evidence(
            current_plan([planned]),
            [observation(planned, valid=3)],
        )
        self.assertEqual(result["state"], "sufficient")
        self.assertEqual(result["required_action"], "stop")
        self.assertFalse(result["unresolved"])

    def test_visual_numeric_conflict_requires_same_aspect_verification(self):
        planned = round_plan()
        result = assess_evidence(
            current_plan([planned]),
            [observation(planned, conflict=True)],
        )
        self.assertEqual(result["state"], "evidence_conflict")
        self.assertEqual(result["required_action"], "verify")
        self.assertEqual(
            result["verification_of"], "object_appearance.color_blue"
        )

    def test_invalid_or_incomplete_aggregate_requires_verification(self):
        planned = round_plan()
        result = assess_evidence(
            current_plan([planned]),
            [observation(planned, valid=1, missing=1, invalid=1)],
        )
        self.assertEqual(result["state"], "aggregate_uncertain")
        self.assertEqual(result["required_action"], "verify")
        self.assertIn(
            "invalid_policy_results", result["checks"]["reasons"]
        )

    def test_semantic_absence_is_not_instrumentation_uncertainty(self):
        planned = round_plan(
            metric="pickup_to_first_contact_time", episodes=2
        )
        semantic_rows = [
            {
                "role": "policy_under_evaluation",
                "value": None,
                "details": {
                    "reason": "contact_not_observed_after_pickup"
                },
            },
            {
                "role": "policy_under_evaluation",
                "value": None,
                "details": {"reason": "pickup_not_observed"},
            },
        ]
        result = assess_evidence(
            current_plan(
                [planned],
                requested=["object_appearance.color_blue"],
            ),
            [
                observation(
                    planned,
                    valid=0,
                    missing=2,
                    planned_episodes=semantic_rows,
                )
            ],
        )
        self.assertEqual(result["state"], "sufficient")
        self.assertEqual(result["checks"]["semantic_missing"], 2)

    def test_sufficient_round_continues_to_uncovered_requested_aspect(self):
        planned = round_plan()
        result = assess_evidence(
            current_plan(
                [planned],
                requested=[
                    "object_appearance.color_blue",
                    "object_position.official_random",
                ],
            ),
            [observation(planned)],
        )
        self.assertEqual(result["required_action"], "continue")
        self.assertEqual(
            result["remaining_template_ids"],
            ["object_position.official_random"],
        )

    def test_pipeline_failure_stops_without_verification(self):
        planned = round_plan()
        result = assess_evidence(
            current_plan([planned]),
            [observation(planned, pipeline_passed=False)],
        )
        self.assertEqual(result["state"], "pipeline_failure")
        self.assertEqual(result["required_action"], "stop")

    def test_second_unresolved_verification_stops(self):
        original = round_plan(episodes=1)
        verification = round_plan(
            round_id="round_2",
            episodes=1,
            verification_of="object_appearance.color_blue",
        )
        result = assess_evidence(
            current_plan([original, verification]),
            [
                observation(original, valid=1, conflict=True),
                observation(verification, valid=1, conflict=True),
            ],
        )
        self.assertEqual(result["required_action"], "stop")
        self.assertTrue(result["unresolved"])
        self.assertIn("verification_already_used", result["reasons"])

    def test_budget_cap_stops_even_when_more_evidence_is_requested(self):
        planned = round_plan()
        plan = current_plan(
            [planned],
            requested=[
                "object_appearance.color_blue",
                "object_position.official_random",
            ],
            max_rounds=1,
        )
        result = assess_evidence(plan, [observation(planned)])
        self.assertEqual(result["required_action"], "stop")
        self.assertTrue(result["unresolved"])


class ConditionalTransitionTests(unittest.TestCase):
    CATALOG = {
        "position": {"template_ids": ["position.left", "position.right"]},
        "instance": {"template_ids": ["instance.base0", "instance.base1"]},
    }

    @staticmethod
    def adaptive_round(template_id, aspect_id):
        value = round_plan(template_id=template_id, episodes=1)
        value["aspect_id"] = aspect_id
        value["sub_aspect"] = aspect_id
        return value

    @staticmethod
    def adaptive_observation(plan, *, success, pipeline_passed=True):
        value = observation(
            plan,
            valid=1,
            pipeline_passed=pipeline_passed,
        )
        value["observations"]["policy_success"] = success
        return value

    def plan(self, rounds, *, max_rounds=3):
        return {
            "rounds": rounds,
            "requested_template_ids": [
                template_id
                for aspect in self.CATALOG.values()
                for template_id in aspect["template_ids"]
            ],
            "requested_aspect_ids": list(self.CATALOG),
            "max_rounds": max_rounds,
        }

    def test_failure_drills_into_same_aspect(self):
        first = self.adaptive_round("position.left", "position")
        result = assess_conditional_transition(
            self.plan([first]),
            [self.adaptive_observation(first, success=0.0)],
            aspect_catalog=self.CATALOG,
        )
        self.assertEqual(result["required_action"], "continue")
        self.assertEqual(result["required_transition"], "drill_down")
        self.assertEqual(result["required_next_aspect_id"], "position")

    def test_success_switches_to_uncovered_aspect(self):
        first = self.adaptive_round("position.left", "position")
        result = assess_conditional_transition(
            self.plan([first]),
            [self.adaptive_observation(first, success=1.0)],
            aspect_catalog=self.CATALOG,
        )
        self.assertEqual(result["required_transition"], "switch_aspect")
        self.assertEqual(result["required_next_aspect_id"], "instance")

    def test_pipeline_failure_stops_before_navigation(self):
        first = self.adaptive_round("position.left", "position")
        result = assess_conditional_transition(
            self.plan([first]),
            [
                self.adaptive_observation(
                    first, success=0.0, pipeline_passed=False
                )
            ],
            aspect_catalog=self.CATALOG,
        )
        self.assertEqual(result["required_action"], "stop")
        self.assertEqual(result["required_transition"], "stop")

    def test_final_conflict_stays_unresolved_when_all_variants_are_exhausted(self):
        catalog = {"position": {"template_ids": ["position.left"]}}
        first = self.adaptive_round("position.left", "position")
        latest = observation(first, valid=1, conflict=True)
        latest["observations"]["policy_success"] = 1.0
        plan = {
            "rounds": [first],
            "requested_template_ids": ["position.left"],
            "requested_aspect_ids": ["position"],
            "max_rounds": 1,
        }
        result = assess_conditional_transition(
            plan,
            [latest],
            aspect_catalog=catalog,
        )
        self.assertEqual(result["state"], "evidence_conflict")
        self.assertEqual(result["required_action"], "stop")
        self.assertEqual(result["required_transition"], "stop")
        self.assertTrue(result["unresolved"])
        self.assertIn(
            "round_budget_exhausted_with_unresolved_evidence",
            result["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
