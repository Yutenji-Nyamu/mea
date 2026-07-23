import unittest

from mea.planner.claim_first_runtime import (
    ClaimFirstRuntimeController,
    ClaimFirstRuntimeError,
)
from mea.round_provenance import canonical_sha256


def target():
    return {
        "task_name": "click_bell",
        "max_rounds": 3,
        "policy": {"policy_name": "ACT"},
        "aspects": [
            {
                "aspect_id": "performance.completion_time_stability",
                "description": "Unchanged official scene control.",
                "template_ids": [
                    "performance.completion_time_stability.official"
                ],
            },
            {
                "aspect_id": "object_position",
                "description": "Generalization across left and right positions.",
                "template_ids": [
                    "object_position.left_fixed",
                    "object_position.right_fixed",
                ],
            },
        ],
    }


def round_plan(round_number, template_id):
    aspect = (
        "performance.completion_time_stability"
        if template_id.endswith(".official")
        else "object_position"
    )
    return {
        "round_id": f"round_{round_number}",
        "template_id": template_id,
        "sub_aspect": aspect,
        "task_instruction": f"Evaluate {template_id}.",
        "execution": {"num_episodes": 1, "seeds": [1000 + round_number]},
        "tool_request": {"metric": "time_to_success"},
        "task_proposal": {
            "aspect_id": aspect,
            "intent": f"Test {template_id}.",
            "changes": (
                {}
                if template_id.endswith(".official")
                else {"bell": {"position_mode": "fixed"}}
            ),
        },
    }


def summary(plan, success_rate, *, pipeline_passed=True):
    return {
        "round_id": plan["round_id"],
        "pipeline_passed": pipeline_passed,
        "observations": {
            "policy_success": success_rate,
            "aggregate": {
                "status": "passed",
                "input_issues": [],
                "metrics": [
                    {
                        "metric": "time_to_success",
                        "cohorts": [
                            {
                                "role": "policy_under_evaluation",
                                "summary": {
                                    "quality": {
                                        "valid": 1,
                                        "missing": 0,
                                        "invalid": 0,
                                    }
                                },
                            }
                        ],
                    }
                ],
            },
            "planned_tool": {
                "route_decision": {"metric": "time_to_success"},
                "episodes": [],
            },
            "execution_vqa": {"evidence_conflict": False},
        },
    }


def bind_provenance(plan, observed):
    binding = {
        "round_id": plan["round_id"],
        "round_plan_sha256": canonical_sha256(plan),
        "artifacts": [
            {
                "kind": "child_manifest",
                "path": f"runs/{plan['round_id']}/manifest.json",
                "sha256": "a" * 64,
                "size_bytes": 10,
            },
            {
                "kind": "round_aggregate",
                "path": f"runs/{plan['round_id']}/aggregate.json",
                "sha256": "b" * 64,
                "size_bytes": 10,
            },
        ],
    }
    provenance = {
        "binding": binding,
        "binding_sha256": canonical_sha256(binding),
    }
    observed["provenance"] = {
        "path": f"runs/{plan['round_id']}/provenance.json",
        "sha256": "c" * 64,
        "binding_sha256": provenance["binding_sha256"],
    }
    return provenance


def semantic_bundle():
    return {
        "schema_version": 1,
        "source": "cached_test_proposal",
        "proposal": {
            "schema_version": 1,
            "action": "continue",
            "sub_aspect": "object_position.left_fixed",
            "hypothesis": "The left fixed position may expose a weakness.",
            "requested_perturbation": {
                "description": "Place the bell at the safe left position.",
                "controlled_changes": ["left position"],
                "preserve": ["task identity", "checkpoint"],
            },
            "task_need": {
                "required": True,
                "description": "Materialize the left scene.",
            },
            "tool_need": {
                "required": True,
                "description": "Measure policy success and completion time.",
                "reuse_first": True,
            },
            "rationale": "A left sentinel is the first diagnostic candidate.",
        },
    }


class ClaimFirstRuntimeTests(unittest.TestCase):
    def test_control_pass_automatically_binds_evidence_and_semantic_step(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        observed = summary(control, 1.0)
        provenance = bind_provenance(control, observed)

        state = controller.observe([control], [observed], [provenance])

        self.assertTrue(state["control_passed"])
        self.assertFalse(state["assessment"]["should_stop"])
        self.assertEqual(
            state["open_query_evidence_history"][0]["outcome"], "success"
        )
        refs = state["records"][0]["evidence_refs"]
        self.assertEqual(
            {item["kind"] for item in refs},
            {"round_provenance", "child_manifest", "round_aggregate"},
        )
        bound = controller.bind_semantic_step(
            semantic_bundle(),
            state,
            executed_template_ids=[control["template_id"]],
        )
        self.assertEqual(
            bound["plan_step"]["template_id"],
            "object_position.left_fixed",
        )
        self.assertFalse(
            bound["resolution"]["catalog_was_model_visible"]
        )

    def test_failed_control_stops_before_property_attribution(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        observed = summary(control, 0.0)
        provenance = bind_provenance(control, observed)

        state = controller.observe([control], [observed], [provenance])

        self.assertTrue(state["assessment"]["should_stop"])
        self.assertEqual(
            state["assessment"]["stop_reason"],
            "control_baseline_policy_failed",
        )
        self.assertFalse(state["query_answer"]["answered"])
        with self.assertRaisesRegex(
            ClaimFirstRuntimeError, "after the query contract stopped"
        ):
            controller.bind_semantic_step(
                semantic_bundle(),
                state,
                executed_template_ids=[control["template_id"]],
            )

    def test_diagnostic_failure_stops_by_sufficiency_not_hard_cap(self):
        controller = ClaimFirstRuntimeController(
            "Where does this policy first expose a weakness?",
            target(),
        )
        control = round_plan(
            1, "performance.completion_time_stability.official"
        )
        candidate = round_plan(2, "object_position.left_fixed")
        control_summary = summary(control, 1.0)
        candidate_summary = summary(candidate, 0.0)
        control_provenance = bind_provenance(control, control_summary)
        candidate_provenance = bind_provenance(candidate, candidate_summary)

        state = controller.observe(
            [control, candidate],
            [control_summary, candidate_summary],
            [control_provenance, candidate_provenance],
        )

        self.assertTrue(state["assessment"]["evidence_sufficient"])
        self.assertEqual(
            state["assessment"]["stop_reason"], "evidence_sufficient"
        )
        self.assertEqual(state["assessment"]["claim_verdict"], "diagnosed")
        self.assertTrue(state["query_answer"]["answered"])
        self.assertNotIn("hard", state["query_answer"]["stop_reason"])
        self.assertIn(
            "object_position.right_fixed",
            state["query_answer"]["untested_candidate_ids"],
        )
        self.assertGreaterEqual(
            len(state["query_answer"]["evidence_refs"]), 6
        )


if __name__ == "__main__":
    unittest.main()
