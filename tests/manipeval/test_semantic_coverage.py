from __future__ import annotations

import unittest

from mea.feedback.answer_scope import build_answer_scope
from mea.planner.semantic_coverage import (
    advance_implementation_trace_with_tool,
    build_candidate_intent_alignment,
    build_evaluation_intent,
    build_implementation_trace,
)


def _intent() -> dict:
    return build_evaluation_intent(
        source_query="Where does unseen bottle geometry expose weakness?",
        original_concern="generalization to unseen bottle geometry",
        hypothesis=(
            "A novel aspect ratio reduces success and increases final pose error."
        ),
        requested_change=(
            "Use a bottle with a novel shape or aspect ratio while preserving "
            "the target and initial placement."
        ),
        required_observation=(
            "Compare success, final pose error, and grasp failures against a "
            "matched familiar bottle."
        ),
        preserved_conditions=[
            {
                "actor": "target",
                "property": "model_identity",
                "axis": None,
                "relation": "preserve",
            },
            {
                "actor": "target",
                "property": "position",
                "axis": "all",
                "relation": "preserve",
            },
        ],
    )


def _candidate(
    *,
    semantic_concern: str,
    scene_description: str,
    checker_description: str,
    tool_description: str,
) -> dict:
    intent = _intent()
    scene_need = {
        "kind": "adapt",
        "description": scene_description,
        "reuse_first": True,
    }
    checker_need = {
        "kind": "generate",
        "description": checker_description,
        "reuse_first": True,
    }
    rule_tool_need = {
        "kind": "measure",
        "description": tool_description,
        "reuse_first": True,
    }
    alignment = build_candidate_intent_alignment(
        intent,
        semantic_concern=semantic_concern,
        scene_need=scene_need,
        checker_need=checker_need,
        tool_need=rule_tool_need,
    )
    return {
        "schema_version": 2,
        "candidate_id": "dynamic.adjust_bottle.geometry",
        "source_query": intent["source_query"],
        "base_task": "adjust_bottle",
        "semantic_concern": semantic_concern,
        "scene_need": scene_need,
        "checker_need": checker_need,
        "rule_tool_need": rule_tool_need,
        "vqa_tool_need": None,
        "tool_need": rule_tool_need,
        "evaluation_intent": intent,
        "intent_alignment": alignment,
    }


class SemanticCoverageTests(unittest.TestCase):
    def test_empty_preserve_set_is_vacuously_covered(self):
        intent = build_evaluation_intent(
            source_query="Does pre-contact motion become jerky?",
            original_concern="pre-contact trajectory smoothness",
            hypothesis="The policy exhibits a high pre-contact jerk peak.",
            requested_change="Reuse the official scene and inspect its trajectory.",
            required_observation="Measure pre-contact jerk peak.",
        )
        rule_need = {
            "kind": "measure",
            "description": "Measure pre-contact jerk peak.",
            "reuse_first": True,
        }
        alignment = build_candidate_intent_alignment(
            intent,
            semantic_concern="pre-contact trajectory smoothness and jerk peak",
            scene_need=None,
            checker_need=None,
            rule_tool_need=rule_need,
        )

        self.assertNotIn(
            "preserved_conditions",
            alignment["unmatched_intent_fields"],
        )
        self.assertEqual(alignment["relationship"], "direct")

    def test_tool_only_intent_can_complete_without_taskgen(self):
        intent = build_evaluation_intent(
            source_query="Does pre-contact motion become jerky?",
            original_concern="pre-contact trajectory smoothness",
            hypothesis="A high pre-contact jerk peak reveals instability.",
            requested_change="Reuse the official scene and inspect its trajectory.",
            required_observation="Measure pre-contact jerk peak.",
        )
        rule_need = {
            "kind": "measure",
            "description": (
                "Measure pre-contact jerk peak to test whether a high peak "
                "reveals instability."
            ),
            "reuse_first": True,
        }
        alignment = build_candidate_intent_alignment(
            intent,
            semantic_concern=(
                "Reuse the official scene and inspect pre-contact trajectory "
                "smoothness for instability."
            ),
            scene_need=None,
            checker_need=None,
            rule_tool_need=rule_need,
        )
        candidate = {
            "candidate_id": "dynamic.tool_only.jerk",
            "scene_need": None,
            "checker_need": None,
            "rule_tool_need": rule_need,
            "vqa_tool_need": None,
            "tool_need": rule_need,
            "evaluation_intent": intent,
            "intent_alignment": alignment,
        }

        trace = build_implementation_trace(candidate)
        self.assertEqual(
            trace["pending_intent_fields"], ["required_observation"]
        )
        completed = advance_implementation_trace_with_tool(
            trace, {"status": "passed"}
        )
        self.assertEqual(completed["coverage_status"], "complete")

    def test_scene_only_taskgen_keeps_unrequested_checker_coverage(self):
        intent = build_evaluation_intent(
            source_query="Does a smaller bell expose an ACT weakness?",
            original_concern="bell scale sensitivity",
            hypothesis="A smaller bell lowers policy success.",
            requested_change=(
                "Use a smaller bell while keeping its position unchanged."
            ),
            preserved_conditions=[
                {
                    "actor": "bell",
                    "property": "position",
                    "axis": "all",
                    "relation": "preserve",
                }
            ],
            required_observation=(
                "Compare policy success for the smaller bell."
            ),
        )
        scene_need = {
            "kind": "adapt",
            "description": (
                "Use a smaller bell while keeping its position unchanged."
            ),
            "reuse_first": True,
        }
        alignment = build_candidate_intent_alignment(
            intent,
            semantic_concern=(
                "Test bell scale sensitivity: a smaller bell may lower policy "
                "success, so compare policy success for the smaller bell."
            ),
            scene_need=scene_need,
            checker_need=None,
        )
        candidate = {
            "candidate_id": "dynamic.scene_only.scale",
            "scene_need": scene_need,
            "checker_need": None,
            "rule_tool_need": None,
            "vqa_tool_need": None,
            "tool_need": None,
            "evaluation_intent": intent,
            "intent_alignment": alignment,
        }

        trace = build_implementation_trace(
            candidate,
            taskgen_validation={
                "preflight": {
                    "scene_change_passed": True,
                    "preserved_conditions_verified": True,
                    "vision_validation": {"passed": True},
                }
            },
        )

        self.assertEqual(trace["uncovered_intent_fields"], [])
        self.assertIn("hypothesis", trace["covered_intent_fields"])
        self.assertEqual(
            trace["pending_intent_fields"], ["required_observation"]
        )

    def test_shift_with_other_layout_unchanged_is_a_scene_change(self):
        intent = build_evaluation_intent(
            source_query="Which bounded variation exposes a weakness?",
            original_concern="target position robustness",
            hypothesis="A shifted bell causes a policy failure.",
            requested_change=(
                "Shift the bell 10 cm left while keeping the overall scene "
                "layout unchanged."
            ),
            preserved_conditions=[
                {
                    "actor": None,
                    "property": "appearance",
                    "axis": None,
                    "relation": "preserve",
                }
            ],
            required_observation="Observe official success after the shift.",
        )
        scene_need = {
            "kind": "adapt",
            "description": (
                "Shift the bell 10 cm left while keeping the overall scene "
                "layout unchanged. Preserve unchanged: the overall scene "
                "layout."
            ),
            "reuse_first": True,
        }

        alignment = build_candidate_intent_alignment(
            intent,
            semantic_concern=(
                "target position robustness: a shifted bell causes a policy "
                "failure"
            ),
            scene_need=scene_need,
            checker_need=None,
            rule_tool_need={
                "kind": "measure",
                "description": "Observe official success after the shift.",
                "reuse_first": True,
            },
        )

        self.assertEqual(alignment["relationship"], "direct")
        self.assertNotIn(
            "requested_change",
            alignment["unmatched_intent_fields"],
        )

    def test_checker_only_taskgen_keeps_unrequested_scene_coverage(self):
        intent = build_evaluation_intent(
            source_query="Did the rollout hit the target without a false hit?",
            original_concern="target-only contact",
            hypothesis=(
                "A rollout succeeds only when it hits the target without a "
                "false hit."
            ),
            requested_change="Reuse the official unchanged scene.",
            required_observation=(
                "Evaluate target contact and false-hit status."
            ),
        )
        checker_need = {
            "kind": "generate",
            "description": (
                "Return success only when the rollout hits the target without "
                "a false hit, and expose target contact and false-hit status."
            ),
            "reuse_first": True,
        }
        alignment = build_candidate_intent_alignment(
            intent,
            semantic_concern=(
                "Reuse the official unchanged scene to test target-only "
                "contact and evaluate target contact and false-hit status."
            ),
            scene_need=None,
            checker_need=checker_need,
        )
        candidate = {
            "candidate_id": "dynamic.checker_only.target_contact",
            "scene_need": None,
            "checker_need": checker_need,
            "rule_tool_need": None,
            "vqa_tool_need": None,
            "tool_need": None,
            "evaluation_intent": intent,
            "intent_alignment": alignment,
        }

        trace = build_implementation_trace(
            candidate,
            taskgen_validation={
                "checker_fixtures": [{"passed": True}],
                "preflight": {},
            },
        )

        self.assertEqual(trace["uncovered_intent_fields"], [])
        self.assertIn("requested_change", trace["covered_intent_fields"])
        self.assertEqual(
            trace["pending_intent_fields"], ["required_observation"]
        )

    def test_checker_only_preservation_requires_taskgen_authority(self):
        intent = build_evaluation_intent(
            source_query="Can a stricter contact checker preserve official success?",
            original_concern="target-only contact semantics",
            hypothesis=(
                "The generated checker identifies target-only contact while "
                "preserving official success semantics."
            ),
            requested_change="Reuse the official unchanged scene.",
            preserved_conditions=[
                {
                    "actor": None,
                    "property": "checker_semantics",
                    "axis": None,
                    "relation": "preserve",
                }
            ],
            required_observation="Evaluate target and false-hit contact.",
        )
        checker_need = {
            "kind": "generate",
            "description": (
                "Identify target-only contact while preserving official "
                "success semantics and expose target and false-hit contact."
            ),
            "reuse_first": True,
        }
        alignment = build_candidate_intent_alignment(
            intent,
            semantic_concern=(
                "Target-only contact semantics preserve official success "
                "semantics and expose target and false-hit contact."
            ),
            scene_need=None,
            checker_need=checker_need,
        )
        candidate = {
            "candidate_id": "dynamic.checker_only.preserve_official",
            "scene_need": None,
            "checker_need": checker_need,
            "rule_tool_need": None,
            "vqa_tool_need": None,
            "tool_need": None,
            "evaluation_intent": intent,
            "intent_alignment": alignment,
        }

        pending = build_implementation_trace(
            candidate,
            taskgen_validation={
                "checker_fixtures": [{"passed": True}],
                "preflight": {},
            },
        )
        self.assertIn(
            "preserved_conditions",
            pending["pending_intent_fields"],
        )

        failed = build_implementation_trace(
            candidate,
            taskgen_validation={
                "checker_fixtures": [{"passed": True}],
                "preflight": {
                    "preserved_conditions_verified": False,
                },
            },
        )
        self.assertIn(
            "preserved_conditions",
            failed["uncovered_intent_fields"],
        )
        self.assertTrue(failed["repair_required"])

        verified = build_implementation_trace(
            candidate,
            taskgen_validation={
                "checker_fixtures": [{"passed": True}],
                "preflight": {
                    "preserved_conditions_verified": True,
                },
            },
        )
        self.assertIn(
            "preserved_conditions",
            verified["covered_intent_fields"],
        )

    def test_vqa_only_execution_does_not_require_rule_success(self):
        intent = build_evaluation_intent(
            source_query="Is there visible oscillation before contact?",
            original_concern="pre-contact visual oscillation",
            hypothesis="Visible oscillation occurs before contact.",
            requested_change="Reuse the official unchanged scene.",
            required_observation=(
                "Visually determine whether visible oscillation occurs before "
                "contact."
            ),
        )
        vqa_need = {
            "kind": "vqa",
            "description": (
                "Visually determine whether visible oscillation occurs before "
                "contact."
            ),
            "reuse_first": True,
        }
        alignment = build_candidate_intent_alignment(
            intent,
            semantic_concern=(
                "Reuse the official unchanged scene and inspect whether "
                "visible oscillation occurs before contact."
            ),
            scene_need=None,
            checker_need=None,
            vqa_tool_need=vqa_need,
        )
        candidate = {
            "candidate_id": "dynamic.vqa_only.oscillation",
            "scene_need": None,
            "checker_need": None,
            "rule_tool_need": None,
            "vqa_tool_need": vqa_need,
            "tool_need": vqa_need,
            "evaluation_intent": intent,
            "intent_alignment": alignment,
        }
        trace = build_implementation_trace(candidate)

        completed = advance_implementation_trace_with_tool(
            trace,
            {"status": "failed"},
            rule_required=False,
            vqa_evaluation={"status": "passed"},
            vqa_required=True,
        )

        self.assertEqual(completed["coverage_status"], "complete")
        self.assertEqual(completed["uncovered_intent_fields"], [])

    def test_structured_needs_replace_lexical_proxy_classification(self):
        candidate = _candidate(
            semantic_concern="Decompose official height and x success margins.",
            scene_description="Keep the official scene unchanged.",
            checker_description="Expose height and x margin components.",
            tool_description="Measure height margin, x margin, and official success.",
        )

        self.assertEqual(candidate["intent_alignment"]["relationship"], "direct")
        trace = build_implementation_trace(candidate)
        self.assertEqual(trace["relationship"], "direct")
        self.assertEqual(trace["uncovered_intent_fields"], [])

    def test_direct_taskgen_trace_waits_for_tool_execution(self):
        candidate = _candidate(
            semantic_concern=(
                "Test unseen bottle geometry with a novel aspect ratio and "
                "compare final pose and grasp failures."
            ),
            scene_description=(
                "Use a bottle with a novel shape or aspect ratio while "
                "preserving the target and initial placement."
            ),
            checker_description=(
                "Decide whether a novel aspect ratio reduces success and "
                "increases final pose error."
            ),
            tool_description=(
                "Compare success, final pose error, and grasp failures against "
                "a matched familiar bottle."
            ),
        )
        self.assertEqual(candidate["intent_alignment"]["relationship"], "direct")
        validation = {
            "checker_fixtures": [{"passed": True}],
            "preflight": {
                "scene_change_passed": True,
                "preserved_conditions_verified": True,
                "vision_validation": {"passed": True},
            },
        }

        taskgen_trace = build_implementation_trace(
            candidate,
            taskgen_validation=validation,
        )
        self.assertFalse(taskgen_trace["repair_required"])
        self.assertEqual(
            taskgen_trace["pending_intent_fields"],
            ["required_observation"],
        )

        execution_trace = build_implementation_trace(
            candidate,
            taskgen_validation=validation,
            tool_evaluation={"status": "passed"},
        )
        self.assertEqual(execution_trace["coverage_status"], "complete")

    def test_answer_scope_keeps_original_intent_trace_informational(self):
        candidate = _candidate(
            semantic_concern="Decompose official height and x success margins.",
            scene_description="Keep the official scene unchanged.",
            checker_description="Expose height and x margin components.",
            tool_description="Measure height margin and x margin.",
        )
        trace = build_implementation_trace(candidate)
        scope = build_answer_scope(
            {
                "implementation_trace": trace,
                "observations": {"pipeline_passed": True},
            }
        )

        self.assertEqual(
            scope["original_intent_ids"],
            [candidate["evaluation_intent"]["intent_id"]],
        )
        self.assertEqual(scope["covered_original_intent_fields"], [])
        self.assertEqual(len(scope["uncovered_original_intent_fields"]), 4)
        self.assertNotIn(
            "original_intent_uncovered",
            [item["code"] for item in scope["required_limitations"]],
        )


if __name__ == "__main__":
    unittest.main()
