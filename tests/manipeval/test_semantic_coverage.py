from __future__ import annotations

import unittest

from mea.planner.semantic_coverage import (
    SemanticCoverageError,
    build_candidate_intent_alignment,
    build_evaluation_intent,
)


def _intent(*, preserved_conditions=()) -> dict:
    return build_evaluation_intent(
        source_query="Where does unseen bottle geometry expose weakness?",
        original_concern="generalization to unseen bottle geometry",
        hypothesis="A novel aspect ratio reduces official task success.",
        requested_change="Use a bottle with a novel aspect ratio.",
        required_observation="Measure official success on the generated scene.",
        preserved_conditions=preserved_conditions,
    )


class SemanticCoverageTests(unittest.TestCase):
    def test_typed_preservation_is_carried_without_lexical_matching(self):
        fact = {
            "actor": "target",
            "property": "position",
            "axis": "y",
            "relation": "preserve",
        }
        intent = _intent(preserved_conditions=[fact])
        scene_need = {
            "kind": "adapt",
            "description": "Generate one bounded geometry variation.",
            "reuse_first": True,
        }

        alignment = build_candidate_intent_alignment(
            intent,
            semantic_concern="unseen geometry",
            scene_need=scene_need,
            checker_need=None,
            rule_tool_need={
                "kind": "measure",
                "description": "Read official success.",
                "reuse_first": True,
            },
        )

        self.assertEqual(intent["preserved_conditions"], [fact])
        self.assertEqual(alignment["relationship"], "direct")
        self.assertEqual(alignment["unmatched_intent_fields"], [])

    def test_preservation_prose_is_rejected_at_intent_boundary(self):
        with self.assertRaisesRegex(
            SemanticCoverageError, "preserved_conditions"
        ):
            _intent(preserved_conditions=["keep target y unchanged"])

    def test_missing_observation_owner_is_a_diagnostic_proxy(self):
        intent = _intent()

        alignment = build_candidate_intent_alignment(
            intent,
            semantic_concern="unseen geometry",
            scene_need=None,
            checker_need=None,
        )

        self.assertEqual(alignment["relationship"], "diagnostic_proxy")
        self.assertEqual(
            alignment["unmatched_intent_fields"], ["required_observation"]
        )

    def test_direct_declaration_cannot_hide_missing_observation_owner(self):
        with self.assertRaisesRegex(
            SemanticCoverageError, "cannot declare direct"
        ):
            build_candidate_intent_alignment(
                _intent(),
                semantic_concern="unseen geometry",
                scene_need=None,
                checker_need=None,
                declared_relationship="direct",
            )


if __name__ == "__main__":
    unittest.main()
