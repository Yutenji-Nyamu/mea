from __future__ import annotations

import unittest

from mea.planner.semantic_coverage import (
    SemanticCoverageError,
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

        self.assertEqual(intent["preserved_conditions"], [fact])

    def test_preservation_prose_is_rejected_at_intent_boundary(self):
        with self.assertRaisesRegex(
            SemanticCoverageError, "preserved_conditions"
        ):
            _intent(preserved_conditions=["keep target y unchanged"])


if __name__ == "__main__":
    unittest.main()
