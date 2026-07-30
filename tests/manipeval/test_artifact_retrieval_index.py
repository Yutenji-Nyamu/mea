import unittest

from mea.artifact_retrieval_index import (
    ArtifactRetrievalIndexError,
    OFFICIAL_CONTROL_TEMPLATE_ID,
    resolve_task_retrieval_index,
)
from mea.capability_adapter import (
    CapabilityAdapterError,
    resolve_task_retrieval_index as resolve_compatibility_index,
)


class ArtifactRetrievalIndexTests(unittest.TestCase):
    def test_reviewed_index_is_non_authoritative_and_menu_free(self):
        index = resolve_task_retrieval_index("click_bell")

        self.assertEqual(index["index_role"], "retrieval_only")
        self.assertIs(index["execution_authority"], False)
        self.assertNotIn("planner_kind", index)
        self.assertNotIn("task_profile", index)
        self.assertNotIn("max_rounds", index)
        self.assertIn(
            "object_position.left_fixed",
            [entry["template_id"] for entry in index["entries"]],
        )

    def test_unknown_runtime_task_gets_no_artificial_capabilities(self):
        index = resolve_task_retrieval_index(
            "runtime_schema_task",
            allow_unregistered=True,
        )

        self.assertEqual(index["task_name"], "runtime_schema_task")
        self.assertEqual(
            index["control_template_id"],
            OFFICIAL_CONTROL_TEMPLATE_ID,
        )
        self.assertEqual(index["entries"], [])
        self.assertEqual(index["vqa_questions"], {})
        self.assertEqual(index["vqa_metric_rules"], {})
        self.assertIs(index["execution_authority"], False)

    def test_unknown_task_is_strict_without_open_world_opt_in(self):
        with self.assertRaisesRegex(
            ArtifactRetrievalIndexError,
            "no reviewed artifact index",
        ):
            resolve_task_retrieval_index("runtime_schema_task")

    def test_returned_entries_are_defensive_copies(self):
        first = resolve_task_retrieval_index("click_bell")
        first["entries"][0]["template_id"] = "mutated"

        second = resolve_task_retrieval_index("click_bell")
        self.assertNotEqual(second["entries"][0]["template_id"], "mutated")

    def test_legacy_export_delegates_without_changing_its_error_boundary(self):
        self.assertEqual(
            resolve_compatibility_index("click_bell"),
            resolve_task_retrieval_index("click_bell"),
        )
        with self.assertRaises(CapabilityAdapterError):
            resolve_compatibility_index("runtime_schema_task")


if __name__ == "__main__":
    unittest.main()
