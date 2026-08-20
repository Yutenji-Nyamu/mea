import json
import tempfile
import unittest
from pathlib import Path

from mea.artifact_registry import ArtifactRegistry, ArtifactRegistryError


class ArtifactRegistryTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.index_path = Path(self._temporary.name) / "artifact_index.json"
        self.registry = ArtifactRegistry(self.index_path)

    def tearDown(self):
        self._temporary.cleanup()

    def test_register_and_retrieve_exact_semantic_key_without_mutation(self):
        registered = self.registry.register(
            kind="task",
            semantic_key={
                "base_task": "adjust_bottle",
                "concern": "robustness.distractor_avoidance",
            },
            artifact_path="runs/taskgen/task.py",
        )
        self.assertEqual(
            set(registered), {"kind", "semantic_key", "artifact_path"}
        )
        encoded = self.index_path.read_text(encoding="utf-8")

        retrieved = self.registry.retrieve(
            kind="task",
            semantic_key={
                "concern": "robustness.distractor_avoidance",
                "base_task": "adjust_bottle",
            },
        )
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["artifact_path"], "runs/taskgen/task.py")
        self.assertEqual(self.index_path.read_text(encoding="utf-8"), encoded)

    def test_same_semantic_key_cannot_silently_change_artifact(self):
        fields = {
            "kind": "tool",
            "semantic_key": {"metric": "precontact_jerk_peak"},
            "artifact_path": "tools/jerk.py",
        }
        first = self.registry.register(**fields)
        self.assertEqual(self.registry.register(**fields), first)
        with self.assertRaisesRegex(
            ArtifactRegistryError, "different artifact"
        ):
            self.registry.register(
                **{**fields, "artifact_path": "tools/other.py"}
            )

    def test_entry_schema_is_the_shared_three_field_contract(self):
        self.registry.register(
            kind="vqa",
            semantic_key={"question": "wrong_object_contact"},
            artifact_path="vqa/question.json",
        )
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(payload[0]),
            {
                "kind",
                "semantic_key",
                "artifact_path",
            },
        )

    def test_missing_registry_is_an_empty_index(self):
        self.assertEqual(self.registry.entries(), [])
        self.assertIsNone(
            self.registry.retrieve(
                kind="task",
                semantic_key={"base_task": "place_phone_stand"},
            )
        )
        self.assertFalse(self.index_path.exists())

    def test_rejects_malformed_index(self):
        self.index_path.write_text('{"entries": []}', encoding="utf-8")
        with self.assertRaisesRegex(ArtifactRegistryError, "JSON list"):
            self.registry.entries()

    def test_explicit_legacy_index_is_not_implicitly_migrated(self):
        self.index_path.write_text(
            json.dumps(
                [
                    {
                        "kind": "task",
                        "semantic_key": {"base_task": "adjust_bottle"},
                        "artifact_path": "runs/legacy/manifest.json",
                        "source_query": "historical query",
                        "validation": {"status": "validated"},
                        "reuse_count": 2,
                    }
                ]
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ArtifactRegistryError, "entry fields are invalid"
        ):
            self.registry.entries()


if __name__ == "__main__":
    unittest.main()
