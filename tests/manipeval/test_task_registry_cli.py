from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from mea.taskgen.reviewed_registry import ReviewedTaskRegistryError


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/manipeval_task_registry.py"
SPEC = importlib.util.spec_from_file_location("manipeval_task_registry", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TaskRegistryCliTests(unittest.TestCase):
    def test_resolution_query_projects_only_lookup_contract(self) -> None:
        semantic_key = {
            "schema_version": 1,
            "task_name": "beat_block_hammer",
        }
        digest = "a" * 64
        self.assertEqual(
            MODULE._resolution_query(
                {
                    "schema_version": 1,
                    "semantic_key": semantic_key,
                    "semantic_key_sha256": digest,
                    "provider_required": False,
                }
            ),
            {
                "schema_version": 1,
                "semantic_key": semantic_key,
                "semantic_key_sha256": digest,
            },
        )

    def test_resolution_query_rejects_missing_identity(self) -> None:
        with self.assertRaisesRegex(
            ReviewedTaskRegistryError, "semantic_key is missing"
        ):
            MODULE._resolution_query({"semantic_key_sha256": "a" * 64})


if __name__ == "__main__":
    unittest.main()
