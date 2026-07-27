from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mea.taskgen.artifact_index import (
    GenericTaskArtifactIndex,
    materialize_reused_generic_task,
)


def _semantic_key() -> dict:
    return {
        "schema_version": 1,
        "base_task": "adjust_bottle",
        "semantic_concern": "object_pose.rotation",
        "scene_need": "rotate the bottle",
        "checker_need": "require upright placement",
        "adapter_contract": {"official_source_sha256": "a" * 64},
    }


def _digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class GenericTaskArtifactIndexTests(unittest.TestCase):
    def test_register_find_and_materialize_clean_reuse_envelope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "mea/generated_tasks/run_source"
            for child in ("generation", "validation", "evidence", "evaluation"):
                (source / child).mkdir(parents=True, exist_ok=True)
            (source / "__init__.py").write_text("", encoding="utf-8")
            task_source = "class adjust_bottle:\n    pass\n"
            task_path = source / "task.py"
            task_path.write_text(task_source, encoding="utf-8")
            module_hash = hashlib.sha256(
                task_path.read_bytes()
            ).hexdigest()
            (source / "overlay.yml").write_text("{}\n", encoding="utf-8")
            (source / "candidate_manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_source",
                        "task_module": (
                            "mea.generated_tasks.run_source.task"
                        ),
                        "module_sha256": module_hash,
                        "codegen_provenance": {},
                    }
                ),
                encoding="utf-8",
            )
            candidate = {
                "schema_version": 1,
                "candidate_id": "dynamic.adjust.rotate.abc",
                "source_query": "Does rotation expose a failure?",
                "base_task": "adjust_bottle",
                "semantic_concern": "object_pose.rotation",
                "scene_need": "rotate the bottle",
                "checker_need": "require upright placement",
                "tool_need": "measure contact",
            }
            manifest = {
                "run_id": "run_source",
                "generation_kind": (
                    "generic_provider_scene_checker_codegen"
                ),
                "mode": "generic_provider_scene_checker_codegen",
                "task_name": "adjust_bottle",
                "task_module": "mea.generated_tasks.run_source.task",
                "candidate_module_sha256": module_hash,
                "scene_validation": {
                    "generic_preflight": {
                        "scene_change_passed": True,
                        "checker_fixtures": [
                            {"passed": True},
                            {"passed": True},
                        ],
                    }
                },
                "task_generation_acceptance": {
                    "status": "accepted",
                    "act_rollouts_started_before_acceptance": 0,
                },
                "act_evaluation": {
                    "artifact": "evaluation/stale.json",
                    "success_rate": 1.0,
                },
            }
            source_manifest = source / "manifest.json"
            source_manifest.write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            semantic_key = _semantic_key()
            resolution = {
                "status": "generated",
                "semantic_key": semantic_key,
                "semantic_key_sha256": _digest(semantic_key),
            }
            index = GenericTaskArtifactIndex(root)
            index.register_generated(
                resolution=resolution,
                manifest_path=source_manifest,
                source_query=candidate["source_query"],
            )
            match = index.find_exact(
                {
                    "semantic_key": semantic_key,
                    "semantic_key_sha256": _digest(semantic_key),
                }
            )
            reused_resolution = {
                "status": "reused",
                "exact_match": match,
            }

            reused = materialize_reused_generic_task(
                root,
                run_id="run_reused",
                user_request="Measure the same experiment another way.",
                candidate={**candidate, "tool_need": "measure distance"},
                resolution=reused_resolution,
            )

            self.assertFalse(reused["provider"]["called"])
            self.assertEqual(
                reused["task_module"],
                "mea.generated_tasks.run_reused.task",
            )
            destination = root / "mea/generated_tasks/run_reused"
            self.assertTrue((destination / "task.py").is_file())
            self.assertTrue((destination / "manifest.json").is_file())
            self.assertNotIn("act_evaluation", reused)
            self.assertEqual(
                reused["task_generation_acceptance"]["status"],
                "pending_current_seed_revalidation",
            )
            self.assertEqual(
                index.registry.entries(kind="task")[0]["reuse_count"],
                0,
            )
            index.mark_reuse(semantic_key)
            self.assertEqual(
                index.registry.entries(kind="task")[0]["reuse_count"],
                1,
            )


if __name__ == "__main__":
    unittest.main()
