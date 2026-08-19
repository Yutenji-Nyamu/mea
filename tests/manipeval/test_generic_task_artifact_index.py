from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mea.taskgen.artifact_index import (
    GenericTaskArtifactError,
    GenericTaskArtifactIndex,
    materialize_reused_generic_task,
)
def _semantic_key() -> dict:
    return {
        "schema_version": 2,
        "base_task": "adjust_bottle",
        "semantic_concern": "object_pose.rotation",
        "scene_need": "rotate the bottle",
        "checker_need": "require upright placement",
        "evaluation_intent": None,
        "adapter_contract": {"official_source": "envs/adjust_bottle.py"},
    }


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
            (source / "overlay.yml").write_text("{}\n", encoding="utf-8")
            candidate = {
                "schema_version": 1,
                "candidate_id": "dynamic.adjust.rotate.abc",
                "source_query": "Does rotation expose a failure?",
                "base_task": "adjust_bottle",
                "semantic_concern": "object_pose.rotation",
                "scene_need": "rotate the bottle",
                "checker_need": "require upright placement",
                "rule_tool_need": "measure contact",
            }
            checker_review = {
                "schema_version": 1,
                "status": "approved",
                "checks": {
                    "implements_every_checker_requirement": True,
                    "preserves_quantifiers_and_temporal_relations": True,
                    "uses_direct_current_simulator_observables": True,
                    "does_not_substitute_correlated_proxy": True,
                },
                "reason": "The checker directly implements upright placement.",
                "authority": "development_agent_proxy",
            }
            (source / "candidate_manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_source",
                        "task_module": (
                            "mea.generated_tasks.run_source.task"
                        ),
                        "checker_semantic_review": checker_review,
                        "codegen_provenance": {},
                    }
                ),
                encoding="utf-8",
            )
            # Historical source artifacts remain readable during exact reuse.
            (source / "generation/experiment_candidate.json").write_text(
                json.dumps(candidate),
                encoding="utf-8",
            )
            manifest = {
                "run_id": "run_source",
                "generation_kind": (
                    "generic_provider_scene_checker_codegen"
                ),
                "mode": "generic_provider_scene_checker_codegen",
                "task_name": "adjust_bottle",
                "task_module": "mea.generated_tasks.run_source.task",
                "experiment_candidate": candidate,
                "experiment_candidate_path": (
                    "generation/experiment_candidate.json"
                ),
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
                    "checker_semantic_review": checker_review,
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
            }
            index = GenericTaskArtifactIndex(root)
            index.register_generated(
                resolution=resolution,
                manifest_path=source_manifest,
                source_query=candidate["source_query"],
            )
            match = index.find_exact(
                {
                    "schema_version": 2,
                    "semantic_key": semantic_key,
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
                candidate={
                    **candidate,
                    "rule_tool_need": "measure distance",
                },
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
            self.assertEqual(reused["proposal_path"], "generation/proposal.json")
            self.assertEqual(
                reused["proposal"]["candidate_id"],
                candidate["candidate_id"],
            )
            self.assertTrue(
                (destination / "generation/proposal.json").is_file()
            )
            self.assertFalse(
                (
                    destination
                    / "generation/experiment_candidate.json"
                ).exists()
            )
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
