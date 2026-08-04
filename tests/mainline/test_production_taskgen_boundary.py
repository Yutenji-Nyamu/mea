from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path

from mea.taskgen.round_materialization import materialize_open_world_round


REPO_ROOT = Path(__file__).resolve().parents[2]
ROUND_MATERIALIZATION = (
    REPO_ROOT / "mea" / "taskgen" / "round_materialization.py"
)
PLANNER_RETRIEVAL_BOUNDARIES = (
    REPO_ROOT / "mea" / "planner" / "context.py",
    REPO_ROOT / "mea" / "planner" / "open_world_session.py",
    REPO_ROOT / "mea" / "planner" / "query_interpretation.py",
)
LEGACY_MODULES = {
    "mea.capability_adapter",
    "mea.proposals",
    "mea.round_contract",
}


def _imported_modules(nodes: list[ast.stmt]) -> set[str]:
    modules: set[str] = set()
    for node in nodes:
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


class ProductionTaskIndependenceTests(unittest.TestCase):
    def test_production_planners_use_retrieval_index_directly(self):
        for source_path in PLANNER_RETRIEVAL_BOUNDARIES:
            with self.subTest(source=source_path.name):
                module = ast.parse(source_path.read_text(encoding="utf-8"))
                imports = _imported_modules(module.body)
                self.assertIn("mea.artifact_retrieval_index", imports)
                self.assertNotIn("mea.capability_adapter", imports)

    def test_legacy_task_menu_imports_are_scoped_to_command_builder(self):
        module = ast.parse(
            ROUND_MATERIALIZATION.read_text(encoding="utf-8")
        )
        top_level_imports = _imported_modules(module.body)
        self.assertTrue(LEGACY_MODULES.isdisjoint(top_level_imports))

        functions = {
            node.name: node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        command_imports = {
            imported
            for node in ast.walk(functions["build_taskgen_command"])
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for imported in _imported_modules([node])
        }
        production_imports = {
            imported
            for node in ast.walk(functions["materialize_open_world_round"])
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for imported in _imported_modules([node])
        }
        self.assertTrue(LEGACY_MODULES.issubset(command_imports))
        self.assertTrue(LEGACY_MODULES.isdisjoint(production_imports))

    def test_unregistered_task_materializes_without_legacy_proposal_fields(self):
        candidate = {
            "schema_version": 2,
            "candidate_id": "press_stapler.official.success",
            "source_query": "Did the policy complete press_stapler?",
            "base_task": "press_stapler",
            "semantic_concern": "task_execution: official task success",
            "scene_need": None,
            "checker_need": None,
            "rule_tool_need": {
                "kind": "reuse",
                "description": "Reuse the official check_success result.",
                "reuse_first": True,
            },
            "vqa_tool_need": None,
        }
        with tempfile.TemporaryDirectory() as temporary:
            round_plan, tool_bundle = materialize_open_world_round(
                REPO_ROOT,
                Path(temporary),
                round_number=2,
                candidate=candidate,
                control_execution={
                    "backend": "act",
                    "seeds": [100000],
                    "num_episodes": 1,
                    "gates": [],
                },
                policy_backend="smolvla",
            )

        self.assertEqual(round_plan["task_name"], "press_stapler")
        self.assertEqual(round_plan["task_module"], "envs.press_stapler")
        self.assertEqual(round_plan["route"], "official")
        self.assertIsNone(round_plan["template_id"])
        self.assertNotIn("capability_contract", round_plan)
        self.assertNotIn("task_proposal", round_plan)
        self.assertEqual(tool_bundle["source"], "official_checker_reuse")


if __name__ == "__main__":
    unittest.main()
