from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_import_probe(source: str) -> dict[str, bool]:
    process = subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise AssertionError(process.stderr)
    return json.loads(process.stdout)


class ProductionCliBoundaryTests(unittest.TestCase):
    def test_agent_import_does_not_load_paper_or_task_specific_planners(self) -> None:
        modules = [
            "mea.strategy_plan",
            "mea.evidence_manifest",
            "experiments.paper.registered_execution_adapter",
            "experiments.paper.legacy_planner_factory",
            "mea.planner.catalog_plan",
            "mea.planner.click_bell",
            "mea.planner.official",
        ]
        probe = (
            "import importlib.util,json,pathlib,sys;"
            "path=pathlib.Path('scripts/manipeval_agent.py');"
            "spec=importlib.util.spec_from_file_location('production_agent_probe',path);"
            "module=importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(module);"
            f"print(json.dumps({{name:name in sys.modules for name in {modules!r}}}))"
        )
        loaded = run_import_probe(probe)
        self.assertEqual(loaded, {name: False for name in modules})

    def test_legacy_factory_still_loads_compatibility_planners(self) -> None:
        modules = [
            "mea.planner.catalog_plan",
            "mea.planner.click_bell",
            "mea.planner.official",
        ]
        probe = (
            "import json,sys;"
            "from experiments.paper.legacy_planner_factory "
            "import build_legacy_planner;"
            f"print(json.dumps({{name:name in sys.modules for name in {modules!r}}}))"
        )
        loaded = run_import_probe(probe)
        self.assertEqual(loaded, {name: True for name in modules})

    def test_paper_only_arguments_are_hidden_from_production_help(self) -> None:
        process = subprocess.run(
            [sys.executable, "scripts/manipeval_agent.py", "--help"],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        for option in (
            "--open-query-planner",
            "--proposal-mode",
            "--task-profile",
            "--planning-policy",
            "--evidence-manifest",
            "--command-plan",
            "--registered-route",
            "--registered-strategy",
        ):
            self.assertNotIn(option, process.stdout)
        self.assertIn("--auto-route", process.stdout)

    def test_default_production_mode_is_claim_first(self) -> None:
        probe = (
            "import argparse,importlib.util,json,pathlib;"
            "path=pathlib.Path('scripts/manipeval_agent.py');"
            "spec=importlib.util.spec_from_file_location('agent_defaults',path);"
            "module=importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(module);"
            "base=dict(open_query_planner=None,registered_strategy=None,"
            "task_profile='official',planning_policy='dynamic_evidence_v1',"
            "proposal_mode='catalog');"
            "production=module.resolve_default_open_query_planner("
            "argparse.Namespace(**base));"
            "base['planning_policy']='fixed_predeclared_v1';"
            "paper=module.resolve_default_open_query_planner("
            "argparse.Namespace(**base));"
            "print(json.dumps({'production':production,'paper':paper}))"
        )
        self.assertEqual(
            run_import_probe(probe),
            {
                "production": "claim_first_v1",
                "paper": "catalog_step_v1",
            },
        )

    def test_no_control_query_keeps_its_only_candidate_round(self) -> None:
        probe = (
            "import importlib.util,json,pathlib;"
            "path=pathlib.Path('scripts/manipeval_agent.py');"
            "spec=importlib.util.spec_from_file_location('agent_budget',path);"
            "module=importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(module);"
            "context={'sub_aspect':'trajectory.terminal_height',"
            "'hypothesis':'Measure the terminal telemetry field.',"
            "'requested_variation':'none',"
            "'measurement_need':'Measure terminal bottle telemetry height.'};"
            "budget=module.resolve_claim_first_candidate_budget("
            "1,user_request='What is the terminal bottle telemetry height?',"
            "query_contract=None,semantic_context=context);"
            "print(json.dumps({'budget':budget}))"
        )
        self.assertEqual(run_import_probe(probe), {"budget": 1})


if __name__ == "__main__":
    unittest.main()
