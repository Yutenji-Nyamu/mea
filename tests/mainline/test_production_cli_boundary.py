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
    def test_agent_exports_cli_contracts_without_internal_acceptance_helpers(
        self,
    ) -> None:
        probe = (
            "import importlib.util,json,pathlib;"
            "from mea import agent_cli;"
            "path=pathlib.Path('scripts/manipeval_agent.py');"
            "spec=importlib.util.spec_from_file_location('agent_reexports',path);"
            "module=importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(module);"
            "print(json.dumps({"
            "'parse_args':module.parse_args is agent_cli.parse_args,"
            "'allowed_aspects':"
            "module.resolve_plan_agent_allowed_aspects "
            "is agent_cli.resolve_plan_agent_allowed_aspects,"
            "'planner_default':module.resolve_default_open_query_planner "
            "is agent_cli.resolve_default_open_query_planner,"
            "'candidate_budget':module.resolve_plan_agent_candidate_budget "
            "is agent_cli.resolve_plan_agent_candidate_budget,"
            "'episode_results_absent':"
            "not hasattr(module,'_episode_tool_results')}))"
        )
        self.assertEqual(
            run_import_probe(probe),
            {
                "parse_args": True,
                "allowed_aspects": True,
                "planner_default": True,
                "candidate_budget": True,
                "episode_results_absent": True,
            },
        )

    def test_query_interpretation_acceptance_allows_one_bounded_schema_repair(
        self,
    ) -> None:
        from mea.agent_acceptance import (
            _valid_query_interpretation_provider_trace,
        )

        self.assertTrue(
            _valid_query_interpretation_provider_trace(
                {"called": True, "attempt_count": 1, "errors": []}
            )
        )
        self.assertTrue(
            _valid_query_interpretation_provider_trace(
                {
                    "called": True,
                    "attempt_count": 2,
                    "errors": ["FreeConcern schema mismatch"],
                }
            )
        )
        self.assertFalse(
            _valid_query_interpretation_provider_trace(
                {
                    "called": True,
                    "attempt_count": 3,
                    "errors": ["first", "second"],
                }
            )
        )

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

    def test_default_production_mode_is_plan_agent(self) -> None:
        probe = (
            "import argparse,importlib.util,json,pathlib;"
            "from experiments.paper.compat_agent_profile "
            "import resolve_compat_agent_profile;"
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
            "paper=resolve_compat_agent_profile("
            "argparse.Namespace(auto_route=False,evidence_manifest=None,"
            "command_plan=None,registered_route=None,evaluation_id=None,**base),"
            "requested_open_query_planner=None)['open_query_planner'];"
            "print(json.dumps({'production':production,'paper':paper}))"
        )
        self.assertEqual(
            run_import_probe(probe),
            {
                "production": "plan_agent_v1",
                "paper": "catalog_step_v1",
            },
        )

    def test_historical_claim_first_value_normalizes_to_plan_agent(self) -> None:
        probe = (
            "import argparse,json;"
            "from mea.agent_cli import resolve_default_open_query_planner;"
            "value=resolve_default_open_query_planner("
            "argparse.Namespace(open_query_planner='claim_first_v1'));"
            "print(json.dumps({'planner':value}))"
        )
        self.assertEqual(
            run_import_probe(probe),
            {"planner": "plan_agent_v1"},
        )

    def test_control_path_plans_next_subaspect_after_evidence(self) -> None:
        source = (
            REPO_ROOT / "mea/plan_agent_application.py"
        ).read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "self.session.propose_semantic_step(",
            source,
        )
        self.assertIn(
            "self.session.bind_evidence_conditioned_semantic_step(",
            source,
        )
        self.assertNotIn("claim_first_controller =", source)
        self.assertNotIn("pending_first_semantic_bundle", source)
        self.assertNotIn("use_pending_first", source)
        self.assertIn(
            'assessment.get("evidence_sufficient") is not True',
            source,
        )
        self.assertIn('if plan_step["action"] == "stop":', source)
        self.assertIn("materialized_round = None", source)
        self.assertNotIn(
            "QueryContract accepted stopping but the Plan Agent did not "
            "propose action=stop",
            source,
        )

    def test_plan_agent_decides_before_the_external_hard_cap(self) -> None:
        source = (
            REPO_ROOT / "mea/plan_agent_application.py"
        ).read_text(
            encoding="utf-8"
        )
        author_decision = source.index(
            "self.session.propose_semantic_step(",
        )
        continue_gate = source.index(
            'raw_proposal.get("action") != "stop"',
            author_decision,
        )
        hard_cap = source.index(
            "apply_external_hard_round_cap(",
            continue_gate,
        )
        bind_continue = source.index(
            "self.session.bind_evidence_conditioned_semantic_step(",
            hard_cap,
        )
        self.assertLess(
            author_decision,
            continue_gate,
            "the Agent must author stop/continue before the cap is checked",
        )
        self.assertLess(
            continue_gate,
            hard_cap,
            "only an Agent continue decision may enter the hard-cap stop",
        )
        self.assertLess(
            hard_cap,
            bind_continue,
            "a capped continue must stop before candidate binding/materialization",
        )

    def test_production_cli_calls_the_extracted_application(self) -> None:
        source = (REPO_ROOT / "scripts/manipeval_agent.py").read_text(
            encoding="utf-8"
        )
        application = source.index("PlanAgentApplication(")
        execution = source.index(").run()", application)
        compat_loop = source.index(
            "round_runs: list[dict[str, Any]] = []",
            execution,
        )
        self.assertLess(application, execution)
        self.assertLess(execution, compat_loop)
        self.assertNotIn(
            "propose_semantic_step(",
            source,
            "the production decision loop must not remain duplicated in the CLI",
        )

    def test_precontrol_concern_does_not_shrink_planner_domain(self) -> None:
        probe = (
            "import importlib.util,json,pathlib;"
            "path=pathlib.Path('scripts/manipeval_agent.py');"
            "spec=importlib.util.spec_from_file_location('agent_domain',path);"
            "module=importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(module);"
            "print(json.dumps({"
            "'open':module.resolve_plan_agent_allowed_aspects(None),"
            "'explicit':module.resolve_plan_agent_allowed_aspects("
            "['object_position','object_position','object_instance'])}))"
        )
        self.assertEqual(
            run_import_probe(probe),
            {
                "open": None,
                "explicit": ["object_position", "object_instance"],
            },
        )

    def test_production_execution_uses_native_method_runtime(
        self,
    ) -> None:
        agent_source = (REPO_ROOT / "scripts/manipeval_agent.py").read_text(
            encoding="utf-8"
        )
        executor_source = (REPO_ROOT / "mea/round_executor.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'native_policy_rounds["act"] = partial(',
            agent_source,
        )
        self.assertIn(
            '"smolvla": partial(',
            agent_source,
        )
        self.assertIn(
            "generated_task_materializer=(",
            agent_source,
        )
        self.assertNotIn(
            "project_executed_round_through_method_runtime(",
            agent_source + executor_source,
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
            "budget=module.resolve_plan_agent_candidate_budget("
            "1,user_request='What is the terminal bottle telemetry height?',"
            "query_contract=None,semantic_context=context);"
            "print(json.dumps({'budget':budget}))"
        )
        self.assertEqual(run_import_probe(probe), {"budget": 1})


if __name__ == "__main__":
    unittest.main()
