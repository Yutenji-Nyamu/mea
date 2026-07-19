import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mea.portfolio import (
    PortfolioError,
    build_portfolio_command_plan,
    build_reused_portfolio,
    render_portfolio_report,
)
from mea.runtime_ledger import (
    record_act_batch_start,
    record_provider_transport_start,
    runtime_ledger_context,
    summarize_runtime_ledger,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_catalog_root(root: Path) -> None:
    runner = root / "scripts/manipeval_agent.py"
    runner.parent.mkdir(parents=True, exist_ok=True)
    runner.write_text("print('stub')\n", encoding="utf-8")
    for task, family in (
        ("click_bell", "button_interaction"),
        ("beat_block_hammer", "tool_use"),
    ):
        write_json(
            root / f"mea/toolkit/schemas/{task}.json",
            {"task_name": task, "task_family": family},
        )
        checkpoint = root / f"policy/ACT/act_ckpt/act-{task}/demo_clean-50"
        checkpoint.mkdir(parents=True)
        (checkpoint / "policy_last.ckpt").write_bytes(b"weights")
        (checkpoint / "dataset_stats.pkl").write_bytes(b"stats")


def make_child(
    root: Path,
    *,
    task_name: str,
    evaluation_id: str,
    pipeline_passed: bool,
    policy_success: float | None,
    seed: int,
    act_starts: int = 0,
) -> None:
    evaluation = root / "mea/evaluation_runs" / evaluation_id
    feedback = {
        "answer": "child answer",
        "findings": ["child finding"],
        "limitations": ["N=1"],
        "recommended_next_step": "repeat",
        "provider_metadata": {"id": "provider-call"},
    }
    round_observations = {
        "execution_backend": "ACT",
        "actual_seeds": [seed],
        "pipeline_passed": pipeline_passed,
        "policy_success": policy_success,
        "whole_round_recovery": {
            "runtime": {
                "provider_called": True,
                "simulator_called": True,
                "act_rollouts_started": max(act_starts, 1),
            }
        },
    }
    runtime_ledgers = None
    if act_starts:
        runtime_ledgers = []
        for attempt in range(1, act_starts + 1):
            context = {
                "schema_version": 1,
                "evaluation_id": evaluation_id,
                "logical_round_id": "round_1",
                "round_attempt_index": attempt,
                "child_run_id": f"run_{evaluation_id}_round_1_attempt_{attempt}",
            }
            ledger = (
                evaluation
                / f"runtime/round_1/attempt_{attempt:02d}/call_starts.jsonl"
            )
            with runtime_ledger_context(ledger, context):
                record_provider_transport_start(
                    logical_call_id=f"{attempt:032x}",
                    transport_attempt=1,
                    modality="text",
                    model="fake-model",
                )
                record_act_batch_start(
                    task_name=task_name,
                    policy_name="ACT",
                    start_seed=seed,
                    num_rollouts=1,
                )
            if attempt == act_starts:
                summary = summarize_runtime_ledger(
                    ledger, expected_context=context
                )
                summary["artifact"] = ledger.relative_to(root).as_posix()
                round_observations["runtime_call_ledger"] = summary
    evidence = {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "user_request": f"historical request for {task_name}",
        "rounds": [
            {
                "round_id": "round_1",
                "seeds": [seed],
                "num_episodes": 1,
                "observations": round_observations,
                "tool_evaluation": {
                    "route_decision": {"provider_called": False}
                },
            }
        ],
        "observations": {
            "execution_backends": ["ACT"],
            "pipeline_passed": pipeline_passed,
            "policy_success": policy_success,
            "policy_success_by_round": [policy_success],
        },
        "total_episodes": 1,
    }
    manifest = {
        "schema_version": 1,
        "evaluation_id": evaluation_id,
        "task_name": task_name,
        "status": (
            "completed" if pipeline_passed else "completed_with_pipeline_failure"
        ),
        "lifecycle_status": "completed",
        "evidence_path": "summary/evidence_bundle.json",
        "feedback_path": "feedback/feedback.json",
        "report_path": "evaluation_report.md",
        "feedback": feedback,
        "planner": {"provider_called": True},
        "plan": {
            "policy": {
                "name": "ACT",
                "checkpoint_setting": "demo_clean",
                "expert_data_num": 50,
            },
            "rounds": [{"round_id": "round_1"}],
        },
    }
    if runtime_ledgers is not None:
        manifest["runtime_ledgers"] = runtime_ledgers
    write_json(evaluation / "manifest.json", manifest)
    write_json(evaluation / "summary/evidence_bundle.json", evidence)
    write_json(evaluation / "feedback/feedback.json", feedback)
    (evaluation / "evaluation_report.md").write_text(
        f"# {task_name} report\n", encoding="utf-8"
    )


class PortfolioTests(unittest.TestCase):
    def test_plan_binds_exact_two_ready_tasks_and_two_act_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_catalog_root(root)
            result = build_portfolio_command_plan(
                root,
                portfolio_id="portfolio_demo",
                user_query="How robust is object interaction?",
                start_seed=9,
                gpu=1,
            )
        self.assertEqual(
            result["task_bindings"], ["click_bell", "beat_block_hammer"]
        )
        self.assertEqual(result["planned_runtime"]["max_act_rollouts"], 2)
        self.assertEqual(result["runtime"]["act_rollouts_started"], 0)
        self.assertEqual(result["runtime"]["provider_calls_started"], 0)
        self.assertFalse(result["paper_table_eligible"])
        self.assertRegex(result["runner"]["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len({row["evaluation_id"] for row in result["children"]}), 2)
        for child in result["children"]:
            argv = child["argv"]
            self.assertIn("--num-episodes", argv)
            self.assertEqual(argv[argv.index("--num-episodes") + 1], "1")
            self.assertEqual(argv[argv.index("--generated-rounds") + 1], "1")
            self.assertEqual(argv[argv.index("--max-agent-rounds") + 1], "1")
            self.assertEqual(
                argv[argv.index("--round-recovery-max-restarts") + 1], "0"
            )
            self.assertEqual(
                child["expected_postconditions"]["act_rollouts_started"], 1
            )
            self.assertEqual(
                child["expected_postconditions"]["hard_agent_round_limit"], 1
            )
            if child["task_name"] == "click_bell":
                self.assertEqual(
                    argv[argv.index("--task-profile") + 1], "adaptive_properties"
                )

    def test_plan_fails_when_one_checkpoint_ready_task_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_catalog_root(root)
            (root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt").unlink()
            with self.assertRaisesRegex(PortfolioError, "missing portfolio tasks"):
                build_portfolio_command_plan(
                    root,
                    portfolio_id="portfolio_missing",
                    user_query="test both tasks",
                )

    def test_reuse_hash_binds_children_without_conflating_pipeline_and_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_child(
                root,
                task_name="click_bell",
                evaluation_id="eval_click_failure",
                pipeline_passed=True,
                policy_success=0.0,
                seed=10,
            )
            make_child(
                root,
                task_name="beat_block_hammer",
                evaluation_id="eval_bbh_success",
                pipeline_passed=True,
                policy_success=1.0,
                seed=11,
            )
            result = build_reused_portfolio(
                root,
                portfolio_id="portfolio_reuse",
                user_query="Compare generalization across both tasks",
                child_evaluation_ids={
                    "click_bell": "eval_click_failure",
                    "beat_block_hammer": "eval_bbh_success",
                },
            )
        self.assertEqual(result["mode"], "reused_completed_children")
        self.assertEqual(result["runtime"]["act_rollouts_started"], 0)
        self.assertEqual(result["historical_child_runtime"]["act_rollouts_started"], 2)
        self.assertTrue(result["historical_child_runtime"]["provider_called"])
        self.assertIn("1 with nonzero success", result["synthesis"]["answer"])
        self.assertTrue(
            any(
                "click_bell" in item and "0.000" in item
                for item in result["synthesis"]["weaknesses"]
            )
        )
        self.assertFalse(
            any(
                "click_bell" in item and "policy_success=1" in item
                for item in result["synthesis"]["strengths"]
            )
        )
        for child in result["children"]:
            self.assertEqual(
                set(child["artifacts"]),
                {"manifest", "evidence", "feedback", "report"},
            )
            for reference in child["artifacts"].values():
                self.assertRegex(reference["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(reference["size_bytes"], 0)
        report = render_portfolio_report(result)
        self.assertIn("## Strengths", report)
        self.assertIn("## Weaknesses", report)
        self.assertIn("## Historical child runtime", report)

    def test_reuse_rejects_aggregate_outcome_that_conflicts_with_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task, evaluation_id in (
                ("click_bell", "eval_click_mismatch"),
                ("beat_block_hammer", "eval_bbh_mismatch"),
            ):
                make_child(
                    root,
                    task_name=task,
                    evaluation_id=evaluation_id,
                    pipeline_passed=True,
                    policy_success=0.0,
                    seed=1,
                )
            evidence_path = root / (
                "mea/evaluation_runs/eval_click_mismatch/summary/evidence_bundle.json"
            )
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            evidence["observations"]["policy_success"] = 1.0
            write_json(evidence_path, evidence)
            with self.assertRaisesRegex(PortfolioError, "weighted rounds"):
                build_reused_portfolio(
                    root,
                    portfolio_id="portfolio_mismatch",
                    user_query="query",
                    child_evaluation_ids={
                        "click_bell": "eval_click_mismatch",
                        "beat_block_hammer": "eval_bbh_mismatch",
                    },
                )

    def test_call_start_ledgers_count_recovered_start_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_child(
                root,
                task_name="click_bell",
                evaluation_id="eval_click_recovered",
                pipeline_passed=True,
                policy_success=1.0,
                seed=3,
                act_starts=2,
            )
            make_child(
                root,
                task_name="beat_block_hammer",
                evaluation_id="eval_bbh_one_start",
                pipeline_passed=True,
                policy_success=1.0,
                seed=4,
                act_starts=1,
            )
            result = build_reused_portfolio(
                root,
                portfolio_id="portfolio_recovered",
                user_query="query",
                child_evaluation_ids={
                    "click_bell": "eval_click_recovered",
                    "beat_block_hammer": "eval_bbh_one_start",
                },
            )
        historical = result["historical_child_runtime"]
        self.assertEqual(historical["act_rollouts_started"], 3)
        self.assertEqual(historical["completed_act_episodes"], 2)
        self.assertTrue(historical["started_count_exact"])

    def test_reuse_rejects_duplicate_children_and_artifact_escape(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for task, evaluation_id in (
                ("click_bell", "eval_click"),
                ("beat_block_hammer", "eval_bbh"),
            ):
                make_child(
                    root,
                    task_name=task,
                    evaluation_id=evaluation_id,
                    pipeline_passed=True,
                    policy_success=1.0,
                    seed=1,
                )
            with self.assertRaisesRegex(PortfolioError, "unique child"):
                build_reused_portfolio(
                    root,
                    portfolio_id="portfolio_duplicate",
                    user_query="query",
                    child_evaluation_ids={
                        "click_bell": "eval_click",
                        "beat_block_hammer": "eval_click",
                    },
                )
            manifest_path = root / "mea/evaluation_runs/eval_click/manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["evidence_path"] = "../outside.json"
            write_json(manifest_path, manifest)
            write_json(root / "mea/evaluation_runs/outside.json", {})
            with self.assertRaisesRegex(PortfolioError, "canonical relative"):
                build_reused_portfolio(
                    root,
                    portfolio_id="portfolio_escape",
                    user_query="query",
                    child_evaluation_ids={
                        "click_bell": "eval_click",
                        "beat_block_hammer": "eval_bbh",
                    },
                )

    def test_cli_reuse_writes_summary_and_report(self):
        script = REPO_ROOT / "scripts/manipeval_portfolio.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_child(
                root,
                task_name="click_bell",
                evaluation_id="eval_click_cli",
                pipeline_passed=True,
                policy_success=0.0,
                seed=5,
            )
            make_child(
                root,
                task_name="beat_block_hammer",
                evaluation_id="eval_bbh_cli",
                pipeline_passed=True,
                policy_success=1.0,
                seed=6,
            )
            process = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--repo-root",
                    str(root),
                    "reuse",
                    "--portfolio-id",
                    "portfolio_cli",
                    "--query",
                    "one query",
                    "--output-dir",
                    "mea/portfolio_runs/portfolio_cli",
                    "--click-bell-evaluation-id",
                    "eval_click_cli",
                    "--bbh-evaluation-id",
                    "eval_bbh_cli",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue(
                (root / "mea/portfolio_runs/portfolio_cli/summary.json").is_file()
            )
            self.assertTrue(
                (root / "mea/portfolio_runs/portfolio_cli/report.md").is_file()
            )


if __name__ == "__main__":
    unittest.main()
