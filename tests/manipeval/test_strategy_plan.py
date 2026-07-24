import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mea.evidence_manifest import canonical_sha256, prepare_evidence_manifest
from mea.strategy_plan import (
    StrategyPlanError,
    build_matched_strategy_plan,
    compare_registered_strategies,
    load_registered_execution,
)
from tests.manipeval.test_strategy_comparison import make_run


REPO_ROOT = Path(__file__).resolve().parents[2]


def write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def evidence_config(root: Path) -> dict:
    candidates = ["object_position.left_fixed", "object_position.right_fixed"]
    return {
        "schema_version": 1,
        "registration_id": "pair_demo",
        "claim_scope": "table1_efficiency_mechanism_plumbing_only",
        "task_name": "click_bell",
        "query": "How robust is ACT to bell position?",
        "base_commit": git_head(root),
        "candidate_suite": candidates,
        "checkpoint_setting": "demo_clean",
        "expert_data_num": 50,
        "checkpoint_files": [
            "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt",
            "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/dataset_stats.pkl",
        ],
        "telemetry_profile": "balanced_v1",
        "sample_schedule": [
            {"strategy": strategy, "variant_id": variant, "seed": 100402}
            for strategy in ("fixed_predeclared_v1", "dynamic_evidence_v1")
            for variant in candidates
        ],
        "source_artifacts": [
            "mea/vqa_query_registry/reviewed/index.json",
            "mea/tool_registry/reviewed/index.json",
        ],
    }


def strategy_config() -> dict:
    return {
        "schema_version": 1,
        "plan_id": "pair_demo_n1",
        "evidence_manifest": "configs/evidence_manifest.json",
        "task_name": "click_bell",
        "model_profile": "legacy",
        "python_executable": "python",
        "gpu": 0,
        "reviewed_tool_registry": "mea/tool_registry/reviewed",
        "reviewed_vqa_registry": "mea/vqa_query_registry/reviewed",
    }


def materialize(root: Path) -> None:
    write(
        root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/policy_last.ckpt",
        b"policy",
    )
    write(
        root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50/dataset_stats.pkl",
        b"stats",
    )
    write(root / "mea/vqa_query_registry/reviewed/index.json", b'{"entries":[]}\n')
    write(root / "mea/tool_registry/reviewed/index.json", b'{"entries":[]}\n')
    write(
        root / "mea/toolkit/schemas/click_bell.json",
        b'{"task_name":"click_bell","task_family":"contact_trigger"}\n',
    )
    write(
        root / ".gitignore",
        b"policy/ACT/\nmea/evaluation_runs/\nmea/generated_tasks/\nmea/validation_runs/\nconfigs/evidence_manifest.json\nconfigs/strategy_plan.json\n",
    )
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "MEA Tests"], cwd=root, check=True
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    manifest = prepare_evidence_manifest(root, evidence_config(root))
    path = root / "configs/evidence_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


def bind_fake_run(root: Path, relative: str, registration: dict) -> None:
    parent_path = root / relative / "manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    parent["registration_identity"] = registration
    parent_path.write_text(json.dumps(parent), encoding="utf-8")
    summary = json.loads(
        (root / relative / "summary/summary.json").read_text(encoding="utf-8")
    )
    for round_value in summary["rounds"]:
        child_id = round_value["taskgen_run_id"]
        child_path = root / f"mea/generated_tasks/{child_id}/manifest.json"
        child = json.loads(child_path.read_text(encoding="utf-8"))
        child["registration_identity"] = registration
        child_path.write_text(json.dumps(child), encoding="utf-8")
        for episode_path in (
            root / f"mea/generated_tasks/{child_id}/evaluation/telemetry"
        ).rglob("episode.json"):
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            episode["registration_identity"] = registration
            episode_path.write_text(json.dumps(episode), encoding="utf-8")


def normalize_fake_child_ids(root: Path, relative: str) -> None:
    summary_path = root / relative / "summary/summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for round_value in summary["rounds"]:
        old_id = round_value["taskgen_run_id"]
        new_id = old_id.replace("run_eval_", "run_", 1)
        old_dir = root / f"mea/generated_tasks/{old_id}"
        new_dir = root / f"mea/generated_tasks/{new_id}"
        old_dir.rename(new_dir)
        child_path = new_dir / "manifest.json"
        child = json.loads(child_path.read_text(encoding="utf-8"))
        child["run_id"] = new_id
        child_path.write_text(json.dumps(child), encoding="utf-8")
        round_value["taskgen_run_id"] = new_id
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


class StrategyPlanTests(unittest.TestCase):
    def test_route_plan_hash_and_comparison_directory_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            plan = build_matched_strategy_plan(root, strategy_config())
            output = root / "mea/validation_runs/pair_demo_n1"
            output.mkdir(parents=True)
            plan_path = output / "command_plan.json"
            route_path = output / "registered_route.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            route_path.write_text(
                json.dumps(plan["registered_route"]["payload"]), encoding="utf-8"
            )

            tampered_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            tampered_plan["claim_scope"] = "changed"
            plan_path.write_text(json.dumps(tampered_plan), encoding="utf-8")
            with self.assertRaisesRegex(StrategyPlanError, "command plan hash"):
                load_registered_execution(
                    root,
                    evidence_manifest_path="configs/evidence_manifest.json",
                    command_plan_path="mea/validation_runs/pair_demo_n1/command_plan.json",
                    registered_route_path="mea/validation_runs/pair_demo_n1/registered_route.json",
                    strategy="fixed_predeclared_v1",
                    evaluation_id="eval_pair_demo_n1_fixed",
                )

            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            drifted_route = json.loads(route_path.read_text(encoding="utf-8"))
            drifted_route["selection"]["evaluation_goal"] = "router drift"
            payload = dict(drifted_route)
            payload.pop("integrity")
            drifted_route["integrity"]["canonical_payload_sha256"] = canonical_sha256(
                payload
            )
            route_path.write_text(json.dumps(drifted_route), encoding="utf-8")
            with self.assertRaisesRegex(StrategyPlanError, "selection drifted"):
                load_registered_execution(
                    root,
                    evidence_manifest_path="configs/evidence_manifest.json",
                    command_plan_path="mea/validation_runs/pair_demo_n1/command_plan.json",
                    registered_route_path="mea/validation_runs/pair_demo_n1/registered_route.json",
                    strategy="fixed_predeclared_v1",
                    evaluation_id="eval_pair_demo_n1_fixed",
                )

            route_path.write_text(
                json.dumps(plan["registered_route"]["payload"]), encoding="utf-8"
            )
            wrong_dirs = json.loads(json.dumps(plan))
            wrong_dirs["posthoc"]["comparison_config"][
                "dynamic_evaluation_dir"
            ] = "mea/evaluation_runs/eval_other"
            wrong_dirs.pop("plan_sha256")
            wrong_dirs["plan_sha256"] = canonical_sha256(wrong_dirs)
            with self.assertRaisesRegex(StrategyPlanError, "comparison directories"):
                compare_registered_strategies(root, wrong_dirs)

    def test_builds_inert_auditable_pair_and_posthoc_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            plan = build_matched_strategy_plan(root, strategy_config())
            self.assertEqual(plan["execution_status"], "planned_not_started")
            self.assertEqual(plan["act_rollouts_started"], 0)
            self.assertEqual(plan["provider_calls_started"], 0)
            self.assertEqual(plan["schedule"]["pair_max_act_rollouts"], 4)
            self.assertIsNone(plan["table2_consistency"])
            self.assertEqual(
                plan["posthoc"]["comparison_module"],
                "mea.strategy_comparison.compare_fixed_dynamic",
            )
            fixed = plan["strategies"]["fixed_predeclared_v1"]["argv"]
            dynamic = plan["strategies"]["dynamic_evidence_v1"]["argv"]
            self.assertIn("fixed_predeclared_v1", fixed)
            self.assertIn("dynamic_evidence_v1", dynamic)
            self.assertEqual(fixed[fixed.index("--start-seed") + 1], "100402")
            self.assertIn("--no-history", fixed)

    def test_unpinned_registry_and_nonuniform_cli_seed_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            bad = strategy_config()
            bad["reviewed_tool_registry"] = "unregistered"
            with self.assertRaisesRegex(StrategyPlanError, "not hash-pinned"):
                build_matched_strategy_plan(root, bad)

            manifest_path = root / "configs/evidence_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            # This second manifest is internally valid but uses a different
            # per-variant seed, which the current single --start-seed CLI
            # cannot execute exactly.
            variant = "object_position.right_fixed"
            custom = evidence_config(root)
            for item in custom["sample_schedule"]:
                if item["variant_id"] == variant:
                    item["seed"] = 100403
            manifest = prepare_evidence_manifest(root, custom)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(StrategyPlanError, "same N=1 start seed"):
                build_matched_strategy_plan(root, strategy_config())

    def test_cli_writes_commands_but_does_not_execute_them(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            config_path = root / "configs/strategy_plan.json"
            config_path.write_text(json.dumps(strategy_config()), encoding="utf-8")
            output = root / "mea/validation_runs/pair_demo_n1"
            process = subprocess.run(
                [
                    sys.executable,
                    str(
                        REPO_ROOT
                        / "experiments/paper/manipeval_plan_strategy_pair.py"
                    ),
                    "--repo-root",
                    str(root),
                    "--config",
                    str(config_path),
                    "--output-dir",
                    str(output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            summary = json.loads(process.stdout)
            self.assertEqual(summary["act_rollouts_started"], 0)
            self.assertEqual(summary["provider_calls_started"], 0)
            self.assertTrue((output / "command_plan.json").is_file())
            self.assertTrue((output / "strategy_comparison_config.json").is_file())
            self.assertTrue((output / "registered_route.json").is_file())
            report = (output / "commands.md").read_text(encoding="utf-8")
            self.assertIn("manipeval_compare_registered_strategies.py", report)
            self.assertIn("Table 2 consistency: unavailable", report)
            self.assertIn("set -euo pipefail", report)
            self.assertFalse((root / "mea/evaluation_runs/eval_pair_demo_n1_fixed").exists())
            plan = json.loads((output / "command_plan.json").read_text(encoding="utf-8"))
            fixed = plan["strategies"]["fixed_predeclared_v1"]["argv"]
            self.assertNotIn("--auto-route", fixed)
            self.assertIn("--registered-route", fixed)
            loaded = load_registered_execution(
                root,
                evidence_manifest_path="configs/evidence_manifest.json",
                command_plan_path="mea/validation_runs/pair_demo_n1/command_plan.json",
                registered_route_path="mea/validation_runs/pair_demo_n1/registered_route.json",
                strategy="fixed_predeclared_v1",
                evaluation_id="eval_pair_demo_n1_fixed",
                observed_argv=fixed[1:],
            )
            self.assertFalse(loaded["route"]["provider_called"])
            drifted_argv = list(fixed[1:]) + ["--auto-route"]
            with self.assertRaisesRegex(StrategyPlanError, "argv differs"):
                load_registered_execution(
                    root,
                    evidence_manifest_path="configs/evidence_manifest.json",
                    command_plan_path="mea/validation_runs/pair_demo_n1/command_plan.json",
                    registered_route_path="mea/validation_runs/pair_demo_n1/registered_route.json",
                    strategy="fixed_predeclared_v1",
                    evaluation_id="eval_pair_demo_n1_fixed",
                    observed_argv=drifted_argv,
                )

    def test_registered_posthoc_wrapper_reuses_strict_comparator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            materialize(root)
            plan = build_matched_strategy_plan(root, strategy_config())
            candidates = evidence_config(root)["candidate_suite"]
            fixed = make_run(
                root,
                "eval_pair_demo_n1_fixed",
                "fixed_predeclared_v1",
                candidates,
                candidates=candidates,
                seed=100402,
            )
            dynamic = make_run(
                root,
                "eval_pair_demo_n1_dynamic",
                "dynamic_evidence_v1",
                candidates[:1],
                candidates=candidates,
                seed=100402,
            )
            normalize_fake_child_ids(root, fixed)
            normalize_fake_child_ids(root, dynamic)
            for relative in (fixed, dynamic):
                path = root / relative / "manifest.json"
                value = json.loads(path.read_text(encoding="utf-8"))
                value["base_commit"] = git_head(root)
                value["user_request"] = evidence_config(root)["query"]
                path.write_text(json.dumps(value), encoding="utf-8")

            plan_path = root / "mea/validation_runs/pair_demo_n1/command_plan.json"
            route_path = root / "mea/validation_runs/pair_demo_n1/registered_route.json"
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            route_path.write_text(
                json.dumps(plan["registered_route"]["payload"]), encoding="utf-8"
            )
            for strategy, relative in (
                ("fixed_predeclared_v1", fixed),
                ("dynamic_evidence_v1", dynamic),
            ):
                registered = load_registered_execution(
                    root,
                    evidence_manifest_path="configs/evidence_manifest.json",
                    command_plan_path="mea/validation_runs/pair_demo_n1/command_plan.json",
                    registered_route_path="mea/validation_runs/pair_demo_n1/registered_route.json",
                    strategy=strategy,
                    evaluation_id=plan["strategies"][strategy]["evaluation_id"],
                )
                bind_fake_run(
                    root, relative, registered["registration_identity"]
                )
            result = compare_registered_strategies(root, plan)
            self.assertEqual(result["status"], "passed")
            self.assertTrue(result["registered_identity_match"])
            self.assertEqual(result["comparison"]["rollout_savings"], 1)
            self.assertIsNone(result["table2_consistency"])

            fixed_manifest_path = root / fixed / "manifest.json"
            fixed_manifest = json.loads(
                fixed_manifest_path.read_text(encoding="utf-8")
            )
            fixed_registration = dict(fixed_manifest["registration_identity"])
            fixed_manifest["registration_identity"]["command_plan_sha256"] = "0" * 64
            fixed_manifest_path.write_text(json.dumps(fixed_manifest), encoding="utf-8")
            with self.assertRaisesRegex(StrategyPlanError, "registration identity"):
                compare_registered_strategies(root, plan)
            bind_fake_run(root, fixed, fixed_registration)

            dynamic_summary_path = root / dynamic / "summary/summary.json"
            dynamic_summary = json.loads(
                dynamic_summary_path.read_text(encoding="utf-8")
            )
            original_variant = dynamic_summary["rounds"][0]["variant_id"]
            dynamic_summary["rounds"][0]["variant_id"] = candidates[1]
            dynamic_summary_path.write_text(
                json.dumps(dynamic_summary), encoding="utf-8"
            )
            with self.assertRaisesRegex(StrategyPlanError, "ordered prefix"):
                compare_registered_strategies(root, plan)
            dynamic_summary["rounds"][0]["variant_id"] = original_variant
            dynamic_summary_path.write_text(
                json.dumps(dynamic_summary), encoding="utf-8"
            )

            registered_output = root / "mea/validation_runs/pair_demo_n1/comparison"
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/manipeval_compare_registered_strategies.py"),
                    "--repo-root",
                    str(root),
                    "--command-plan",
                    str(plan_path),
                    "--output-dir",
                    str(registered_output),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertTrue(json.loads(process.stdout)["registered_identity_match"])
            self.assertTrue((registered_output / "report.md").is_file())

            for run_name in (
                "run_pair_demo_n1_fixed_1",
                "run_pair_demo_n1_dynamic_1",
            ):
                episode_path = (
                    root
                    / f"mea/generated_tasks/{run_name}"
                    / "evaluation/telemetry/act/episode_1/episode.json"
                )
                episode = json.loads(episode_path.read_text(encoding="utf-8"))
                episode["seed"] = 9
                episode_path.write_text(json.dumps(episode), encoding="utf-8")
            with self.assertRaisesRegex(StrategyPlanError, "preregistered N=1 schedule"):
                compare_registered_strategies(root, plan)


if __name__ == "__main__":
    unittest.main()
