import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.planner import OfficialTaskPlanAgent
from mea.taskgen import create_official_task_run
from scripts.manipeval_agent import (
    build_evidence_bundle,
    build_taskgen_command,
    finish_unsupported_global_route,
    run_round_execution_vqa,
    summarize_round,
)
from scripts.manipeval_taskgen import (
    run_act,
    run_official_expert_episodes,
    run_probe,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_schema_repo(root: Path) -> None:
    (root / "envs").mkdir(parents=True)
    (root / "envs/click_bell.py").write_text(
        "class click_bell:\n    pass\n", encoding="utf-8"
    )
    schema_dir = root / "mea/toolkit/schemas"
    schema_dir.mkdir(parents=True)
    shutil.copy2(
        REPO_ROOT / "mea/toolkit/schemas/click_bell.json",
        schema_dir / "click_bell.json",
    )


def official_round(execution_backend: str | None = None) -> dict:
    execution = {"seeds": [7], "num_episodes": 1}
    if execution_backend is not None:
        execution["backend"] = execution_backend
    return {
        "round_id": "round_1",
        "template_id": "task_execution.official_baseline",
        "sub_aspect": "task_execution.official_baseline",
        "task_instruction": "evaluate click_bell",
        "task_name": "click_bell",
        "task_module": "envs.click_bell",
        "route": "official",
        "execution": execution,
        "tool_request": {
            "schema_version": 1,
            "task_name": "click_bell",
            "metric": "official_check_success",
            "question": "Did the bell task succeed?",
        },
    }


class CrossTaskEntrypointTests(unittest.TestCase):
    def test_auto_route_rejects_task_module_override_before_provider_setup(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment.pop("UIUI_API_KEY", None)
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/manipeval_agent.py"),
                    "--repo-root",
                    temporary,
                    "--request",
                    "evaluate a bell",
                    "--auto-route",
                    "--task-module",
                    "envs.click_bell",
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("do not pass --task-module", process.stderr)

    def test_unsupported_global_route_rejects_path_escape_evaluation_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, "evaluation_id"):
                finish_unsupported_global_route(
                    root,
                    evaluation_id="../escape",
                    user_request="unsupported query",
                    catalog={},
                    route_result={},
                    router=object(),
                    history_retrieval={},
                )
            self.assertFalse((root / "mea/escape").exists())

    def test_official_plan_only_does_not_require_provider_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema_dir = root / "mea/toolkit/schemas"
            schema_dir.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "mea/toolkit/schemas/click_bell.json",
                schema_dir / "click_bell.json",
            )
            environment = dict(os.environ)
            environment.pop("UIUI_API_KEY", None)
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/manipeval_agent.py"),
                    "--repo-root",
                    str(root),
                    "--request",
                    "evaluate click_bell",
                    "--task-name",
                    "click_bell",
                    "--evaluation-id",
                    "eval_click_bell_no_key",
                    "--plan-only",
                    "--no-history",
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            plan = json.loads(process.stdout)
            self.assertEqual(plan["task_name"], "click_bell")

    def test_bound_claim_first_plan_only_is_providerless_control_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema_dir = root / "mea/toolkit/schemas"
            schema_dir.mkdir(parents=True)
            shutil.copy2(
                REPO_ROOT / "mea/toolkit/schemas/click_bell.json",
                schema_dir / "click_bell.json",
            )
            checkpoint_dir = (
                root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            environment = dict(os.environ)
            environment.pop("UIUI_API_KEY", None)
            process = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/manipeval_agent.py"),
                    "--repo-root",
                    str(root),
                    "--request",
                    "Where does this policy first expose a weakness?",
                    "--open-query-planner",
                    "claim_first_v1",
                    "--bound-task-name",
                    "click_bell",
                    "--generated-rounds",
                    "2",
                    "--evaluation-id",
                    "eval_claim_first_bound_plan_only",
                    "--plan-only",
                    "--no-history",
                ],
                cwd=REPO_ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            plan = json.loads(process.stdout)
            self.assertEqual(plan["task_name"], "click_bell")
            self.assertEqual(
                plan["rounds"][0]["template_id"],
                "performance.completion_time_stability.official",
            )
            manifest = json.loads(
                (
                    root
                    / "mea/evaluation_runs/eval_claim_first_bound_plan_only/"
                    "manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["planner"]["public_planner"],
                "ClaimFirstOpenQueryAgent",
            )
            self.assertFalse(manifest["planner"]["provider_called"])

    def test_official_task_run_records_no_codegen(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_schema_repo(root)
            manifest = create_official_task_run(
                root,
                "evaluate the official bell task",
                task_name="click_bell",
                run_id="run_click_bell_test",
                telemetry_profile="balanced_v1",
            )
            run_dir = root / "mea/generated_tasks/run_click_bell_test"
            self.assertEqual(manifest["mode"], "official")
            self.assertEqual(manifest["task_module"], "envs.click_bell")
            self.assertFalse(manifest["provider"]["called"])
            self.assertFalse(
                manifest["static_validation"]["code_generation"]["performed"]
            )
            self.assertEqual(
                (run_dir / "overlay.yml").read_text(encoding="utf-8"), "{}\n"
            )
            self.assertTrue(
                (run_dir / "generation/official_source.json").is_file()
            )
            bundle = json.loads(
                (run_dir / "generation/task_artifact_bundle.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(bundle["scene_method"]["origin"], "official_reuse")
            self.assertEqual(bundle["success_method"]["origin"], "official_reuse")

    def test_official_planner_materializes_one_expert_round(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_schema_repo(root)
            planner = OfficialTaskPlanAgent(
                root,
                task_name="click_bell",
                start_seed=10,
                num_episodes=2,
                telemetry_profile="legacy_v1",
            )
            manifest = planner.plan(
                "evaluate click_bell",
                evaluation_id="eval_click_bell_test",
            )
            plan = manifest["plan"]
            round_plan = plan["rounds"][0]
            self.assertEqual(plan["policy"]["name"], "expert")
            self.assertEqual(round_plan["execution"]["seeds"], [10, 11])
            self.assertEqual(round_plan["route"], "official")
            self.assertEqual(round_plan["telemetry_profile"], "legacy_v1")
            updated, decision = planner.decide_next_round(
                evaluation_id="eval_click_bell_test",
                user_request="evaluate click_bell",
                current_plan=plan,
                observation_history=[
                    {"round_id": "round_1", "pipeline_passed": True}
                ],
            )
            self.assertEqual(decision["action"], "stop")
            self.assertEqual(updated["planning_state"], "stopped_after_round_1")

    def test_official_planner_materializes_requested_execution_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_schema_repo(root)
            for backend in ("act", "both"):
                planner = OfficialTaskPlanAgent(
                    root,
                    task_name="click_bell",
                    start_seed=20,
                    num_episodes=2,
                    execution_backend=backend,
                )
                manifest = planner.plan(
                    "evaluate click_bell",
                    evaluation_id=f"eval_click_bell_{backend}",
                )
                plan = manifest["plan"]
                round_plan = plan["rounds"][0]
                self.assertEqual(round_plan["route"], "official")
                self.assertEqual(round_plan["execution"]["backend"], backend)
                self.assertEqual(round_plan["execution"]["seeds"], [20, 21])
                self.assertIn("act", round_plan["execution"]["gates"])
                self.assertEqual(plan["policy"]["name"], "ACT")
                if backend == "both":
                    self.assertIn("expert", round_plan["execution"]["gates"])

    def test_official_command_uses_expert_probe_without_act_or_codegen_vqa(self):
        command, _ = build_taskgen_command(
            Path("/repo"),
            "eval_click",
            official_round(),
            text_model="text",
            vision_model="vision",
            base_url=None,
            gpu=0,
            max_reflections=2,
            telemetry_profile="legacy_v1",
        )
        self.assertIn("official", command)
        self.assertIn("envs.click_bell", command)
        self.assertIn("legacy_v1", command)
        self.assertIn("--expert", command)
        self.assertNotIn("--run-act", command)
        self.assertNotIn("--vision-check", command)

    def test_official_command_flags_follow_execution_backend(self):
        expected = {
            "expert": {"--expert"},
            "act": {"--run-act"},
            "both": {"--expert", "--run-act"},
        }
        for backend, expected_flags in expected.items():
            with self.subTest(backend=backend):
                command, _ = build_taskgen_command(
                    Path("/repo"),
                    f"eval_click_{backend}",
                    official_round(backend),
                    text_model="text",
                    vision_model="vision",
                    base_url=None,
                    gpu=0,
                    max_reflections=2,
                )
                actual_flags = {
                    flag
                    for flag in ("--expert", "--run-act")
                    if flag in command
                }
                self.assertEqual(actual_flags, expected_flags)
                self.assertNotIn("--vision-check", command)

    def test_click_bell_vqa_query_is_saved_even_when_execution_vqa_skips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child = root / "mea/generated_tasks/run_click"
            execution = root / "mea/evaluation_runs/e/execution/round_1"
            child.mkdir(parents=True)
            result = run_round_execution_vqa(
                repo_root=root,
                child_manifest={"task_name": "click_bell"},
                child_dir=child,
                tool_evaluation=None,
                execution_dir=execution,
                provider=object(),
                model="vision",
                round_plan=official_round(),
            )
            query = json.loads(
                (execution / "execution_vqa_query.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["status"], "skipped")
            self.assertEqual(query["phenomenon_ids"], ["bell_visibly_pressed"])

    def test_official_summary_uses_its_declared_gates(self):
        round_plan = official_round()
        child_manifest = {
            "run_id": "run_click",
            "status": "completed_without_act",
            "scene_validation": {
                "render_success": True,
                "rule_check": {"passed": True},
                "expert": {"passed": True},
                "expert_batch": {"passed": True},
            },
            "trusted_tool_evaluation": {"episode_count": 1, "episodes": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            child_dir = root / "mea/generated_tasks/run_click"
            (child_dir / "evaluation").mkdir(parents=True)
            summary = summarize_round(
                round_plan,
                child_manifest,
                child_dir,
                {"status": "passed", "episodes": []},
                {"status": "passed", "metrics": []},
                {"status": "skipped", "evidence_conflict": False},
                0,
            )
            evidence = build_evidence_bundle(
                root,
                "eval_click",
                "evaluate click_bell",
                {
                    "max_rounds": 1,
                    "requested_template_ids": [round_plan["template_id"]],
                    "planning_state": "stopped_after_round_1",
                    "round_decisions": [],
                },
                [
                    {
                        "round_plan": round_plan,
                        "child_manifest": child_manifest,
                        "child_dir": child_dir,
                        "round_summary": summary,
                        "tool_evaluation": {"status": "passed"},
                    }
                ],
            )
            self.assertTrue(summary["pipeline_passed"])
            self.assertEqual(
                summary["observations"]["execution_backend"], "expert"
            )
            self.assertIsNone(summary["observations"]["act_pipeline_status"])
            self.assertIsNone(summary["observations"]["policy_success"])
            self.assertEqual(
                summary["observations"]["scene_clutter"],
                {
                    "expected": False,
                    "counts": [],
                    "all_matched": None,
                    "authority": None,
                },
            )
            self.assertEqual(
                evidence["observations"]["execution_backends"], ["expert"]
            )
            self.assertIsNone(evidence["observations"]["act_pipeline_status"])

    def test_official_act_policy_failure_is_not_pipeline_failure(self):
        round_plan = official_round("act")
        child_manifest = {
            "run_id": "run_click_act",
            "status": "completed",
            "scene_validation": {
                "render_success": True,
                "rule_check": {"passed": True},
            },
            "act_evaluation": {"passed": True, "actual_seeds": [8]},
            "trusted_tool_evaluation": {
                "episode_count": 1,
                "episodes": [],
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            child_dir = Path(temporary) / "mea/generated_tasks/run_click_act"
            evaluation_dir = child_dir / "evaluation"
            evaluation_dir.mkdir(parents=True)
            (evaluation_dir / "_result.txt").write_text(
                "0.0\n", encoding="utf-8"
            )
            summary = summarize_round(
                round_plan,
                child_manifest,
                child_dir,
                {"status": "passed", "episodes": []},
                {"status": "passed", "metrics": []},
                {"status": "passed", "evidence_conflict": False},
                0,
            )

        self.assertTrue(summary["pipeline_passed"])
        self.assertEqual(summary["observations"]["execution_backend"], "ACT")
        self.assertTrue(summary["observations"]["act_pipeline_status"])
        self.assertEqual(summary["observations"]["policy_success"], 0.0)
        self.assertIsNone(summary["observations"]["expert_solvable"])
        self.assertEqual(summary["observations"]["requested_seeds"], [7])
        self.assertEqual(summary["observations"]["actual_seeds"], [8])

    def test_official_episode_index_is_forwarded_to_recorder_probe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click"
            scenes = [
                {
                    "setup_success": True,
                    "render_success": True,
                    "rule_check": {"passed": True},
                    "expert": {"passed": True},
                    "image": f"image-{index}",
                    "telemetry": {
                        "episode_dir": f"episode-{index}",
                        "metadata": {
                            "artifacts": {"video": "video.mp4"},
                            "visual_capture": {
                                "profile_id": "event_keyframes_v1",
                                "status": "completed",
                            },
                        },
                    },
                }
                for index in range(2)
            ]
            with patch(
                "scripts.manipeval_taskgen.run_probe",
                side_effect=scenes,
            ) as probe:
                result = run_official_expert_episodes(
                    root,
                    run_dir,
                    {"task_name": "click_bell"},
                    start_seed=10,
                    num_episodes=2,
                    telemetry_profile="balanced_v1",
                )
            self.assertTrue(result["expert_batch"]["passed"])
            self.assertEqual(
                [call.kwargs["episode_index"] for call in probe.call_args_list],
                [0, 1],
            )
            self.assertEqual(
                [
                    call.kwargs["visual_capture_profile_id"]
                    for call in probe.call_args_list
                ],
                ["event_keyframes_v1", "event_keyframes_v1"],
            )
            self.assertEqual(
                result["expert_batch"]["episodes"][0]["video"],
                str(Path("episode-0") / "video.mp4"),
            )
            self.assertEqual(result["expert_batch"]["rejected_seed_count"], 0)

    def test_official_expert_skips_unstable_seed_with_audit_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_adjust"

            def accepted_scene(index):
                return {
                    "returncode": 0,
                    "setup_success": True,
                    "render_success": True,
                    "rule_check": {"passed": True},
                    "expert": {"passed": True},
                    "image": f"image-{index}",
                    "telemetry": {
                        "episode_dir": f"episode-{index}",
                        "metadata": {
                            "artifacts": {"video": "video.mp4"},
                            "visual_capture": {"status": "completed"},
                        },
                    },
                }

            scenes = [
                {
                    "returncode": 1,
                    "error": {
                        "type": "UnStableError",
                        "message": "bottle unstable",
                    },
                },
                accepted_scene(0),
                accepted_scene(1),
            ]
            with patch(
                "scripts.manipeval_taskgen.run_probe", side_effect=scenes
            ) as probe:
                result = run_official_expert_episodes(
                    root,
                    run_dir,
                    {"task_name": "adjust_bottle"},
                    start_seed=100,
                    num_episodes=2,
                    telemetry_profile="balanced_v1",
                    max_seed_candidates=3,
                )

            self.assertEqual(
                [call.kwargs["seed"] for call in probe.call_args_list],
                [100, 101, 102],
            )
            self.assertEqual(
                [call.kwargs["episode_index"] for call in probe.call_args_list],
                [0, 0, 1],
            )
            self.assertTrue(
                all(
                    call.kwargs["raise_on_failure"] is False
                    for call in probe.call_args_list
                )
            )
            self.assertEqual(result["expert_batch"]["episode_count"], 2)
            self.assertEqual(result["expert_batch"]["candidate_count"], 3)
            self.assertEqual(result["expert_batch"]["rejected_seed_count"], 1)
            self.assertEqual(
                result["expert_batch"]["rejected_seeds"][0]["seed"], 100
            )

    def test_official_expert_skips_unsolvable_seed_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click"
            accepted = {
                "returncode": 0,
                "setup_success": True,
                "render_success": True,
                "rule_check": {"passed": True},
                "expert": {"passed": True},
                "telemetry": {"episode_dir": "expert/accepted", "metadata": {}},
            }
            with patch(
                "scripts.manipeval_taskgen.run_probe",
                side_effect=[
                    {"returncode": 2, "expert": {"passed": False}},
                    accepted,
                ],
            ) as probe:
                result = run_official_expert_episodes(
                    root,
                    run_dir,
                    {"task_name": "click_bell"},
                    start_seed=7,
                    num_episodes=1,
                    telemetry_profile="balanced_v1",
                    max_seed_candidates=2,
                )
            self.assertEqual(result["expert_batch"]["episodes"][0]["seed"], 8)
            self.assertEqual(
                result["expert_batch"]["rejected_seeds"][0]["reason"],
                "expert_unsolvable",
            )
            self.assertTrue(
                all(
                    call.kwargs["max_expert_attempts"] == 1
                    for call in probe.call_args_list
                )
            )

    def test_probe_command_forwards_visual_capture_only_when_requested(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click"
            (run_dir / "validation").mkdir(parents=True)
            manifest = {
                "task_name": "click_bell",
                "task_module": "envs.click_bell",
            }
            with patch(
                "scripts.manipeval_taskgen.run_command", return_value=0
            ) as invoked:
                run_probe(
                    root,
                    run_dir,
                    manifest,
                    seed=7,
                    expert=True,
                    visual_capture_profile_id="event_keyframes_v1",
                )
                visual_command = invoked.call_args.args[0]
                run_probe(
                    root,
                    run_dir,
                    manifest,
                    seed=8,
                    expert=False,
                )
                default_command = invoked.call_args.args[0]
            flag_index = visual_command.index("--visual-capture-profile")
            self.assertEqual(
                visual_command[flag_index + 1], "event_keyframes_v1"
            )
            self.assertNotIn("--visual-capture-profile", default_command)

    def test_act_wrapper_receives_telemetry_profile_as_twelfth_argument(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_act_profile"
            (run_dir / "evaluation").mkdir(parents=True)
            checkpoint_dir = (
                root
                / "policy/ACT/act_ckpt/act-beat_block_hammer/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            with patch(
                "scripts.manipeval_taskgen.run_command", return_value=0
            ) as invoked:
                with self.assertRaises(RuntimeError):
                    run_act(
                        root,
                        run_dir,
                        {
                            "task_name": "beat_block_hammer",
                            "task_module": "mea.tasks.beat_block_hammer",
                        },
                        seed=7,
                        gpu=0,
                        num_episodes=1,
                        telemetry_profile="legacy_v1",
                    )
            command = invoked.call_args.args[0]
            self.assertEqual(command[-1], "legacy_v1")

    def test_act_wrapper_uses_click_bell_checkpoint_and_eval_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click_act"
            (run_dir / "evaluation").mkdir(parents=True)
            checkpoint_dir = (
                root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            eval_root = root / "eval_result/click_bell/ACT/demo_clean/demo_clean"
            telemetry_episode = (
                run_dir / "evaluation/telemetry/act/episode_000_seed_7"
            )

            def fake_run(command, *, cwd, log_path):
                self.assertEqual(cwd, root)
                output = eval_root / "mock_eval"
                output.mkdir(parents=True)
                (output / "episode0.mp4").write_bytes(b"video")
                (output / "_result.txt").write_text(
                    "1.0\n", encoding="utf-8"
                )
                telemetry_episode.mkdir(parents=True)
                (telemetry_episode / "episode.json").write_text(
                    json.dumps({"seed": 7}), encoding="utf-8"
                )
                return 0

            with patch(
                "scripts.manipeval_taskgen.run_command",
                side_effect=fake_run,
            ) as invoked:
                result = run_act(
                    root,
                    run_dir,
                    {
                        "task_name": "click_bell",
                        "task_module": "envs.click_bell",
                    },
                    seed=7,
                    gpu=0,
                    num_episodes=1,
                    telemetry_profile="legacy_v1",
                )

            command = invoked.call_args.args[0]
            self.assertEqual(
                command[4:10],
                ["click_bell", "demo_clean", "demo_clean", "50", "0", "0"],
            )
            self.assertEqual(command[11], "envs.click_bell")
            self.assertEqual(command[-1], "legacy_v1")
            self.assertTrue(result["passed"])
            self.assertEqual(result["task_name"], "click_bell")
            self.assertEqual(result["actual_seeds"], [7])
            self.assertTrue(result["checkpoint"]["preflight_passed"])
            metadata = json.loads(
                (telemetry_episode / "episode.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["artifacts"]["video"], "video.mp4")

    def test_act_checkpoint_preflight_fails_before_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click_act"
            (run_dir / "evaluation").mkdir(parents=True)
            with patch("scripts.manipeval_taskgen.run_command") as invoked:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "ACT checkpoint preflight failed for click_bell",
                ):
                    run_act(
                        root,
                        run_dir,
                        {
                            "task_name": "click_bell",
                            "task_module": "envs.click_bell",
                        },
                        seed=7,
                        gpu=0,
                        num_episodes=1,
                    )
            invoked.assert_not_called()

    def test_act_video_association_uses_numeric_episode_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "mea/generated_tasks/run_click_act"
            (run_dir / "evaluation").mkdir(parents=True)
            checkpoint_dir = (
                root / "policy/ACT/act_ckpt/act-click_bell/demo_clean-50"
            )
            checkpoint_dir.mkdir(parents=True)
            (checkpoint_dir / "policy_last.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "dataset_stats.pkl").write_bytes(b"stats")
            eval_root = root / "eval_result/click_bell/ACT/demo_clean/demo_clean"

            def fake_run(command, *, cwd, log_path):
                output = eval_root / "mock_eval"
                output.mkdir(parents=True)
                (output / "episode2.mp4").write_bytes(b"video-two")
                (output / "episode10.mp4").write_bytes(b"video-ten")
                (output / "_result.txt").write_text("0.5\n", encoding="utf-8")
                for index, seed in ((2, 22), (10, 110)):
                    episode = (
                        run_dir
                        / "evaluation/telemetry/act"
                        / f"episode_{index:03d}_seed_{seed}"
                    )
                    episode.mkdir(parents=True)
                    (episode / "episode.json").write_text(
                        json.dumps({"seed": seed}), encoding="utf-8"
                    )
                return 0

            with patch(
                "scripts.manipeval_taskgen.run_command",
                side_effect=fake_run,
            ):
                result = run_act(
                    root,
                    run_dir,
                    {
                        "task_name": "click_bell",
                        "task_module": "envs.click_bell",
                    },
                    seed=7,
                    gpu=0,
                    num_episodes=2,
                )
            self.assertTrue(result["episode_index_alignment"]["passed"])
            self.assertEqual(result["actual_seeds"], [22, 110])
            self.assertEqual(
                (
                    run_dir
                    / "evaluation/telemetry/act/episode_002_seed_22/video.mp4"
                ).read_bytes(),
                b"video-two",
            )
            self.assertEqual(
                (
                    run_dir
                    / "evaluation/telemetry/act/episode_010_seed_110/video.mp4"
                ).read_bytes(),
                b"video-ten",
            )


if __name__ == "__main__":
    unittest.main()
