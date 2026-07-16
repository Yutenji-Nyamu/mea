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


def official_round() -> dict:
    return {
        "round_id": "round_1",
        "template_id": "task_execution.official_baseline",
        "sub_aspect": "task_execution.official_baseline",
        "task_instruction": "evaluate click_bell",
        "task_name": "click_bell",
        "task_module": "envs.click_bell",
        "route": "official",
        "execution": {"seeds": [7], "num_episodes": 1},
        "tool_request": {
            "schema_version": 1,
            "task_name": "click_bell",
            "metric": "official_check_success",
            "question": "Did the bell task succeed?",
        },
    }


class CrossTaskEntrypointTests(unittest.TestCase):
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
                evidence["observations"]["execution_backends"], ["expert"]
            )
            self.assertIsNone(evidence["observations"]["act_pipeline_status"])

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
            with patch(
                "scripts.manipeval_taskgen.run_command", return_value=0
            ) as invoked:
                with self.assertRaises(RuntimeError):
                    run_act(
                        root,
                        run_dir,
                        {"task_module": "mea.tasks.beat_block_hammer"},
                        seed=7,
                        gpu=0,
                        num_episodes=1,
                        telemetry_profile="legacy_v1",
                    )
            command = invoked.call_args.args[0]
            self.assertEqual(command[-1], "legacy_v1")


if __name__ == "__main__":
    unittest.main()
