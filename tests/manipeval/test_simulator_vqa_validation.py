import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from mea.execution_vqa import build_execution_vqa_query
from mea.simulator_vqa_validation import (
    CLUTTER_ASPECT,
    CLUTTER_PHENOMENA,
    CLUTTER_TEMPLATE,
    CLEAN_ASPECT,
    CLEAN_PHENOMENA,
    CLEAN_TEMPLATE,
    PROTOCOL,
    SimulatorVQAValidationError,
    summarize_simulator_vqa_suite,
    validate_simulator_vqa_suite,
)


class SimulatorVQAValidationTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.reviewer_id = "codex-development-agent-proxy"
        self.seed = 100401
        bell_pose = [0.2234336882829666, -0.08040805160999298, 0.741]
        bell_quaternion = [0.5, 0.5, 0.5, 0.5]
        self.clean = self._make_case(
            condition="clean",
            evaluation_id="eval_real_clean_n1",
            run_id="run_real_clean_n1",
            template_id=CLEAN_TEMPLATE,
            sub_aspect=CLEAN_ASPECT,
            phenomenon_ids=list(CLEAN_PHENOMENA),
            predicted={"bell_visibly_pressed": True},
            labels={"bell_visibly_pressed": True},
            bell_pose=bell_pose,
            bell_quaternion=bell_quaternion,
        )
        self.clutter = self._make_case(
            condition="scene_clutter",
            evaluation_id="eval_real_clutter_n1",
            run_id="run_real_clutter_n1",
            template_id=CLUTTER_TEMPLATE,
            sub_aspect=CLUTTER_ASPECT,
            phenomenon_ids=list(CLUTTER_PHENOMENA),
            predicted={
                "bell_visibly_pressed": False,
                "bell_target_selected_among_clutter": True,
            },
            labels={
                "bell_visibly_pressed": False,
                "bell_target_selected_among_clutter": False,
            },
            bell_pose=bell_pose,
            bell_quaternion=bell_quaternion,
        )
        self.suite = {
            "schema_version": 1,
            "suite_id": "simvqa_click_bell_clean_clutter_n1",
            "protocol": PROTOCOL,
            "reviewer": {
                "id": self.reviewer_id,
                "kind": "development_agent_proxy",
            },
            "cases": [self.clean, self.clutter],
        }

    def tearDown(self):
        self._temporary.cleanup()

    def _write_json(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _make_case(
        self,
        *,
        condition,
        evaluation_id,
        run_id,
        template_id,
        sub_aspect,
        phenomenon_ids,
        predicted,
        labels,
        bell_pose,
        bell_quaternion,
    ):
        episode_dir = (
            f"mea/generated_tasks/{run_id}/evaluation/telemetry/act/"
            f"episode_000_seed_{self.seed}"
        )
        video = f"{episode_dir}/video.mp4"
        video_path = self.root / video
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"real-simulator-video")
        self._write_json(
            f"{episode_dir}/episode.json",
            {
                "task_name": "click_bell",
                "policy_name": "ACT",
                "seed": self.seed,
                "success": bool(predicted["bell_visibly_pressed"]),
                "checkpoint_setting": "demo_clean",
                "telemetry_profile_id": "balanced_v1",
                "telemetry_profile_sha256": "balanced-profile-sha256",
            },
        )
        if condition == "clean":
            randomization = {
                "cluttered_table": False,
                "clean_background_rate": 1.0,
                "cluttered_object_count": 0,
                "cluttered_objects": [],
                "authority": "simulator_task_info:cluttered_table_info",
            }
            manifest_identity = {
                "mode": "official",
                "generation_kind": "official_passthrough",
                "task_module": "envs.click_bell",
            }
            position_samples = {
                "status": "not_applicable",
                "passed": True,
                "samples": [],
                "metrics": {},
            }
        else:
            objects = [{"object_type": "021_cup", "object_index": 0}]
            randomization = {
                "cluttered_table": True,
                "clean_background_rate": 0.0,
                "cluttered_object_count": 1,
                "cluttered_objects": objects,
                "authority": "simulator_task_info:cluttered_table_info",
            }
            manifest_identity = {
                "mode": "reuse",
                "generation_kind": "bounded_variant_overlay",
                "task_module": "mea.tasks.click_bell",
                "variant_id": CLUTTER_TEMPLATE,
                "capability_id": CLUTTER_ASPECT,
            }
            position_samples = {
                "passed": True,
                "variant_contract": {
                    "domain_randomization": {
                        "cluttered_table": True,
                        "clean_background_rate": 0.0,
                    }
                },
                "samples": [{"clutter_count": 1, "clutter_matched": True}],
                "metrics": {
                    "expected_clutter": True,
                    "all_clutter_matched": True,
                },
            }
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "completed",
            "task_name": "click_bell",
            **manifest_identity,
            "scene_validation": {
                "seed": self.seed,
                "setup_success": True,
                "render_success": True,
                "rule_check": {"passed": True},
                "domain_randomization": randomization,
                "tracked_actors": [
                    {
                        "id": "bell",
                        "position": bell_pose,
                        "quaternion": bell_quaternion,
                    }
                ],
                "task_attributes": {"bell_id": 0},
            },
            "position_samples": position_samples,
            "act_evaluation": {
                "passed": True,
                "task_name": "click_bell",
                "task_config": "demo_clean",
                "checkpoint_setting": "demo_clean",
                "num_episodes": 1,
                "actual_seeds": [self.seed],
                "video_associations": [
                    {"episode_dir": episode_dir, "video": video, "episode_index": 0}
                ],
            },
            "failure": None,
        }
        manifest_relative = f"mea/generated_tasks/{run_id}/manifest.json"
        self._write_json(manifest_relative, manifest)
        self._write_json(
            f"mea/evaluation_runs/{evaluation_id}/manifest.json",
            {
                "evaluation_id": evaluation_id,
                "status": "completed",
                "lifecycle_status": "completed",
                "active_child_run_id": run_id,
                "task_name": "click_bell",
                "telemetry_profile": "balanced_v1",
                "base_commit": "test-base-commit",
            },
        )

        query = build_execution_vqa_query(
            task_name="click_bell",
            template_id=template_id,
            sub_aspect=sub_aspect,
            tool_contract={"metric": "official_check_success"},
        )
        self.assertEqual(query["phenomenon_ids"], phenomenon_ids)
        execution_dir = (
            f"mea/evaluation_runs/{evaluation_id}/execution/round_1/execution_vqa"
        )
        montage_relative = f"{execution_dir}/execution_montage.png"
        montage_path = self.root / montage_relative
        montage_path.parent.mkdir(parents=True, exist_ok=True)
        montage_path.write_bytes(b"real-rollout-montage")
        query_relative = f"{execution_dir}/execution_vqa_query.json"
        self._write_json(query_relative, query)
        selection = {
            "video_path": str((self.root / video).resolve()),
            "frame_count": 2,
            "fps": 10.0,
            "selected_frames": [
                {"frame_id": "initial", "frame_index": 0, "source": "boundary"},
                {"frame_id": "final", "frame_index": 1, "source": "boundary"},
            ],
            "reference_scene": None,
            "montage_path": str(montage_path.resolve()),
        }
        phenomena = [
            {
                "id": phenomenon_id,
                "observed": predicted[phenomenon_id],
                "description": f"Observed {phenomenon_id}.",
                "confidence": 0.8,
                "frame_ids": ["initial", "final"],
            }
            for phenomenon_id in phenomenon_ids
        ]
        artifact = {
            "schema_version": 1,
            "selection": selection,
            "query": query,
            "numeric_tool_results": [],
            "observation": {
                "phenomena": phenomena,
                "confidence": 0.8,
                "frame_ids": ["initial", "final"],
                "numeric_consistency": "consistent",
                "conflicts": [],
                "evidence_conflict": False,
            },
            "evidence_conflict": False,
            "provider_metadata": {"model": "already-completed-model"},
            "artifacts": {
                "result": f"{execution_dir}/execution_vqa.json",
                "montage": montage_relative,
                "query": query_relative,
            },
            "status": "passed",
            "representative_episode": episode_dir,
        }
        artifact_relative = f"{execution_dir}/execution_vqa.json"
        self._write_json(artifact_relative, artifact)
        return {
            "id": f"{condition}_n1",
            "condition": condition,
            "source_evaluation_id": evaluation_id,
            "source_execution_vqa": artifact_relative,
            "source_taskgen_manifest": manifest_relative,
            "source_montage": montage_relative,
            "seed": self.seed,
            "expected_query": {
                "task_name": "click_bell",
                "template_id": template_id,
                "sub_aspect": sub_aspect,
                "phenomenon_ids": phenomenon_ids,
            },
            "labels": [
                {
                    "phenomenon_id": phenomenon_id,
                    "observed": labels[phenomenon_id],
                    "label_source": "development_agent_proxy",
                    "reviewer_id": self.reviewer_id,
                }
                for phenomenon_id in phenomenon_ids
            ],
        }

    def test_valid_real_simulator_pair_aggregates_proxy_accuracy(self):
        summary = summarize_simulator_vqa_suite(self.root, self.suite)
        self.assertTrue(summary["target_identity"]["passed"])
        self.assertTrue(summary["protocol_identity"]["passed"])
        self.assertEqual(
            summary["accuracy"],
            {
                "value": 2 / 3,
                "correct": 2,
                "total": 3,
            },
        )
        self.assertEqual(summary["by_condition"]["clean"]["evaluation_count"], 1)
        self.assertIsNone(summary["by_condition"]["clean"]["auroc"])
        self.assertIsNone(summary["by_condition"]["scene_clutter"]["auroc"])
        self.assertIsNone(summary["auroc"])
        self.assertFalse(summary["paper_table_eligible"])
        self.assertEqual(summary["human_reviewer_count"], 0)
        self.assertFalse(summary["provider_called"])
        self.assertFalse(summary["simulator_called"])
        self.assertFalse(summary["image_proxy_used"])

    def test_suite_rejects_human_or_nonbinary_proxy_labels(self):
        human = deepcopy(self.suite)
        human["reviewer"]["kind"] = "human"
        with self.assertRaisesRegex(
            SimulatorVQAValidationError, "development_agent_proxy"
        ):
            validate_simulator_vqa_suite(human)
        nonbinary = deepcopy(self.suite)
        nonbinary["cases"][0]["labels"][0]["observed"] = None
        with self.assertRaisesRegex(SimulatorVQAValidationError, "must be boolean"):
            validate_simulator_vqa_suite(nonbinary)

    def test_query_or_montage_provenance_mismatch_is_rejected(self):
        bad_query = deepcopy(self.suite)
        bad_query["cases"][1]["expected_query"]["template_id"] = CLEAN_TEMPLATE
        with self.assertRaisesRegex(
            SimulatorVQAValidationError, "expected_query does not match"
        ):
            summarize_simulator_vqa_suite(self.root, bad_query)

        bad_montage = deepcopy(self.suite)
        other = self.root / "other_montage.png"
        other.write_bytes(b"image-proxy-like-substitute")
        bad_montage["cases"][0]["source_montage"] = "other_montage.png"
        with self.assertRaisesRegex(
            SimulatorVQAValidationError, "source montage paths"
        ):
            summarize_simulator_vqa_suite(self.root, bad_montage)

    def test_scene_clutter_requires_simulator_authority_and_objects(self):
        manifest_path = self.root / self.clutter["source_taskgen_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["scene_validation"]["domain_randomization"].update(
            {
                "cluttered_object_count": 0,
                "cluttered_objects": [],
            }
        )
        self._write_json(self.clutter["source_taskgen_manifest"], manifest)
        with self.assertRaisesRegex(
            SimulatorVQAValidationError, "does not prove scene_clutter"
        ):
            summarize_simulator_vqa_suite(self.root, self.suite)

    def test_parent_child_and_vqa_video_must_match(self):
        parent_path = (
            self.root
            / "mea/evaluation_runs/eval_real_clean_n1/manifest.json"
        )
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        parent["active_child_run_id"] = "run_wrong_child"
        self._write_json(
            "mea/evaluation_runs/eval_real_clean_n1/manifest.json", parent
        )
        with self.assertRaisesRegex(
            SimulatorVQAValidationError, "parent evaluation"
        ):
            summarize_simulator_vqa_suite(self.root, self.suite)

        parent["active_child_run_id"] = "run_real_clean_n1"
        self._write_json(
            "mea/evaluation_runs/eval_real_clean_n1/manifest.json", parent
        )
        vqa_path = self.root / self.clean["source_execution_vqa"]
        vqa = json.loads(vqa_path.read_text(encoding="utf-8"))
        alternate = self.root / "alternate_video.mp4"
        alternate.write_bytes(b"different-real-video")
        vqa["selection"]["video_path"] = str(alternate.resolve())
        self._write_json(self.clean["source_execution_vqa"], vqa)
        with self.assertRaisesRegex(
            SimulatorVQAValidationError, "selection video"
        ):
            summarize_simulator_vqa_suite(self.root, self.suite)

    def test_pair_rejects_changed_target_identity(self):
        manifest_path = self.root / self.clutter["source_taskgen_manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["scene_validation"]["tracked_actors"][0]["position"][0] += 0.01
        self._write_json(self.clutter["source_taskgen_manifest"], manifest)
        with self.assertRaisesRegex(
            SimulatorVQAValidationError, "target identity differs"
        ):
            summarize_simulator_vqa_suite(self.root, self.suite)

    def test_cli_writes_offline_summary_without_model_configuration(self):
        from scripts import manipeval_vqa_simulator_validate as cli

        suite_path = self._write_json("suite.json", self.suite)
        output_relative = Path("outputs/simulator_vqa_summary.json")
        with patch.object(
            sys,
            "argv",
            [
                "manipeval_vqa_simulator_validate.py",
                "--repo-root",
                str(self.root),
                "--suite",
                str(suite_path),
                "--output",
                str(output_relative),
            ],
        ), redirect_stdout(io.StringIO()):
            cli.main()
        summary = json.loads((self.root / output_relative).read_text(encoding="utf-8"))
        self.assertFalse(summary["provider_called"])
        self.assertFalse(summary["image_proxy_used"])
        self.assertEqual(summary["human_reviewer_count"], 0)


if __name__ == "__main__":
    unittest.main()
