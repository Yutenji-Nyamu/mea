import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from mea.execution_vqa import build_execution_vqa_query
from mea.scene_shift_vqa_validation import (
    CONDITION_CONTRACTS,
    PROTOCOL,
    SceneShiftVQAValidationError,
    summarize_scene_shift_vqa_suite,
    validate_scene_shift_vqa_suite,
)


class SceneShiftVQAValidationTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.reviewer_id = "codex-development-agent-proxy"
        self.cases = [
            self._make_case(
                condition="scene_background_texture.unseen",
                suffix="texture_positive",
                seed=12001,
                primary_label=True,
                primary_prediction=True,
            ),
            self._make_case(
                condition="scene_background_texture.unseen",
                suffix="texture_negative",
                seed=12002,
                primary_label=False,
                primary_prediction=False,
            ),
            self._make_case(
                condition="scene_lighting.static_random",
                suffix="lighting_positive",
                seed=13001,
                primary_label=True,
                primary_prediction=False,
            ),
            self._make_case(
                condition="scene_lighting.static_random",
                suffix="lighting_negative",
                seed=13002,
                primary_label=False,
                primary_prediction=False,
            ),
        ]
        self.suite = {
            "schema_version": 1,
            "suite_id": "sceneshiftvqa_click_bell_texture_light_dev_n2",
            "protocol": PROTOCOL,
            "reviewer": {
                "id": self.reviewer_id,
                "kind": "development_agent_proxy",
            },
            "cases": self.cases,
        }

    def tearDown(self):
        self._temporary.cleanup()

    @staticmethod
    def _digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_json(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def _ref(self, relative):
        path = self.root / relative
        return {"path": Path(relative).as_posix(), "sha256": self._digest(path)}

    def _make_case(
        self,
        *,
        condition,
        suffix,
        seed,
        primary_label,
        primary_prediction,
    ):
        contract = CONDITION_CONTRACTS[condition]
        run_id = f"run_{suffix}"
        evaluation_id = f"eval_{suffix}"
        episode_dir = (
            f"mea/generated_tasks/{run_id}/evaluation/telemetry/act/"
            f"episode_000_seed_{seed}"
        )
        episode_relative = f"{episode_dir}/episode.json"
        video_relative = f"{episode_dir}/video.mp4"
        video_path = self.root / video_relative
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(f"real-simulator-video-{suffix}".encode())
        self._write_json(
            episode_relative,
            {
                "task_name": "click_bell",
                "policy_name": "ACT",
                "seed": seed,
                "success": primary_prediction,
                "checkpoint_setting": "demo_clean",
                "telemetry_profile_id": "balanced_v1",
                "telemetry_profile_sha256": "balanced-profile-sha256",
            },
        )
        if condition == "scene_background_texture.unseen":
            randomization = {
                "random_background": True,
                "clean_background_rate": 0.0,
                "wall_texture": "unseen/3",
                "table_texture": "unseen/7",
                "texture_split": "unseen",
                "background_authority": "simulator_task_info:texture_info",
            }
        else:
            randomization = {
                "random_light": True,
                "crazy_random_light_rate": 0.0,
                "crazy_random_light": False,
                "direction_light_count": 1,
                "point_light_count": 2,
                "direction_light_colors": [[0.2, 0.4, 0.6]],
                "point_light_colors": [[0.1, 0.3, 0.5], [0.7, 0.8, 0.9]],
                "lighting_authority": (
                    "simulator_task_attributes:random_light,"
                    "crazy_random_light_rate,crazy_random_light;"
                    "simulator_light_components:get_color"
                ),
            }
        taskgen_manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "completed",
            "failure": None,
            "task_name": "click_bell",
            "task_module": "mea.tasks.click_bell",
            "mode": "reuse",
            "generation_kind": "bounded_variant_overlay",
            "variant_id": contract["template_id"],
            "capability_id": contract["capability_id"],
            "base_commit": "same-test-base-commit",
            "scene_validation": {
                "seed": seed,
                "eval_mode": condition == "scene_background_texture.unseen",
                "setup_success": True,
                "render_success": True,
                "rule_check": {"passed": True},
                "domain_randomization": randomization,
            },
            "position_samples": {
                "passed": True,
                "controlled_axis": contract["controlled_axis"],
                "variant_contract": contract["changes"],
            },
            "act_evaluation": {
                "passed": True,
                "task_name": "click_bell",
                "task_config": "demo_clean",
                "checkpoint_setting": "demo_clean",
                "num_episodes": 1,
                "actual_seeds": [seed],
                "video_associations": [
                    {
                        "episode_dir": episode_dir,
                        "video": video_relative,
                        "episode_index": 0,
                    }
                ],
            },
        }
        taskgen_relative = f"mea/generated_tasks/{run_id}/manifest.json"
        self._write_json(taskgen_relative, taskgen_manifest)
        evaluation_relative = (
            f"mea/evaluation_runs/{evaluation_id}/manifest.json"
        )
        self._write_json(
            evaluation_relative,
            {
                "evaluation_id": evaluation_id,
                "status": "completed",
                "lifecycle_status": "completed",
                "active_child_run_id": run_id,
                "task_name": "click_bell",
                "telemetry_profile": "balanced_v1",
                "base_commit": "same-test-base-commit",
            },
        )

        query = build_execution_vqa_query(
            task_name="click_bell",
            template_id=contract["template_id"],
            sub_aspect=contract["sub_aspect"],
            tool_contract={"metric": "official_check_success"},
        )
        self.assertEqual(query["phenomenon_ids"], contract["phenomenon_ids"])
        execution_dir = (
            f"mea/evaluation_runs/{evaluation_id}/execution/round_1/execution_vqa"
        )
        query_relative = f"{execution_dir}/execution_vqa_query.json"
        montage_relative = f"{execution_dir}/execution_montage.png"
        execution_relative = f"{execution_dir}/execution_vqa.json"
        self._write_json(query_relative, query)
        montage = self.root / montage_relative
        montage.parent.mkdir(parents=True, exist_ok=True)
        montage.write_bytes(f"real-rollout-montage-{suffix}".encode())
        predictions = {
            "bell_visibly_pressed": primary_prediction,
            contract["primary_visibility_phenomenon_id"]: primary_prediction,
        }
        artifact = {
            "schema_version": 1,
            "status": "passed",
            "selection": {
                "video_path": str(video_path.resolve()),
                "montage_path": str(montage.resolve()),
                "selected_frames": [
                    {"frame_id": "initial", "frame_index": 0},
                    {"frame_id": "final", "frame_index": 1},
                ],
            },
            "query": query,
            "observation": {
                "phenomena": [
                    {
                        "id": phenomenon_id,
                        "observed": predictions[phenomenon_id],
                        "description": f"Observed {phenomenon_id}.",
                        "confidence": 0.8,
                        "frame_ids": ["initial", "final"],
                    }
                    for phenomenon_id in contract["phenomenon_ids"]
                ],
                "confidence": 0.8,
                "frame_ids": ["initial", "final"],
                "numeric_consistency": "consistent",
                "conflicts": [],
            },
            "provider_metadata": {"model": "already-completed-model"},
            "artifacts": {
                "result": execution_relative,
                "query": query_relative,
                "montage": montage_relative,
            },
            "representative_episode": episode_dir,
        }
        self._write_json(execution_relative, artifact)
        labels = {
            "bell_visibly_pressed": primary_label,
            contract["primary_visibility_phenomenon_id"]: primary_label,
        }
        return {
            "id": suffix,
            "condition": condition,
            "seed": seed,
            "source_evaluation_id": evaluation_id,
            "sources": {
                "taskgen_manifest": self._ref(taskgen_relative),
                "evaluation_manifest": self._ref(evaluation_relative),
                "episode": self._ref(episode_relative),
                "video": self._ref(video_relative),
                "execution_vqa": self._ref(execution_relative),
                "query": self._ref(query_relative),
                "montage": self._ref(montage_relative),
            },
            "expected_query": {
                "task_name": "click_bell",
                "template_id": contract["template_id"],
                "sub_aspect": contract["sub_aspect"],
                "primary_visibility_phenomenon_id": contract[
                    "primary_visibility_phenomenon_id"
                ],
                "phenomenon_ids": contract["phenomenon_ids"],
            },
            "labels": [
                {
                    "phenomenon_id": phenomenon_id,
                    "observed": labels[phenomenon_id],
                    "label_source": "development_agent_proxy",
                    "reviewer_id": self.reviewer_id,
                }
                for phenomenon_id in contract["phenomenon_ids"]
            ],
        }

    def test_valid_suite_audits_four_real_source_cases(self):
        summary = summarize_scene_shift_vqa_suite(self.root, self.suite)
        self.assertEqual(summary["accuracy"], {"value": 6 / 8, "correct": 6, "total": 8})
        for condition in CONDITION_CONTRACTS:
            condition_summary = summary["by_condition"][condition]
            self.assertEqual(condition_summary["evaluation_count"], 2)
            self.assertEqual(
                condition_summary["primary_visibility_label_balance"],
                {"false": 1, "true": 1},
            )
            self.assertIsNone(condition_summary["auroc"])
        self.assertFalse(summary["paper_table_eligible"])
        self.assertEqual(summary["human_reviewer_count"], 0)
        self.assertFalse(summary["provider_called"])
        self.assertFalse(summary["simulator_called"])
        self.assertFalse(summary["act_called"])
        self.assertFalse(summary["image_proxy_used"])

    def test_earlier_round_child_is_bound_by_parent_child_list(self):
        case = self.cases[0]
        parent_path = self.root / case["sources"]["evaluation_manifest"]["path"]
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        first_child = parent["active_child_run_id"]
        parent["child_run_ids"] = [first_child, "run_later_scene_round"]
        parent["active_child_run_id"] = "run_later_scene_round"
        self._write_json(case["sources"]["evaluation_manifest"]["path"], parent)
        suite = deepcopy(self.suite)
        suite["cases"][0]["sources"]["evaluation_manifest"]["sha256"] = (
            self._digest(parent_path)
        )

        summary = summarize_scene_shift_vqa_suite(self.root, suite)
        self.assertEqual(len(summary["cases"]), 4)

        parent["child_run_ids"] = ["run_later_scene_round"]
        self._write_json(case["sources"]["evaluation_manifest"]["path"], parent)
        suite["cases"][0]["sources"]["evaluation_manifest"]["sha256"] = (
            self._digest(parent_path)
        )
        with self.assertRaisesRegex(
            SceneShiftVQAValidationError, "does not bind the TaskGen child"
        ):
            summarize_scene_shift_vqa_suite(self.root, suite)

    def test_each_condition_requires_two_cases_and_true_false_primary_labels(self):
        missing = deepcopy(self.suite)
        missing["cases"] = [
            case for case in missing["cases"] if case["id"] != "texture_negative"
        ]
        with self.assertRaisesRegex(
            SceneShiftVQAValidationError, "at least two completed cases"
        ):
            validate_scene_shift_vqa_suite(missing)

        no_negative = deepcopy(self.suite)
        primary = CONDITION_CONTRACTS[
            "scene_background_texture.unseen"
        ]["primary_visibility_phenomenon_id"]
        for case in no_negative["cases"]:
            if case["condition"] == "scene_background_texture.unseen":
                for label in case["labels"]:
                    if label["phenomenon_id"] == primary:
                        label["observed"] = True
        with self.assertRaisesRegex(
            SceneShiftVQAValidationError, "require both true and false"
        ):
            validate_scene_shift_vqa_suite(no_negative)

    def test_missing_or_tampered_hash_is_rejected(self):
        missing = deepcopy(self.suite)
        del missing["cases"][0]["sources"]["video"]["sha256"]
        with self.assertRaisesRegex(SceneShiftVQAValidationError, "sha256"):
            validate_scene_shift_vqa_suite(missing)

        tampered = self.root / self.cases[0]["sources"]["video"]["path"]
        tampered.write_bytes(b"tampered-video")
        with self.assertRaisesRegex(SceneShiftVQAValidationError, "does not match"):
            summarize_scene_shift_vqa_suite(self.root, self.suite)

    def test_vqa_result_must_self_bind_to_the_hashed_result(self):
        case = self.cases[0]
        result_path = self.root / case["sources"]["execution_vqa"]["path"]
        artifact = json.loads(result_path.read_text(encoding="utf-8"))
        artifact["artifacts"]["result"] = case["sources"]["query"]["path"]
        self._write_json(case["sources"]["execution_vqa"]["path"], artifact)
        suite = deepcopy(self.suite)
        suite["cases"][0]["sources"]["execution_vqa"]["sha256"] = self._digest(
            result_path
        )
        with self.assertRaisesRegex(
            SceneShiftVQAValidationError, "hashed result/query/montage"
        ):
            summarize_scene_shift_vqa_suite(self.root, suite)

    def test_vqa_observation_extensions_and_conflict_fail_closed(self):
        case = self.cases[0]
        result_path = self.root / case["sources"]["execution_vqa"]["path"]
        artifact = json.loads(result_path.read_text(encoding="utf-8"))
        artifact["observation"]["untrusted_extension"] = True
        self._write_json(case["sources"]["execution_vqa"]["path"], artifact)
        suite = deepcopy(self.suite)
        suite["cases"][0]["sources"]["execution_vqa"]["sha256"] = self._digest(
            result_path
        )
        with self.assertRaisesRegex(
            SceneShiftVQAValidationError, "observation has invalid fields"
        ):
            summarize_scene_shift_vqa_suite(self.root, suite)

        del artifact["observation"]["untrusted_extension"]
        artifact["observation"]["evidence_conflict"] = True
        self._write_json(case["sources"]["execution_vqa"]["path"], artifact)
        suite["cases"][0]["sources"]["execution_vqa"]["sha256"] = self._digest(
            result_path
        )
        with self.assertRaisesRegex(
            SceneShiftVQAValidationError, "evidence_conflict does not match"
        ):
            summarize_scene_shift_vqa_suite(self.root, suite)

    def test_escape_and_symlink_sources_are_rejected(self):
        escaped = deepcopy(self.suite)
        escaped["cases"][0]["sources"]["video"]["path"] = "../video.mp4"
        with self.assertRaisesRegex(SceneShiftVQAValidationError, "repo-relative"):
            summarize_scene_shift_vqa_suite(self.root, escaped)

        link = self.root / "video-link.mp4"
        try:
            link.symlink_to(self.root / self.cases[0]["sources"]["video"]["path"])
        except (OSError, NotImplementedError):
            return
        linked = deepcopy(self.suite)
        linked["cases"][0]["sources"]["video"] = {
            "path": "video-link.mp4",
            "sha256": self.cases[0]["sources"]["video"]["sha256"],
        }
        with self.assertRaisesRegex(SceneShiftVQAValidationError, "symlink"):
            summarize_scene_shift_vqa_suite(self.root, linked)

    def test_simulator_native_texture_and_lighting_authorities_are_required(self):
        texture_case = self.cases[0]
        texture_path = self.root / texture_case["sources"]["taskgen_manifest"]["path"]
        texture = json.loads(texture_path.read_text(encoding="utf-8"))
        texture["scene_validation"]["domain_randomization"]["wall_texture"] = "seen/3"
        self._write_json(texture_case["sources"]["taskgen_manifest"]["path"], texture)
        changed = deepcopy(self.suite)
        changed["cases"][0]["sources"]["taskgen_manifest"]["sha256"] = self._digest(
            texture_path
        )
        with self.assertRaisesRegex(
            SceneShiftVQAValidationError, "simulator-authoritative"
        ):
            summarize_scene_shift_vqa_suite(self.root, changed)

        # Restore before testing the independent lighting authority branch.
        texture["scene_validation"]["domain_randomization"]["wall_texture"] = "unseen/3"
        self._write_json(texture_case["sources"]["taskgen_manifest"]["path"], texture)
        lighting_case = self.cases[2]
        lighting_path = self.root / lighting_case["sources"]["taskgen_manifest"]["path"]
        lighting = json.loads(lighting_path.read_text(encoding="utf-8"))
        lighting["scene_validation"]["domain_randomization"][
            "lighting_authority"
        ] = "image_proxy"
        self._write_json(lighting_case["sources"]["taskgen_manifest"]["path"], lighting)
        changed = deepcopy(self.suite)
        changed["cases"][0]["sources"]["taskgen_manifest"]["sha256"] = self._digest(
            texture_path
        )
        changed["cases"][2]["sources"]["taskgen_manifest"]["sha256"] = self._digest(
            lighting_path
        )
        with self.assertRaisesRegex(
            SceneShiftVQAValidationError, "simulator-authoritative"
        ):
            summarize_scene_shift_vqa_suite(self.root, changed)

    def test_boolean_zero_scene_rates_are_rejected(self):
        for case_index, rate_field in (
            (0, "clean_background_rate"),
            (2, "crazy_random_light_rate"),
        ):
            case = self.cases[case_index]
            manifest_path = self.root / case["sources"]["taskgen_manifest"]["path"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            original = deepcopy(manifest)
            manifest["scene_validation"]["domain_randomization"][rate_field] = False
            self._write_json(case["sources"]["taskgen_manifest"]["path"], manifest)
            suite = deepcopy(self.suite)
            suite["cases"][case_index]["sources"]["taskgen_manifest"]["sha256"] = (
                self._digest(manifest_path)
            )
            with self.assertRaisesRegex(
                SceneShiftVQAValidationError, "simulator-authoritative"
            ):
                summarize_scene_shift_vqa_suite(self.root, suite)
            self._write_json(case["sources"]["taskgen_manifest"]["path"], original)

    def test_act_video_and_vqa_video_must_be_the_hashed_source(self):
        bad_association = deepcopy(self.suite)
        manifest_ref = bad_association["cases"][0]["sources"]["taskgen_manifest"]
        manifest_path = self.root / manifest_ref["path"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["act_evaluation"]["video_associations"][0]["video"] = "other.mp4"
        self._write_json(manifest_ref["path"], manifest)
        manifest_ref["sha256"] = self._digest(manifest_path)
        with self.assertRaisesRegex(SceneShiftVQAValidationError, "association"):
            summarize_scene_shift_vqa_suite(self.root, bad_association)

    def test_cli_writes_only_an_offline_summary(self):
        from scripts import manipeval_vqa_scene_shift_validate as cli

        suite_path = self._write_json("configs/scene_shift_suite.json", self.suite)
        output = Path("outputs/scene_shift_summary.json")
        with patch.object(
            sys,
            "argv",
            [
                "manipeval_vqa_scene_shift_validate.py",
                "--repo-root",
                str(self.root),
                "--suite",
                str(suite_path),
                "--output",
                str(output),
            ],
        ), redirect_stdout(io.StringIO()):
            cli.main()
        saved = json.loads((self.root / output).read_text(encoding="utf-8"))
        self.assertEqual(saved["mode"], "offline_completed_artifact_audit")
        self.assertFalse(saved["provider_called"])
        self.assertFalse(saved["simulator_called"])
        self.assertFalse(saved["act_called"])


if __name__ == "__main__":
    unittest.main()
