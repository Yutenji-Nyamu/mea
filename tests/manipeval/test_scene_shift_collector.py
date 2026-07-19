import json
import tempfile
import unittest
from pathlib import Path

from mea.execution_vqa import build_execution_vqa_query
from mea.scene_shift_collector import collect_scene_shift_candidates
from mea.scene_shift_vqa_validation import CONDITION_CONTRACTS


class SceneShiftCollectorTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        (self.root / "mea/evaluation_runs").mkdir(parents=True)

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

    def _make_round(self, evaluation_id, round_number, condition, seed):
        contract = CONDITION_CONTRACTS[condition]
        round_id = f"round_{round_number}"
        run_id = f"run_{evaluation_id.removeprefix('eval_')}_{round_id}"
        episode_dir = (
            f"mea/generated_tasks/{run_id}/evaluation/telemetry/act/"
            f"episode_000_seed_{seed}"
        )
        episode_path = f"{episode_dir}/episode.json"
        video_path = f"{episode_dir}/video.mp4"
        self._write_json(
            episode_path,
            {
                "task_name": "click_bell",
                "policy_name": "ACT",
                "seed": seed,
                "success": True,
                "checkpoint_setting": "demo_clean",
                "telemetry_profile_id": "balanced_v1",
                "telemetry_profile_sha256": "balanced-profile-hash",
            },
        )
        video = self.root / video_path
        video.write_bytes(f"video-{condition}-{seed}".encode())
        if condition == "scene_background_texture.unseen":
            randomization = {
                "random_background": True,
                "clean_background_rate": 0.0,
                "texture_split": "unseen",
                "wall_texture": "unseen/2",
                "table_texture": "unseen/5",
                "background_authority": "simulator_task_info:texture_info",
            }
        else:
            randomization = {
                "random_light": True,
                "crazy_random_light_rate": 0.0,
                "crazy_random_light": False,
                "direction_light_count": 1,
                "point_light_count": 2,
                "direction_light_colors": [[0.1, 0.2, 0.3]],
                "point_light_colors": [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                "lighting_authority": (
                    "simulator_task_attributes:random_light,"
                    "crazy_random_light_rate,crazy_random_light;"
                    "simulator_light_components:get_color"
                ),
            }
        self._write_json(
            f"mea/generated_tasks/{run_id}/manifest.json",
            {
                "schema_version": 1,
                "run_id": run_id,
                "status": "completed",
                "failure": None,
                "task_name": "click_bell",
                "task_module": "mea.tasks.click_bell",
                "mode": "reuse",
                "generation_kind": "bounded_variant_overlay",
                "variant_id": condition,
                "capability_id": contract["capability_id"],
                "base_commit": "same-commit",
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
                    "actual_seeds": [seed],
                    "video_associations": [
                        {
                            "episode_dir": episode_dir,
                            "video": video_path,
                            "episode_index": 0,
                        }
                    ],
                },
            },
        )
        execution_dir = f"mea/evaluation_runs/{evaluation_id}/execution/{round_id}"
        query = build_execution_vqa_query(
            task_name="click_bell",
            template_id=condition,
            sub_aspect=contract["sub_aspect"],
            tool_contract={"metric": "official_check_success"},
        )
        query_path = f"{execution_dir}/execution_vqa_query.json"
        montage_path = f"{execution_dir}/execution_vqa/execution_montage.png"
        result_path = f"{execution_dir}/execution_vqa/execution_vqa.json"
        self._write_json(query_path, query)
        montage = self.root / montage_path
        montage.parent.mkdir(parents=True, exist_ok=True)
        montage.write_bytes(f"montage-{condition}-{seed}".encode())
        self._write_json(
            result_path,
            {
                "schema_version": 1,
                "status": "passed",
                "selection": {
                    "video_path": video_path,
                    "montage_path": montage_path,
                    "selected_frames": [
                        {"frame_id": "initial", "frame_index": 0},
                        {"frame_id": "final", "frame_index": 1},
                    ],
                },
                "query": query,
                "observation": {
                    "phenomena": [
                        {
                            "id": phenomenon,
                            "observed": True,
                            "description": f"Observed {phenomenon}.",
                            "confidence": 0.9,
                            "frame_ids": ["initial", "final"],
                        }
                        for phenomenon in contract["phenomenon_ids"]
                    ],
                    "confidence": 0.9,
                    "frame_ids": ["initial", "final"],
                    "numeric_consistency": "consistent",
                    "conflicts": [],
                    "evidence_conflict": False,
                },
                "provider_metadata": {"model": "completed-model"},
                "artifacts": {
                    "result": result_path,
                    "query": query_path,
                    "montage": montage_path,
                },
                "representative_episode": episode_dir,
            },
        )
        return {
            "round_id": round_id,
            "child_run_id": run_id,
            "variant_id": condition,
            "seeds": [seed],
            "artifacts": {
                "child_manifest": f"mea/generated_tasks/{run_id}/manifest.json",
                "execution_vqa": result_path,
                "execution_vqa_query": query_path,
                "execution_vqa_montage": montage_path,
            },
        }

    def _make_evaluation(self, evaluation_id="eval_scene_shift_complete"):
        conditions = list(CONDITION_CONTRACTS)
        rounds = [
            self._make_round(evaluation_id, index, condition, 12000 + index)
            for index, condition in enumerate(conditions, start=1)
        ]
        child_ids = [item["child_run_id"] for item in rounds]
        self._write_json(
            f"mea/evaluation_runs/{evaluation_id}/summary/evidence_bundle.json",
            {"schema_version": 2, "rounds": rounds},
        )
        self._write_json(
            f"mea/evaluation_runs/{evaluation_id}/manifest.json",
            {
                "schema_version": 6,
                "evaluation_id": evaluation_id,
                "status": "completed",
                "lifecycle_status": "completed",
                "task_name": "click_bell",
                "telemetry_profile": "balanced_v1",
                "base_commit": "same-commit",
                "active_child_run_id": child_ids[-1],
                "child_run_ids": child_ids,
                "evidence_path": "summary/evidence_bundle.json",
                "plan": {
                    "requested_template_ids": conditions,
                },
            },
        )
        return rounds

    def test_collects_both_rounds_and_hashes_every_regular_source(self):
        rounds = self._make_evaluation()
        result = collect_scene_shift_candidates(self.root)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["ready_candidate_count"], 2)
        self.assertEqual(result["diagnostics"], [])
        self.assertFalse(result["act_called"])
        self.assertFalse(result["provider_called"])
        self.assertFalse(result["suite_validated"])
        self.assertIsNone(result["suite_draft"])
        for candidate, expected_round in zip(result["candidates"], rounds):
            self.assertEqual(candidate["round_id"], expected_round["round_id"])
            self.assertEqual(len(candidate["sources"]), 7)
            for reference in candidate["sources"].values():
                self.assertRegex(reference["sha256"], r"^[0-9a-f]{64}$")
        repeated = collect_scene_shift_candidates(self.root)
        self.assertEqual(result["inventory_sha256"], repeated["inventory_sha256"])

    def test_missing_video_is_an_exact_incomplete_diagnostic(self):
        rounds = self._make_evaluation()
        child = json.loads(
            (
                self.root
                / f"mea/generated_tasks/{rounds[0]['child_run_id']}/manifest.json"
            ).read_text(encoding="utf-8")
        )
        video = self.root / child["act_evaluation"]["video_associations"][0]["video"]
        video.unlink()
        result = collect_scene_shift_candidates(self.root)
        first = result["candidates"][0]
        self.assertEqual(first["status"], "incomplete")
        self.assertIn("video_missing", first["diagnostic_codes"])
        matching = [
            item for item in result["diagnostics"] if item["code"] == "video_missing"
        ]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["artifact"], "video")
        self.assertIn("video.mp4", matching[0]["path"])
        self.assertEqual(result["ready_candidate_count"], 1)

    def test_parent_child_list_must_be_unique_and_end_at_active_child(self):
        rounds = self._make_evaluation()
        parent_path = self.root / (
            "mea/evaluation_runs/eval_scene_shift_complete/manifest.json"
        )
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        parent["child_run_ids"].append(parent["child_run_ids"][-1])
        self._write_json(
            "mea/evaluation_runs/eval_scene_shift_complete/manifest.json", parent
        )
        duplicate = collect_scene_shift_candidates(self.root)
        self.assertEqual(duplicate["ready_candidate_count"], 0)
        self.assertEqual(
            {item["code"] for item in duplicate["diagnostics"]},
            {"parent_child_binding_invalid"},
        )

        parent["child_run_ids"] = [item["child_run_id"] for item in rounds]
        parent["active_child_run_id"] = parent["child_run_ids"][0]
        self._write_json(
            "mea/evaluation_runs/eval_scene_shift_complete/manifest.json", parent
        )
        wrong_active = collect_scene_shift_candidates(self.root)
        self.assertEqual(wrong_active["ready_candidate_count"], 0)
        self.assertEqual(
            {item["code"] for item in wrong_active["diagnostics"]},
            {"parent_child_binding_invalid"},
        )

    def test_missing_vqa_result_preserves_other_resolvable_source_hashes(self):
        rounds = self._make_evaluation()
        result_path = self.root / rounds[0]["artifacts"]["execution_vqa"]
        result_path.unlink()
        result = collect_scene_shift_candidates(self.root)
        first = result["candidates"][0]
        self.assertEqual(first["status"], "incomplete")
        self.assertEqual(
            set(first["sources"]),
            {
                "taskgen_manifest",
                "evaluation_manifest",
                "episode",
                "video",
                "query",
                "montage",
            },
        )
        self.assertIn("execution_vqa_missing", first["diagnostic_codes"])
        self.assertNotIn("episode_missing", first["diagnostic_codes"])
        self.assertNotIn("video_missing", first["diagnostic_codes"])

    def test_unknown_vqa_observation_extension_fails_closed(self):
        rounds = self._make_evaluation()
        result_path = self.root / rounds[0]["artifacts"]["execution_vqa"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["observation"]["untrusted_extension"] = True
        self._write_json(rounds[0]["artifacts"]["execution_vqa"], result)

        collected = collect_scene_shift_candidates(self.root)

        self.assertEqual(collected["ready_candidate_count"], 1)
        self.assertIn(
            "execution_vqa_response_invalid",
            collected["candidates"][0]["diagnostic_codes"],
        )

    def test_derived_vqa_conflict_must_match_recomputed_response(self):
        rounds = self._make_evaluation()
        result_path = self.root / rounds[0]["artifacts"]["execution_vqa"]
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["observation"]["evidence_conflict"] = True
        self._write_json(rounds[0]["artifacts"]["execution_vqa"], result)

        collected = collect_scene_shift_candidates(self.root)

        self.assertEqual(collected["ready_candidate_count"], 1)
        self.assertIn(
            "execution_vqa_response_invalid",
            collected["candidates"][0]["diagnostic_codes"],
        )

    def test_boolean_zero_scene_rates_fail_closed(self):
        rounds = self._make_evaluation()
        rate_fields = ("clean_background_rate", "crazy_random_light_rate")
        for row, rate_field in zip(rounds, rate_fields):
            child_path = self.root / row["artifacts"]["child_manifest"]
            child = json.loads(child_path.read_text(encoding="utf-8"))
            child["scene_validation"]["domain_randomization"][rate_field] = False
            self._write_json(row["artifacts"]["child_manifest"], child)

        collected = collect_scene_shift_candidates(self.root)

        self.assertEqual(collected["ready_candidate_count"], 0)
        self.assertEqual(
            {item["code"] for item in collected["diagnostics"]},
            {"scene_contract_invalid"},
        )

    def test_completed_plan_without_evidence_reports_each_missing_condition(self):
        evaluation_id = "eval_scene_shift_missing_evidence"
        conditions = list(CONDITION_CONTRACTS)
        self._write_json(
            f"mea/evaluation_runs/{evaluation_id}/manifest.json",
            {
                "evaluation_id": evaluation_id,
                "status": "completed",
                "lifecycle_status": "completed",
                "task_name": "click_bell",
                "telemetry_profile": "balanced_v1",
                "base_commit": "same-commit",
                "evidence_path": "summary/evidence_bundle.json",
                "plan": {"requested_template_ids": conditions},
            },
        )
        result = collect_scene_shift_candidates(
            self.root, evaluation_ids=[evaluation_id]
        )
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["ready_candidate_count"], 0)
        self.assertEqual(
            {item["code"] for item in result["diagnostics"]},
            {"evidence_bundle_missing"},
        )

    def test_explicit_nonexistent_or_incomplete_evaluation_is_diagnosed(self):
        incomplete_id = "eval_scene_shift_incomplete"
        self._write_json(
            f"mea/evaluation_runs/{incomplete_id}/manifest.json",
            {
                "evaluation_id": incomplete_id,
                "status": "executing_round_1",
                "lifecycle_status": "running",
            },
        )
        result = collect_scene_shift_candidates(
            self.root,
            evaluation_ids=[incomplete_id, "eval_scene_shift_absent"],
        )
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(
            {item["code"] for item in result["diagnostics"]},
            {"evaluation_not_completed", "evaluation_missing"},
        )

    def test_suite_draft_requires_complete_external_labels_and_stays_unvalidated(self):
        self._make_evaluation()
        inventory = collect_scene_shift_candidates(self.root)
        incomplete = {
            inventory["candidates"][0]["candidate_id"]: {
                "bell_visibly_pressed": True
            }
        }
        result = collect_scene_shift_candidates(
            self.root,
            labels=incomplete,
            reviewer_id="codex-development-agent-proxy",
        )
        self.assertEqual(result["label_status"], "labels_incomplete")
        self.assertIsNone(result["suite_draft"])

        complete = {
            candidate["candidate_id"]: {
                phenomenon: True
                for phenomenon in candidate["expected_query"]["phenomenon_ids"]
            }
            for candidate in inventory["candidates"]
        }
        result = collect_scene_shift_candidates(
            self.root,
            labels=complete,
            reviewer_id="codex-development-agent-proxy",
        )
        self.assertEqual(result["label_status"], "emitted_unvalidated")
        self.assertIsNotNone(result["suite_draft"])
        self.assertEqual(len(result["suite_draft"]["cases"]), 2)
        self.assertFalse(result["suite_validated"])
        self.assertFalse(result["paper_table_eligible"])
        self.assertFalse(result["labels_inferred_from_vqa"])


if __name__ == "__main__":
    unittest.main()
