import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from mea.history import (
    EvaluationHistoryDB,
    IncompleteEvaluationError,
    HistoryRecordError,
    build_history_record,
)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def make_evaluation(
    repo_root: Path,
    evaluation_id: str,
    *,
    request: str,
    task_name: str = "beat_block_hammer",
    policy_name: str = "ACT",
    templates=None,
    lifecycle_status: str = "completed",
) -> Path:
    templates = templates or ["object_appearance.color_blue"]
    directory = repo_root / "mea/evaluation_runs" / evaluation_id
    manifest = {
        "schema_version": 5,
        "evaluation_id": evaluation_id,
        "lifecycle_status": lifecycle_status,
        "status": "completed",
        "created_at": "2026-07-16T10:00:00+08:00",
        "execution_finished_at": "2026-07-16T10:05:00+08:00",
        "user_request": request,
        "base_commit": "abc123",
        "plan_path": "plan/evaluation_plan.json",
        "evidence_path": "summary/evidence_bundle.json",
        "report_path": "evaluation_report.md",
    }
    plan = {
        "schema_version": 5,
        "task_name": task_name,
        "policy": {
            "name": policy_name,
            "checkpoint_setting": "demo_clean",
            "expert_data_num": 50,
            "language_conditioned": False,
        },
        "evaluation_goal": "evaluate requested aspects",
        "requested_template_ids": templates,
        "rounds": [
            {
                "round_id": f"round_{index}",
                "template_id": template,
                "sub_aspect": template.split(".")[0],
                "route": "reuse",
            }
            for index, template in enumerate(templates, start=1)
        ],
        "round_decisions": [],
        "planning_state": f"stopped_after_round_{len(templates)}",
    }
    evidence = {
        "schema_version": 2,
        "evaluation_id": evaluation_id,
        "user_request": request,
        "plan": {
            "executed_rounds": len(templates),
            "completed_template_ids": templates,
        },
        "rounds": [{"round_id": f"round_{index}"} for index in range(1, len(templates) + 1)],
        "observations": {
            "pipeline_passed": True,
            "execution_vqa_conflict": False,
        },
    }
    write_json(directory / "manifest.json", manifest)
    write_json(directory / "plan/evaluation_plan.json", plan)
    write_json(directory / "summary/evidence_bundle.json", evidence)
    (directory / "evaluation_report.md").write_text("# report\n", encoding="utf-8")
    return directory


class EvaluationHistoryTests(unittest.TestCase):
    def test_builds_canonical_record_with_repo_relative_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = make_evaluation(
                root,
                "eval_blue",
                request="把红色方块改为蓝色",
            )
            record = build_history_record(root, evaluation)
            self.assertEqual(record["evaluation_id"], "eval_blue")
            self.assertEqual(record["policy"]["name"], "ACT")
            self.assertEqual(
                record["planning"]["requested_template_ids"],
                ["object_appearance.color_blue"],
            )
            for value in record["artifacts"].values():
                if value is not None:
                    self.assertFalse(Path(value).is_absolute())

            database = EvaluationHistoryDB(
                root / "mea/evaluation_runs/history.sqlite3",
                repo_root=root,
            )
            indexed = database.index_evaluation_dir(evaluation)
            history_path = root / indexed["history_record"]
            self.assertTrue(history_path.is_file())
            encoded = history_path.read_text(encoding="utf-8")
            self.assertTrue(encoded.endswith("\n"))
            self.assertEqual(json.loads(encoded), record)

    def test_only_completed_lifecycle_is_indexed_and_upsert_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = make_evaluation(
                root, "eval_complete", request="蓝色方块评估"
            )
            incomplete = make_evaluation(
                root,
                "eval_running",
                request="still running",
                lifecycle_status="executing",
            )
            database = EvaluationHistoryDB(
                root / "mea/evaluation_runs/history.sqlite3",
                repo_root=root,
            )
            first = database.index_evaluation_dir(complete)
            second = database.index_evaluation_dir(complete)
            self.assertEqual(first["action"], "inserted")
            self.assertEqual(second["action"], "unchanged")
            self.assertEqual(database.count(), 1)
            with self.assertRaises(IncompleteEvaluationError):
                database.index_evaluation_dir(incomplete)
            self.assertEqual(database.count(), 1)

    def test_legacy_completed_manifest_requires_final_timestamp_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            legacy = make_evaluation(
                root,
                "eval_legacy_complete",
                request="旧版蓝色方块评估",
                lifecycle_status=None,
            )
            plan_path = legacy / "plan/evaluation_plan.json"
            legacy_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            legacy_plan.pop("requested_template_ids")
            legacy_plan["rounds"][0].pop("template_id")
            legacy_plan["rounds"][0]["sub_aspect"] = "object_appearance.color"
            write_json(plan_path, legacy_plan)
            database = EvaluationHistoryDB(
                root / "mea/evaluation_runs/history.sqlite3",
                repo_root=root,
            )
            indexed = database.index_evaluation_dir(legacy)
            self.assertEqual(indexed["action"], "inserted")
            record = json.loads(
                (legacy / "summary/history_record.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(
                record["compatibility"]["legacy_completion_inferred"]
            )
            self.assertTrue(
                record["compatibility"]["legacy_template_ids_inferred"]
            )
            self.assertEqual(
                record["planning"]["requested_template_ids"],
                ["object_appearance.color_blue"],
            )

            unfinished = make_evaluation(
                root,
                "eval_legacy_without_finish",
                request="未完成旧评估",
                lifecycle_status=None,
            )
            manifest_path = unfinished / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["execution_finished_at"] = None
            write_json(manifest_path, manifest)
            with self.assertRaises(IncompleteEvaluationError):
                database.index_evaluation_dir(unfinished)
            self.assertEqual(database.count(), 1)

    def test_similarity_is_deterministic_task_filtered_and_policy_labeled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            blue_act = make_evaluation(
                root,
                "eval_blue_act",
                request="把 beat_block_hammer 的红色方块改成蓝色",
                policy_name="ACT",
            )
            blue_other = make_evaluation(
                root,
                "eval_blue_other",
                request="把 beat_block_hammer 的红色方块改成蓝色",
                policy_name="DiffusionPolicy",
            )
            position = make_evaluation(
                root,
                "eval_position",
                request="评估方块位置随机化",
                templates=["object_position.official_random"],
            )
            other_task = make_evaluation(
                root,
                "eval_other_task",
                request="把红色方块改成蓝色",
                task_name="pick_dual_bottles",
            )
            database = EvaluationHistoryDB(
                root / "mea/evaluation_runs/history.sqlite3",
                repo_root=root,
            )
            for directory in (blue_act, blue_other, position, other_task):
                database.index_evaluation_dir(directory)

            first = database.retrieve_similar(
                "把 beat_block_hammer 的红色方块改成蓝色",
                task_name="beat_block_hammer",
                policy_name="ACT",
                checkpoint_setting="demo_clean",
                limit=3,
            )
            second = database.retrieve_similar(
                "把 beat_block_hammer 的红色方块改成蓝色",
                task_name="beat_block_hammer",
                policy_name="ACT",
                checkpoint_setting="demo_clean",
                limit=3,
            )
            self.assertEqual(first, second)
            ids = [item["evaluation_id"] for item in first["candidates"]]
            self.assertEqual(ids[0], "eval_blue_act")
            self.assertIn("eval_blue_other", ids)
            self.assertNotIn("eval_other_task", ids)
            cross_policy = next(
                item
                for item in first["candidates"]
                if item["evaluation_id"] == "eval_blue_other"
            )
            self.assertEqual(cross_policy["policy"]["name"], "DiffusionPolicy")
            self.assertFalse(cross_policy["compatibility"]["same_policy"])

    def test_global_retrieval_uses_only_trusted_task_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bbh = make_evaluation(
                root,
                "eval_bbh",
                request="evaluate object generalization",
            )
            bell = make_evaluation(
                root,
                "eval_bell",
                request="evaluate object generalization for click bell",
                task_name="click_bell",
            )
            excluded = make_evaluation(
                root,
                "eval_excluded",
                request="evaluate object generalization",
                task_name="untrusted_task",
            )
            database = EvaluationHistoryDB(
                root / "mea/evaluation_runs/history.sqlite3",
                repo_root=root,
            )
            for directory in (bbh, bell, excluded):
                database.index_evaluation_dir(directory)

            result = database.retrieve_similar_global(
                "evaluate object generalization",
                allowed_task_names=["click_bell", "beat_block_hammer"],
                policy_name="ACT",
                checkpoint_setting="demo_clean",
                limit=3,
            )
            ids = {item["evaluation_id"] for item in result["candidates"]}
            self.assertEqual(ids, {"eval_bbh", "eval_bell"})
            self.assertEqual(
                result["selection_policy"]["task_filter"],
                "trusted_allowlist",
            )

    def test_rebuild_skips_corrupt_and_incomplete_entries_without_aborting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_evaluation(root, "eval_valid", request="蓝色方块")
            make_evaluation(
                root,
                "eval_incomplete",
                request="incomplete",
                lifecycle_status="executing",
            )
            corrupt = root / "mea/evaluation_runs/eval_corrupt"
            corrupt.mkdir(parents=True)
            (corrupt / "manifest.json").write_text("{bad json", encoding="utf-8")
            database = EvaluationHistoryDB(
                root / "mea/evaluation_runs/history.sqlite3",
                repo_root=root,
            )
            result = database.rebuild(root / "mea/evaluation_runs", reset=True)
            self.assertEqual(result["counts"]["inserted"], 1)
            self.assertEqual(result["counts"]["skipped"], 2)
            self.assertEqual(result["database_record_count"], 1)
            self.assertEqual(
                {issue["kind"] for issue in result["issues"]},
                {"incomplete", "invalid"},
            )

            # A second rebuild is deterministic and idempotent at row level.
            again = database.rebuild(root / "mea/evaluation_runs")
            self.assertEqual(again["counts"]["unchanged"], 1)
            self.assertEqual(database.count(), 1)

    def test_rebuild_prefers_durable_canonical_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = make_evaluation(
                root,
                "eval_canonical",
                request="蓝色方块 canonical history",
            )
            database = EvaluationHistoryDB(
                root / "mea/evaluation_runs/history.sqlite3",
                repo_root=root,
            )
            database.index_evaluation_dir(evaluation)

            # The compact record is the durable rebuild source. Raw planning
            # artifacts may be archived or removed after completion.
            (evaluation / "plan/evaluation_plan.json").unlink()
            rebuilt = database.rebuild(
                root / "mea/evaluation_runs", reset=True
            )

            self.assertEqual(rebuilt["counts"]["inserted"], 1)
            self.assertEqual(rebuilt["database_record_count"], 1)
            self.assertEqual(
                rebuilt["indexed"][0]["source"],
                "canonical_history_record",
            )
            retrieved = database.retrieve_similar(
                "蓝色方块", task_name="beat_block_hammer"
            )
            self.assertEqual(
                retrieved["candidates"][0]["evaluation_id"],
                "eval_canonical",
            )

    def test_build_rejects_cross_artifact_identity_and_round_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation = make_evaluation(
                root,
                "eval_consistent",
                request="蓝色方块 consistency",
            )
            evidence_path = evaluation / "summary/evidence_bundle.json"
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

            evidence["evaluation_id"] = "eval_other"
            write_json(evidence_path, evidence)
            with self.assertRaisesRegex(
                HistoryRecordError, "evidence.evaluation_id"
            ):
                build_history_record(root, evaluation)

            evidence["evaluation_id"] = "eval_consistent"
            evidence["plan"]["executed_rounds"] = 0
            write_json(evidence_path, evidence)
            with self.assertRaisesRegex(
                HistoryRecordError, "executed_rounds"
            ):
                build_history_record(root, evaluation)

            evidence["plan"]["executed_rounds"] = 1
            evidence["rounds"][0]["round_id"] = "round_other"
            write_json(evidence_path, evidence)
            with self.assertRaisesRegex(HistoryRecordError, "round_id mismatch"):
                build_history_record(root, evaluation)

    def test_retrieval_survives_one_corrupt_cached_row(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = make_evaluation(root, "eval_valid", request="蓝色方块")
            database_path = root / "mea/evaluation_runs/history.sqlite3"
            database = EvaluationHistoryDB(database_path, repo_root=root)
            database.index_evaluation_dir(valid)
            with closing(sqlite3.connect(database_path)) as connection, connection:
                connection.execute(
                    "UPDATE evaluations SET record_json = ? WHERE evaluation_id = ?",
                    ("{bad json", "eval_valid"),
                )
            result = database.retrieve_similar(
                "蓝色方块", task_name="beat_block_hammer"
            )
            self.assertEqual(result["selected_count"], 0)
            self.assertEqual(result["issues"][0]["evaluation_id"], "eval_valid")


if __name__ == "__main__":
    unittest.main()
