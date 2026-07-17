import json
import os
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from mea.execution_vqa.reviewed_registry import (
    ANSWER_CONTRACT_ID,
    REGISTRY_SCOPE,
    REVIEW_CHECK_KEYS,
    REVIEW_SCOPE,
    ReviewedVQAQuerySpecError,
    canonical_sha256,
    file_sha256,
    load_reviewed_vqa_query_specs,
    match_reviewed_vqa_query_spec,
    validate_vqa_query_review,
    validate_vqa_query_spec,
)


class ReviewedVQAQueryRegistryTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.registry = self.root / "reviewed_vqa"
        self.spec = {
            "schema_version": 1,
            "spec_id": "vqa_click_bell_scene_clutter_v1",
            "task_name": "click_bell",
            "template_ids": ["robustness.scene_clutter.official_table"],
            "sub_aspect_prefixes": ["robustness.scene_clutter"],
            "tool_metrics": ["official_check_success"],
            "phenomenon_ids": [
                "bell_visibly_pressed",
                "bell_target_selected_among_clutter",
            ],
            "answer_contract_id": ANSWER_CONTRACT_ID,
        }

    def tearDown(self):
        self._temporary.cleanup()

    def _review(self, spec_hash, *, reviewer_kind="development_agent_proxy"):
        return {
            "schema_version": 1,
            "decision": "approved",
            "review_scope": REVIEW_SCOPE,
            "reviewer": {
                "id": "codex-development-proxy-test",
                "kind": reviewer_kind,
            },
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "spec_sha256": spec_hash,
            "checks": {key: True for key in sorted(REVIEW_CHECK_KEYS)},
            "notes": "Test-only proxy review; this is not human gold.",
        }

    @staticmethod
    def _write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_registry(self, *, spec=None, review=None, entries=None):
        spec = deepcopy(spec or self.spec)
        normalized = validate_vqa_query_spec(spec)
        spec_hash = canonical_sha256(normalized)
        review = deepcopy(review or self._review(spec_hash))
        spec_path = self.registry / "entries/clutter/spec.json"
        review_path = self.registry / "entries/clutter/review.json"
        self._write_json(spec_path, spec)
        self._write_json(review_path, review)
        default_entry = {
            "spec_id": spec["spec_id"],
            "spec_artifact": "entries/clutter/spec.json",
            "spec_artifact_sha256": file_sha256(spec_path),
            "spec_sha256": spec_hash,
            "review_artifact": "entries/clutter/review.json",
            "review_artifact_sha256": file_sha256(review_path),
        }
        self._write_json(
            self.registry / "index.json",
            {
                "schema_version": 1,
                "scope": REGISTRY_SCOPE,
                "entries": entries if entries is not None else [default_entry],
            },
        )
        return default_entry, spec_path, review_path

    def test_loads_proxy_review_and_matches_exact_clutter_context(self):
        self._write_registry()
        loaded = load_reviewed_vqa_query_specs(self.registry)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(
            loaded[0]["review"]["reviewer"]["kind"],
            "development_agent_proxy",
        )
        match = match_reviewed_vqa_query_spec(
            loaded,
            task_name="click_bell",
            template_id="robustness.scene_clutter.official_table",
            sub_aspect="robustness.scene_clutter",
            tool_metric="official_check_success",
        )
        self.assertEqual(
            match["spec"]["phenomenon_ids"],
            [
                "bell_visibly_pressed",
                "bell_target_selected_among_clutter",
            ],
        )

    def test_human_review_is_also_allowed_but_development_agent_is_not(self):
        spec_hash = canonical_sha256(validate_vqa_query_spec(self.spec))
        human = validate_vqa_query_review(
            self._review(spec_hash, reviewer_kind="human"),
            spec_sha256=spec_hash,
        )
        self.assertEqual(human["reviewer"]["kind"], "human")
        with self.assertRaisesRegex(
            ReviewedVQAQuerySpecError, "human or development_agent_proxy"
        ):
            validate_vqa_query_review(
                self._review(spec_hash, reviewer_kind="development_agent"),
                spec_sha256=spec_hash,
            )

    def test_selection_only_contract_rejects_unknown_phenomenon(self):
        tampered = deepcopy(self.spec)
        tampered["phenomenon_ids"].append("free_form_prompt_injection")
        with self.assertRaisesRegex(
            ReviewedVQAQuerySpecError, "non-allowlisted phenomena"
        ):
            validate_vqa_query_spec(tampered)

    def test_pending_or_spec_hash_mismatched_review_is_rejected(self):
        spec_hash = canonical_sha256(validate_vqa_query_spec(self.spec))
        pending = self._review(spec_hash)
        pending["decision"] = "pending"
        with self.assertRaisesRegex(
            ReviewedVQAQuerySpecError, "decision must be approved"
        ):
            validate_vqa_query_review(pending, spec_sha256=spec_hash)
        mismatched = self._review("0" * 64)
        with self.assertRaisesRegex(
            ReviewedVQAQuerySpecError, "not pinned to the exact spec"
        ):
            validate_vqa_query_review(mismatched, spec_sha256=spec_hash)

    def test_tampered_spec_and_review_artifacts_are_rejected(self):
        _, spec_path, review_path = self._write_registry()
        spec_path.write_text(
            spec_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ReviewedVQAQuerySpecError, "spec artifact hash mismatch"
        ):
            load_reviewed_vqa_query_specs(self.registry)

        self.registry = self.root / "reviewed_vqa_review_tamper"
        _, _, review_path = self._write_registry()
        review_path.write_text(
            review_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ReviewedVQAQuerySpecError, "review artifact hash mismatch"
        ):
            load_reviewed_vqa_query_specs(self.registry)

    def test_path_traversal_is_rejected(self):
        entry, _, _ = self._write_registry()
        escaped = deepcopy(entry)
        escaped["spec_artifact"] = "../outside.json"
        self._write_json(
            self.registry / "index.json",
            {
                "schema_version": 1,
                "scope": REGISTRY_SCOPE,
                "entries": [escaped],
            },
        )
        with self.assertRaisesRegex(ReviewedVQAQuerySpecError, "escapes registry root"):
            load_reviewed_vqa_query_specs(self.registry)

    @unittest.skipIf(os.name == "nt", "symlink creation is not portable on Windows")
    def test_symlinked_artifact_is_rejected(self):
        entry, spec_path, _ = self._write_registry()
        external = self.root / "external_spec.json"
        external.write_bytes(spec_path.read_bytes())
        spec_path.unlink()
        spec_path.symlink_to(external)
        entry["spec_artifact_sha256"] = file_sha256(external)
        self._write_json(
            self.registry / "index.json",
            {
                "schema_version": 1,
                "scope": REGISTRY_SCOPE,
                "entries": [entry],
            },
        )
        with self.assertRaisesRegex(
            ReviewedVQAQuerySpecError, "must not contain symlinks"
        ):
            load_reviewed_vqa_query_specs(self.registry)

    def test_nonempty_selector_groups_are_conjunctive(self):
        self._write_registry()
        loaded = load_reviewed_vqa_query_specs(self.registry)
        for context in (
            {
                "task_name": "click_bell",
                "template_id": "task_execution.official_baseline",
                "sub_aspect": "robustness.scene_clutter",
                "tool_metric": "official_check_success",
            },
            {
                "task_name": "click_bell",
                "template_id": "robustness.scene_clutter.official_table",
                "sub_aspect": "object_position",
                "tool_metric": "official_check_success",
            },
            {
                "task_name": "click_bell",
                "template_id": "robustness.scene_clutter.official_table",
                "sub_aspect": "robustness.scene_clutter",
                "tool_metric": "other_metric",
            },
        ):
            with self.subTest(context=context):
                self.assertIsNone(match_reviewed_vqa_query_spec(loaded, **context))

    def test_ambiguous_matches_fail_closed(self):
        self._write_registry()
        loaded = load_reviewed_vqa_query_specs(self.registry)
        with self.assertRaisesRegex(
            ReviewedVQAQuerySpecError, "multiple reviewed VQAQuerySpecs"
        ):
            match_reviewed_vqa_query_spec(
                [*loaded, deepcopy(loaded[0])],
                task_name="click_bell",
                template_id="robustness.scene_clutter.official_table",
                sub_aspect="robustness.scene_clutter",
                tool_metric="official_check_success",
            )

    def test_missing_registry_is_an_empty_safe_miss(self):
        loaded = load_reviewed_vqa_query_specs(self.root / "missing")
        self.assertEqual(loaded, [])
        self.assertIsNone(
            match_reviewed_vqa_query_spec(
                loaded,
                task_name="click_bell",
                template_id="robustness.scene_clutter.official_table",
                sub_aspect="robustness.scene_clutter",
                tool_metric="official_check_success",
            )
        )


if __name__ == "__main__":
    unittest.main()
