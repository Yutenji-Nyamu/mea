import json
import tempfile
import unittest
from pathlib import Path

from mea.round_provenance import (
    RoundProvenanceError,
    bind_round_provenance,
    verify_round_provenance,
)


class RoundProvenanceTests(unittest.TestCase):
    def _fixture(self, root: Path):
        evaluation = root / "mea/evaluation_runs/eval_test"
        child = root / "mea/taskgen_runs/run_test_round_01"
        (evaluation / "summary").mkdir(parents=True)
        (evaluation / "execution/round_01").mkdir(parents=True)
        child.mkdir(parents=True)
        (child / "manifest.json").write_text(
            json.dumps({"run_id": "run_test_round_01"}), encoding="utf-8"
        )
        (evaluation / "execution/round_01/aggregate_result.json").write_text(
            json.dumps({"passed": True}), encoding="utf-8"
        )
        plan = {"round_id": "round_01", "sub_aspect": "object_position"}
        summary = {
            "evaluation_id": "eval_test",
            "child_run_id": "run_test_round_01",
            "pipeline_passed": True,
            "observations": {
                "whole_round_recovery": {
                    "attempts": [{"attempt_index": 1}]
                }
            },
        }
        return evaluation, child, plan, summary

    def test_round_plan_summary_and_files_are_hash_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, child, plan, summary = self._fixture(root)
            bound, provenance = bind_round_provenance(
                root,
                evaluation,
                round_plan=plan,
                child_dir=child,
                round_summary=summary,
            )
            sidecar = root / bound["provenance"]["path"]
            verified = verify_round_provenance(
                root, sidecar, round_plan=plan, round_summary=bound
            )
            self.assertEqual(verified["binding_sha256"], provenance["binding_sha256"])
            self.assertEqual(verified["binding"]["final_round_attempt_index"], 1)

    def test_tampering_and_overwrite_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation, child, plan, summary = self._fixture(root)
            bound, _ = bind_round_provenance(
                root,
                evaluation,
                round_plan=plan,
                child_dir=child,
                round_summary=summary,
            )
            with self.assertRaisesRegex(RoundProvenanceError, "already exists"):
                bind_round_provenance(
                    root,
                    evaluation,
                    round_plan=plan,
                    child_dir=child,
                    round_summary=summary,
                )
            (child / "manifest.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RoundProvenanceError, "artifact hash mismatch"):
                verify_round_provenance(
                    root,
                    root / bound["provenance"]["path"],
                    round_plan=plan,
                    round_summary=bound,
                )


if __name__ == "__main__":
    unittest.main()
