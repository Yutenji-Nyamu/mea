import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.cached_finalization import finalize_cached_evaluation


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FinalSummary:
    def __init__(self, _root, _provider, *, model):
        self.model = model

    def generate(self, _evidence, *, output_dir):
        feedback = {
            "answer": "bounded answer",
            "evaluation_scope": "one cached round",
            "findings": ["official success"],
            "limitations": ["N=1"],
            "recommended_next_step": "repeat with more seeds",
        }
        _write_json(output_dir / "answer.json", feedback)
        (output_dir / "answer.md").write_text("bounded answer\n", encoding="utf-8")
        return feedback


class CachedFinalizationTests(unittest.TestCase):
    def test_recovers_final_answer_without_mutating_method_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evaluation_id = "eval_cached_finalization"
            evaluation = root / "mea/evaluation_runs" / evaluation_id
            aggregate = evaluation / "summary/aggregate_result.json"
            summary = evaluation / "summary/summary.json"
            evidence = evaluation / "summary/evidence_bundle.json"
            _write_json(
                evaluation / "manifest.json",
                {
                    "evaluation_id": evaluation_id,
                    "status": "failed",
                    "lifecycle_status": "failed",
                    "failure_stage": "final_answer",
                    "failure": {"type": "ProviderError", "message": "503"},
                    "history_retrieval_status": "disabled",
                },
            )
            _write_json(aggregate, {"status": "passed", "source_count": 1})
            _write_json(
                summary,
                {
                    "evaluation_id": evaluation_id,
                    "status": "completed",
                    "rounds": [{"round_id": "round_1"}],
                },
            )
            _write_json(
                evidence,
                {
                    "evaluation_id": evaluation_id,
                    "rounds": [{"round_id": "round_1"}],
                },
            )
            before = {path: _sha256(path) for path in (aggregate, summary, evidence)}

            def _write_report(_root, _evaluation, *, destination):
                destination.write_text("compact evidence\n", encoding="utf-8")
                return {"report": "evidence_report.md"}

            with (
                patch(
                    "mea.cached_finalization.PlanAgentFinalSummary",
                    _FinalSummary,
                ),
                patch(
                    "mea.cached_finalization.render_evaluation_report",
                    return_value="evaluation report\n",
                ),
                patch(
                    "mea.cached_finalization.write_evidence_report",
                    side_effect=_write_report,
                ),
            ):
                result = finalize_cached_evaluation(
                    root,
                    evaluation_id,
                    provider=object(),
                    feedback_model="fixture-model",
                )

            self.assertEqual(result["rollouts_executed"], 0)
            self.assertEqual(
                before,
                {path: _sha256(path) for path in (aggregate, summary, evidence)},
            )
            manifest = json.loads(
                (evaluation / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["lifecycle_status"], "completed")
            self.assertIsNone(manifest["failure"])
            self.assertEqual(manifest["history_index"], {"status": "disabled"})
            self.assertTrue((evaluation / "answer/answer.json").is_file())
            self.assertTrue((evaluation / "evaluation_report.md").is_file())
            self.assertTrue((evaluation / "evidence_report.md").is_file())


if __name__ == "__main__":
    unittest.main()
