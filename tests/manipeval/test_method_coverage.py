import binascii
import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from mea.method_coverage import (
    VALID_STATUSES,
    build_method_coverage_report,
    validate_matched_strategy_evidence,
    validate_run_local_vqa_evidence,
)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_png(path, *, width=32, height=32):
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", binascii.crc32(kind + payload) & 0xFFFFFFFF)
        )

    # Flatten the RGB tuples without depending on Pillow in this audit test.
    rows = b"".join(
        b"\x00"
        + bytes(
            channel
            for x in range(width)
            for channel in (x % 256, y % 256, (x + y) % 256)
        )
        for y in range(height)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class MethodCoverageTests(unittest.TestCase):
    def test_report_has_sixteen_ranked_claims_and_derived_statuses(self):
        root = Path(__file__).resolve().parents[2]
        report = build_method_coverage_report(root)
        self.assertEqual(report["claim_count"], 16)
        self.assertEqual(
            [item["rank"] for item in report["claims"]], list(range(1, 17))
        )
        self.assertEqual(len({item["claim_id"] for item in report["claims"]}), 16)
        for claim in report["claims"]:
            self.assertIn(claim["status"], VALID_STATUSES)
            code_ready = all(item["passed"] for item in claim["code_checks"])
            evidence_ready = all(item["passed"] for item in claim["evidence_checks"])
            self.assertEqual(claim["code_ready"], code_ready)
            self.assertEqual(claim["evidence_ready"], evidence_ready)
            if claim["status"] == "implemented":
                self.assertTrue(code_ready)
                self.assertTrue(evidence_ready)
            elif claim["status"] == "evidence_pending":
                self.assertTrue(code_ready)
                self.assertTrue(claim["evidence_checks"])
                self.assertFalse(evidence_ready)
            else:
                self.assertFalse(code_ready)

        by_id = {item["claim_id"]: item for item in report["claims"]}
        self.assertEqual(by_id["semantic_history_reuse"]["status"], "implemented")
        self.assertEqual(
            by_id["taxonomy_unsupported_boundary"]["status"], "implemented"
        )
        self.assertEqual(by_id["stage_recovery_resume"]["status"], "implemented")
        self.assertEqual(by_id["proposal_every_round"]["status"], "implemented")
        self.assertEqual(by_id["complete_task_codegen"]["status"], "implemented")

    def test_run_local_vqa_rejects_synthetic_shape_and_accepts_full_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode_dir = (
                root
                / "mea/generated_tasks/run_live/evaluation/telemetry/act/episode_000_seed_7"
            )
            write_json(
                episode_dir / "episode.json",
                {
                    "policy_name": "ACT",
                    "task_name": "click_bell",
                    "seed": 7,
                    "success": False,
                },
            )
            (episode_dir / "video.mp4").write_bytes(b"real-cached-video-provenance")
            output = root / "mea/validation_runs/live_vqa"
            montage = output / "execution_montage.png"
            write_png(montage)
            selection = {
                "video_path": str(episode_dir / "video.mp4"),
                "frame_count": 100,
                "fps": 10.0,
                "selected_frames": [
                    {
                        "frame_id": "initial",
                        "frame_index": 0,
                        "source": "video_boundary",
                    },
                    {
                        "frame_id": "final",
                        "frame_index": 99,
                        "source": "video_boundary",
                    },
                ],
                "reference_scene": None,
                "montage_path": str(montage),
            }
            query = {
                "questions": [
                    {
                        "id": "run_local.bell_progress",
                        "question": "Did the robot make visible progress?",
                    }
                ]
            }
            write_json(output / "keyframe_selection.json", selection)
            write_json(output / "execution_vqa_query.json", query)
            (output / "execution_vqa_prompt.md").write_text(
                "provider prompt with initial and final keyframes", encoding="utf-8"
            )
            (output / "execution_vqa_response.txt").write_text(
                '{"phenomena": []}', encoding="utf-8"
            )
            artifact = output / "execution_vqa.json"
            value = {
                "status": "passed",
                "model_requested": "vision-model",
                "provider_metadata": {
                    "id": "chatcmpl-live-provider-call",
                    "model": "served-vision-model",
                    "usage": {"total_tokens": 123},
                },
                "query": query,
                "observation": {
                    "phenomena": [
                        {
                            "id": "run_local.bell_progress",
                            "observed": False,
                        }
                    ],
                    "frame_ids": ["initial", "final"],
                },
                "representative_episode": str(episode_dir.relative_to(root)),
                "selection": selection,
                "artifacts": {
                    "result": str(artifact.relative_to(root)),
                    "prompt": str(
                        (output / "execution_vqa_prompt.md").relative_to(root)
                    ),
                    "response": str(
                        (output / "execution_vqa_response.txt").relative_to(root)
                    ),
                    "montage": str(montage.relative_to(root)),
                    "selection": str(
                        (output / "keyframe_selection.json").relative_to(root)
                    ),
                    "query": str(
                        (output / "execution_vqa_query.json").relative_to(root)
                    ),
                },
            }
            write_json(artifact, value)
            passed, _ = validate_run_local_vqa_evidence(
                root, artifact, value
            )
            self.assertTrue(passed)

            # This was accepted by the old validator: a model string plus any
            # three bytes called a montage, with no provider or rollout trace.
            synthetic = {
                "status": "passed",
                "model_requested": "vision-model",
                "query": query,
                "observation": value["observation"],
                "artifacts": {"montage": "fake.png"},
            }
            (root / "fake.png").write_bytes(b"png")
            passed, reason = validate_run_local_vqa_evidence(
                root, root / "fake_execution_vqa.json", synthetic
            )
            self.assertFalse(passed)
            self.assertIn("provider", reason)

    def test_matched_strategy_requires_registered_evaluation_child_episode_chain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            policy = {"name": "ACT", "checkpoint_setting": "act-click-bell"}

            def build_run(strategy, evaluation_id, variants):
                registration = {
                    "registration_id": "registered-pair-1",
                    "strategy": strategy,
                    "expected_evaluation_id": evaluation_id,
                    "expected_child_run_prefix": f"run_{strategy}_",
                }
                evaluation_dir = root / "mea/evaluation_runs" / evaluation_id
                samples = []
                rounds = []
                for index, (variant, success) in enumerate(variants, start=1):
                    child_id = f"run_{strategy}_{index}"
                    episode_relative = "act/episode_000_seed_7"
                    child_dir = root / "mea/generated_tasks" / child_id
                    write_json(
                        child_dir / "manifest.json",
                        {
                            "run_id": child_id,
                            "status": "completed",
                            "registration_identity": registration,
                            "trusted_tool_evaluation": {
                                "episodes": [
                                    {
                                        "episode_dir": episode_relative,
                                        "policy_name": "ACT",
                                        "seed": 7,
                                    }
                                ]
                            },
                        },
                    )
                    episode_path = (
                        child_dir
                        / "evaluation/telemetry"
                        / episode_relative
                        / "episode.json"
                    )
                    write_json(
                        episode_path,
                        {
                            "policy_name": "ACT",
                            "task_name": "click_bell",
                            "seed": 7,
                            "success": success,
                            "registration_identity": registration,
                        },
                    )
                    samples.append(
                        {
                            "variant_id": variant,
                            "seed": 7,
                            "success": success,
                            "episode": str(episode_path.relative_to(root)),
                        }
                    )
                    rounds.append(
                        {
                            "variant_id": variant,
                            "taskgen_run_id": child_id,
                            "pipeline_passed": True,
                        }
                    )
                write_json(
                    evaluation_dir / "manifest.json",
                    {
                        "evaluation_id": evaluation_id,
                        "lifecycle_status": "completed",
                        "status": "completed",
                        "task_name": "click_bell",
                        "planning_policy": strategy,
                        "registration_identity": registration,
                        "plan": {"policy": policy},
                    },
                )
                write_json(
                    evaluation_dir / "summary/summary.json",
                    {"status": "completed", "rounds": rounds},
                )
                return {
                    "evaluation_dir": str(evaluation_dir.relative_to(root)),
                    "evaluation_id": evaluation_id,
                    "task_name": "click_bell",
                    "registration_identity": registration,
                    "samples": samples,
                    "totals": {"act_rollouts": len(samples)},
                }

            fixed = build_run(
                "fixed_predeclared_v1",
                "eval_fixed",
                [("left", False), ("right", True)],
            )
            dynamic = build_run(
                "dynamic_evidence_v1", "eval_dynamic", [("left", False)]
            )
            comparison = {
                "protocol": "click_bell_fixed_vs_dynamic_n1",
                "identity": {"task_name": "click_bell", "policy": policy},
                "strategies": {
                    "fixed_predeclared_v1": fixed,
                    "dynamic_evidence_v1": dynamic,
                },
                "overlap": [
                    {
                        "variant_id": "left",
                        "seed": 7,
                        "exact_success_agreement": True,
                    }
                ],
            }
            value = {
                "status": "passed",
                "registered_identity_match": True,
                "evidence": {"registration_id": "registered-pair-1"},
                "comparison": comparison,
            }
            passed, _ = validate_matched_strategy_evidence(
                root, root / "summary.json", value
            )
            self.assertTrue(passed)

            # The old validator accepted this raw comparison with empty
            # manifests because absent registered_identity_match was not False.
            legacy_false_positive = dict(comparison)
            passed, reason = validate_matched_strategy_evidence(
                root, root / "summary.json", legacy_false_positive
            )
            self.assertFalse(passed)
            self.assertIn("status", reason)


if __name__ == "__main__":
    unittest.main()
