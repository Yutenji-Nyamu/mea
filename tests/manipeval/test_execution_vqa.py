import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mea.execution_vqa import (
    ExecutionVQAError,
    analyze_execution_montage,
    build_execution_montage,
    select_keyframes,
    validate_execution_vqa_response,
)


def tool_result(name, before, after, value=True):
    return {
        "tool": name,
        "value": value,
        "evidence": [
            {
                "policy_step": before,
                "physics_step": before * 25,
                "video_frame_before": before,
                "video_frame_after": after,
            }
        ],
    }


def valid_response(*, conflict=False):
    return {
        "phenomena": [
            {
                "id": "block_color_blue",
                "observed": True,
                "description": "The target block is blue.",
                "confidence": 0.95,
                "frame_ids": ["initial", "final"],
            },
            {
                "id": "hammer_visibly_lifted",
                "observed": not conflict,
                "description": "The hammer is visible above its initial height.",
                "confidence": 0.8,
                "frame_ids": ["pickup_after"],
            },
            {
                "id": "block_visibly_displaced",
                "observed": None,
                "description": "The selected views do not establish displacement.",
                "confidence": 0.5,
                "frame_ids": ["initial", "final"],
            },
        ],
        "confidence": 0.87,
        "frame_ids": ["initial", "pickup_after", "final"],
        "numeric_consistency": "conflict" if conflict else "consistent",
        "conflicts": (
            [
                {
                    "phenomenon": "hammer_visibly_lifted",
                    "description": "Numeric pickup is true but no lift is visible.",
                    "frame_ids": ["pickup_after"],
                }
            ]
            if conflict
            else []
        ),
    }


class FakeVisionProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []
        self.last_metadata = {"model": "served-model", "usage": {"total_tokens": 5}}

    def vision(self, prompt, image_path, **kwargs):
        self.calls.append((prompt, Path(image_path), kwargs))
        return json.dumps(self.response)


class ExecutionVQATests(unittest.TestCase):
    def test_selects_initial_pickup_contact_and_final_frames(self):
        selected = select_keyframes(
            frame_count=100,
            tool_results=[
                tool_result("first_hammer_pickup_step", 20, 21),
                tool_result("first_contact_step", 50, 51),
            ],
        )
        self.assertEqual(
            [(item["frame_id"], item["frame_index"]) for item in selected],
            [
                ("initial", 0),
                ("pickup_before", 20),
                ("pickup_after", 21),
                ("contact_before", 50),
                ("contact_after", 51),
                ("final", 99),
            ],
        )
        self.assertLessEqual(len(selected), 8)

    def test_contact_event_is_used_when_tool_has_no_evidence(self):
        selected = select_keyframes(
            frame_count=40,
            tool_results=[],
            events=[
                {
                    "type": "contact_interval",
                    "actors": ["020_hammer", "box"],
                    "physical_contact": True,
                    "first_physical_policy_step": 17,
                }
            ],
        )
        frames = {item["frame_id"]: item["frame_index"] for item in selected}
        self.assertEqual(frames["contact_before"], 17)
        self.assertEqual(frames["contact_after"], 18)
        self.assertIn("initial", frames)
        self.assertIn("final", frames)

    def test_projection_evidence_steps_use_semantic_physics_mapping(self):
        selected = select_keyframes(
            frame_count=100,
            tool_results=[
                {
                    "tool": "generated_pickup_contact_duration_v1",
                    "value": 1.0,
                    "evidence_steps": [500, 900],
                    "details": {
                        "pickup_detected": True,
                        "contact_detected": True,
                        "pickup_physics_step": 500,
                        "contact_physics_step": 900,
                    },
                }
            ],
            physics_to_policy={500: 20, 900: 50},
        )
        frames = {item["frame_id"]: item["frame_index"] for item in selected}
        self.assertEqual(frames["pickup_before"], 20)
        self.assertEqual(frames["pickup_after"], 21)
        self.assertEqual(frames["contact_before"], 50)
        self.assertEqual(frames["contact_after"], 51)

    def test_official_success_event_uses_exact_sparse_video_frame(self):
        selected = select_keyframes(
            frame_count=4,
            tool_results=[
                {
                    "tool": "official_check_success",
                    "value": True,
                    "evidence_steps": [843],
                    "evidence": [
                        {
                            "type": "success_transition",
                            "physics_step": 843,
                            "video_frame_index": 2,
                        }
                    ],
                }
            ],
        )
        frames = {item["frame_id"]: item["frame_index"] for item in selected}
        self.assertEqual(frames["success_before"], 1)
        self.assertEqual(frames["success_after"], 2)
        self.assertEqual(frames["initial"], 0)
        self.assertEqual(frames["final"], 3)

    def test_duration_without_contact_never_creates_contact_frame_labels(self):
        selected = select_keyframes(
            frame_count=100,
            tool_results=[
                {
                    "tool": "generated_pickup_contact_duration_v1",
                    "value": None,
                    "evidence_steps": [500],
                    "details": {
                        "pickup_detected": True,
                        "contact_detected": False,
                        "pickup_physics_step": 500,
                        "contact_physics_step": None,
                        "reason": "contact_not_observed_after_pickup",
                    },
                }
            ],
            events=[
                {
                    "type": "contact_interval",
                    "actors": ["020_hammer", "table"],
                    "physical_contact": True,
                    "start_policy_step": 4,
                }
            ],
            physics_to_policy={500: 20},
        )
        frames = {item["frame_id"]: item["frame_index"] for item in selected}
        self.assertEqual(frames["pickup_before"], 20)
        self.assertEqual(frames["pickup_after"], 21)
        self.assertNotIn("contact_before", frames)
        self.assertNotIn("contact_after", frames)

    def test_sparse_timeline_adds_four_deterministic_context_frames(self):
        selected = select_keyframes(frame_count=10)
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected[0]["frame_id"], "initial")
        self.assertEqual(selected[-1]["frame_id"], "final")
        self.assertEqual(
            [item["frame_index"] for item in selected], [0, 3, 6, 9]
        )

    def test_builds_one_reference_and_rollout_montage(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"mock-video")
            reference = root / "reference.png"
            Image.new("RGB", (80, 60), "red").save(reference)

            def fake_reader(path, indices=None):
                return (
                    10,
                    10.0,
                    {
                        int(index): Image.new("RGB", (80, 60), "blue")
                        for index in (indices or [])
                    },
                )

            with patch(
                "mea.execution_vqa.prototype._read_video_frames",
                side_effect=fake_reader,
            ):
                result = build_execution_montage(
                    video_path=video,
                    destination=root / "execution_montage.png",
                    reference_scene=reference,
                )

            self.assertEqual(len(result["selected_frames"]), 4)
            self.assertEqual(result["reference_scene"], str(reference.resolve()))
            with Image.open(result["montage_path"]) as montage:
                self.assertGreater(montage.width, 320)
                self.assertGreater(montage.height, 240)

    def test_schema_rejects_unknown_frame_and_derives_conflict(self):
        result = validate_execution_vqa_response(
            valid_response(conflict=True),
            allowed_frame_ids=["initial", "pickup_after", "final"],
        )
        self.assertTrue(result["evidence_conflict"])
        self.assertEqual(result["numeric_consistency"], "conflict")

        malformed = valid_response()
        malformed["frame_ids"] = ["invented"]
        with self.assertRaisesRegex(ExecutionVQAError, "unknown frame_id"):
            validate_execution_vqa_response(
                malformed,
                allowed_frame_ids=["initial", "pickup_after", "final"],
            )

    def test_schema_requires_each_phenomenon_exactly_once(self):
        missing = valid_response()
        missing["phenomena"].pop()
        with self.assertRaisesRegex(ExecutionVQAError, "every allowlisted id"):
            validate_execution_vqa_response(
                missing,
                allowed_frame_ids=["initial", "pickup_after", "final"],
            )

        duplicate = valid_response()
        duplicate["phenomena"][2] = dict(duplicate["phenomena"][0])
        with self.assertRaisesRegex(ExecutionVQAError, "duplicate id"):
            validate_execution_vqa_response(
                duplicate,
                allowed_frame_ids=["initial", "pickup_after", "final"],
            )

    def test_provider_model_is_forwarded_and_numeric_result_is_not_overwritten(self):
        numeric = [tool_result("first_hammer_pickup_step", 20, 21)]
        selection = {
            "selected_frames": [
                {"frame_id": "initial", "frame_index": 0},
                {"frame_id": "pickup_after", "frame_index": 21},
                {"frame_id": "final", "frame_index": 99},
            ]
        }
        provider = FakeVisionProvider(valid_response(conflict=True))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            montage = root / "montage.png"
            montage.write_bytes(b"mock-png")
            result = analyze_execution_montage(
                provider=provider,
                model="gpt-5.6-balanced",
                montage_path=montage,
                selection=selection,
                numeric_tool_results=numeric,
                destination=root / "execution_vqa.json",
            )

            self.assertEqual(result["numeric_tool_results"], numeric)
            self.assertTrue(result["evidence_conflict"])
            self.assertEqual(provider.calls[0][2]["model"], "gpt-5.6-balanced")
            self.assertIn("authoritative", provider.calls[0][0])
            self.assertTrue((root / "execution_vqa.json").is_file())
            self.assertTrue((root / "execution_vqa_prompt.md").is_file())
            self.assertTrue((root / "execution_vqa_response.txt").is_file())

    def test_numeric_guard_records_conflict_even_if_vision_claims_consistent(self):
        numeric = [tool_result("first_hammer_pickup_step", 20, 21)]
        response = valid_response()
        response["phenomena"][1]["observed"] = False
        provider = FakeVisionProvider(response)
        selection = {
            "selected_frames": [
                {"frame_id": "initial", "frame_index": 0},
                {"frame_id": "pickup_after", "frame_index": 21},
                {"frame_id": "final", "frame_index": 99},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            montage = Path(directory) / "montage.png"
            montage.write_bytes(b"mock-png")
            result = analyze_execution_montage(
                provider=provider,
                model="vision-model",
                montage_path=montage,
                selection=selection,
                numeric_tool_results=numeric,
            )
        self.assertTrue(result["evidence_conflict"])
        self.assertEqual(
            result["observation"]["numeric_consistency"], "conflict"
        )
        self.assertIn(
            "Deterministic numeric guard",
            result["observation"]["conflicts"][0]["description"],
        )


if __name__ == "__main__":
    unittest.main()
