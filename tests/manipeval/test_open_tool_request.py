from __future__ import annotations

import json
import unittest
from pathlib import Path

from mea.toolgen.open_request import (
    OpenToolRequestAgent,
    OpenToolRequestError,
    validate_open_tool_request,
)


class _Provider:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0
        self.last_metadata = {"provider": "fixture"}

    def text(self, *_args, **_kwargs):
        self.calls += 1
        return json.dumps(self.payload)


class _RetryProvider:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0
        self.last_metadata = {"provider": "fixture"}

    def text(self, *_args, **_kwargs):
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return json.dumps(payload)


def _typed(task_name: str = "runtime_task"):
    return {
        "schema_version": 2,
        "task_name": task_name,
        "metric": "runtime_xy_gap",
        "question": "How close did the two actors get?",
        "metric_spec": {
            "schema_version": 1,
            "operation": "minimum_distance",
            "left_signal": "left_position",
            "right_signal": "right_position",
            "dimensions": ["x", "y"],
            "unit": "m",
            "null_semantics": "null_if_no_finite_sample",
        },
    }


class OpenToolRequestTest(unittest.TestCase):
    def test_novel_metric_requires_typed_spec(self):
        with self.assertRaisesRegex(OpenToolRequestError, "typed MetricSpec"):
            validate_open_tool_request(
                {
                    "schema_version": 1,
                    "task_name": "runtime_task",
                    "metric": "unknown_metric",
                    "question": "Unknown?",
                },
                task_name="runtime_task",
            )

    def test_typed_request_is_task_bound(self):
        self.assertEqual(
            validate_open_tool_request(
                _typed(), task_name="runtime_task"
            )["metric"],
            "runtime_xy_gap",
        )
        with self.assertRaisesRegex(OpenToolRequestError, "changed the bound task"):
            validate_open_tool_request(
                _typed("other_task"), task_name="runtime_task"
            )

    def test_agent_uses_schema_not_task_catalog(self):
        root = Path(__file__).resolve().parents[2]
        payload = _typed("adjust_bottle")
        payload["metric_spec"]["left_signal"] = "left_tcp_position"
        payload["metric_spec"]["right_signal"] = "bottle_position"
        provider = _Provider(payload)
        agent = OpenToolRequestAgent(
            root, provider, model="fixture-model"
        )
        bundle = agent.propose(
            source_query="Which condition fails?",
            semantic_concern="runtime gap",
            tool_need="measure actor proximity",
            task_name="adjust_bottle",
            reusable_tool_requests=[
                {
                    "registration_id": "runlocal_fixture",
                    "request": payload,
                    "validation": {"scope": "evaluation_local"},
                }
            ],
        )
        self.assertEqual(
            bundle["tool_request"]["metric"], "runtime_xy_gap"
        )
        self.assertEqual(provider.calls, 1)
        self.assertNotIn(
            "template",
            json.dumps(bundle["context"], ensure_ascii=False),
        )
        self.assertEqual(
            bundle["context"]["validated_generated_tools"][0][
                "registration_id"
            ],
            "runlocal_fixture",
        )

    def test_agent_rejects_missing_schema_signal(self):
        root = Path(__file__).resolve().parents[2]
        payload = _typed("adjust_bottle")
        provider = _Provider(payload)
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")
        with self.assertRaisesRegex(
            OpenToolRequestError, "unavailable telemetry signals"
        ):
            agent.propose(
                source_query="Which condition fails?",
                semantic_concern="runtime gap",
                tool_need="measure actor proximity",
                task_name="adjust_bottle",
            )

    def test_invalid_metric_id_is_repaired_inside_provider_loop(self):
        root = Path(__file__).resolve().parents[2]
        invalid = _typed("adjust_bottle")
        invalid["metric"] = "Bottle Gap"
        invalid["metric_spec"]["left_signal"] = "left_tcp_position"
        invalid["metric_spec"]["right_signal"] = "bottle_position"
        repaired = json.loads(json.dumps(invalid))
        repaired["metric"] = "bottle_gap"
        provider = _RetryProvider([invalid, repaired])
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")

        bundle = agent.propose(
            source_query="Which condition fails?",
            semantic_concern="runtime gap",
            tool_need="measure actor proximity",
            task_name="adjust_bottle",
        )

        self.assertEqual(bundle["tool_request"]["metric"], "bottle_gap")
        self.assertEqual(provider.calls, 2)
        self.assertIn("lower_snake_case", bundle["provider"]["errors"][0])

    def test_generated_checker_context_forbids_official_success_alias(self):
        root = Path(__file__).resolve().parents[2]
        forbidden = {
            "schema_version": 1,
            "task_name": "adjust_bottle",
            "metric": "official_check_success",
            "question": "Did the task succeed?",
        }
        repaired = {
            "schema_version": 1,
            "task_name": "adjust_bottle",
            "metric": "generated_check_success",
            "question": "Did the generated checker pass?",
        }
        provider = _RetryProvider([forbidden, repaired])
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")

        bundle = agent.propose(
            source_query="Which condition fails?",
            semantic_concern="query-derived experimental scene",
            tool_need="measure the generated checker outcome",
            task_name="adjust_bottle",
            generated_checker_semantics=True,
        )

        self.assertEqual(
            bundle["tool_request"]["metric"],
            "generated_check_success",
        )
        self.assertEqual(provider.calls, 2)
        registry_names = {
            item["name"]
            for item in bundle["context"]["tool_registry"]["trusted_tools"]
        }
        self.assertNotIn("official_check_success", registry_names)
        self.assertNotIn("time_to_success", registry_names)

    def test_event_count_and_time_between_event_shapes_are_executable(self):
        selector = {
            "event_type": "contact_interval",
            "actors": ["bottle", "left_tcp"],
            "physical_only": True,
        }
        count_request = {
            "schema_version": 2,
            "task_name": "adjust_bottle",
            "metric": "bottle_contact_count",
            "question": "How many physical contact intervals occurred?",
            "metric_spec": {
                "schema_version": 1,
                "operation": "event_count",
                "event": selector,
                "unit": "count",
                "null_semantics": "zero_if_absent",
            },
        }
        elapsed_request = {
            "schema_version": 2,
            "task_name": "adjust_bottle",
            "metric": "contact_to_generated_success_time",
            "question": "How long elapsed from contact to success?",
            "metric_spec": {
                "schema_version": 1,
                "operation": "time_between_events",
                "start_event": selector,
                "end_event": {
                    "event_type": "success_transition",
                    "actors": None,
                    "physical_only": False,
                },
                "unit": "s",
                "null_semantics": "null_if_missing_or_reversed",
            },
        }
        actors = {"bottle", "left_tcp"}

        self.assertEqual(
            validate_open_tool_request(
                count_request,
                task_name="adjust_bottle",
                available_actor_ids=actors,
            )["metric_spec"]["operation"],
            "event_count",
        )
        self.assertEqual(
            validate_open_tool_request(
                elapsed_request,
                task_name="adjust_bottle",
                available_actor_ids=actors,
            )["metric_spec"]["operation"],
            "time_between_events",
        )

    def test_active_arm_need_rejects_fixed_side_metric_spec(self):
        request = _typed("click_bell")
        request["metric_spec"]["left_signal"] = "left_tcp_position"
        request["metric_spec"]["right_signal"] = "bell_contact_position"

        with self.assertRaisesRegex(
            OpenToolRequestError,
            "fixed-side minimum_distance",
        ):
            validate_open_tool_request(
                request,
                task_name="click_bell",
                available_signal_names={
                    "left_tcp_position",
                    "right_tcp_position",
                    "bell_contact_position",
                },
                available_signal_sides={
                    "left_tcp_position": "left",
                    "right_tcp_position": "right",
                },
                measurement_need=(
                    "Measure target-contact distance with the active gripper."
                ),
            )

    def test_active_arm_need_retries_to_registered_composite_target(self):
        root = Path(__file__).resolve().parents[2]
        invalid = _typed("click_bell")
        invalid["metric_spec"]["left_signal"] = "left_tcp_position"
        invalid["metric_spec"]["right_signal"] = "bell_contact_position"
        repaired = {
            "schema_version": 1,
            "task_name": "click_bell",
            "metric": "bell_active_tcp_min_xy_error",
            "question": (
                "What was the minimum XY distance between the official "
                "active-arm TCP and the bell contact point?"
            ),
        }
        provider = _RetryProvider([invalid, repaired])
        agent = OpenToolRequestAgent(root, provider, model="fixture-model")

        bundle = agent.propose(
            source_query="Where does bell-position generalization fail?",
            semantic_concern="bounded bell translation",
            tool_need=(
                "Measure target-contact distance with the active gripper."
            ),
            task_name="click_bell",
        )

        self.assertEqual(
            bundle["tool_request"]["metric"],
            "bell_active_tcp_min_xy_error",
        )
        self.assertEqual(provider.calls, 2)
        self.assertIn(
            "fixed-side minimum_distance",
            bundle["provider"]["errors"][0],
        )


if __name__ == "__main__":
    unittest.main()
