import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from mea.providers import OpenAICompatibleProvider
from mea.runtime_ledger import (
    LEDGER_CONTEXT_ENV,
    LEDGER_PATH_ENV,
    RuntimeLedgerError,
    read_runtime_ledger,
    record_act_batch_start,
    runtime_ledger_context,
    summarize_runtime_ledger,
    validate_runtime_context,
)


def context(**updates):
    value = {
        "schema_version": 1,
        "evaluation_id": "eval_ledger",
        "logical_round_id": "round_1",
        "round_attempt_index": 1,
        "child_run_id": "run_eval_ledger_round_1",
    }
    value.update(updates)
    return value


def successful_response(content="ok"):
    response = Mock(status_code=200)
    response.json.return_value = {
        "id": "provider-response-id",
        "model": "gateway-model",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"total_tokens": 3},
    }
    return response


def provider_with(session):
    return OpenAICompatibleProvider(
        api_key="api-key-must-not-be-logged",
        base_url="https://secret-host.invalid/v1",
        session=session,
        max_retries=2,
        retry_delay=0,
    )


class RuntimeLedgerTests(unittest.TestCase):
    def test_context_requires_exact_fields_and_positive_attempt(self):
        self.assertEqual(validate_runtime_context(context()), context())
        with self.assertRaisesRegex(RuntimeLedgerError, "fields must be exactly"):
            validate_runtime_context({**context(), "prompt": "must not exist"})
        with self.assertRaisesRegex(RuntimeLedgerError, "positive integer"):
            validate_runtime_context(context(round_attempt_index=True))

    def test_text_call_is_durably_logged_without_request_secrets(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "provider_starts.jsonl"
            session = Mock()
            session.post.return_value = successful_response()
            provider = provider_with(session)
            prompt = "PRIVATE PROMPT CONTENT"
            system = "PRIVATE SYSTEM CONTENT"

            with runtime_ledger_context(ledger, context()):
                self.assertEqual(provider.text(prompt, system=system), "ok")

            self.assertEqual(session.post.call_count, 1)
            events = read_runtime_ledger(ledger, expected_context=context())
            self.assertEqual(len(events), 1)
            event = events[0]
            self.assertEqual(event["event_type"], "provider_transport_started")
            self.assertEqual(event["transport_attempt"], 1)
            self.assertEqual(event["modality"], "text")
            self.assertEqual(event["model"], "gpt-4o-mini")
            self.assertRegex(event["logical_call_id"], r"^[0-9a-f]{32}$")
            self.assertEqual(
                set(event),
                {
                    "schema_version",
                    "evaluation_id",
                    "logical_round_id",
                    "round_attempt_index",
                    "child_run_id",
                    "event_type",
                    "recorded_at",
                    "logical_call_id",
                    "transport_attempt",
                    "modality",
                    "model",
                },
            )
            raw = ledger.read_text(encoding="utf-8")
            for forbidden in (
                prompt,
                system,
                "api-key-must-not-be-logged",
                "secret-host.invalid",
                "Authorization",
                "messages",
            ):
                self.assertNotIn(forbidden, raw)

    def test_vision_call_does_not_log_prompt_image_or_data_url(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "provider_starts.jsonl"
            image = root / "private-frame-name.png"
            image.write_bytes(b"private-image-bytes")
            session = Mock()
            session.post.return_value = successful_response("visible")
            provider = provider_with(session)

            with runtime_ledger_context(ledger, context()):
                self.assertEqual(
                    provider.vision("PRIVATE VISION PROMPT", image), "visible"
                )

            events = read_runtime_ledger(ledger, expected_context=context())
            self.assertEqual(events[0]["modality"], "vision")
            raw = ledger.read_text(encoding="utf-8")
            for forbidden in (
                "PRIVATE VISION PROMPT",
                image.name,
                "private-image-bytes",
                "data:image",
                "image_url",
            ):
                self.assertNotIn(forbidden, raw)

    def test_retries_share_one_logical_call_and_count_each_transport_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "provider_starts.jsonl"
            session = Mock()
            session.post.side_effect = [
                requests.ReadTimeout("transient"),
                successful_response("recovered"),
                successful_response("second logical call"),
            ]
            provider = provider_with(session)

            with runtime_ledger_context(ledger, context()):
                self.assertEqual(provider.text("first"), "recovered")
                self.assertEqual(provider.text("second"), "second logical call")

            events = read_runtime_ledger(ledger, expected_context=context())
            self.assertEqual(session.post.call_count, 3)
            self.assertEqual([item["transport_attempt"] for item in events], [1, 2, 1])
            self.assertEqual(events[0]["logical_call_id"], events[1]["logical_call_id"])
            self.assertNotEqual(events[1]["logical_call_id"], events[2]["logical_call_id"])
            summary = summarize_runtime_ledger(
                ledger, expected_context=context()
            )
            self.assertTrue(summary["provider_called"])
            self.assertEqual(summary["provider_calls_started"], 2)
            self.assertEqual(summary["provider_transport_attempts_started"], 3)
            self.assertEqual(
                summary["by_modality"]["text"],
                {
                    "logical_calls_started": 2,
                    "transport_attempts_started": 3,
                },
            )
            self.assertRegex(summary["ledger_sha256"], r"^[0-9a-f]{64}$")

    def test_act_batch_is_recorded_before_launch_and_counted_conservatively(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "runtime_starts.jsonl"
            with runtime_ledger_context(ledger, context()):
                event = record_act_batch_start(
                    task_name="click_bell",
                    policy_name="ACT",
                    start_seed=100401,
                    num_rollouts=3,
                )
            self.assertEqual(event["event_type"], "act_batch_started")
            self.assertNotIn("checkpoint", ledger.read_text(encoding="utf-8"))
            summary = summarize_runtime_ledger(ledger, expected_context=context())
            self.assertEqual(summary["provider_calls_started"], 0)
            self.assertEqual(summary["act_batches_started"], 1)
            self.assertEqual(summary["act_rollouts_started"], 3)

    def test_http_status_retry_is_logged_before_both_posts(self):
        unavailable = Mock(status_code=502, text="temporary")
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "provider_starts.jsonl"
            session = Mock()
            session.post.side_effect = [unavailable, successful_response("recovered")]
            provider = provider_with(session)

            observed_counts = []

            def observe_then_respond(*args, **kwargs):
                observed_counts.append(len(ledger.read_text(encoding="utf-8").splitlines()))
                return [unavailable, successful_response("recovered")][
                    len(observed_counts) - 1
                ]

            session.post.side_effect = observe_then_respond
            with runtime_ledger_context(ledger, context()):
                self.assertEqual(provider.text("retry"), "recovered")
            self.assertEqual(observed_counts, [1, 2])

    def test_ledger_write_failure_prevents_external_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "provider_starts.jsonl"
            session = Mock()
            session.post.return_value = successful_response()
            provider = provider_with(session)
            with runtime_ledger_context(ledger, context()):
                with patch(
                    "mea.runtime_ledger.os.open",
                    side_effect=PermissionError("read only"),
                ):
                    with self.assertRaisesRegex(RuntimeLedgerError, "cannot open"):
                        provider.text("must not be sent")
            session.post.assert_not_called()

    def test_partial_environment_configuration_fails_before_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            session = Mock()
            provider = provider_with(session)
            with patch.dict(os.environ, {LEDGER_PATH_ENV: str(Path(temporary) / "x")}, clear=False):
                os.environ.pop(LEDGER_CONTEXT_ENV, None)
                with self.assertRaisesRegex(RuntimeLedgerError, "must be set together"):
                    provider.text("must not be sent")
            session.post.assert_not_called()

    def test_existing_context_mismatch_and_unknown_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "provider_starts.jsonl"
            first_session = Mock()
            first_session.post.return_value = successful_response()
            with runtime_ledger_context(ledger, context()):
                provider_with(first_session).text("first")
            first_line = ledger.read_text(encoding="utf-8")

            second_session = Mock()
            second_session.post.return_value = successful_response()
            with runtime_ledger_context(
                ledger, context(round_attempt_index=2)
            ):
                with self.assertRaisesRegex(RuntimeLedgerError, "context does not match"):
                    provider_with(second_session).text("second")
            second_session.post.assert_not_called()
            self.assertEqual(ledger.read_text(encoding="utf-8"), first_line)

            event = json.loads(first_line)
            event["prompt"] = "forbidden extra field"
            ledger.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeLedgerError, "fields must be exactly"):
                read_runtime_ledger(ledger, expected_context=context())

    def test_context_manager_restores_environment_and_never_truncates(self):
        with tempfile.TemporaryDirectory() as temporary:
            ledger = Path(temporary) / "provider_starts.jsonl"
            ledger.write_text("existing-prefix\n", encoding="utf-8")
            previous_path = os.environ.get(LEDGER_PATH_ENV)
            previous_context = os.environ.get(LEDGER_CONTEXT_ENV)
            with runtime_ledger_context(ledger, context()):
                self.assertEqual(Path(os.environ[LEDGER_PATH_ENV]), ledger.resolve())
            self.assertEqual(ledger.read_text(encoding="utf-8"), "existing-prefix\n")
            self.assertEqual(os.environ.get(LEDGER_PATH_ENV), previous_path)
            self.assertEqual(os.environ.get(LEDGER_CONTEXT_ENV), previous_context)


if __name__ == "__main__":
    unittest.main()
