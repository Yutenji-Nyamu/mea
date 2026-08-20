from unittest.mock import Mock, patch

import pytest

from mea.providers import OpenAICompatibleProvider, ProviderError


def _response(payload, *, text="response"):
    response = Mock(status_code=200, text=text)
    response.json.return_value = payload
    return response


def test_retries_invalid_completion_content_within_existing_budget():
    session = Mock()
    session.post.side_effect = [
        _response({"choices": [{"message": {"content": "  "}}]}),
        _response({"choices": []}, text="missing choice"),
        _response(
            {
                "model": "test-model",
                "choices": [{"message": {"content": "recovered"}}],
            }
        ),
    ]
    provider = OpenAICompatibleProvider(
        api_key="test-key", session=session, max_retries=2, retry_delay=0
    )

    assert provider.text("hello") == "recovered"
    assert session.post.call_count == 3
    assert provider.last_metadata["retry_count"] == 2


def test_invalid_completion_exhausts_existing_retry_budget():
    session = Mock()
    session.post.side_effect = [
        _response({"choices": []}, text="missing choice"),
        _response({"choices": []}, text="missing choice"),
    ]
    provider = OpenAICompatibleProvider(
        api_key="test-key", session=session, max_retries=1, retry_delay=0
    )

    with pytest.raises(ProviderError, match="invalid chat completion"):
        provider.text("hello")
    assert session.post.call_count == 2


def test_default_retry_window_recovers_from_short_gateway_saturation():
    saturated = Mock(status_code=429, text="upstream capacity saturated")
    recovered = _response(
        {
            "model": "test-model",
            "choices": [{"message": {"content": "recovered"}}],
        }
    )
    session = Mock()
    session.post.side_effect = [
        saturated,
        saturated,
        recovered,
    ]
    provider = OpenAICompatibleProvider(api_key="test-key", session=session)

    with patch("mea.providers.openai_compatible.time.sleep") as sleep:
        assert provider.text("hello") == "recovered"

    assert session.post.call_count == 3
    assert [call.args[0] for call in sleep.call_args_list] == [1.0, 2.0]
    assert provider.last_metadata["retry_count"] == 2
