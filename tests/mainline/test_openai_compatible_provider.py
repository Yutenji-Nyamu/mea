from unittest.mock import Mock

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
