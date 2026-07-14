import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from mea.providers import OpenAICompatibleProvider, ProviderError


class OpenAICompatibleProviderTests(unittest.TestCase):
    def make_provider(self, response):
        session = Mock()
        session.post.return_value = response
        return OpenAICompatibleProvider(api_key="test-key", session=session), session

    def test_text_returns_content_and_metadata(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "id": "request-1",
            "model": "test-model",
            "choices": [
                {"message": {"content": "ok"}, "finish_reason": "stop"}
            ],
            "usage": {"total_tokens": 3},
        }
        provider, session = self.make_provider(response)

        self.assertEqual(provider.text("hello"), "ok")
        self.assertEqual(provider.last_metadata["model"], "test-model")
        headers = session.post.call_args.kwargs["headers"]
        self.assertEqual(headers["Authorization"], "Bearer test-key")

    def test_vision_embeds_local_image_as_data_url(self):
        response = Mock(status_code=200)
        response.json.return_value = {
            "choices": [{"message": {"content": "blue"}}]
        }
        provider, session = self.make_provider(response)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            path.write_bytes(b"png-bytes")
            self.assertEqual(provider.vision("color?", path), "blue")

        payload = session.post.call_args.kwargs["json"]
        url = payload["messages"][0]["content"][1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        self.assertTrue(url.endswith(base64.b64encode(b"png-bytes").decode("ascii")))

    def test_http_error_does_not_include_api_key(self):
        response = Mock(status_code=401, text="unauthorized")
        provider, _ = self.make_provider(response)

        with self.assertRaises(ProviderError) as raised:
            provider.text("hello")
        self.assertNotIn("test-key", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
