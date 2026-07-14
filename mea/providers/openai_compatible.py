"""OpenAI-compatible text and vision provider used for UIUI and similar gateways."""

import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

import requests

from .base import MultimodalProvider


class ProviderError(RuntimeError):
    """Raised when an OpenAI-compatible gateway returns an invalid response."""


class OpenAICompatibleProvider(MultimodalProvider):
    """Small provider with no gateway-specific dependency or persisted credentials."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "UIUI_API_KEY",
        text_model: str = "gpt-4o-mini",
        vision_model: str = "gpt-4o",
        timeout: float = 60.0,
        max_retries: int = 2,
        retry_delay: float = 1.0,
        session: requests.Session | None = None,
    ):
        self.base_url = (
            base_url or os.getenv("UIUI_BASE_URL") or "https://api.uiuihao.com/v1"
        ).rstrip("/")
        self.api_key = api_key or os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing API key: set {api_key_env}")

        self.text_model = text_model
        self.vision_model = vision_model
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.session = session or requests.Session()
        self.last_metadata: dict[str, Any] = {}

    def _complete(self, payload: dict[str, Any]) -> str:
        response = None
        retry_count = 0
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                retry_count = attempt
                break
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt >= self.max_retries:
                    raise ProviderError(
                        "Gateway transient request failed after "
                        f"{attempt + 1} attempts: {type(exc).__name__}"
                    ) from exc
                time.sleep(self.retry_delay * (attempt + 1))
        if response is None:
            raise ProviderError("Gateway request did not produce a response")
        if response.status_code >= 400:
            body = response.text[:1000]
            raise ProviderError(
                f"Gateway request failed with HTTP {response.status_code}: {body}"
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(
                f"Gateway returned an invalid chat completion: {response.text[:1000]}"
            ) from exc

        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Gateway returned empty assistant content")

        self.last_metadata = {
            "id": data.get("id"),
            "model": data.get("model"),
            "finish_reason": data.get("choices", [{}])[0].get("finish_reason"),
            "usage": data.get("usage"),
            "retry_count": retry_count,
        }
        return content.strip()

    def text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self._complete(
            {
                "model": model or self.text_model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

    def vision(
        self,
        prompt: str,
        image_path: str | Path,
        *,
        model: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)

        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        image_data = base64.b64encode(path.read_bytes()).decode("ascii")
        return self._complete(
            {
                "model": model or self.vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
