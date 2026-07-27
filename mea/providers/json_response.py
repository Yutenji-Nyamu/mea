"""Small shared parser for JSON objects returned by model providers."""

from __future__ import annotations

import json
import re
from typing import Any


class ProviderJSONError(ValueError):
    """Raised when a provider response contains no parseable JSON object."""


def extract_json_response(response: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating a Markdown JSON fence or surrounding text."""

    source = str(response)
    candidates = re.findall(
        r"```(?:json)?\s*(.*?)```",
        source,
        flags=re.DOTALL | re.IGNORECASE,
    )
    candidates.append(source.strip())
    match = re.search(r"\{.*\}", source, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            value = json.loads(candidate.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ProviderJSONError("provider did not return a parseable JSON object")


__all__ = ["ProviderJSONError", "extract_json_response"]
