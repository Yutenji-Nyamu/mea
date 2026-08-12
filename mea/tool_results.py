"""Small compatibility projection for episode Tool results.

Current runtimes store a list under ``tool_results``.  Historical episodes
may contain one direct ``result`` instead.  Keeping this reader next to the
round evidence code avoids importing paper-only flagship acceptance logic into
the production method loop.
"""

from __future__ import annotations

from typing import Any, Mapping


def episode_tool_results(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return current or historical Tool-result envelopes uniformly."""

    raw_results = episode.get("tool_results")
    if not isinstance(raw_results, list):
        direct_result = episode.get("result")
        raw_results = [direct_result] if isinstance(direct_result, dict) else []
    return [result for result in raw_results if isinstance(result, dict)]


__all__ = ["episode_tool_results"]
