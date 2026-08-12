"""Evidence aggregation shared by round execution and paper replays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mea.tool_results import episode_tool_results
from mea.toolkit import aggregate_tool_executions


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def compact_tool_evaluation(
    tool_evaluation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep planned Tool evidence compact while preserving episode roles."""

    if not tool_evaluation:
        return None
    compact_episodes: list[dict[str, Any]] = []
    for item in tool_evaluation.get("episodes", []):
        if not isinstance(item, Mapping):
            continue
        for result in episode_tool_results(item):
            compact_episodes.append(
                {
                    "policy_name": item.get("policy_name"),
                    "seed": item.get("seed"),
                    "role": item.get("role"),
                    "metric": (
                        result.get("tool")
                        or tool_evaluation.get("reference_tool")
                    ),
                    "value": result.get("value"),
                    "unit": result.get("unit"),
                    "passed": result.get("passed"),
                    "evidence_steps": result.get("evidence_steps", []),
                    "details": result.get("details", {}),
                }
            )
    return {
        "status": tool_evaluation.get("status"),
        "requested_route": tool_evaluation.get("requested_route"),
        "route": tool_evaluation.get("route"),
        "reference_tool": tool_evaluation.get("reference_tool"),
        "route_decision": tool_evaluation.get("route_decision", {}),
        "source": tool_evaluation.get("source", {}),
        "episodes": compact_episodes,
        "validation": tool_evaluation.get("validation", {}),
    }


def aggregate_sources(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    tool_evaluation: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Build one de-duplicated set of episode ToolResult sources."""

    context = {
        "round_id": round_plan["round_id"],
        "variant": round_plan.get("template_id")
        or round_plan.get("sub_aspect")
        or round_plan.get("route"),
    }
    sources: list[dict[str, Any]] = []
    trusted = child_manifest.get("trusted_tool_evaluation") or {}
    trusted_tools = {
        result.get("tool")
        for episode in trusted.get("episodes", [])
        if isinstance(episode, Mapping)
        for result in episode_tool_results(episode)
        if result.get("tool")
    }
    if trusted.get("episodes"):
        sources.append(
            {
                **trusted,
                "context": {
                    **context,
                    "source_artifact": trusted.get("artifact"),
                },
            }
        )
    if tool_evaluation and tool_evaluation.get("episodes"):
        request = tool_evaluation.get("tool_request") or tool_evaluation.get(
            "tool_spec", {}
        )
        metric = request.get("metric") if isinstance(request, dict) else None
        if metric not in trusted_tools:
            sources.append(
                {
                    "tool_execution": tool_evaluation,
                    "context": {
                        **context,
                        "source_artifact": tool_evaluation.get(
                            "artifacts", {}
                        ).get("tool_execution"),
                    },
                }
            )
    return sources


def aggregate_round_results(
    round_plan: dict[str, Any],
    child_manifest: dict[str, Any],
    tool_evaluation: dict[str, Any] | None,
    output_path: Path,
) -> dict[str, Any]:
    sources = aggregate_sources(round_plan, child_manifest, tool_evaluation)
    if not sources:
        result = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no episode ToolResult rows were available",
            "metrics": [],
        }
        _write_json(output_path, result)
        return result
    return aggregate_tool_executions(sources, output_path=output_path)


def aggregate_evaluation_results(
    round_runs: list[dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    sources = [
        source
        for item in round_runs
        for source in aggregate_sources(
            item["round_plan"],
            item["child_manifest"],
            item["tool_evaluation"],
        )
    ]
    if not sources:
        result = {
            "schema_version": 1,
            "status": "skipped",
            "reason": "no completed round ToolResult rows were available",
            "metrics": [],
        }
        _write_json(output_path, result)
        return result
    return aggregate_tool_executions(sources, output_path=output_path)


__all__ = [
    "aggregate_evaluation_results",
    "aggregate_round_results",
    "aggregate_sources",
    "compact_tool_evaluation",
]
