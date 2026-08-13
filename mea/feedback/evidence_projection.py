"""Pure projections from raw evaluation evidence to compact report data."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


class EvidenceReportError(RuntimeError):
    """Raised when an evaluation cannot be represented without guessing."""


_SAFE_ARTIFACT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _read_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise EvidenceReportError(f"required JSON artifact is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceReportError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceReportError(f"JSON artifact must contain an object: {path}")
    return value


def _safe_artifact_id(raw: Any, *, label: str) -> str:
    value = str(raw)
    if value in {".", ".."} or _SAFE_ARTIFACT_ID.fullmatch(value) is None:
        raise EvidenceReportError(f"{label} is not a safe artifact id: {value!r}")
    return value


def _target_scope(target: Mapping[str, Any]) -> dict[str, Any]:
    """Project legacy or schema-v3 targets into one report-only scope."""

    binding = target.get("policy_task_binding")
    if isinstance(binding, Mapping):
        return {
            "binding_mode": target.get("binding_mode"),
            "task_name": binding.get("task_name"),
            "task_profile": None,
            "policy": deepcopy(
                dict(binding.get("policy"))
                if isinstance(binding.get("policy"), Mapping)
                else {}
            ),
            "checkpoint": deepcopy(
                dict(binding.get("checkpoint"))
                if isinstance(binding.get("checkpoint"), Mapping)
                else {}
            ),
        }
    return {
        "binding_mode": target.get("binding_mode"),
        "task_name": target.get("task_name"),
        "task_profile": target.get("task_profile"),
        "policy": deepcopy(
            dict(target.get("policy"))
            if isinstance(target.get("policy"), Mapping)
            else {}
        ),
        "checkpoint": deepcopy(
            dict(target.get("checkpoint"))
            if isinstance(target.get("checkpoint"), Mapping)
            else {}
        ),
    }



def _compact_tool_rows(tool: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode in tool.get("episodes") or []:
        if not isinstance(episode, Mapping):
            continue
        result = episode.get("result") if isinstance(episode.get("result"), Mapping) else {}
        rows.append(
            {
                "role": episode.get("role"),
                "policy_name": episode.get("policy_name"),
                "seed": episode.get("seed"),
                "value": result.get("value"),
                "unit": result.get("unit"),
                "passed": result.get("passed"),
            }
        )
    return rows


def _compact_vqa(vqa: Mapping[str, Any]) -> dict[str, Any]:
    query = vqa.get("query") if isinstance(vqa.get("query"), Mapping) else {}
    observation = (
        vqa.get("observation")
        if isinstance(vqa.get("observation"), Mapping)
        else {}
    )
    return {
        "status": vqa.get("status"),
        "questions": [
            {"id": item.get("id"), "question": item.get("question")}
            for item in query.get("questions") or []
            if isinstance(item, Mapping)
        ],
        "phenomena": [
            {
                "id": item.get("id"),
                "observed": item.get("observed"),
                "description": item.get("description"),
                "confidence": item.get("confidence"),
                "frame_ids": item.get("frame_ids"),
            }
            for item in observation.get("phenomena") or []
            if isinstance(item, Mapping)
        ],
        "numeric_consistency": observation.get("numeric_consistency"),
        "evidence_conflict": vqa.get("evidence_conflict"),
    }


def _compact_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": aggregate.get("status"),
        "source_count": aggregate.get("source_count"),
        "episode_result_count": aggregate.get("episode_result_count"),
        "unique_episode_count": aggregate.get("unique_episode_count"),
        "metric_ids": [
            item.get("metric")
            for item in aggregate.get("metrics") or []
            if isinstance(item, Mapping) and item.get("metric")
        ],
        "input_issue_count": len(aggregate.get("input_issues") or []),
    }


def _without_provenance(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_provenance(item)
            for key, item in value.items()
            if key != "provenance"
        }
    if isinstance(value, list):
        return [_without_provenance(item) for item in value]
    return deepcopy(value)


def _semantic_aggregate(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    """Keep Aggregate semantics while dropping repeated sample provenance."""

    metrics = []
    for raw_metric in aggregate.get("metrics") or []:
        if not isinstance(raw_metric, Mapping):
            continue
        cohorts = []
        for raw_cohort in raw_metric.get("cohorts") or []:
            if not isinstance(raw_cohort, Mapping):
                continue
            cohort = {
                "role": raw_cohort.get("role"),
                "policy_names": deepcopy(raw_cohort.get("policy_names") or []),
                "summary": _without_provenance(
                    raw_cohort.get("summary") or {}
                ),
            }
            if raw_cohort.get("passed_summary") is not None:
                cohort["passed_summary"] = _without_provenance(
                    raw_cohort.get("passed_summary")
                )
            cohorts.append(cohort)
        metrics.append(
            {
                "metric": raw_metric.get("metric"),
                "tools": deepcopy(raw_metric.get("tools") or []),
                "unit": raw_metric.get("unit"),
                "value_kind": raw_metric.get("value_kind"),
                "cohorts": cohorts,
            }
        )
    return {
        "schema_version": 1,
        **_compact_aggregate(aggregate),
        "input_issues": _without_provenance(
            aggregate.get("input_issues") or []
        ),
        "metrics": metrics,
    }


def _compact_decision(value: Any) -> dict[str, Any]:
    decision = dict(value) if isinstance(value, Mapping) else {}
    assessment = (
        dict(decision.get("query_assessment"))
        if isinstance(decision.get("query_assessment"), Mapping)
        else dict(decision.get("evidence_assessment"))
        if isinstance(decision.get("evidence_assessment"), Mapping)
        else {}
    )
    return {
        "action": decision.get("action"),
        "transition": decision.get("transition"),
        "decision_reason": decision.get("decision_reason"),
        "observation_summary": decision.get("observation_summary"),
        "answered_query": decision.get("answered_query"),
        "evidence_sufficient": assessment.get("evidence_sufficient"),
        "claim_verdict": assessment.get("claim_verdict"),
        "stop_reason": assessment.get("stop_reason"),
    }


def _resolve_child_ids(manifest: Mapping[str, Any], rounds: list[dict[str, Any]]) -> list[str | None]:
    ids = list(manifest.get("child_run_ids") or [])
    result: list[str | None] = []
    for index, round_plan in enumerate(rounds):
        run_id = ids[index] if index < len(ids) else None
        if run_id is None:
            run_id = round_plan.get("taskgen_run_id")
        result.append(str(run_id) if run_id else None)
    return result




__all__ = [
    "EvidenceReportError",
    "_compact_aggregate",
    "_compact_decision",
    "_compact_tool_rows",
    "_compact_vqa",
    "_read_json",
    "_resolve_child_ids",
    "_safe_artifact_id",
    "_semantic_aggregate",
    "_target_scope",
]
