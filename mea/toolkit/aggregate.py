"""Deterministic cross-episode aggregation for MEA Tool results.

The aggregate layer consumes normalized Tool execution envelopes.  It does not
read simulator state and does not use an LLM: all counts and statistics are
computed by trusted Python code.  Policy-under-evaluation and expert-validation
episodes are intentionally emitted as separate cohorts so an expert result can
never enter a policy mean or rate.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class AggregateToolkitError(ValueError):
    """Raised when Aggregate Toolkit input violates its public contract."""


_ROLE_ORDER = {
    "policy_under_evaluation": 0,
    "expert_validation": 1,
    "validation_control": 2,
}
_GROUP_DIMENSIONS = ("seed", "round_id", "variant", "policy_name")
_CONTACT_BEFORE_PICKUP_REASONS = {
    "contact_before_pickup",
    "contact_precedes_pickup",
}


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _stable_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, _canonical(value)


def _role(policy_name: Any) -> str:
    normalized = str(policy_name or "").casefold()
    if normalized == "act":
        return "policy_under_evaluation"
    if normalized == "expert":
        return "expert_validation"
    return "validation_control"


def _source_payload(source: Any, source_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return one execution payload and its optional aggregate context."""

    source_artifact: str | None = None
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve()
        try:
            source = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AggregateToolkitError(
                f"cannot load aggregate source {path}: {exc}"
            ) from exc
        source_artifact = str(path)
    if not isinstance(source, Mapping):
        raise AggregateToolkitError(
            f"aggregate source {source_index} must be an object or JSON path"
        )

    wrapper = dict(source)
    context = wrapper.get("context", {})
    if context is None:
        context = {}
    if not isinstance(context, Mapping):
        raise AggregateToolkitError(
            f"aggregate source {source_index}.context must be an object"
        )
    normalized_context = dict(context)
    for key in ("round_id", "variant", "variant_id", "source_artifact"):
        if key in wrapper and key not in normalized_context:
            normalized_context[key] = wrapper[key]
    if source_artifact is not None:
        normalized_context.setdefault("source_artifact", source_artifact)

    payload: Any
    if "tool_execution" in wrapper:
        payload = wrapper["tool_execution"]
    elif "execution" in wrapper:
        payload = wrapper["execution"]
    else:
        payload = wrapper
    if not isinstance(payload, Mapping):
        raise AggregateToolkitError(
            f"aggregate source {source_index} execution must be an object"
        )
    return dict(payload), normalized_context


def _pick_context(
    episode: Mapping[str, Any],
    execution: Mapping[str, Any],
    context: Mapping[str, Any],
    *names: str,
) -> Any:
    for container in (episode, context, execution):
        for name in names:
            if name in container:
                return container[name]
    return None


def _metric_for(
    execution: Mapping[str, Any], result: Mapping[str, Any]
) -> str | None:
    for container in (
        execution.get("tool_request"),
        execution.get("tool_spec"),
        execution.get("route_decision"),
    ):
        if isinstance(container, Mapping):
            metric = container.get("metric")
            if isinstance(metric, str) and metric.strip():
                return metric.strip()
    tool = result.get("tool")
    return tool.strip() if isinstance(tool, str) and tool.strip() else None


def _episode_results(episode: Mapping[str, Any]) -> list[Any]:
    if "result" in episode:
        return [episode["result"]]
    results = episode.get("tool_results")
    if isinstance(results, list):
        return list(results)
    return [None]


def _normalize_evidence_steps(value: Any) -> tuple[list[int], str | None]:
    if value is None:
        return [], None
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in value
    ):
        return [], "evidence_steps_must_be_integer_list"
    return sorted(set(value)), None


def _normalize_rows(sources: Sequence[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    input_issues: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        execution, context = _source_payload(source, source_index)
        episodes = execution.get("episodes")
        if not isinstance(episodes, list):
            raise AggregateToolkitError(
                f"aggregate source {source_index}.episodes must be a list"
            )
        execution_status = execution.get("status")
        if execution_status not in (None, "passed"):
            input_issues.append(
                {
                    "source_index": source_index,
                    "status": execution_status,
                    "reason": "execution_not_passed",
                    "source_artifact": context.get("source_artifact"),
                }
            )
        for episode_index, raw_episode in enumerate(episodes):
            if not isinstance(raw_episode, Mapping):
                raise AggregateToolkitError(
                    f"source {source_index} episode {episode_index} must be an object"
                )
            episode = dict(raw_episode)
            metadata = episode.get("metadata")
            if isinstance(metadata, Mapping):
                merged = dict(metadata)
                merged.update(episode)
                episode = merged
            policy_name = episode.get("policy_name")
            declared_role = episode.get("role")
            derived_role = _role(policy_name)
            recognized_policy = derived_role != "validation_control"
            role_mismatch = bool(
                recognized_policy
                and declared_role is not None
                and declared_role != derived_role
            )
            role = (
                derived_role
                if recognized_policy
                else declared_role or derived_role
            )
            for result_index, raw_result in enumerate(_episode_results(episode)):
                result = dict(raw_result) if isinstance(raw_result, Mapping) else {}
                metric = _metric_for(execution, result)
                if metric is None:
                    metric = "__unknown_metric__"
                evidence_steps, evidence_error = _normalize_evidence_steps(
                    result.get("evidence_steps")
                )
                details = result.get("details")
                details = dict(details) if isinstance(details, Mapping) else {}
                reason = details.get("reason")
                value = result.get("value")
                passed_present = "passed" in result and result.get("passed") is not None
                passed_value = result.get("passed")
                status: str
                invalid_reason: str | None = None
                candidate_kind: str | None = None
                if execution_status not in (None, "passed"):
                    status = "invalid"
                    invalid_reason = "execution_not_passed"
                elif role_mismatch:
                    status = "invalid"
                    invalid_reason = "policy_name_role_mismatch"
                elif not isinstance(raw_result, Mapping):
                    status = "invalid"
                    invalid_reason = "result_must_be_an_object"
                elif evidence_error:
                    status = "invalid"
                    invalid_reason = evidence_error
                elif value is None and reason in _CONTACT_BEFORE_PICKUP_REASONS:
                    status = "invalid"
                    invalid_reason = str(reason)
                elif value is None:
                    status = "missing"
                elif isinstance(value, bool):
                    status = "candidate_valid"
                    candidate_kind = "boolean"
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    if math.isfinite(float(value)):
                        status = "candidate_valid"
                        candidate_kind = "numeric"
                        value = float(value)
                    else:
                        status = "invalid"
                        invalid_reason = "numeric_value_must_be_finite"
                else:
                    status = "invalid"
                    invalid_reason = "value_must_be_boolean_numeric_or_null"

                if not passed_present:
                    passed_status = "missing"
                    passed_invalid_reason = None
                elif status == "invalid" and invalid_reason in {
                    "execution_not_passed",
                    "policy_name_role_mismatch",
                    "result_must_be_an_object",
                    "evidence_steps_must_be_integer_list",
                }:
                    passed_status = "invalid"
                    passed_invalid_reason = invalid_reason
                elif isinstance(passed_value, bool):
                    passed_status = "valid"
                    passed_invalid_reason = None
                else:
                    passed_status = "invalid"
                    passed_invalid_reason = "passed_must_be_boolean_or_null"

                variant = _pick_context(
                    episode, execution, context, "variant", "variant_id"
                )
                if isinstance(variant, (dict, list)):
                    variant = _canonical(variant)
                rows.append(
                    {
                        "metric": metric,
                        "tool": result.get("tool"),
                        "unit": result.get("unit"),
                        "value": value,
                        "candidate_kind": candidate_kind,
                        "status": status,
                        "invalid_reason": invalid_reason,
                        "details_reason": reason,
                        "passed": passed_value,
                        "passed_present": passed_present,
                        "passed_status": passed_status,
                        "passed_invalid_reason": passed_invalid_reason,
                        "episode_dir": episode.get("episode_dir"),
                        "seed": episode.get("seed"),
                        "round_id": _pick_context(
                            episode, execution, context, "round_id"
                        ),
                        "variant": variant,
                        "policy_name": policy_name,
                        "role": str(role),
                        "evidence_steps": evidence_steps,
                        "source_artifact": context.get("source_artifact"),
                        "source_index": source_index,
                        "episode_index": episode_index,
                        "result_index": result_index,
                    }
                )
    return rows, input_issues


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["metric"]),
        _ROLE_ORDER.get(str(row["role"]), 99),
        str(row["role"]),
        _stable_key(row.get("policy_name")),
        _stable_key(row.get("round_id")),
        _stable_key(row.get("variant")),
        _stable_key(row.get("seed")),
        str(row.get("episode_dir") or ""),
        int(row["source_index"]),
        int(row["episode_index"]),
        int(row["result_index"]),
    )


def _finalize_metric_rows(rows: list[dict[str, Any]]) -> tuple[str, Any]:
    kinds = sorted(
        {row["candidate_kind"] for row in rows if row["candidate_kind"]}
    )
    units = sorted(
        {row["unit"] for row in rows if row["status"] == "candidate_valid"},
        key=_stable_key,
    )
    if len(kinds) > 1:
        for row in rows:
            if row["status"] == "candidate_valid":
                row["status"] = "invalid"
                row["invalid_reason"] = "mixed_boolean_and_numeric_values"
        return "mixed", None
    kind = kinds[0] if kinds else "unknown"
    if len(units) > 1:
        for row in rows:
            if row["status"] == "candidate_valid":
                row["status"] = "invalid"
                row["invalid_reason"] = "mixed_units"
        return kind, None
    for row in rows:
        if row["status"] == "candidate_valid":
            row["status"] = "valid"
    return kind, units[0] if units else None


def _provenance(row: Mapping[str, Any], *, include_value: bool = False) -> dict[str, Any]:
    result = {
        "episode_dir": row.get("episode_dir"),
        "seed": row.get("seed"),
        "round_id": row.get("round_id"),
        "variant": row.get("variant"),
        "policy_name": row.get("policy_name"),
        "role": row.get("role"),
        "evidence_steps": list(row.get("evidence_steps", [])),
        "source_artifact": row.get("source_artifact"),
    }
    if row.get("details_reason") is not None:
        result["details_reason"] = row["details_reason"]
    if row.get("invalid_reason") is not None:
        result["invalid_reason"] = row["invalid_reason"]
    if include_value:
        result["value"] = row.get("value")
    return result


def _statistic(value: Any, rows: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "value": value,
        **extra,
        "provenance": [_provenance(row, include_value=True) for row in rows],
    }


def _summary(rows: list[dict[str, Any]], kind: str, unit: Any) -> dict[str, Any]:
    ordered = sorted(rows, key=_row_sort_key)
    valid = [row for row in ordered if row["status"] == "valid"]
    missing = [row for row in ordered if row["status"] == "missing"]
    invalid = [row for row in ordered if row["status"] == "invalid"]
    result: dict[str, Any] = {
        "value_kind": kind,
        "unit": unit,
        "episode_result_count": len(ordered),
        "quality": {
            "valid": _statistic(len(valid), valid),
            "missing": _statistic(len(missing), missing),
            "invalid": _statistic(len(invalid), invalid),
        },
        "statistics": {},
    }
    if kind == "boolean":
        true_rows = [row for row in valid if row["value"] is True]
        false_rows = [row for row in valid if row["value"] is False]
        denominator = len(valid)
        result["statistics"] = {
            "true_count": _statistic(len(true_rows), true_rows),
            "true_rate": _statistic(
                len(true_rows) / denominator if denominator else None,
                valid,
                numerator=len(true_rows),
                denominator=denominator,
            ),
            "false_count": _statistic(len(false_rows), false_rows),
            "false_rate": _statistic(
                len(false_rows) / denominator if denominator else None,
                valid,
                numerator=len(false_rows),
                denominator=denominator,
            ),
        }
    elif kind == "numeric":
        values = [float(row["value"]) for row in valid]
        if values:
            minimum = min(values)
            maximum = max(values)
            min_rows = [row for row in valid if float(row["value"]) == minimum]
            max_rows = [row for row in valid if float(row["value"]) == maximum]
            result["statistics"] = {
                "mean": _statistic(statistics.fmean(values), valid),
                "median": _statistic(statistics.median(values), valid),
                "min": _statistic(minimum, min_rows),
                "max": _statistic(maximum, max_rows),
                "population_stddev": _statistic(
                    statistics.pstdev(values), valid
                ),
            }
        else:
            result["statistics"] = {
                name: _statistic(None, [])
                for name in (
                    "mean",
                    "median",
                    "min",
                    "max",
                    "population_stddev",
                )
            }
    return result


def _passed_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not any(row.get("passed_present") for row in rows):
        return None
    predicate_rows: list[dict[str, Any]] = []
    for row in rows:
        predicate = dict(row)
        predicate["value"] = row.get("passed")
        predicate["status"] = row.get("passed_status", "missing")
        predicate["invalid_reason"] = row.get("passed_invalid_reason")
        predicate_rows.append(predicate)
    return _summary(predicate_rows, "boolean", None)


def _cohort(
    metric: str,
    role: str,
    rows: list[dict[str, Any]],
    kind: str,
    unit: Any,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for dimension in _GROUP_DIMENSIONS:
        buckets: dict[str, tuple[Any, list[dict[str, Any]]]] = {}
        for row in rows:
            value = row.get(dimension)
            key = _canonical(value)
            buckets.setdefault(key, (value, []))[1].append(row)
        groups[dimension] = [
            {
                "value": value,
                "summary": _summary(group_rows, kind, unit),
            }
            for _, (value, group_rows) in sorted(buckets.items())
        ]
    cohort = {
        "role": role,
        "policy_names": sorted(
            {row.get("policy_name") for row in rows}, key=_stable_key
        ),
        "summary": _summary(rows, kind, unit),
        "groups": groups,
    }
    if metric == "official_check_success" and kind == "boolean":
        cohort["summary"]["statistics"]["success_count"] = dict(
            cohort["summary"]["statistics"]["true_count"]
        )
        cohort["summary"]["statistics"]["success_rate"] = dict(
            cohort["summary"]["statistics"]["true_rate"]
        )
        for dimension_groups in groups.values():
            for group in dimension_groups:
                group["summary"]["statistics"]["success_count"] = dict(
                    group["summary"]["statistics"]["true_count"]
                )
                group["summary"]["statistics"]["success_rate"] = dict(
                    group["summary"]["statistics"]["true_rate"]
                )
    passed = _passed_summary(rows)
    if passed is not None:
        cohort["passed_summary"] = passed
        for dimension, dimension_groups in groups.items():
            bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                bucket_rows[_canonical(row.get(dimension))].append(row)
            for group in dimension_groups:
                group_passed = _passed_summary(
                    bucket_rows[_canonical(group["value"])]
                )
                if group_passed is not None:
                    group["passed_summary"] = group_passed
    return cohort


def aggregate_tool_executions(
    sources: Sequence[Any],
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate ToolResult episodes and optionally write ``aggregate_result.json``.

    Each source may be a Tool execution envelope, a JSON path, or a wrapper:

    ``{"tool_execution": envelope, "context": {"round_id": ..., "variant": ...}}``

    Trusted-tool summaries with ``episode.tool_results`` are also accepted.
    Statistics are never computed across roles.
    """

    if isinstance(sources, (str, bytes, Path)) or not isinstance(
        sources, Sequence
    ):
        raise AggregateToolkitError("sources must be a sequence")
    rows, input_issues = _normalize_rows(sources)
    if not rows:
        raise AggregateToolkitError("no episode ToolResult rows were provided")
    rows.sort(key=_row_sort_key)
    metrics: list[dict[str, Any]] = []
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_metric[row["metric"]].append(row)
    for metric in sorted(by_metric):
        metric_rows = by_metric[metric]
        kind, unit = _finalize_metric_rows(metric_rows)
        by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in metric_rows:
            by_role[row["role"]].append(row)
        cohorts = [
            _cohort(metric, role, role_rows, kind, unit)
            for role, role_rows in sorted(
                by_role.items(),
                key=lambda item: (
                    _ROLE_ORDER.get(item[0], 99),
                    item[0],
                ),
            )
        ]
        metrics.append(
            {
                "metric": metric,
                "tools": sorted(
                    {row.get("tool") for row in metric_rows}, key=_stable_key
                ),
                "value_kind": kind,
                "unit": unit,
                "cohorts": cohorts,
            }
        )
    aggregate = {
        "schema_version": 1,
        "status": "passed" if not input_issues else "passed_with_input_issues",
        "aggregation_policy": {
            "role_isolation": "strict",
            "numeric_standard_deviation": "population",
            "missing_values_enter_numeric_statistics": False,
            "invalid_values_enter_numeric_statistics": False,
            "group_dimensions": list(_GROUP_DIMENSIONS),
        },
        "source_count": len(sources),
        "episode_result_count": len(rows),
        "unique_episode_count": len(
            {
                (
                    row.get("episode_dir"),
                    row.get("role"),
                    row.get("seed"),
                    row.get("round_id"),
                    row.get("variant"),
                )
                for row in rows
            }
        ),
        "input_issues": sorted(
            input_issues,
            key=lambda item: (item["source_index"], str(item.get("reason"))),
        ),
        "metrics": metrics,
    }
    if output_path is not None:
        write_aggregate_result(aggregate, output_path)
    return aggregate


def write_aggregate_result(
    aggregate: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Write one deterministic, human-readable aggregate JSON artifact."""

    if not isinstance(aggregate, Mapping):
        raise AggregateToolkitError("aggregate must be an object")
    path = Path(output_path).expanduser().resolve()
    if path.exists() and path.is_dir():
        path = path / "aggregate_result.json"
    elif not path.suffix:
        path = path / "aggregate_result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = json.dumps(
            dict(aggregate),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise AggregateToolkitError(
            f"aggregate is not deterministic JSON data: {exc}"
        ) from exc
    path.write_text(encoded, encoding="utf-8")
    return path
