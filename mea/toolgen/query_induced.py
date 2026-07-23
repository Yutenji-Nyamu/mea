"""Bounded query-induced ToolGen for pre-contact motion evidence.

The provider is allowed to describe a metric in a small compositional DSL.  It
never selects a robot arm and never supplies executable code.  The runtime
derives the active arm from the recorded block position, compiles the validated
DSL deterministically, exact-AST checks the result, validates three explicit
oracles, and only then installs the tool in the evaluation-local registry.

Exact-language reuse is resolved before consulting a provider.  For a
paraphrase, the provider sees summaries of compatible registered metrics and
may select one by returning its exact executable DSL; this reuses the existing
registration without code generation or registration.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np

from mea.toolkit.tools import TrajectoryView

from .registry import (
    canonical_sha256,
    find_run_local_registration,
    load_registry,
    register_run_local_tool,
)


class QueryInducedToolError(RuntimeError):
    """Raised when the bounded query-induced ToolGen contract is not met."""


class TextProvider(Protocol):
    """The small provider surface needed by this independently testable slice."""

    last_metadata: dict[str, Any]

    def text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> str: ...


_PROPOSAL_KEYS = {
    "schema_version",
    "metric_id",
    "finite_difference_order",
    "window_seconds",
    "reducer",
    "time_normalization",
    "threshold",
    "unit",
    "null_semantics",
    "rationale",
}
_EXECUTABLE_KEYS = _PROPOSAL_KEYS - {"rationale"}
_METRIC_ID = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_ORDER_TO_UNIT = {
    2: "m_per_second_squared",
    3: "m_per_second_cubed",
}
_NULL_SEMANTICS = (
    "null_if_no_target_contact_or_insufficient_precontact_samples"
)
_REQUIRED_TRACE_SIGNALS = {
    "block_position",
    "left_tcp_position",
    "right_tcp_position",
    "physics_step",
    "policy_step",
    "simulation_time_seconds",
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalized_query(query: str) -> str:
    normalized = " ".join(
        unicodedata.normalize("NFKC", str(query)).casefold().split()
    )
    if not normalized:
        raise QueryInducedToolError("query must be non-empty")
    if len(normalized) > 2000:
        raise QueryInducedToolError("query exceeds the 2000-character bound")
    return normalized


def query_fingerprint(query: str) -> str:
    """Hash normalized *language*, without hard-coded semantic aliases."""

    return "qit_v2_" + _sha256_text(_normalized_query(query))[:24]


def _json_object(response: str) -> dict[str, Any]:
    text = str(response).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise QueryInducedToolError("provider response contains no JSON object")
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise QueryInducedToolError(
            f"provider response is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise QueryInducedToolError("provider proposal must be a JSON object")
    return value


def _validate_dsl(
    value: Mapping[str, Any],
    *,
    require_rationale: bool,
) -> dict[str, Any]:
    proposal = dict(value)
    expected = _PROPOSAL_KEYS if require_rationale else _EXECUTABLE_KEYS
    if set(proposal) != expected:
        raise QueryInducedToolError(
            "proposal fields must be exactly " + ", ".join(sorted(expected))
        )
    if proposal.get("schema_version") != 2:
        raise QueryInducedToolError("proposal.schema_version must be 2")
    metric_id = proposal.get("metric_id")
    if not isinstance(metric_id, str) or not _METRIC_ID.fullmatch(metric_id):
        raise QueryInducedToolError(
            "proposal.metric_id must match ^[a-z][a-z0-9_]{2,63}$"
        )
    order = proposal.get("finite_difference_order")
    if type(order) is not int or order not in _ORDER_TO_UNIT:
        raise QueryInducedToolError(
            "proposal.finite_difference_order must be 2 or 3"
        )
    window = proposal.get("window_seconds")
    if (
        type(window) not in (int, float)
        or not np.isfinite(float(window))
        or not 0.02 <= float(window) <= 5.0
    ):
        raise QueryInducedToolError(
            "proposal.window_seconds must be finite and in [0.02, 5.0]"
        )
    if proposal.get("reducer") != "peak_l2":
        raise QueryInducedToolError("proposal.reducer must be peak_l2")
    if proposal.get("time_normalization") != "physical_seconds":
        raise QueryInducedToolError(
            "proposal.time_normalization must be physical_seconds"
        )
    threshold = proposal.get("threshold")
    if (
        type(threshold) not in (int, float)
        or not np.isfinite(float(threshold))
        or not 0.0 < float(threshold) <= 1.0e12
    ):
        raise QueryInducedToolError(
            "proposal.threshold must be finite and in (0, 1e12]"
        )
    expected_unit = _ORDER_TO_UNIT[order]
    if proposal.get("unit") != expected_unit:
        raise QueryInducedToolError(
            f"proposal.unit must be {expected_unit} for order {order}"
        )
    if proposal.get("null_semantics") != _NULL_SEMANTICS:
        raise QueryInducedToolError("proposal.null_semantics is unsupported")
    if require_rationale:
        rationale = proposal.get("rationale")
        if (
            not isinstance(rationale, str)
            or not rationale.strip()
            or len(rationale) > 500
        ):
            raise QueryInducedToolError(
                "proposal.rationale must contain 1-500 characters"
            )
        proposal["rationale"] = rationale.strip()
    proposal["window_seconds"] = float(window)
    proposal["threshold"] = float(threshold)
    return proposal


def validate_query_metric_proposal(
    value: Mapping[str, Any],
    *,
    available_signals: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a provider proposal; ``signal`` is intentionally not a field.

    ``available_signals`` is accepted only for compatibility with callers of
    the v1 helper.  When supplied, it must contain the three runtime-owned
    traces used to derive the active arm.
    """

    proposal = _validate_dsl(value, require_rationale=True)
    if available_signals is not None:
        required = {
            "block_position",
            "left_tcp_position",
            "right_tcp_position",
        }
        missing = sorted(required - set(available_signals))
        if missing:
            raise QueryInducedToolError(
                "telemetry lacks runtime-owned active-arm signals: "
                + ", ".join(missing)
            )
    return proposal


def _executable_spec(proposal: Mapping[str, Any]) -> dict[str, Any]:
    return _validate_dsl(
        {key: proposal[key] for key in _EXECUTABLE_KEYS},
        require_rationale=False,
    )


def _provider_prompt(
    *,
    query: str,
    registered_summaries: list[dict[str, Any]],
) -> str:
    contract = {
        "schema_version": 2,
        "metric_id": "descriptive_snake_case_id",
        "finite_difference_order": 3,
        "window_seconds": 0.5,
        "reducer": "peak_l2",
        "time_normalization": "physical_seconds",
        "threshold": 50.0,
        "unit": "m_per_second_cubed",
        "null_semantics": _NULL_SEMANTICS,
        "rationale": "semantic explanation",
    }
    return (
        "Propose one bounded pre-target-contact motion metric for the Query.\n"
        f"Query: {query}\n"
        "The runtime, not you, selects the active TCP: it reads the initial "
        "block_position x coordinate (x < 0 => left, otherwise right). Do not "
        "return a signal name or Python code.\n"
        "The only compiler family is a physical-time-normalized finite "
        "difference of order 2 or 3 over a bounded pre-contact window, reduced "
        "by peak L2. finite_difference_order, window_seconds, and threshold "
        "must be JSON numbers, not strings. Order 2 requires "
        "m_per_second_squared; order 3 requires m_per_second_cubed. Choose the "
        "semantics and a meaningful metric_id within that family.\n"
        "Compatible validated run-local metrics are listed below. A paraphrase "
        "may reuse one only by copying all its executable fields exactly. "
        "Otherwise choose a new metric_id.\n"
        "registered_metric_summaries:\n"
        + json.dumps(registered_summaries, ensure_ascii=False, indent=2)
        + "\nReturn JSON only with these exact fields. The values below are a "
        "type-correct example, not values you must copy:\n"
        + json.dumps(contract, ensure_ascii=False, indent=2)
    )


def _float_literal(value: float) -> str:
    return format(float(value), ".17g")


def compile_metric_source(proposal: Mapping[str, Any]) -> str:
    """Compile a validated DSL spec to deterministic, pure-Python source."""

    spec = _executable_spec(proposal)
    order = int(spec["finite_difference_order"])
    coefficients = {
        2: "(1.0, -2.0, 1.0)",
        3: "(1.0, -3.0, 3.0, -1.0)",
    }[order]
    return f'''def generated_tool(positions, simulation_times, contact_index):
    METRIC_ID = {spec["metric_id"]!r}
    ORDER = {order}
    WINDOW_SECONDS = {_float_literal(spec["window_seconds"])}
    REDUCER = 'peak_l2'
    TIME_NORMALIZATION = 'physical_seconds'
    THRESHOLD = {_float_literal(spec["threshold"])}
    UNIT = {spec["unit"]!r}
    NULL_SEMANTICS = {spec["null_semantics"]!r}
    COEFFICIENTS = {coefficients}
    if contact_index is None:
        return {{"metric_id": METRIC_ID, "value": None, "peak_index": None, "passed": False, "unit": UNIT, "null_semantics": NULL_SEMANTICS, "null_reason": "no_target_contact_event"}}
    if contact_index <= ORDER or contact_index >= len(positions):
        return {{"metric_id": METRIC_ID, "value": None, "peak_index": None, "passed": False, "unit": UNIT, "null_semantics": NULL_SEMANTICS, "null_reason": "insufficient_precontact_samples"}}
    contact_time = simulation_times[contact_index]
    window_start = contact_time - WINDOW_SECONDS
    peak = None
    peak_index = None
    for index in range(ORDER, contact_index):
        if simulation_times[index - ORDER] < window_start:
            continue
        elapsed = simulation_times[index] - simulation_times[index - ORDER]
        if elapsed <= 0.0:
            continue
        dt = elapsed / ORDER
        total = 0.0
        for axis in range(len(positions[index])):
            delta = 0.0
            for offset in range(ORDER + 1):
                delta = delta + COEFFICIENTS[offset] * positions[index - offset][axis]
            normalized = delta / (dt ** ORDER)
            total = total + normalized * normalized
        magnitude = total ** 0.5
        if peak is None or magnitude > peak:
            peak = magnitude
            peak_index = index
    if peak is None:
        return {{"metric_id": METRIC_ID, "value": None, "peak_index": None, "passed": False, "unit": UNIT, "null_semantics": NULL_SEMANTICS, "null_reason": "insufficient_precontact_samples"}}
    return {{"metric_id": METRIC_ID, "value": peak, "peak_index": peak_index, "passed": peak <= THRESHOLD, "unit": UNIT, "null_semantics": NULL_SEMANTICS, "null_reason": None, "reducer": REDUCER, "time_normalization": TIME_NORMALIZATION}}
'''


def validate_compiled_source(
    source: str,
    proposal: Mapping[str, Any],
) -> str:
    """Exact-AST gate against the compiler output for this validated spec."""

    try:
        candidate = ast.parse(source, mode="exec")
        expected = ast.parse(compile_metric_source(proposal), mode="exec")
    except SyntaxError as exc:
        raise QueryInducedToolError(
            f"compiled source is not valid Python: {exc}"
        ) from exc
    if ast.dump(candidate, include_attributes=False) != ast.dump(
        expected,
        include_attributes=False,
    ):
        raise QueryInducedToolError(
            "compiled source violates the exact DSL-to-AST contract"
        )
    return _sha256_text(source)


def _load_compiled(source: str, proposal: Mapping[str, Any]):
    validate_compiled_source(source, proposal)
    namespace: dict[str, Any] = {
        "__builtins__": {"len": len, "range": range}
    }
    exec(compile(source, "<query-induced-tool>", "exec"), namespace, namespace)
    return namespace["generated_tool"]


def _first_target_contact_index(trajectory: TrajectoryView) -> int | None:
    """Find task-target contact, excluding incidental robot/table contacts."""

    task_name = str(trajectory.metadata.get("task_name") or "")
    required_actors = {
        "beat_block_hammer": {"020_hammer", "box"},
    }.get(task_name)
    steps: list[int] = []
    for event in trajectory.contact_intervals:
        if event.get("physical_contact") is not True:
            continue
        actors = set(event.get("actors") or [])
        if required_actors is not None and not required_actors.issubset(actors):
            continue
        value = event.get("first_physical_physics_step")
        if isinstance(value, (int, float)):
            steps.append(int(value))
    if not steps:
        return None
    trace_steps = trajectory.trace["physics_step"].astype(np.int64)
    index = int(np.searchsorted(trace_steps, min(steps), side="left"))
    return index if index < len(trace_steps) else None


def _active_arm_trace(
    trajectory: TrajectoryView,
) -> tuple[str, str, np.ndarray, float]:
    missing = sorted(_REQUIRED_TRACE_SIGNALS - set(trajectory.trace))
    if missing:
        raise QueryInducedToolError(
            "recorded telemetry lacks required signals: " + ", ".join(missing)
        )
    block = np.asarray(trajectory.trace["block_position"], dtype=float)
    if (
        block.ndim != 2
        or block.shape[0] < 1
        or block.shape[1] < 1
        or not np.isfinite(block[0, 0])
    ):
        raise QueryInducedToolError(
            "block_position must contain a finite initial x coordinate"
        )
    block_x = float(block[0, 0])
    active_arm = "left" if block_x < 0.0 else "right"
    signal = f"{active_arm}_tcp_position"
    positions = np.asarray(trajectory.trace[signal], dtype=float)
    if (
        positions.ndim != 2
        or positions.shape[1] < 2
        or not np.all(np.isfinite(positions))
    ):
        raise QueryInducedToolError(f"invalid TCP trace for {signal}")
    return active_arm, signal, positions, block_x


def evaluate_precontact_motion_peak(
    trajectory: TrajectoryView,
    *,
    proposal: Mapping[str, Any],
    source: str | None = None,
) -> dict[str, Any]:
    """Run one compiled metric on caller-supplied recorded telemetry."""

    validated = validate_query_metric_proposal(proposal)
    active_arm, signal, positions, block_x = _active_arm_trace(trajectory)
    times = np.asarray(
        trajectory.trace["simulation_time_seconds"],
        dtype=float,
    )
    if (
        times.ndim != 1
        or len(times) != len(positions)
        or not np.all(np.isfinite(times))
        or (len(times) > 1 and not np.all(np.diff(times) > 0.0))
    ):
        raise QueryInducedToolError(
            "simulation_time_seconds must be finite, one-dimensional, and "
            "strictly increasing"
        )
    compiled_source = source or compile_metric_source(validated)
    compiled = _load_compiled(compiled_source, validated)
    contact_index = _first_target_contact_index(trajectory)
    raw = compiled(positions.tolist(), times.tolist(), contact_index)
    peak_index = raw["peak_index"]
    evidence: list[dict[str, Any]] = []
    if peak_index is not None:
        policy_step = int(trajectory.trace["policy_step"][peak_index])
        evidence.append(
            {
                "trace_index": int(peak_index),
                "physics_step": int(
                    trajectory.trace["physics_step"][peak_index]
                ),
                "policy_step": policy_step,
                "simulation_time_seconds": float(times[peak_index]),
                "video_frame_before": max(policy_step, 0),
                "video_frame_after": max(policy_step + 1, 0),
            }
        )
    return {
        "tool": validated["metric_id"],
        "value": raw["value"],
        "unit": validated["unit"],
        "passed": bool(raw["passed"]),
        "null_semantics": validated["null_semantics"],
        "null_reason": raw.get("null_reason"),
        "finite_difference_order": validated["finite_difference_order"],
        "window_seconds": validated["window_seconds"],
        "reducer": validated["reducer"],
        "time_normalization": validated["time_normalization"],
        "active_arm": active_arm,
        "active_arm_rule": "initial_block_position_x_lt_0_left_else_right",
        "initial_block_position_x": block_x,
        "signal": signal,
        "first_target_contact_trace_index": contact_index,
        "evidence": evidence,
        "evidence_steps": [item["physics_step"] for item in evidence],
    }


def evaluate_precontact_jerk_peak(
    trajectory: TrajectoryView,
    *,
    proposal: Mapping[str, Any],
    source: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible name for the order-2/order-3 motion evaluator."""

    return evaluate_precontact_motion_peak(
        trajectory,
        proposal=proposal,
        source=source,
    )


def _oracle_fixtures(
    proposal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Explicit validation-only fixtures, never live-result fallbacks."""

    spec = _executable_spec(proposal)
    order = int(spec["finite_difference_order"])
    contact_index = order + 5
    dt = float(spec["window_seconds"]) / float(contact_index)
    times = [dt * index for index in range(contact_index + 1)]
    smooth = [[0.0, 0.0, 0.0] for _ in times]
    amplitude = (
        float(spec["threshold"]) * (dt ** order) * 4.0
        + max(dt ** order, 1.0e-12)
    )
    oscillatory = [
        [0.0 if index % 2 == 0 else amplitude, 0.0, 0.0]
        for index in range(len(times))
    ]
    return [
        {
            "name": "smooth",
            "positions": smooth,
            "simulation_times": times,
            "contact_index": contact_index,
            "expected_value": 0.0,
            "expected_passed": True,
            "expected_null_reason": None,
        },
        {
            "name": "oscillatory",
            "positions": oscillatory,
            "simulation_times": times,
            "contact_index": contact_index,
            "expected_value_min": float(spec["threshold"]),
            "expected_passed": False,
            "expected_null_reason": None,
        },
        {
            "name": "missing_target_contact",
            "positions": smooth,
            "simulation_times": times,
            "contact_index": None,
            "expected_value": None,
            "expected_passed": False,
            "expected_null_reason": "no_target_contact_event",
        },
    ]


def validate_precontact_motion_oracles(
    *,
    proposal: Mapping[str, Any],
    source: str | None = None,
) -> list[dict[str, Any]]:
    validated = validate_query_metric_proposal(proposal)
    compiled_source = source or compile_metric_source(validated)
    compiled = _load_compiled(compiled_source, validated)
    results: list[dict[str, Any]] = []
    for fixture in _oracle_fixtures(validated):
        result = compiled(
            fixture["positions"],
            fixture["simulation_times"],
            fixture["contact_index"],
        )
        if "expected_value" in fixture:
            expected_value = fixture["expected_value"]
            value_ok = (
                result["value"] is None
                if expected_value is None
                else result["value"] is not None
                and math.isclose(
                    float(result["value"]),
                    float(expected_value),
                    abs_tol=1.0e-12,
                )
            )
        else:
            value_ok = (
                result["value"] is not None
                and float(result["value"])
                > float(fixture["expected_value_min"])
            )
        passed = bool(
            value_ok
            and result["passed"] is fixture["expected_passed"]
            and result.get("null_reason")
            == fixture["expected_null_reason"]
        )
        results.append(
            {
                "fixture": fixture["name"],
                "validation_only": True,
                "expected": {
                    key: value
                    for key, value in fixture.items()
                    if key.startswith("expected_")
                },
                "observed": result,
                "passed": passed,
            }
        )
    if not all(item["passed"] for item in results):
        raise QueryInducedToolError(
            "pre-contact motion oracle validation failed"
        )
    return results


def validate_precontact_jerk_oracles(
    *,
    proposal: Mapping[str, Any],
    source: str | None = None,
) -> list[dict[str, Any]]:
    """Backward-compatible name for generic order-2/order-3 oracles."""

    return validate_precontact_motion_oracles(
        proposal=proposal,
        source=source,
    )


def _tool_spec(
    *,
    trajectory: TrajectoryView,
    query: str,
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    spec = _executable_spec(proposal)
    return {
        "task_name": str(trajectory.metadata["task_name"]),
        "metric": spec["metric_id"],
        "question": query,
        "metric_spec": spec,
        "required_signals": [
            "semantic_trace.block_position",
            "semantic_trace.left_tcp_position",
            "semantic_trace.right_tcp_position",
            "semantic_trace.simulation_time_seconds",
            "semantic_trace.physics_step",
            "events.contact_interval",
        ],
        "output": {
            "value": "number_or_null",
            "unit": spec["unit"],
            "passed": "boolean",
            "null_reason": "string_or_null",
            "active_arm": "left_or_right",
        },
    }


def _semantic_index_path(registry_dir: Path) -> Path:
    return registry_dir / "query_induced_index.json"


def _empty_semantic_index() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "fingerprint_policy": "normalized_exact_language_no_semantic_aliases",
        "entries": {},
    }


def _load_semantic_index(registry_dir: Path) -> dict[str, Any]:
    path = _semantic_index_path(registry_dir)
    if not path.is_file():
        return _empty_semantic_index()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueryInducedToolError(
            f"invalid query-induced index: {exc}"
        ) from exc
    if value.get("schema_version") == 1 and isinstance(
        value.get("entries"),
        dict,
    ):
        migrated = _empty_semantic_index()
        migrated["legacy_v1_entries"] = value["entries"]
        migrated["migration_note"] = (
            "v1 aliases and sample-normalized code are audit-only; one v2 "
            "generation is required before executable reuse"
        )
        return migrated
    if value.get("schema_version") != 2 or not isinstance(
        value.get("entries"),
        dict,
    ):
        raise QueryInducedToolError("unsupported query-induced index schema")
    return value


def _registered_metric_records(
    registry_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    registry_index = load_registry(registry_dir)
    for entry in registry_index["entries"]:
        if (
            entry.get("scope") != "run_local"
            or entry.get("status") != "validated"
        ):
            continue
        artifact = entry.get("registration_artifact")
        if not isinstance(artifact, str):
            continue
        path = (registry_dir / artifact).resolve()
        if registry_dir != path and registry_dir not in path.parents:
            continue
        try:
            registration = json.loads(path.read_text(encoding="utf-8"))
            tool_spec = registration["tool_contract"]["tool_spec"]
            spec = _validate_dsl(
                tool_spec["metric_spec"],
                require_rationale=False,
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError, QueryInducedToolError):
            continue
        records.append(
            {
                "registration_id": registration.get("registration_id"),
                "task_name": tool_spec.get("task_name"),
                "spec": spec,
            }
        )
    records.sort(
        key=lambda item: (
            str(item["spec"]["metric_id"]),
            str(item["registration_id"]),
        )
    )
    return records


def _compatible_registered_summaries(
    *,
    registry_dir: Path,
    trajectory: TrajectoryView,
    episode_path: Path,
    query: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_records = _registered_metric_records(registry_dir)
    compatible: list[dict[str, Any]] = []
    for record in all_records:
        proposal = dict(record["spec"])
        proposal["rationale"] = "registered executable metric"
        tool_spec = _tool_spec(
            trajectory=trajectory,
            query=query,
            proposal=proposal,
        )
        match = find_run_local_registration(
            registry_dir,
            tool_spec=tool_spec,
            episode_dirs=[episode_path],
        )
        if match is None:
            continue
        compatible.append(
            {
                "registration_id": match["registration"]["registration_id"],
                "task_name": record["task_name"],
                "executable_spec": record["spec"],
            }
        )
    return compatible, all_records


def _write_provider_artifacts(
    *,
    output: Path,
    query: str,
    prompt: str,
    response: str,
    registered_summaries: list[dict[str, Any]],
) -> None:
    _write_json(
        output / "provider_request.json",
        {
            "query": query,
            "prompt": prompt,
            "registered_metric_summaries": registered_summaries,
        },
    )
    response_path = output / "provider_response.txt"
    response_path.parent.mkdir(parents=True, exist_ok=True)
    response_path.write_text(response + "\n", encoding="utf-8")


def run_query_induced_toolgen(
    *,
    query: str,
    episode_dir: str | Path,
    output_dir: str | Path,
    registry_dir: str | Path,
    provider: TextProvider | None,
    model: str | None = None,
) -> dict[str, Any]:
    """Generate or reuse a DSL metric, then evaluate recorded telemetry."""

    episode_path = Path(episode_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    registry = Path(registry_dir).expanduser().resolve()
    trajectory = TrajectoryView(episode_path)
    _active_arm_trace(trajectory)
    fingerprint = query_fingerprint(query)
    index = _load_semantic_index(registry)
    cached = index["entries"].get(fingerprint)
    provider_metadata: dict[str, Any] | None = None
    provider_called = False
    codegen_performed = False
    registration_performed = False
    oracle_validation_performed = False
    registered_summary_count = 0

    if cached is not None:
        proposal = validate_query_metric_proposal(cached["proposal"])
        tool_spec = _tool_spec(
            trajectory=trajectory,
            query=query,
            proposal=proposal,
        )
        registration = find_run_local_registration(
            registry,
            tool_spec=tool_spec,
            episode_dirs=[episode_path],
        )
        if registration is None:
            raise QueryInducedToolError(
                "exact-query index points to an unvalidated or incompatible "
                "registry entry"
            )
        if (
            cached.get("registration_id")
            != registration["registration"]["registration_id"]
        ):
            raise QueryInducedToolError(
                "exact-query index registration id does not match the "
                "validated registry entry"
            )
        source = registration["source_path"].read_text(encoding="utf-8")
        route = "exact_query_registry_reuse"
    else:
        if provider is None:
            raise QueryInducedToolError(
                "provider is required for an unseen exact-language query; "
                "reuse-only mode cannot infer paraphrase semantics"
            )
        compatible, all_records = _compatible_registered_summaries(
            registry_dir=registry,
            trajectory=trajectory,
            episode_path=episode_path,
            query=query,
        )
        registered_summary_count = len(compatible)
        prompt = _provider_prompt(
            query=query,
            registered_summaries=compatible,
        )
        response = provider.text(
            prompt,
            model=model,
            max_tokens=420,
            temperature=0.0,
        )
        provider_called = True
        _write_provider_artifacts(
            output=output,
            query=query,
            prompt=prompt,
            response=response,
            registered_summaries=compatible,
        )
        provider_metadata = dict(
            getattr(provider, "last_metadata", {}) or {}
        )
        proposal = validate_query_metric_proposal(_json_object(response))
        executable = _executable_spec(proposal)
        for record in all_records:
            registered_spec = record["spec"]
            if (
                registered_spec["metric_id"] == executable["metric_id"]
                and registered_spec != executable
            ):
                raise QueryInducedToolError(
                    "provider metric_id collides with a registered metric that "
                    "has different executable semantics"
                )
        tool_spec = _tool_spec(
            trajectory=trajectory,
            query=query,
            proposal=proposal,
        )
        registration = find_run_local_registration(
            registry,
            tool_spec=tool_spec,
            episode_dirs=[episode_path],
        )
        if registration is not None:
            source = registration["source_path"].read_text(encoding="utf-8")
            route = "provider_semantic_registry_reuse"
        else:
            source = compile_metric_source(proposal)
            source_hash = validate_compiled_source(source, proposal)
            source_path = output / "generated_tool.py"
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_text(source, encoding="utf-8")
            codegen_performed = True
            oracle = validate_precontact_motion_oracles(
                proposal=proposal,
                source=source,
            )
            oracle_validation_performed = True
            registration = register_run_local_tool(
                registry,
                tool_spec=tool_spec,
                episode_dirs=[episode_path],
                source_path=source_path,
                generation_registration={
                    "tool": proposal["metric_id"],
                    "validated_episode_count": 0,
                    "validated_property_scenario_count": len(oracle),
                    "oracle_kind": (
                        "explicit_physical_time_finite_difference_fixtures"
                    ),
                },
                generation_manifest={
                    "model_requested": model,
                    "contract_sha256": canonical_sha256(proposal),
                    "generator_source_sha256": source_hash,
                    "example_validation": oracle,
                },
                validation_episodes=oracle,
            )
            registration_performed = True
            route = "provider_generate_validate_register"
        index["entries"][fingerprint] = {
            "query_fingerprint": fingerprint,
            "normalized_query_sha256": _sha256_text(
                _normalized_query(query)
            ),
            "proposal": proposal,
            "registration_id": registration["registration"][
                "registration_id"
            ],
            "proposal_sha256": canonical_sha256(proposal),
            "resolution_route": route,
        }
        _write_json(_semantic_index_path(registry), index)

    validate_compiled_source(source, proposal)
    live = evaluate_precontact_motion_peak(
        trajectory,
        proposal=proposal,
        source=source,
    )
    result = {
        "schema_version": 2,
        "query": query,
        "query_fingerprint": fingerprint,
        "route": route,
        "proposal": proposal,
        "provider_called": provider_called,
        "provider_metadata": provider_metadata,
        "registered_summary_count_presented": registered_summary_count,
        "codegen_performed": codegen_performed,
        "oracle_validation_performed": oracle_validation_performed,
        "registration_performed": registration_performed,
        "registration_id": registration["registration"]["registration_id"],
        "live_telemetry": {
            "episode_dir": str(episode_path),
            "task_name": trajectory.metadata.get("task_name"),
            "synthetic_fallback_used": False,
        },
        "tool_result": live,
    }
    _write_json(output / "query_induced_toolgen_result.json", result)
    return result


def query_induced_result_to_tool_execution(
    value: Mapping[str, Any],
    *,
    policy_name: str = "ACT",
    role: str = "policy_under_evaluation",
    seed: int | None = None,
) -> dict[str, Any]:
    """Adapt one validated live ToolGen result to the shared Aggregate schema.

    Synthetic oracle rows are intentionally absent.  A null live result stays
    null and carries its explicit reason into ``details.reason`` so Aggregate
    counts it as missing evidence instead of silently turning it into failure
    or a numeric sample.
    """

    if not isinstance(value, Mapping) or value.get("schema_version") != 2:
        raise QueryInducedToolError(
            "query-induced result must be a schema v2 object"
        )
    proposal = value.get("proposal")
    result = value.get("tool_result")
    live = value.get("live_telemetry")
    if (
        not isinstance(proposal, Mapping)
        or not isinstance(result, Mapping)
        or not isinstance(live, Mapping)
    ):
        raise QueryInducedToolError(
            "query-induced result lacks proposal, tool_result, or live_telemetry"
        )
    validated = validate_query_metric_proposal(proposal)
    metric = validated["metric_id"]
    if result.get("tool") != metric:
        raise QueryInducedToolError(
            "query-induced result tool differs from the validated proposal"
        )
    episode_dir = live.get("episode_dir")
    if not isinstance(episode_dir, str) or not episode_dir.strip():
        raise QueryInducedToolError(
            "query-induced result lacks a live episode directory"
        )
    if live.get("synthetic_fallback_used") is not False:
        raise QueryInducedToolError(
            "Aggregate bridge requires real recorded telemetry"
        )
    evidence_steps = result.get("evidence_steps")
    if not isinstance(evidence_steps, list) or any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in evidence_steps
    ):
        raise QueryInducedToolError(
            "query-induced evidence_steps must be an integer list"
        )
    null_reason = result.get("null_reason")
    if null_reason is not None and not isinstance(null_reason, str):
        raise QueryInducedToolError(
            "query-induced null_reason must be a string or null"
        )
    tool_result = {
        "tool": metric,
        "value": result.get("value"),
        "unit": result.get("unit"),
        "passed": result.get("passed"),
        "evidence_steps": list(evidence_steps),
        "details": {
            "reason": null_reason,
            "null_semantics": result.get("null_semantics"),
            "active_arm": result.get("active_arm"),
            "signal": result.get("signal"),
            "first_target_contact_trace_index": result.get(
                "first_target_contact_trace_index"
            ),
            "registration_id": value.get("registration_id"),
            "resolution_route": value.get("route"),
            "oracle_rows_included": False,
            "synthetic_fallback_used": False,
        },
    }
    return {
        "schema_version": 1,
        "status": "passed",
        "route": str(value.get("route") or ""),
        "tool_spec": {
            "task_name": live.get("task_name"),
            "metric": metric,
            "query": value.get("query"),
        },
        "episodes": [
            {
                "episode_dir": episode_dir,
                "policy_name": str(policy_name),
                "role": str(role),
                "seed": seed,
                "result": tool_result,
            }
        ],
    }
