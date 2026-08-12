"""MetricSpec compatibility compiler and provider-code validation."""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.task_guide import load_task_guide
from mea.toolkit.tools import TrajectoryView

from .metric_evaluation import _event_fields
from .metric_schema import (
    MetricSpecError,
    _DIMENSION_INDEX,
    _file_sha256,
    validate_metric_spec,
)

_DERIVED_STANDARD_TRACE_KEYS = {
    "physics_step",
    "policy_step",
    "simulation_time_seconds",
}

def _compiled_event_filter(selector: Mapping[str, Any]) -> str:
    clauses = [f"item.get('type') == {selector['event_type']!r}"]
    if selector["physical_only"]:
        clauses.append("item.get('physical_contact') is True")
    if selector["actors"] is not None:
        clauses.append(
            "sorted(item.get('actor_ids') or item.get('actors', [])) == "
            f"{selector['actors']!r}"
        )
    return " and ".join(clauses)


def _compile_event_count_source(spec: Mapping[str, Any]) -> str:
    selector = spec["event"]
    _, step_field = _event_fields(selector)
    event_filter = _compiled_event_filter(selector)
    return f'''def generated_tool(trajectory):
    events = [
        item for item in trajectory.events
        if {event_filter}
    ]
    steps = sorted(list(set([
        int(item.get({step_field!r}))
        for item in events
        if item.get({step_field!r}) is not None
    ])))
    return {{
        "value": len(events),
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": steps,
        "details": {{
            "operation": {spec["operation"]!r},
            "event": {selector!r},
            "matching_count": len(events),
            "reason": "measured",
        }},
    }}
'''


def _compile_time_between_events_source(spec: Mapping[str, Any]) -> str:
    start_selector = spec["start_event"]
    end_selector = spec["end_event"]
    start_time, start_step = _event_fields(start_selector)
    end_time, end_step = _event_fields(end_selector)
    start_filter = _compiled_event_filter(start_selector)
    end_filter = _compiled_event_filter(end_selector)
    return f'''def generated_tool(trajectory):
    start_events = [
        item for item in trajectory.events
        if {start_filter}
    ]
    end_events = [
        item for item in trajectory.events
        if {end_filter}
    ]
    start_points = [
        (float(item.get({start_time!r})), int(item.get({start_step!r})))
        for item in start_events
        if item.get({start_time!r}) is not None
        and item.get({start_step!r}) is not None
    ]
    end_points = [
        (float(item.get({end_time!r})), int(item.get({end_step!r})))
        for item in end_events
        if item.get({end_time!r}) is not None
        and item.get({end_step!r}) is not None
    ]
    start = min(start_points) if start_points else None
    end = min(end_points) if end_points else None
    valid = bool(start is not None and end is not None and end[0] >= start[0])
    steps = sorted(list(set([
        item[1] for item in [start, end] if item is not None
    ])))
    if start is None:
        reason = "start_event_missing"
    elif end is None:
        reason = "end_event_missing"
    elif end[0] < start[0]:
        reason = "end_event_precedes_start_event"
    else:
        reason = "measured"
    return {{
        "value": float(end[0] - start[0]) if valid else None,
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": steps,
        "details": {{
            "operation": {spec["operation"]!r},
            "start_event": {start_selector!r},
            "end_event": {end_selector!r},
            "start_simulation_time_seconds": start[0] if start else None,
            "end_simulation_time_seconds": end[0] if end else None,
            "start_physics_step": start[1] if start else None,
            "end_physics_step": end[1] if end else None,
            "reason": reason,
        }},
    }}
'''


def _compile_terminal_signal_component_source(spec: Mapping[str, Any]) -> str:
    component_index = _DIMENSION_INDEX[spec["component"]]
    value_expression = "abs(raw_value)" if spec["absolute"] else "raw_value"
    return f'''def generated_tool(trajectory):
    signal = np.asarray(trajectory.trace[{spec["signal"]!r}], dtype=float)
    terminal_index = len(signal) - 1
    raw_value = float(signal[terminal_index, {component_index}])
    finite = bool(np.isfinite(raw_value))
    value = {value_expression} if finite else None
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    return {{
        "value": value,
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": [int(physics[terminal_index])] if finite else [],
        "details": {{
            "operation": {spec["operation"]!r},
            "signal": {spec["signal"]!r},
            "component": {spec["component"]!r},
            "absolute": {spec["absolute"]!r},
            "terminal_index": terminal_index,
            "reason": "measured" if finite else "terminal_not_finite",
        }},
    }}
'''


def _compile_terminal_signal_difference_source(spec: Mapping[str, Any]) -> str:
    component_index = _DIMENSION_INDEX[spec["component"]]
    value_expression = (
        "abs(signed_difference)"
        if spec["absolute"]
        else "signed_difference"
    )
    return f'''def generated_tool(trajectory):
    left = np.asarray(trajectory.trace[{spec["left_signal"]!r}], dtype=float)
    right = np.asarray(trajectory.trace[{spec["right_signal"]!r}], dtype=float)
    terminal_index = len(left) - 1
    left_value = float(left[terminal_index, {component_index}])
    right_value = float(right[terminal_index, {component_index}])
    finite = bool(np.isfinite(left_value) and np.isfinite(right_value))
    signed_difference = left_value - right_value
    value = {value_expression} if finite else None
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    return {{
        "value": value,
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": [int(physics[terminal_index])] if finite else [],
        "details": {{
            "operation": {spec["operation"]!r},
            "left_signal": {spec["left_signal"]!r},
            "right_signal": {spec["right_signal"]!r},
            "component": {spec["component"]!r},
            "absolute": {spec["absolute"]!r},
            "left_terminal_value": left_value if finite else None,
            "right_terminal_value": right_value if finite else None,
            "terminal_index": terminal_index,
            "reason": "measured" if finite else "terminal_not_finite",
        }},
    }}
'''


def _compile_terminal_minimum_distance_source(spec: Mapping[str, Any]) -> str:
    indices = [_DIMENSION_INDEX[item] for item in spec["dimensions"]]
    return f'''def generated_tool(trajectory):
    left_signals = {spec["left_signals"]!r}
    target = np.asarray(trajectory.trace[{spec["right_signal"]!r}], dtype=float)
    terminal_index = len(target) - 1
    target_terminal = target[terminal_index, {indices!r}]
    left_terminal = np.asarray([
        np.asarray(trajectory.trace[signal], dtype=float)[terminal_index, {indices!r}]
        for signal in left_signals
    ])
    distances = np.linalg.norm(left_terminal - target_terminal, axis=1)
    finite = np.isfinite(distances)
    winner_index = int(np.argmin(np.where(finite, distances, np.inf))) if np.any(finite) else None
    winner = left_signals[winner_index] if winner_index is not None else None
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    return {{
        "value": float(distances[winner_index]) if winner_index is not None else None,
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": [int(physics[terminal_index])] if winner is not None else [],
        "details": {{
            "operation": {spec["operation"]!r},
            "left_signals": left_signals,
            "right_signal": {spec["right_signal"]!r},
            "dimensions": {spec["dimensions"]!r},
            "selected_left_signal": winner,
            "terminal_index": terminal_index,
            "reason": "measured" if winner is not None else "terminal_not_finite",
        }},
    }}
'''


def compile_metric_spec_source(metric_spec: Mapping[str, Any]) -> str:
    """Compile a MetricSpec to auditable Python accepted by ToolGen's AST gate."""

    spec = validate_metric_spec(metric_spec)
    if spec["operation"] == "derived_observable":
        raise MetricSpecError(
            "derived_observable has no compatibility compiler; provide a "
            "ToolGen provider or an exact validated registry match"
        )
    if spec["operation"] == "event_count":
        return _compile_event_count_source(spec)
    if spec["operation"] == "time_between_events":
        return _compile_time_between_events_source(spec)
    if spec["operation"] == "terminal_signal_component":
        return _compile_terminal_signal_component_source(spec)
    if spec["operation"] == "terminal_signal_difference":
        return _compile_terminal_signal_difference_source(spec)
    if spec["operation"] == "terminal_minimum_distance":
        return _compile_terminal_minimum_distance_source(spec)
    indices = [_DIMENSION_INDEX[item] for item in spec["dimensions"]]
    return f'''def generated_tool(trajectory):
    left = np.asarray(trajectory.trace[{spec["left_signal"]!r}], dtype=float)
    right = np.asarray(trajectory.trace[{spec["right_signal"]!r}], dtype=float)
    left_view = left[:, {indices!r}]
    right_view = right[:, {indices!r}]
    valid = np.all(np.isfinite(left_view) & np.isfinite(right_view), axis=1)
    distances = np.linalg.norm(left_view - right_view, axis=1)
    masked = np.where(valid, distances, np.inf)
    index = int(np.argmin(masked))
    value = float(masked[index])
    if not np.isfinite(value):
        return {{
            "value": None,
            "unit": {spec["unit"]!r},
            "passed": None,
            "evidence_steps": [],
            "details": {{
                "operation": {spec["operation"]!r},
                "left_signal": {spec["left_signal"]!r},
                "right_signal": {spec["right_signal"]!r},
                "dimensions": {spec["dimensions"]!r},
                "min_index": None,
                "reason": "no_finite_sample",
            }},
        }}
    physics = np.asarray(trajectory.trace["physics_step"], dtype=int)
    step = int(physics[index])
    return {{
        "value": value,
        "unit": {spec["unit"]!r},
        "passed": None,
        "evidence_steps": [step],
        "details": {{
            "operation": {spec["operation"]!r},
            "left_signal": {spec["left_signal"]!r},
            "right_signal": {spec["right_signal"]!r},
            "dimensions": {spec["dimensions"]!r},
            "min_index": index,
            "reason": "measured",
        }},
    }}
'''


def build_task_code_context(
    child_run_dir: str | Path,
    *,
    task_proposal: Mapping[str, Any] | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build the compact TaskGen-code context consumed by typed ToolGen."""

    child = Path(child_run_dir).expanduser().resolve()
    manifest_path = child / "manifest.json"
    if not manifest_path.is_file():
        raise MetricSpecError(f"TaskGen manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = child / "task.py"
    bundle_path = child / "generation/task_artifact_bundle.json"
    bundle = (
        json.loads(bundle_path.read_text(encoding="utf-8"))
        if bundle_path.is_file()
        else None
    )
    task_name = manifest.get("task_name")
    guide = (
        load_task_guide(repo_root, task_name)
        if repo_root is not None and isinstance(task_name, str)
        else ""
    )
    return {
        "schema_version": 1,
        "task_name": task_name,
        "task_implementation_guide": guide or None,
        "task_module": manifest.get("task_module"),
        "generation_kind": manifest.get("generation_kind"),
        "task_proposal": deepcopy(dict(task_proposal)) if task_proposal else None,
        "task_source": (
            {
                "path": "task.py",
                "sha256": _file_sha256(source),
                "excerpt": source.read_text(encoding="utf-8")[:4000],
            }
            if source.is_file()
            else None
        ),
        "task_artifact_bundle": bundle,
    }


def _provider_codegen_prompt(
    *,
    repo_root: Path,
    metric: str,
    question: str,
    metric_spec: Mapping[str, Any],
    trajectory: TrajectoryView,
    task_code_context: Mapping[str, Any] | None,
    previous_error: str | None,
    previous_source: str | None = None,
) -> str:
    """Build the compact prompt for a real Python ToolGen attempt."""

    contract = (repo_root / "mea/toolgen/README.Agent.md").read_text(
        encoding="utf-8"
    )
    operation = str(metric_spec["operation"])
    normal_reason = "measured"
    null_reasons = {
        "derived_observable": ["no_finite_sample"],
        "minimum_distance": ["no_finite_sample"],
        "terminal_minimum_distance": ["terminal_not_finite"],
        "terminal_signal_component": ["terminal_not_finite"],
        "terminal_signal_difference": ["terminal_not_finite"],
        "event_count": [],
        "time_between_events": [
            "start_event_missing",
            "end_event_missing",
            "end_event_precedes_start_event",
        ],
    }[operation]
    result_contract = {
        "passed": None,
        "evidence_steps": "plain Python int physics steps",
        "details.operation": operation,
        "details.reason": normal_reason,
        "details.allowed_null_reasons": null_reasons,
        "json_native_scalars_only": True,
    }
    telemetry = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        for key, value in sorted(trajectory.trace.items())
    }
    task_guide = (
        str(task_code_context.get("task_implementation_guide") or "").strip()
        if isinstance(task_code_context, Mapping)
        else ""
    )
    compact_task_context = (
        {
            key: value
            for key, value in task_code_context.items()
            if key != "task_implementation_guide"
        }
        if isinstance(task_code_context, Mapping)
        else task_code_context
    )
    repair = (
        "\nPREVIOUS VALIDATION FAILURE:\n"
        + previous_error
        + (
            "\nFAILURE STAGE: generated Tool validation/oracle execution.\n"
            "PREVIOUS FUNCTION:\n```python\n"
            + previous_source.strip()
            + "\n```\n"
            if previous_source
            else "\n"
        )
        + "Repair only the reported failure and return the complete function.\n"
        if previous_error
        else ""
    )
    return f"""You are ToolGen in ManipEvalAgent.
Write the Python implementation of one Query-induced Rule Tool.  The typed
MetricSpec below is the semantic contract, not source code to copy.
For derived_observable, implement the provider-proposed description over only
its declared telemetry signals; it is not a pre-registered metric operator.
Implement the observable from the recorded trajectory.  For a new derived
observable, a separate semantic reviewer will inspect this complete source
without changing it.  The result is also restricted to declared signals,
checked by an AST allowlist, executed twice on real telemetry, validated for a
finite scalar/null value, the requested unit and trace-bound evidence steps,
and rejected if episode artifacts change.  This Tool is measurement evidence
only and must not define task success or reward.

METRIC ID:
{metric}

QUESTION:
{question}

SEMANTIC ORACLE CONTRACT:
{json.dumps(metric_spec, ensure_ascii=False, indent=2)}

REQUIRED RESULT SEMANTICS:
{json.dumps(result_contract, ensure_ascii=False, indent=2)}

REAL TELEMETRY SURFACE:
{json.dumps(telemetry, ensure_ascii=False, indent=2)}

TASKGEN CONTEXT:
{json.dumps(compact_task_context, ensure_ascii=False, indent=2)}

BOUND TASK IMPLEMENTATION GUIDE:
{task_guide or "No task-local implementation guide is available."}

TOOL CONTRACT:
{contract}
{repair}
Return exactly one Python fenced block containing the complete
def generated_tool(trajectory): function and nothing else.
"""


def _validate_derived_signal_access(
    source_text: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Restrict a derived Tool to its declared trace surface."""

    tree = ast.parse(source_text)
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    used: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "trajectory":
                if node.attr != "trace":
                    raise MetricSpecError(
                        "derived_observable may access only trajectory.trace"
                    )
                parent = parents.get(node)
                if not (
                    isinstance(parent, ast.Subscript)
                    and parent.value is node
                ):
                    raise MetricSpecError(
                        "derived_observable must access trajectory.trace "
                        "directly with a literal field name"
                    )
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "trajectory"
            and value.attr == "trace"
        ):
            continue
        key = node.slice
        if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
            raise MetricSpecError(
                "derived_observable trace access must use a literal field name"
            )
        used.add(key.value)
    declared = set(spec["required_signals"])
    undeclared = sorted(used - declared - _DERIVED_STANDARD_TRACE_KEYS)
    missing = sorted(declared - used)
    if undeclared:
        raise MetricSpecError(
            "derived_observable source uses undeclared telemetry signals: "
            f"{undeclared}"
        )
    if missing:
        raise MetricSpecError(
            "derived_observable source does not use declared signals: "
            f"{missing}"
        )
    return {
        "required_signals": sorted(declared),
        "used_trace_keys": sorted(used),
    }
