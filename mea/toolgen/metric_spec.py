"""Typed, proposal-derived ToolGen metrics with a deterministic compiler.

The paper permits ToolGen to synthesize a rule metric from proposal context and
task code.  Arbitrary Python is unnecessarily broad for the first functional
slice, so this module exposes one small declarative operator, compiles it to the
same ``generated_tool`` contract used by ToolGen, validates it on cached
trajectories, and can register the result in the existing run-local registry.

This is intentionally not a replacement for model-generated tools.  It is the
smallest task-agnostic path proving that a metric need not be pre-enumerated in
``COMPOSITE_TARGETS`` before it can be generated, gated, and reused.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from mea.toolkit.tools import TrajectoryView


class MetricSpecError(RuntimeError):
    """Raised when a typed metric request or its validation evidence is invalid."""


_METRIC_SPEC_KEYS = {
    "schema_version",
    "operation",
    "left_signal",
    "right_signal",
    "dimensions",
    "unit",
    "null_semantics",
}
_SIGNAL = re.compile(r"^[a-z][a-z0-9_]{1,79}$")
_METRIC = re.compile(r"^[a-z][a-z0-9_]{2,79}$")
_DIMENSION_INDEX = {"x": 0, "y": 1, "z": 2}
_CORE_ARTIFACTS = (
    "episode.json",
    "schema.json",
    "states.csv",
    "semantic_trace.npz",
    "events.jsonl",
)


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_metric_spec(value: Any) -> dict[str, Any]:
    """Validate the first deliberately tiny numeric MetricSpec DSL."""

    if not isinstance(value, Mapping) or set(value) != _METRIC_SPEC_KEYS:
        raise MetricSpecError(
            f"MetricSpec fields must be exactly {sorted(_METRIC_SPEC_KEYS)}"
        )
    spec = deepcopy(dict(value))
    if spec.get("schema_version") != 1:
        raise MetricSpecError("MetricSpec.schema_version must be 1")
    if spec.get("operation") != "minimum_distance":
        raise MetricSpecError("MetricSpec.operation must be minimum_distance")
    for field in ("left_signal", "right_signal"):
        signal = spec.get(field)
        if not isinstance(signal, str) or not _SIGNAL.fullmatch(signal):
            raise MetricSpecError(f"MetricSpec.{field} is not a safe trace signal")
        spec[field] = signal
    if spec["left_signal"] == spec["right_signal"]:
        raise MetricSpecError("MetricSpec signals must be distinct")
    dimensions = spec.get("dimensions")
    if dimensions not in (["x", "y"], ["x", "y", "z"]):
        raise MetricSpecError("MetricSpec.dimensions must be [x,y] or [x,y,z]")
    if spec.get("unit") != "m":
        raise MetricSpecError("minimum_distance currently requires unit=m")
    if spec.get("null_semantics") != "null_if_no_finite_sample":
        raise MetricSpecError(
            "MetricSpec.null_semantics must be null_if_no_finite_sample"
        )
    return spec


def metric_spec_tool_spec(
    *,
    task_name: str,
    metric: str,
    question: str,
    metric_spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one MetricSpec into the existing routeful ToolSpec envelope."""

    task = str(task_name).strip()
    metric_id = str(metric).strip()
    prompt = str(question).strip()
    if not task:
        raise MetricSpecError("task_name must be non-empty")
    if not _METRIC.fullmatch(metric_id):
        raise MetricSpecError("metric must be a safe lower_snake_case identifier")
    if not prompt:
        raise MetricSpecError("question must be non-empty")
    spec = validate_metric_spec(metric_spec)
    required = [
        f"semantic_trace.{spec['left_signal']}",
        f"semantic_trace.{spec['right_signal']}",
        "semantic_trace.physics_step",
    ]
    return {
        "schema_version": 1,
        "task_name": task,
        "metric": metric_id,
        "question": prompt,
        "route": "force_codegen",
        "reference_tool": None,
        "required_signals": required,
        "output_contract": {
            "source": "typed_metric_spec_v1",
            "metric_spec": spec,
            "value_type": "number_or_null",
            "unit": spec["unit"],
            "passed": None,
            "evidence": "argmin_physics_step",
        },
        "validation_requirements": {
            "min_episodes": 2,
            "distinct_reference_values": True,
            "required_reference_values": [],
        },
    }


def evaluate_metric_spec(
    metric_spec: Mapping[str, Any], trajectory: TrajectoryView
) -> dict[str, Any]:
    """Evaluate the private deterministic oracle for a typed metric."""

    spec = validate_metric_spec(metric_spec)
    try:
        left = np.asarray(trajectory.trace[spec["left_signal"]], dtype=float)
        right = np.asarray(trajectory.trace[spec["right_signal"]], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise MetricSpecError(f"trajectory is missing a declared signal: {exc}") from exc
    indices = [_DIMENSION_INDEX[item] for item in spec["dimensions"]]
    if left.ndim != 2 or right.ndim != 2 or left.shape[0] != right.shape[0]:
        raise MetricSpecError("declared signals must be aligned two-dimensional arrays")
    if not len(left) or max(indices) >= left.shape[1] or max(indices) >= right.shape[1]:
        raise MetricSpecError("declared signals do not contain the requested dimensions")
    left_view = left[:, indices]
    right_view = right[:, indices]
    valid = np.all(np.isfinite(left_view) & np.isfinite(right_view), axis=1)
    distances = np.linalg.norm(left_view - right_view, axis=1)
    masked = np.where(valid, distances, np.inf)
    index = int(np.argmin(masked))
    value = float(masked[index])
    if not math.isfinite(value):
        return {
            "value": None,
            "unit": spec["unit"],
            "passed": None,
            "evidence_steps": [],
            "details": {
                "operation": spec["operation"],
                "left_signal": spec["left_signal"],
                "right_signal": spec["right_signal"],
                "dimensions": list(spec["dimensions"]),
                "min_index": None,
                "reason": "no_finite_sample",
            },
        }
    physics = np.asarray(
        trajectory.trace.get("physics_step", np.arange(len(left))), dtype=int
    )
    step = int(physics[index])
    return {
        "value": value,
        "unit": spec["unit"],
        "passed": None,
        "evidence_steps": [step],
        "details": {
            "operation": spec["operation"],
            "left_signal": spec["left_signal"],
            "right_signal": spec["right_signal"],
            "dimensions": list(spec["dimensions"]),
            "min_index": index,
            "reason": "measured",
        },
    }


def compile_metric_spec_source(metric_spec: Mapping[str, Any]) -> str:
    """Compile a MetricSpec to auditable Python accepted by ToolGen's AST gate."""

    spec = validate_metric_spec(metric_spec)
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
    return {
        "schema_version": 1,
        "task_name": manifest.get("task_name"),
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


def execute_metric_spec(
    *,
    task_name: str,
    metric: str,
    question: str,
    metric_spec: Mapping[str, Any],
    episode_dirs: Iterable[str | Path],
    output_dir: str | Path,
    task_code_context: Mapping[str, Any] | None = None,
    registry_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Compile, differentially validate, and optionally register one typed Tool."""

    from mea.toolgen.prototype import execute_generated_tool, validate_generated_tool
    from mea.toolgen.registry import (
        find_run_local_registration,
        public_registration_summary,
        register_run_local_tool,
    )

    spec = validate_metric_spec(metric_spec)
    tool_spec = metric_spec_tool_spec(
        task_name=task_name,
        metric=metric,
        question=question,
        metric_spec=spec,
    )
    context = deepcopy(dict(task_code_context)) if task_code_context else None
    if context is not None and context.get("task_name") != task_name:
        raise MetricSpecError("TaskGen code context belongs to a different task")
    episodes = [Path(item).expanduser().resolve() for item in episode_dirs]
    if len(episodes) < 2 or len(set(episodes)) != len(episodes):
        raise MetricSpecError("MetricSpec differential validation needs two episodes")
    trajectories = [TrajectoryView(path) for path in episodes]
    for trajectory in trajectories:
        if (
            trajectory.metadata.get("task_name") != task_name
            or trajectory.schema.get("task_name") != task_name
        ):
            raise MetricSpecError("MetricSpec episode task/schema does not match")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise MetricSpecError(f"MetricSpec output already exists: {destination}")
    destination.mkdir(parents=True)
    _write_json(destination / "metric_spec.json", spec)
    _write_json(destination / "tool_spec.json", tool_spec)
    if context is not None:
        _write_json(destination / "task_code_context.json", context)

    registry_match = None
    if registry_dir is not None:
        registry_match = find_run_local_registration(
            registry_dir, tool_spec=tool_spec, episode_dirs=episodes
        )
    if registry_match is not None:
        source_path = registry_match["source_path"]
        route = "run_local_reuse"
    else:
        source_path = destination / "generated_tool.py"
        source_path.write_text(compile_metric_spec_source(spec), encoding="utf-8")
        validate_generated_tool(source_path.read_text(encoding="utf-8"))
        route = "typed_metric_spec_compile"

    source_text = source_path.read_text(encoding="utf-8")
    rows = []
    values = []
    for episode, trajectory in zip(episodes, trajectories):
        before = {
            name: _file_sha256(episode / name)
            for name in _CORE_ARTIFACTS
            if (episode / name).is_file()
        }
        first = execute_generated_tool(source_text, episode, tool_name=metric)
        second = execute_generated_tool(source_text, episode, tool_name=metric)
        oracle = evaluate_metric_spec(spec, trajectory)
        generated = {
            key: first.get(key)
            for key in ("value", "unit", "passed", "evidence_steps", "details")
        }
        deterministic = _canonical(first) == _canonical(second)
        oracle_agreement = _canonical(generated) == _canonical(oracle)
        after = {
            name: _file_sha256(episode / name)
            for name in _CORE_ARTIFACTS
            if (episode / name).is_file()
        }
        if not deterministic or not oracle_agreement or before != after:
            raise MetricSpecError("compiled MetricSpec failed deterministic gates")
        values.append(oracle.get("value"))
        rows.append(
            {
                "episode_dir": str(episode),
                "policy_name": trajectory.metadata.get("policy_name"),
                "seed": trajectory.metadata.get("seed"),
                "generated_result": first,
                "oracle_projection": oracle,
                "deterministic": deterministic,
                "oracle_agreement": oracle_agreement,
                "artifacts_unchanged": before == after,
            }
        )
    finite_values = [float(item) for item in values if isinstance(item, (int, float))]
    if len({_canonical(item) for item in values}) < 2 or any(
        not math.isfinite(item) for item in finite_values
    ):
        raise MetricSpecError("MetricSpec differential oracle values are insufficient")

    registration = None
    if registry_match is None and registry_dir is not None:
        registration = register_run_local_tool(
            registry_dir,
            tool_spec=tool_spec,
            episode_dirs=episodes,
            source_path=source_path,
            generation_registration={
                "tool": metric,
                "validated_episode_count": len(rows),
                "validated_property_scenario_count": 0,
                "oracle_kind": "typed_metric_spec_v1",
            },
            generation_manifest={
                "successful_attempt": None,
                "model_requested": None,
                "generator_source_sha256": _file_sha256(source_path),
                "contract_sha256": hashlib.sha256(_canonical(spec).encode()).hexdigest(),
                "example_validation": [],
            },
            validation_episodes=[
                {
                    "episode_dir": str(episode),
                    "policy_name": row["policy_name"],
                    "seed": row["seed"],
                    "oracle_value": row["oracle_projection"].get("value"),
                }
                for episode, row in zip(episodes, rows)
            ],
        )
    elif registry_match is not None:
        registration = registry_match
    result = {
        "schema_version": 1,
        "status": "passed",
        "route": route,
        "provider_called": False,
        "tool_spec": tool_spec,
        "task_code_context_consumed": context is not None,
        "episodes": rows,
        "registration": (
            public_registration_summary(registration) if registration else None
        ),
        "limitations": [
            "one typed numeric operator only",
            "development compiler path, not arbitrary Python generation",
        ],
    }
    _write_json(destination / "execution.json", result)
    return result


__all__ = [
    "MetricSpecError",
    "build_task_code_context",
    "compile_metric_spec_source",
    "evaluate_metric_spec",
    "execute_metric_spec",
    "metric_spec_tool_spec",
    "validate_metric_spec",
]
