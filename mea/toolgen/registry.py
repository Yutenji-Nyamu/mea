"""Simple generated Rule Tool library.

The paper method needs retrieve, generate, validate, register, and reuse.  One
readable semantic key maps to one executable artifact.  Reuse is never trust by
metadata: the source is statically checked and executed twice against every
current episode, with the current oracle remaining authoritative.
"""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from .prototype import ToolGenError, validate_generated_tool


REGISTRY_SCHEMA_VERSION = 2
REGISTRATION_SCHEMA_VERSION = 4


class RunLocalRegistryError(RuntimeError):
    """Raised when a generated Tool library entry is malformed."""


def _normalized(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunLocalRegistryError("semantic key fields must be non-empty strings")
    return " ".join(value.casefold().split())


def tool_contract(tool_spec: dict[str, Any]) -> dict[str, Any]:
    spec = semantic_tool_spec(tool_spec)
    return {
        "tool_spec": spec,
        "required_signals": list(tool_spec.get("required_signals", [])),
    }


def semantic_tool_spec(tool_spec: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(tool_spec)
    result.pop("question", None)
    return result


def tool_semantic_key(tool_spec: dict[str, Any]) -> str:
    """Use readable executable semantics instead of an identity hash."""

    task = _normalized(tool_spec.get("task_name"))
    metric = _normalized(tool_spec.get("metric"))
    metric_spec = tool_spec.get("output_contract", {}).get("metric_spec")
    if metric_spec is None:
        metric_spec = tool_spec.get("metric_spec")
    detail = json.dumps(
        metric_spec or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return f"{task} :: {metric} :: {detail}"


def infer_registry_dir(output_dir: str | Path) -> Path | None:
    output = Path(output_dir).expanduser().resolve()
    for parent in output.parents:
        if parent.name == "execution":
            # The generated library is shared by evaluations.  Current-episode
            # validation below is what makes a retrieved artifact executable.
            return parent.parent.parent / "generated_tool_library"
    return None


def _schema_signal_names(value: Any) -> set[str]:
    """Read both mapping schemas and recorder ``[{name: ...}]`` fields."""

    if isinstance(value, dict):
        return {str(name) for name in value if str(name).strip()}
    if not isinstance(value, list):
        return set()
    names: set[str] = set()
    for item in value:
        if isinstance(item, str) and item.strip():
            names.add(item.strip())
        elif isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                names.add(name.strip())
    return names


def telemetry_schema_compatibility(
    episode_dirs: Iterable[str | Path], *, required_signals: Iterable[str]
) -> dict[str, Any]:
    """Check that current telemetry exposes the Tool's required signals."""

    required = list(required_signals)
    schemas: list[dict[str, Any]] = []
    for episode_dir in episode_dirs:
        path = Path(episode_dir).expanduser().resolve() / "schema.json"
        try:
            schema = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RunLocalRegistryError(f"invalid telemetry schema: {path}: {exc}") from exc
        schema_signals = _schema_signal_names(schema.get("signals"))
        schema_signals.update(
            _schema_signal_names(schema.get("semantic_fields"))
        )
        trace_signals: set[str] = set()
        trace_path = path.parent / "semantic_trace.npz"
        if trace_path.is_file():
            try:
                with zipfile.ZipFile(trace_path) as archive:
                    trace_signals = {
                        Path(name).stem for name in archive.namelist() if name.endswith(".npy")
                    }
            except (OSError, zipfile.BadZipFile) as exc:
                raise RunLocalRegistryError(
                    f"invalid semantic telemetry trace: {trace_path}: {exc}"
                ) from exc
        available = set(schema_signals)
        available.update(trace_signals)
        available.update(f"semantic_trace.{name}" for name in trace_signals)
        available.update(f"schema.{name}" for name in schema)
        if (path.parent / "events.jsonl").is_file():
            available.update(
                signal for signal in required if signal.startswith("events.")
            )
        missing = sorted(set(required) - available)
        if missing:
            raise RunLocalRegistryError(
                f"current telemetry is missing required signals: {missing}"
            )
        schemas.append({"task_name": schema.get("task_name"), "signals": sorted(available)})
    if not schemas:
        raise RunLocalRegistryError("telemetry schema set must not be empty")
    return {"required_signals": required, "schemas": schemas}


def _empty_index() -> dict[str, Any]:
    return {"schema_version": REGISTRY_SCHEMA_VERSION, "entries": []}


def load_registry(registry_dir: str | Path) -> dict[str, Any]:
    path = Path(registry_dir).expanduser().resolve() / "index.json"
    if not path.is_file():
        return _empty_index()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RunLocalRegistryError(f"invalid generated Tool library: {exc}") from exc
    if value.get("schema_version") != REGISTRY_SCHEMA_VERSION or not isinstance(
        value.get("entries"), list
    ):
        # Older index formats are not executable in the current library.
        return _empty_index()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _entry_paths(root: Path, entry: dict[str, Any]) -> tuple[Path, Path]:
    return root / entry["artifact"], root / entry["tool_spec_artifact"]


def find_run_local_registration(
    registry_dir: str | Path,
    *,
    tool_spec: dict[str, Any],
    episode_dirs: Iterable[str | Path],
) -> dict[str, Any] | None:
    root = Path(registry_dir).expanduser().resolve()
    key = tool_semantic_key(tool_spec)
    telemetry_schema_compatibility(
        episode_dirs, required_signals=tool_spec.get("required_signals", [])
    )
    for entry in load_registry(root)["entries"]:
        if entry.get("semantic_key") != key:
            continue
        try:
            source_path, spec_path = _entry_paths(root, entry)
            stored_spec = json.loads(spec_path.read_text(encoding="utf-8"))
            source = source_path.read_text(encoding="utf-8")
            validate_generated_tool(source)
        except (KeyError, OSError, json.JSONDecodeError, ValueError, ToolGenError):
            continue
        if semantic_tool_spec(stored_spec) != semantic_tool_spec(tool_spec):
            continue
        registration = {
            "schema_version": REGISTRATION_SCHEMA_VERSION,
            "registration_id": key,
            "semantic_key": key,
            "scope": "generated_artifact_library",
            "status": "validated_on_registration",
            "tool_id": entry.get("tool_id"),
            "target_metric": tool_spec.get("metric"),
            "tool_contract": tool_contract(tool_spec),
            "required_signals": list(tool_spec.get("required_signals", [])),
            "validation": {
                "semantic_review": deepcopy(entry.get("semantic_review")),
            },
        }
        return {
            "registration": registration,
            "registration_path": spec_path,
            "source_path": source_path,
            "source": source,
            "registry_dir": root,
        }
    return None


def compatible_run_local_tool_requests(
    registry_dir: str | Path,
    *,
    task_name: str,
    episode_dirs: Iterable[str | Path],
    include_derived_observables: bool = False,
) -> list[dict[str, Any]]:
    root = Path(registry_dir).expanduser().resolve()
    result: list[dict[str, Any]] = []
    for entry in load_registry(root)["entries"]:
        try:
            _source, spec_path = _entry_paths(root, entry)
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
        except (KeyError, OSError, json.JSONDecodeError):
            continue
        output = spec.get("output_contract", {})
        metric_spec = output.get("metric_spec")
        if spec.get("task_name") != task_name or not isinstance(metric_spec, dict):
            continue
        if metric_spec.get("operation") == "derived_observable" and not include_derived_observables:
            continue
        match = find_run_local_registration(root, tool_spec=spec, episode_dirs=episode_dirs)
        if match is None:
            continue
        result.append(
            {
                "registration_id": entry["semantic_key"],
                "request": {
                    "schema_version": 2,
                    "task_name": task_name,
                    "metric": spec["metric"],
                    "question": f"Reuse the validated {spec['metric']} measurement.",
                    "metric_spec": deepcopy(metric_spec),
                },
                "validation": {"current_telemetry_compatible": True},
            }
        )
    return result


def register_run_local_tool(
    registry_dir: str | Path,
    *,
    tool_spec: dict[str, Any],
    episode_dirs: Iterable[str | Path],
    source_path: str | Path,
    tool_id: str | None = None,
    semantic_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(registry_dir).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise RunLocalRegistryError(f"generated Tool source is missing: {source}")
    validate_generated_tool(source.read_text(encoding="utf-8"))
    telemetry_schema_compatibility(
        episode_dirs, required_signals=tool_spec.get("required_signals", [])
    )
    key = tool_semantic_key(tool_spec)
    index = load_registry(root)
    existing = next((item for item in index["entries"] if item.get("semantic_key") == key), None)
    if existing is None:
        slug = re.sub(r"[^a-z0-9]+", "_", _normalized(tool_spec["metric"])).strip("_")
        base_name = f"{_normalized(tool_spec['task_name']).replace(' ', '_')}__{slug}"
        entry_dir = root / "entries" / base_name
        suffix = 2
        while entry_dir.exists():
            entry_dir = root / "entries" / f"{base_name}__{suffix}"
            suffix += 1
        entry_dir.mkdir(parents=True, exist_ok=True)
        stored_source = entry_dir / "generated_tool.py"
        stored_spec = entry_dir / "tool_spec.json"
        shutil.copyfile(source, stored_source)
        _write_json(stored_spec, tool_spec)
        existing = {
            "semantic_key": key,
            "task_name": tool_spec["task_name"],
            "metric": tool_spec["metric"],
            "tool_id": tool_id or tool_spec.get("metric"),
            "semantic_review": deepcopy(semantic_review),
            "artifact": str(stored_source.relative_to(root)),
            "tool_spec_artifact": str(stored_spec.relative_to(root)),
        }
        index["entries"].append(existing)
        index["entries"].sort(key=lambda item: item["semantic_key"])
        _write_json(root / "index.json", index)
    match = find_run_local_registration(root, tool_spec=tool_spec, episode_dirs=episode_dirs)
    if match is None:
        raise RunLocalRegistryError("registered Tool failed immediate retrieval validation")
    return match


def public_registration_summary(match: dict[str, Any]) -> dict[str, Any]:
    registration = match["registration"]
    return {
        "registration_id": registration["registration_id"],
        "semantic_key": registration["semantic_key"],
        "scope": registration["scope"],
        "status": registration["status"],
        "tool_id": registration["tool_id"],
        "target_metric": registration["target_metric"],
    }
