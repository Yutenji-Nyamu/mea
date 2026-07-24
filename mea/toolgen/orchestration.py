"""Resolve and execute a Plan Agent ToolSpec over recorded trajectories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mea.toolkit.tools import TOOL_CATALOG, TrajectoryView

from .prototype import ToolGenPrototype, execute_generated_tool
from .registry import (
    RunLocalRegistryError,
    find_run_local_registration,
    infer_registry_dir,
    public_registration_summary,
    register_run_local_tool,
)
from .reviewed_registry import (
    ReviewedRegistryError,
    find_reviewed_registration,
    public_reviewed_registration_summary,
)
from .metric_spec import (
    MetricSpecError,
    build_task_code_context,
    execute_metric_spec,
)
from .router import (
    ToolRouterError,
    route_tool_request,
)
from .targets import (
    BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC,
    COMPOSITE_TARGETS,
    PICKUP_TO_CONTACT_METRIC,
    evaluate_target_oracle,
    target_definition,
)


class ToolOrchestrationError(RuntimeError):
    """Raised when a ToolSpec or its runtime inputs violate the contract."""


CONTACT_METRIC = "hammer_block_contact_ever"
CONTACT_QUESTION = "蓝色方块场景中，锤子是否与方块发生过严格物理接触？"
CONTACT_REQUIRED_SIGNALS = [
    "hammer_block_contact_intervals",
    "physics_step_index",
]
CONTACT_OUTPUT_CONTRACT = {
    "value_type": "boolean",
    "unit": None,
    "passed_rule": "equals_value",
    "evidence_rule": "first_physical_contact_physics_step_or_empty",
}
CONTACT_VALIDATION_REQUIREMENTS = {
    "force_codegen": {
        "min_episodes": 2,
        "distinct_reference_values": True,
        "required_reference_values": [False, True],
    },
    "reuse": {
        "min_episodes": 1,
        "distinct_reference_values": False,
        "required_reference_values": [],
    },
}
PICKUP_TO_CONTACT_QUESTION = (
    "蓝色方块场景中，从锤子首次抬升达到 pickup 阈值到首次严格物理接触方块，"
    "经过多少秒？"
)
PICKUP_TO_CONTACT_REQUIRED_SIGNALS = [
    "semantic_trace.hammer_position",
    "semantic_trace.physics_step",
    "semantic_trace.simulation_time_seconds",
    "events.hammer_block_contact_intervals",
    "schema.pickup_height_threshold_m",
]
PICKUP_TO_CONTACT_OUTPUT_CONTRACT = {
    "value_type": "number_or_null",
    "unit": "s",
    "passed_rule": "always_null",
    "evidence_rule": "pickup_and_contact_physics_steps_or_available",
    "null_rule": "missing_pickup_or_contact_or_invalid_order",
}
PICKUP_TO_CONTACT_VALIDATION_REQUIREMENTS = {
    "min_episodes": 2,
    "distinct_reference_values": True,
    "required_reference_values": [],
}
BELL_ACTIVE_TCP_MIN_XY_ERROR_QUESTION = (
    "What was the minimum XY distance between the official active-arm TCP "
    "and the bell contact point during this rollout?"
)
TOOL_SPEC_KEYS = {
    "schema_version",
    "task_name",
    "metric",
    "question",
    "route",
    "reference_tool",
    "required_signals",
    "output_contract",
    "validation_requirements",
}


def contact_tool_request() -> dict[str, Any]:
    """Return a route-free request for strict hammer-block contact."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": CONTACT_METRIC,
        "question": CONTACT_QUESTION,
    }


def pickup_to_contact_tool_request() -> dict[str, Any]:
    """Return a route-free request for pickup-to-contact duration."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": PICKUP_TO_CONTACT_METRIC,
        "question": PICKUP_TO_CONTACT_QUESTION,
    }


def bell_active_tcp_min_xy_error_tool_request() -> dict[str, Any]:
    """Request the object-position diagnostic used for both bell sides."""

    return {
        "schema_version": 1,
        "task_name": "click_bell",
        "metric": BELL_ACTIVE_TCP_MIN_XY_ERROR_METRIC,
        "question": BELL_ACTIVE_TCP_MIN_XY_ERROR_QUESTION,
    }


def official_success_tool_request(task_name: str) -> dict[str, Any]:
    """Return a route-free request for one schema-backed official outcome."""

    if not isinstance(task_name, str) or not task_name.strip():
        raise ToolOrchestrationError("task_name must be a non-empty string")
    return {
        "schema_version": 1,
        "task_name": task_name.strip(),
        "metric": "official_check_success",
        "question": "Did the rollout satisfy the official RoboTwin success check?",
    }


def bbh_distractor_success_tool_request() -> dict[str, Any]:
    """Request the outcome from the validated provider-written BBH checker."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": "bbh_target_without_distractor_success",
        "question": (
            "Did the rollout hit the target block while avoiding every "
            "contact with the look-alike distractor?"
        ),
    }


def hammer_left_camera_contact_count_tool_request() -> dict[str, Any]:
    """Request the bounded BBH unintended-contact proxy from Trusted Tools."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": "hammer_left_camera_contact_count",
        "question": (
            "How many physical hammer-left_camera contact intervals occurred?"
        ),
    }


def time_to_success_tool_request(task_name: str) -> dict[str, Any]:
    """Request the trusted first-success timestamp for an official task.

    Aggregate Toolkit computes the cross-seed mean and dispersion.  Keeping
    this request route-free lets the Tool router prove that the implementation
    came from the audited Trusted Tool catalog rather than model-generated
    measurement code.
    """

    if not isinstance(task_name, str) or not task_name.strip():
        raise ToolOrchestrationError("task_name must be a non-empty string")
    return {
        "schema_version": 1,
        "task_name": task_name.strip(),
        "metric": "time_to_success",
        "question": "When did the rollout first satisfy the official success check?",
    }


def contact_tool_spec(route: str) -> dict[str, Any]:
    """Return the exact first-version contact ToolSpec for a route."""

    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": CONTACT_METRIC,
        "question": CONTACT_QUESTION,
        "route": route,
        "reference_tool": CONTACT_METRIC,
        "required_signals": list(CONTACT_REQUIRED_SIGNALS),
        "output_contract": dict(CONTACT_OUTPUT_CONTRACT),
        "validation_requirements": {
            **CONTACT_VALIDATION_REQUIREMENTS[route],
            "required_reference_values": list(
                CONTACT_VALIDATION_REQUIREMENTS[route][
                    "required_reference_values"
                ]
            ),
        },
    }


def pickup_to_contact_tool_spec(route: str = "force_codegen") -> dict[str, Any]:
    """Return the first genuinely new, composition-validated ToolSpec."""

    if route != "force_codegen":
        raise ToolOrchestrationError(
            "pickup_to_first_contact_time 尚未进入 Trusted catalog，只允许 force_codegen"
        )
    return {
        "schema_version": 1,
        "task_name": "beat_block_hammer",
        "metric": PICKUP_TO_CONTACT_METRIC,
        "question": PICKUP_TO_CONTACT_QUESTION,
        "route": route,
        "reference_tool": None,
        "required_signals": list(PICKUP_TO_CONTACT_REQUIRED_SIGNALS),
        "output_contract": dict(PICKUP_TO_CONTACT_OUTPUT_CONTRACT),
        "validation_requirements": {
            **PICKUP_TO_CONTACT_VALIDATION_REQUIREMENTS,
            "required_reference_values": [],
        },
    }


def _composite_tool_spec(
    metric: str,
    question: str,
    task_name: str,
    *,
    route: str = "force_codegen",
) -> dict[str, Any]:
    if route != "force_codegen" or metric not in COMPOSITE_TARGETS:
        raise ToolOrchestrationError("composite targets require force_codegen")
    definition = target_definition(metric)
    supported = set(definition.get("supported_task_names", []))
    if task_name not in supported:
        raise ToolOrchestrationError(
            f"ToolSpec metric {metric!r} is incompatible with task {task_name!r}"
        )
    if metric == PICKUP_TO_CONTACT_METRIC:
        spec = pickup_to_contact_tool_spec(route)
        spec["question"] = question
        return spec
    return {
        "schema_version": 1,
        "task_name": task_name,
        "metric": metric,
        "question": question,
        "route": route,
        "reference_tool": None,
        "required_signals": list(definition.get("required_signals", [])),
        "output_contract": dict(definition.get("output_contract", {})),
        "validation_requirements": {
            **definition.get("validation_requirements", {}),
            "required_reference_values": list(
                definition.get("validation_requirements", {}).get(
                    "required_reference_values", []
                )
            ),
        },
    }


def _generic_trusted_tool_spec(
    metric: str,
    question: str,
    task_name: str,
) -> dict[str, Any]:
    """Build the internal routeful envelope for any exact catalog match."""

    if metric not in TOOL_CATALOG:
        raise ToolOrchestrationError(f"unknown Trusted Tool metric: {metric}")
    return {
        "schema_version": 1,
        "task_name": task_name,
        "metric": metric,
        "question": question,
        "route": "reuse",
        "reference_tool": metric,
        "required_signals": [],
        "output_contract": {"source": "trusted_tool_catalog"},
        "validation_requirements": {
            "min_episodes": 1,
            "distinct_reference_values": False,
            "required_reference_values": [],
        },
    }


def validate_tool_spec(
    value: Any,
    *,
    expected_route: str | None = None,
    expected_metric: str | None = None,
) -> dict[str, Any]:
    """Validate the intentionally narrow ToolSpec emitted by the Plan Agent."""

    if not isinstance(value, dict):
        raise ToolOrchestrationError("ToolSpec 必须是 JSON object")
    keys = set(value)
    if keys != TOOL_SPEC_KEYS:
        missing = sorted(TOOL_SPEC_KEYS - keys)
        extra = sorted(keys - TOOL_SPEC_KEYS)
        raise ToolOrchestrationError(
            f"ToolSpec fields 不匹配，missing={missing}, extra={extra}"
        )
    route = value.get("route")
    if route not in {"reuse", "force_codegen"}:
        raise ToolOrchestrationError("ToolSpec.route 只允许 reuse 或 force_codegen")
    if expected_route is not None and route != expected_route:
        raise ToolOrchestrationError(
            f"ToolSpec.route 必须是本轮约定的 {expected_route}"
        )
    metric = value.get("metric")
    if expected_metric is not None and metric != expected_metric:
        raise ToolOrchestrationError(
            f"ToolSpec.metric 必须是本轮约定的 {expected_metric}"
        )
    if metric == CONTACT_METRIC:
        expected = contact_tool_spec(route)
    elif metric in COMPOSITE_TARGETS:
        question = value.get("question")
        task_name = value.get("task_name")
        if not isinstance(question, str) or not question.strip():
            raise ToolOrchestrationError("ToolSpec.question must be non-empty")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ToolOrchestrationError("ToolSpec.task_name must be non-empty")
        expected = _composite_tool_spec(
            metric, question.strip(), task_name.strip(), route=route
        )
    elif route == "reuse" and metric in TOOL_CATALOG:
        question = value.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolOrchestrationError("ToolSpec.question must be non-empty")
        task_name = value.get("task_name")
        if not isinstance(task_name, str) or not task_name.strip():
            raise ToolOrchestrationError("ToolSpec.task_name must be non-empty")
        supported = set(
            TOOL_CATALOG[metric].get("supported_task_names", [])
        )
        if "*" not in supported and task_name not in supported:
            raise ToolOrchestrationError(
                f"ToolSpec metric {metric!r} is incompatible with task {task_name!r}"
            )
        expected = _generic_trusted_tool_spec(
            metric, question.strip(), task_name.strip()
        )
    else:
        raise ToolOrchestrationError(f"当前未注册 ToolSpec metric: {metric}")
    question = value.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ToolOrchestrationError("ToolSpec.question must be non-empty")
    expected["question"] = question.strip()
    for field in TOOL_SPEC_KEYS - {"route", "question"}:
        if value.get(field) != expected[field]:
            raise ToolOrchestrationError(
                f"ToolSpec.{field} 必须等于已验证的 {metric} contract"
            )
    return expected


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _relative(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _role(policy_name: Any) -> str:
    normalized = str(policy_name or "").casefold()
    if normalized == "act":
        return "policy_under_evaluation"
    if normalized == "expert":
        return "expert_validation"
    return "validation_control"


def _result_projection(result: dict[str, Any]) -> dict[str, Any]:
    projected = {
        "tool": result.get("tool"),
        "value": result.get("value"),
        "unit": result.get("unit"),
        "passed": result.get("passed"),
        "evidence_steps": list(result.get("evidence_steps", [])),
        "details": dict(result.get("details", {})),
    }
    if result.get("tool_sha256"):
        projected["tool_sha256"] = result["tool_sha256"]
    return projected


def _discover_episodes(
    child_run_dir: Path,
    target_metric: str,
    reference_tool: str | None,
    task_name: str,
) -> list[dict[str, Any]]:
    telemetry_root = child_run_dir / "evaluation/telemetry"
    episodes: list[dict[str, Any]] = []
    for metadata_path in sorted(telemetry_root.glob("*/episode_*/episode.json")):
        episode_dir = metadata_path.parent
        try:
            trajectory = TrajectoryView(episode_dir)
            if (
                reference_tool in TOOL_CATALOG
                and target_metric == reference_tool
            ):
                # Preserve Trusted Tool provenance for reuse.  The generic
                # oracle projection intentionally strips tool/version/hash for
                # differential comparison, but those fields belong in the
                # user-facing reuse result and source envelope.
                oracle_result = TOOL_CATALOG[reference_tool]["function"](
                    trajectory
                )
            else:
                oracle_result = evaluate_target_oracle(
                    target_metric,
                    trajectory,
                    reference_tool=reference_tool,
                )
        except Exception as exc:
            raise ToolOrchestrationError(
                f"无法加载 telemetry episode {episode_dir}: {exc}"
            ) from exc
        if trajectory.metadata.get("error") is not None:
            raise ToolOrchestrationError(
                f"不接受带 error 的 telemetry episode: {episode_dir}"
            )
        if (
            trajectory.metadata.get("task_name") != task_name
            or trajectory.schema.get("task_name") != task_name
        ):
            raise ToolOrchestrationError(
                f"metadata/schema task 不匹配: {episode_dir}"
            )
        episodes.append(
            {
                "episode_dir_path": episode_dir,
                "policy_name": trajectory.metadata.get("policy_name"),
                "seed": trajectory.metadata.get("seed"),
                "role": _role(trajectory.metadata.get("policy_name")),
                "oracle_result": oracle_result,
            }
        )
    if not episodes:
        raise ToolOrchestrationError(
            f"没有在 {telemetry_root} 下发现完整 telemetry episode"
        )
    episodes.sort(
        key=lambda item: (
            {"policy_under_evaluation": 0, "expert_validation": 1}.get(
                item["role"], 2
            ),
            int(item.get("seed") or -1),
            str(item["episode_dir_path"]),
        )
    )
    return episodes


def _resolve(
    repo_root: Path,
    child_run_dir: Path,
    tool_spec: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episodes = _discover_episodes(
        child_run_dir,
        tool_spec["metric"],
        tool_spec["reference_tool"],
        tool_spec["task_name"],
    )
    reference_values = [
        item["oracle_result"].get("value") for item in episodes
    ]
    requirements = tool_spec["validation_requirements"]
    if len(episodes) < int(requirements["min_episodes"]):
        raise ToolOrchestrationError(
            "telemetry episode 数量不足以满足 ToolSpec validation"
        )
    if requirements["distinct_reference_values"] and len(
        set(reference_values)
    ) < 2:
        raise ToolOrchestrationError(
            "telemetry 缺少 ToolGen differential gate 所需的不同 reference 输出"
        )
    required_values = set(requirements["required_reference_values"])
    if not required_values.issubset(set(reference_values)):
        raise ToolOrchestrationError(
            "telemetry 缺少 contact Tool 所需的 reference false/true 对照"
        )
    resolved = {
        "schema_version": 1,
        "tool_spec": tool_spec,
        "resolved_route": tool_spec["route"],
        "resolved_tool_name": (
            tool_spec["reference_tool"]
            if tool_spec["route"] == "reuse"
            else f"generated_{tool_spec['metric']}"
        ),
        "resolved_episodes": [
            {
                "role": item["role"],
                "policy_name": item["policy_name"],
                "seed": item["seed"],
                "oracle_value": item["oracle_result"].get("value"),
                "episode_dir": _relative(item["episode_dir_path"], repo_root),
            }
            for item in episodes
        ],
    }
    return resolved, episodes


def execute_tool_spec(
    repo_root: str | Path,
    child_run_dir: str | Path,
    output_dir: str | Path,
    tool_spec: dict[str, Any],
    *,
    provider: Any | None = None,
    model: str | None = None,
    max_attempts: int = 2,
    _precreated_destination: bool = False,
) -> dict[str, Any]:
    """Execute reuse or force-codegen and emit one normalized envelope."""

    repo = Path(repo_root).expanduser().resolve()
    child = Path(child_run_dir).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and not _precreated_destination:
        raise ToolOrchestrationError(f"tool output directory 已存在: {destination}")
    spec = validate_tool_spec(tool_spec)
    if _precreated_destination and not destination.is_dir():
        raise ToolOrchestrationError(
            f"precreated tool output directory does not exist: {destination}"
        )
    resolved, episodes = _resolve(repo, child, spec)
    destination.mkdir(parents=True, exist_ok=_precreated_destination)
    _write_json(destination / "tool_spec.json", spec)
    _write_json(destination / "resolved_tool_spec.json", resolved)

    if spec["route"] == "reuse":
        if spec["reference_tool"] not in TOOL_CATALOG:
            raise ToolOrchestrationError(
                "reuse route 必须解析到 Trusted catalog tool"
            )
        normalized_episodes = [
            {
                "episode_dir": _relative(item["episode_dir_path"], repo),
                "policy_name": item["policy_name"],
                "seed": item["seed"],
                "role": item["role"],
                "result": _result_projection(item["oracle_result"]),
            }
            for item in episodes
        ]
        first_result = normalized_episodes[0]["result"]
        execution = {
            "schema_version": 1,
            "status": "passed",
            "route": "reuse",
            "reference_tool": spec["reference_tool"],
            "tool_spec": spec,
            "source": {
                "scope": "trusted_catalog",
                "tool": spec["reference_tool"],
                "reference_tool": spec["reference_tool"],
                "tool_sha256": first_result.get("tool_sha256"),
                "artifact": None,
            },
            "episodes": normalized_episodes,
            "validation": {
                "provider_called": False,
                "catalog_tool_found": True,
                "episode_count": len(normalized_episodes),
                "required_reference_values_observed": True,
            },
            "artifacts": {
                "tool_spec": _relative(destination / "tool_spec.json", repo),
                "resolved_tool_spec": _relative(
                    destination / "resolved_tool_spec.json", repo
                ),
            },
        }
    else:
        if provider is None or not model:
            raise ToolOrchestrationError(
                "force_codegen route 必须提供 provider 和 model"
            )
        generated_dir = destination / "generated"
        try:
            manifest = ToolGenPrototype(repo, provider, model=model).generate(
                spec["question"],
                reference_tool=spec["reference_tool"],
                target_metric=spec["metric"],
                episode_dirs=[item["episode_dir_path"] for item in episodes],
                output_dir=generated_dir,
                tool_name=resolved["resolved_tool_name"],
                max_attempts=max_attempts,
            )
        except Exception as exc:
            raise ToolOrchestrationError(f"planned ToolGen failed: {exc}") from exc
        raw_results = json.loads(
            (generated_dir / "execution_results.json").read_text(encoding="utf-8")
        )
        normalized_episodes = []
        validation_rows = []
        episode_lookup = {
            str(item["episode_dir_path"].resolve()): item for item in episodes
        }
        for raw in raw_results:
            episode = episode_lookup.get(str(Path(raw["episode_dir"]).resolve()))
            if episode is None:
                raise ToolOrchestrationError(
                    f"ToolGen 返回未知 episode: {raw['episode_dir']}"
                )
            normalized_episodes.append(
                {
                    "episode_dir": _relative(episode["episode_dir_path"], repo),
                    "policy_name": episode["policy_name"],
                    "seed": episode["seed"],
                    "role": episode["role"],
                    "result": _result_projection(raw["generated_result"]),
                }
            )
            validation_rows.append(
                {
                    "episode_dir": _relative(episode["episode_dir_path"], repo),
                    "deterministic": bool(raw.get("deterministic")),
                    "oracle_agreement": bool(raw.get("oracle_agreement")),
                    "artifacts_unchanged": bool(raw.get("artifacts_unchanged")),
                }
            )
        all_gates_passed = all(
            row[gate]
            for row in validation_rows
            for gate in (
                "deterministic",
                "oracle_agreement",
                "artifacts_unchanged",
            )
        )
        if not all_gates_passed:
            raise ToolOrchestrationError("ToolGen result 未通过 deterministic gates")
        execution = {
            "schema_version": 1,
            "status": "passed",
            "route": "force_codegen",
            "reference_tool": spec["reference_tool"],
            "tool_spec": spec,
            "source": {
                "scope": "run_local_generated",
                "tool": manifest["registration"]["tool"],
                "reference_tool": spec["reference_tool"],
                "tool_sha256": manifest["registration"]["tool_sha256"],
                "artifact": _relative(generated_dir / "generated_tool.py", repo),
            },
            "episodes": normalized_episodes,
            "validation": {
                "provider_called": True,
                "successful_attempt": manifest.get("successful_attempt"),
                "all_gates_passed": all_gates_passed,
                "episodes": validation_rows,
            },
            "artifacts": {
                "tool_spec": _relative(destination / "tool_spec.json", repo),
                "resolved_tool_spec": _relative(
                    destination / "resolved_tool_spec.json", repo
                ),
                "toolgen_manifest": _relative(
                    generated_dir / "manifest.json", repo
                ),
                "registration": _relative(
                    generated_dir / "registration.json", repo
                ),
                "execution_results": _relative(
                    generated_dir / "execution_results.json", repo
                ),
                "property_validation": _relative(
                    generated_dir / "property_validation.json", repo
                ),
                "generated_tool": _relative(
                    generated_dir / "generated_tool.py", repo
                ),
            },
        }

    _write_json(destination / "tool_execution.json", execution)
    execution["artifacts"]["tool_execution"] = _relative(
        destination / "tool_execution.json", repo
    )
    _write_json(destination / "tool_execution.json", execution)
    return execution


def _oracle_projection(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "value": result.get("value"),
        "unit": result.get("unit"),
        "passed": result.get("passed"),
        "evidence_steps": list(result.get("evidence_steps", [])),
        "details": dict(result.get("details", {})),
    }


def _same_projection(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(
        _oracle_projection(left),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) == json.dumps(
        _oracle_projection(right),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _execute_registry_match(
    repo_root: Path,
    destination: Path,
    spec: dict[str, Any],
    match: dict[str, Any],
    episodes: list[dict[str, Any]],
    *,
    route: str,
    source_scope: str,
    registration_id_field: str,
    registry_artifact_key: str,
) -> dict[str, Any]:
    """Execute one exact generated-code registry match without a provider."""

    registration = match["registration"]
    source_path = match["source_path"]
    # Reviewed lookup returns the exact text whose hash was checked.  Run-local
    # matches retain their legacy path-backed behavior.
    source = match.get("source")
    if not isinstance(source, str):
        source = source_path.read_text(encoding="utf-8")
    normalized_episodes: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for episode in episodes:
        generated = execute_generated_tool(
            source,
            episode["episode_dir_path"],
            tool_name=registration["tool_id"],
        )
        repeated = execute_generated_tool(
            source,
            episode["episode_dir_path"],
            tool_name=registration["tool_id"],
        )
        expected = evaluate_target_oracle(
            spec["metric"],
            TrajectoryView(episode["episode_dir_path"]),
            reference_tool=spec["reference_tool"],
        )
        agreement = _same_projection(generated, expected)
        deterministic = generated == repeated
        if not deterministic or not agreement:
            raise ToolOrchestrationError(
                f"{route} Tool failed deterministic/oracle revalidation"
            )
        normalized_episodes.append(
            {
                "episode_dir": _relative(episode["episode_dir_path"], repo_root),
                "policy_name": episode["policy_name"],
                "seed": episode["seed"],
                "role": episode["role"],
                "result": _result_projection(generated),
            }
        )
        validation_rows.append(
            {
                "episode_dir": _relative(episode["episode_dir_path"], repo_root),
                "deterministic": deterministic,
                "oracle_agreement": agreement,
            }
        )
    resolved = {
        "schema_version": 1,
        "tool_spec": spec,
        "resolved_route": route,
        "resolved_tool_name": registration["tool_id"],
        registration_id_field: registration["registration_id"],
        "resolved_episodes": [
            {
                "role": item["role"],
                "policy_name": item["policy_name"],
                "seed": item["seed"],
                "oracle_value": item["oracle_result"].get("value"),
                "episode_dir": _relative(item["episode_dir_path"], repo_root),
            }
            for item in episodes
        ],
    }
    _write_json(destination / "tool_spec.json", spec)
    _write_json(destination / "resolved_tool_spec.json", resolved)
    execution = {
        "schema_version": 1,
        "status": "passed",
        "route": route,
        "reference_tool": spec["reference_tool"],
        "tool_spec": spec,
        "source": {
            "scope": source_scope,
            "tool": registration["tool_id"],
            "reference_tool": spec["reference_tool"],
            "tool_sha256": registration["code_sha256"],
            "registration_id": registration["registration_id"],
            "artifact": _relative(source_path, repo_root),
        },
        "episodes": normalized_episodes,
        "validation": {
            "provider_called": False,
            "registry_match": True,
            "integrity_hashes_matched": True,
            "all_gates_passed": all(
                item[gate]
                for item in validation_rows
                for gate in ("deterministic", "oracle_agreement")
            ),
            "episodes": validation_rows,
        },
        "artifacts": {
            "tool_spec": _relative(destination / "tool_spec.json", repo_root),
            "resolved_tool_spec": _relative(
                destination / "resolved_tool_spec.json", repo_root
            ),
            "registration": _relative(match["registration_path"], repo_root),
            "generated_tool": _relative(source_path, repo_root),
        },
    }
    if route == "reviewed_persistent_reuse":
        execution["validation"]["review_manifest_approved"] = (
            match.get("review_manifest", {}).get("decision") == "approved"
        )
    execution["artifacts"][registry_artifact_key] = _relative(
        match["registry_dir"] / "index.json", repo_root
    )
    review_path = match.get("review_manifest_path")
    if isinstance(review_path, Path):
        execution["artifacts"]["review_manifest"] = _relative(
            review_path, repo_root
        )
    _write_json(destination / "tool_execution.json", execution)
    execution["artifacts"]["tool_execution"] = _relative(
        destination / "tool_execution.json", repo_root
    )
    _write_json(destination / "tool_execution.json", execution)
    return execution


def _execute_run_local_match(
    repo_root: Path,
    destination: Path,
    spec: dict[str, Any],
    match: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve the established evaluation-local reuse envelope."""

    return _execute_registry_match(
        repo_root,
        destination,
        spec,
        match,
        episodes,
        route="run_local_reuse",
        source_scope="run_local_registry",
        registration_id_field="run_local_registration_id",
        registry_artifact_key="run_local_registry",
    )


def _execute_reviewed_match(
    repo_root: Path,
    destination: Path,
    spec: dict[str, Any],
    match: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reuse explicitly reviewed code while retaining current-data gates."""

    return _execute_registry_match(
        repo_root,
        destination,
        spec,
        match,
        episodes,
        route="reviewed_persistent_reuse",
        source_scope="reviewed_persistent_registry",
        registration_id_field="reviewed_registration_id",
        registry_artifact_key="reviewed_registry",
    )


def _resolved_spec_from_request(
    tool_request: dict[str, Any],
    resolved_route: str,
) -> dict[str, Any]:
    """Translate a semantic request into the legacy internal execution spec."""

    metric = tool_request["metric"]
    question = tool_request["question"]
    if resolved_route == "reuse":
        if metric == CONTACT_METRIC:
            spec = contact_tool_spec("reuse")
        else:
            spec = _generic_trusted_tool_spec(
                metric, question, tool_request["task_name"]
            )
    elif resolved_route == "force_codegen":
        if metric not in COMPOSITE_TARGETS:
            raise ToolOrchestrationError(
                f"no executable composite ToolSpec for metric: {metric}"
            )
        spec = _composite_tool_spec(
            metric,
            question,
            tool_request["task_name"],
            route="force_codegen",
        )
    else:
        raise ToolOrchestrationError(
            f"automatic Tool route is not executable: {resolved_route}"
        )
    spec["question"] = question
    return spec


def _execute_typed_metric_request(
    repo: Path,
    child_run_dir: Path,
    destination: Path,
    request: dict[str, Any],
    decision: dict[str, Any],
    *,
    registry_root: Path | None,
    task_proposal: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compile one ToolProposal v3 metric over the round's real telemetry."""

    telemetry_root = child_run_dir / "evaluation/telemetry"
    episode_dirs = [
        path.parent
        for path in sorted(telemetry_root.glob("*/episode_*/episode.json"))
    ]
    if not episode_dirs:
        raise ToolOrchestrationError(
            f"no complete telemetry episode found under {telemetry_root}"
        )
    try:
        context = build_task_code_context(
            child_run_dir, task_proposal=task_proposal
        )
        raw = execute_metric_spec(
            task_name=request["task_name"],
            metric=request["metric"],
            question=request["question"],
            metric_spec=request["metric_spec"],
            episode_dirs=episode_dirs,
            output_dir=destination / "typed_metric_spec",
            task_code_context=context,
            registry_dir=registry_root,
        )
    except MetricSpecError as exc:
        raise ToolOrchestrationError(f"typed MetricSpec execution failed: {exc}") from exc

    actual_route = str(raw["route"])
    decision["resolved_route"] = actual_route
    if actual_route == "run_local_reuse":
        decision["matched_registry"] = "evaluation_local_tool_registry"
        decision["reason"] = (
            "exact typed MetricSpec, task, and telemetry schema matched a "
            "validated evaluation-local Tool"
        )
    decision["provider_called"] = False
    _write_json(destination / "route_decision.json", decision)

    normalized_episodes = [
        {
            "episode_dir": _relative(Path(row["episode_dir"]), repo),
            "policy_name": row.get("policy_name"),
            "seed": row.get("seed"),
            "role": _role(row.get("policy_name")),
            "result": _result_projection(row["generated_result"]),
        }
        for row in raw["episodes"]
    ]
    generated_source = destination / "typed_metric_spec/generated_tool.py"
    registration = raw.get("registration")
    execution = {
        "schema_version": 1,
        "status": "passed",
        "requested_route": "auto",
        "route": actual_route,
        "reference_tool": None,
        "tool_spec": raw["tool_spec"],
        "tool_request": request,
        "route_decision": decision,
        "source": {
            "scope": (
                "run_local_registry"
                if actual_route == "run_local_reuse"
                else "run_local_generated"
            ),
            "tool": request["metric"],
            "reference_tool": None,
            "artifact": (
                _relative(generated_source, repo)
                if generated_source.is_file()
                else None
            ),
            "registration_id": (
                registration.get("registration_id")
                if isinstance(registration, dict)
                else None
            ),
        },
        "episodes": normalized_episodes,
        "validation": {
            "provider_called": False,
            "typed_metric_spec": True,
            "task_code_context_consumed": bool(
                raw.get("task_code_context_consumed")
            ),
            "episode_count": len(normalized_episodes),
            "differential_gates_passed": True,
        },
        "artifacts": {
            "tool_request": _relative(destination / "tool_request.json", repo),
            "catalog_snapshot": _relative(
                destination / "catalog_snapshot.json", repo
            ),
            "route_decision": _relative(
                destination / "route_decision.json", repo
            ),
            "metric_spec_execution": _relative(
                destination / "typed_metric_spec/execution.json", repo
            ),
        },
    }
    _write_json(destination / "resolved_tool_spec.json", execution["tool_spec"])
    execution["artifacts"]["resolved_tool_spec"] = _relative(
        destination / "resolved_tool_spec.json", repo
    )
    execution["artifacts"]["tool_execution"] = _relative(
        destination / "tool_execution.json", repo
    )
    _write_json(destination / "tool_execution.json", execution)
    return execution


def _register_generated_for_evaluation(
    repo: Path,
    child_run_dir: str | Path,
    destination: Path,
    registry_root: Path,
    spec: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    generated_dir = destination / "generated"
    manifest_path = generated_dir / "manifest.json"
    generation_registration_path = generated_dir / "registration.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation_registration = json.loads(
        generation_registration_path.read_text(encoding="utf-8")
    )
    episodes = _discover_episodes(
        Path(child_run_dir).expanduser().resolve(),
        spec["metric"],
        spec["reference_tool"],
        spec["task_name"],
    )
    validation_episodes = [
        {
            "episode_dir": _relative(item["episode_dir_path"], repo),
            "policy_name": item["policy_name"],
            "seed": item["seed"],
            "role": item["role"],
            "oracle_value": item["oracle_result"].get("value"),
        }
        for item in episodes
    ]
    try:
        match = register_run_local_tool(
            registry_root,
            tool_spec=spec,
            episode_dirs=[item["episode_dir_path"] for item in episodes],
            source_path=generated_dir / "generated_tool.py",
            generation_registration=generation_registration,
            generation_manifest=manifest,
            validation_episodes=validation_episodes,
        )
    except RunLocalRegistryError as exc:
        raise ToolOrchestrationError(
            f"failed to register generated Tool for this evaluation: {exc}"
        ) from exc
    registration = match["registration"]
    # Preserve the legacy per-generation registration path while enriching its
    # contents with the exact reusable contract and compatibility hashes.
    _write_json(generation_registration_path, registration)
    manifest["registration"] = registration
    _write_json(manifest_path, manifest)
    execution["source"].update(
        {
            "registration_id": registration["registration_id"],
            "registration_scope": "run_local",
        }
    )
    execution["run_local_registration"] = public_registration_summary(match)
    execution["artifacts"].update(
        {
            "run_local_registry": _relative(registry_root / "index.json", repo),
            "run_local_registration": _relative(
                match["registration_path"], repo
            ),
            "run_local_generated_tool": _relative(match["source_path"], repo),
        }
    )
    return execution


def execute_tool_request(
    repo_root: str | Path,
    child_run_dir: str | Path,
    output_dir: str | Path,
    tool_request: dict[str, Any],
    *,
    provider: Any | None = None,
    model: str | None = None,
    max_attempts: int = 2,
    run_local_registry_dir: str | Path | None = None,
    reviewed_registry_dir: str | Path | None = None,
    task_proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Automatically route and execute one route-free semantic Tool request."""

    repo = Path(repo_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise ToolOrchestrationError(
            f"tool output directory already exists: {destination}"
        )
    try:
        routing = route_tool_request(tool_request)
    except ToolRouterError as exc:
        raise ToolOrchestrationError(f"invalid tool_request: {exc}") from exc

    request = routing["tool_request"]
    snapshot = routing["catalog_snapshot"]
    decision = routing["route_decision"]
    if decision["status"] != "resolved":
        destination.mkdir(parents=True)
        decision["provider_called"] = False
        _write_json(destination / "tool_request.json", request)
        _write_json(destination / "catalog_snapshot.json", snapshot)
        _write_json(destination / "route_decision.json", decision)
        raise ToolOrchestrationError(
            "automatic Tool Router found no supported exact metric match"
        )

    destination.mkdir(parents=True)
    _write_json(destination / "tool_request.json", request)
    _write_json(destination / "catalog_snapshot.json", snapshot)
    _write_json(destination / "route_decision.json", decision)
    registry_root = (
        Path(run_local_registry_dir).expanduser().resolve()
        if run_local_registry_dir is not None
        else infer_registry_dir(destination)
    )
    reviewed_root = (
        Path(reviewed_registry_dir).expanduser().resolve()
        if reviewed_registry_dir is not None
        else None
    )
    if decision["resolved_route"] == "typed_metric_spec_compile":
        try:
            return _execute_typed_metric_request(
                repo,
                Path(child_run_dir).expanduser().resolve(),
                destination,
                request,
                decision,
                registry_root=registry_root,
                task_proposal=task_proposal,
            )
        except Exception as exc:
            decision["status"] = "execution_failed"
            decision["provider_called"] = False
            decision["failure"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            _write_json(destination / "route_decision.json", decision)
            raise

    spec = _resolved_spec_from_request(request, decision["resolved_route"])
    run_local_match = None
    reviewed_match = None
    registry_episodes: list[dict[str, Any]] | None = None
    if decision["resolved_route"] == "force_codegen":
        if registry_root is not None or reviewed_root is not None:
            try:
                registry_episodes = _discover_episodes(
                    Path(child_run_dir).expanduser().resolve(),
                    spec["metric"],
                    spec["reference_tool"],
                    spec["task_name"],
                )
            except ToolOrchestrationError:
                # Preserve the established execution-failure audit.  The normal
                # force-codegen path below will report the telemetry error.
                registry_episodes = None

        if registry_root is not None and registry_episodes is not None:
            try:
                run_local_match = find_run_local_registration(
                    registry_root,
                    tool_spec=spec,
                    episode_dirs=[
                        item["episode_dir_path"] for item in registry_episodes
                    ],
                )
            except RunLocalRegistryError as exc:
                decision["run_local_lookup"] = {
                    "status": "invalid_registry",
                    "message": str(exc),
                }
            if run_local_match is None and "run_local_lookup" not in decision:
                decision["run_local_lookup"] = {
                    "status": "miss",
                    "registry_dir": _relative(registry_root, repo),
                }

        if (
            run_local_match is None
            and reviewed_root is not None
            and registry_episodes is not None
        ):
            try:
                reviewed_match = find_reviewed_registration(
                    reviewed_root,
                    tool_spec=spec,
                    episode_dirs=[
                        item["episode_dir_path"] for item in registry_episodes
                    ],
                )
            except (ReviewedRegistryError, RunLocalRegistryError) as exc:
                decision["reviewed_lookup"] = {
                    "status": "invalid_registry",
                    "message": str(exc),
                }
            if reviewed_match is None and "reviewed_lookup" not in decision:
                decision["reviewed_lookup"] = {
                    "status": "miss",
                    "registry_dir": _relative(reviewed_root, repo),
                }

        lookup_audit = {
            key: decision[key]
            for key in ("run_local_lookup", "reviewed_lookup")
            if key in decision
        }
        if run_local_match is not None:
            routing = route_tool_request(
                request,
                run_local_registration=public_registration_summary(
                    run_local_match
                ),
            )
            snapshot = routing["catalog_snapshot"]
            decision = routing["route_decision"]
        elif reviewed_match is not None:
            routing = route_tool_request(
                request,
                reviewed_registration=public_reviewed_registration_summary(
                    reviewed_match
                ),
            )
            snapshot = routing["catalog_snapshot"]
            decision = routing["route_decision"]
        decision.update(lookup_audit)
        _write_json(destination / "catalog_snapshot.json", snapshot)
        _write_json(destination / "route_decision.json", decision)
    try:
        if run_local_match is not None and registry_episodes is not None:
            execution = _execute_run_local_match(
                repo,
                destination,
                spec,
                run_local_match,
                registry_episodes,
            )
        elif reviewed_match is not None and registry_episodes is not None:
            execution = _execute_reviewed_match(
                repo,
                destination,
                spec,
                reviewed_match,
                registry_episodes,
            )
        else:
            execution = execute_tool_spec(
                repo,
                child_run_dir,
                destination,
                spec,
                provider=provider,
                model=model,
                max_attempts=max_attempts,
                _precreated_destination=True,
            )
            if (
                decision["resolved_route"] == "force_codegen"
                and registry_root is not None
            ):
                execution = _register_generated_for_evaluation(
                    repo,
                    child_run_dir,
                    destination,
                    registry_root,
                    spec,
                    execution,
                )
    except Exception as exc:
        decision["status"] = "execution_failed"
        decision["provider_called"] = bool(
            decision["provider_required"] and provider is not None
        )
        decision["failure"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(destination / "route_decision.json", decision)
        raise

    decision["provider_called"] = bool(
        execution.get("validation", {}).get("provider_called")
    )
    _write_json(destination / "route_decision.json", decision)

    resolved_path = destination / "resolved_tool_spec.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["requested_route"] = "auto"
    resolved["route_decision"] = decision
    _write_json(resolved_path, resolved)

    execution["requested_route"] = "auto"
    execution["route"] = decision["resolved_route"]
    execution["tool_request"] = request
    execution["route_decision"] = decision
    execution["artifacts"].update(
        {
            "tool_request": _relative(destination / "tool_request.json", repo),
            "catalog_snapshot": _relative(
                destination / "catalog_snapshot.json", repo
            ),
            "route_decision": _relative(
                destination / "route_decision.json", repo
            ),
        }
    )
    _write_json(destination / "tool_execution.json", execution)
    return execution
