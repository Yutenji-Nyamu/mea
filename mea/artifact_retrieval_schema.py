"""Schema and semantic validation for reviewed artifact records."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .aspects import AspectError, aspect_semantics, canonicalize_aspect_id
from .artifact_retrieval_records import (
    CapabilityAdapterError,
    _ASPECT_KEYS,
    _CHANGE_ROOT_SCOPES,
    _CONTRACT_KEYS,
    _CONTRACTS,
    _CONTROLLED_AXIS_SCOPES,
    _OPERATIONS,
    _SEMANTIC_SCOPES,
    _TARGET_ROLES,
    _TASK_ADAPTER_KEYS,
    _TASK_ADAPTER_METADATA,
    _TASKGEN_KEYS,
    _TOOL_KEYS,
    _VQA_KEYS,
    _VQA_QUESTION_KEYS,
    _raw_task_adapter,
    _text,
)


def _validate_change_roots(
    *,
    change_scope: Any,
    allowed_roots: Any,
    changes: Any,
) -> dict[str, Any]:
    if change_scope is None:
        if allowed_roots != [] or changes != {}:
            raise CapabilityAdapterError(
                "official passthrough must have no allowed roots or changes"
            )
        return {}
    if change_scope not in {"object", "scene"}:
        raise CapabilityAdapterError("taskgen.change_scope must be object, scene, or null")
    if (
        not isinstance(allowed_roots, list)
        or not allowed_roots
        or any(not isinstance(item, str) or not item for item in allowed_roots)
        or len(allowed_roots) != len(set(allowed_roots))
    ):
        raise CapabilityAdapterError(
            "taskgen.allowed_change_roots must be a non-empty unique string list"
        )
    unknown_roots = sorted(set(allowed_roots) - set(_CHANGE_ROOT_SCOPES))
    if unknown_roots:
        raise CapabilityAdapterError(f"unknown taskgen change roots: {unknown_roots}")
    wrong_scope = sorted(
        root for root in allowed_roots if _CHANGE_ROOT_SCOPES[root] != change_scope
    )
    if wrong_scope:
        raise CapabilityAdapterError(
            f"change roots do not belong to {change_scope!r}: {wrong_scope}"
        )
    if not isinstance(changes, Mapping) or not changes:
        raise CapabilityAdapterError("generated/reused task changes must be non-empty")
    extra = sorted(set(changes) - set(allowed_roots))
    if extra:
        raise CapabilityAdapterError(f"changes exceed allowed roots: {extra}")
    missing = sorted(set(allowed_roots) - set(changes))
    if missing:
        raise CapabilityAdapterError(f"changes omit required roots: {missing}")
    return deepcopy(dict(changes))


def _validate_structure(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONTRACT_KEYS:
        raise CapabilityAdapterError(
            f"capability contract fields must be exactly {sorted(_CONTRACT_KEYS)}"
        )
    contract = deepcopy(dict(value))
    if contract.get("schema_version") != 1:
        raise CapabilityAdapterError("capability contract schema_version must be 1")
    task_name = _text(contract.get("task_name"), field="task_name")
    template_id = _text(contract.get("template_id"), field="template_id")

    aspect = contract.get("aspect")
    if not isinstance(aspect, dict) or set(aspect) != _ASPECT_KEYS:
        raise CapabilityAdapterError(
            f"aspect fields must be exactly {sorted(_ASPECT_KEYS)}"
        )
    try:
        canonical_aspect = canonicalize_aspect_id(aspect.get("aspect_id"))
        expected_semantics = aspect_semantics(canonical_aspect)
    except AspectError as exc:
        raise CapabilityAdapterError(str(exc)) from exc
    scope = aspect.get("semantic_scope")
    if scope not in _SEMANTIC_SCOPES or scope != expected_semantics["semantic_scope"]:
        raise CapabilityAdapterError("aspect semantic_scope does not match the ontology")
    role = aspect.get("target_role")
    if role not in _TARGET_ROLES[scope]:
        raise CapabilityAdapterError(
            f"target_role {role!r} is not valid for semantic_scope {scope!r}"
        )
    aspect["aspect_id"] = canonical_aspect

    taskgen = contract.get("taskgen")
    if not isinstance(taskgen, dict) or set(taskgen) != _TASKGEN_KEYS:
        raise CapabilityAdapterError(
            f"taskgen fields must be exactly {sorted(_TASKGEN_KEYS)}"
        )
    operation = taskgen.get("operation")
    if operation not in _OPERATIONS:
        raise CapabilityAdapterError(f"unsupported taskgen operation: {operation!r}")
    expected_generation_mode = {
        "force_codegen": "force_codegen",
        "provider_scene_checker_codegen": "provider_scene_checker_codegen",
        "bounded_variant_overlay": "bounded_variant_overlay",
        "reuse_variant": "reuse",
        "official_passthrough": None,
    }[operation]
    if taskgen.get("generation_mode") != expected_generation_mode:
        raise CapabilityAdapterError(
            "taskgen operation and generation_mode do not match"
        )
    if operation == "official_passthrough":
        if taskgen.get("capability_id") != "task_execution.official_passthrough":
            raise CapabilityAdapterError(
                "official passthrough must use its trusted capability id"
            )
        for field in (
            "task_variant_id",
            "controlled_axis",
            "change_scope",
            "generation_mode",
        ):
            if taskgen.get(field) is not None:
                raise CapabilityAdapterError(
                    f"official passthrough requires taskgen.{field}=null"
                )
    else:
        for field in (
            "capability_id",
            "task_variant_id",
            "controlled_axis",
            "generation_mode",
        ):
            _text(taskgen.get(field), field=f"taskgen.{field}")
        controlled_scope = _CONTROLLED_AXIS_SCOPES.get(taskgen.get("controlled_axis"))
        if controlled_scope is None:
            raise CapabilityAdapterError(
                f"unknown controlled_axis: {taskgen.get('controlled_axis')!r}"
            )
        if controlled_scope != taskgen.get("change_scope"):
            raise CapabilityAdapterError(
                "controlled_axis and taskgen.change_scope do not match"
            )
        if scope in {"object", "scene"} and scope != taskgen.get("change_scope"):
            raise CapabilityAdapterError(
                "evaluation object/scene scope and TaskGen change scope do not match"
            )
    taskgen["changes"] = _validate_change_roots(
        change_scope=taskgen.get("change_scope"),
        allowed_roots=taskgen.get("allowed_change_roots"),
        changes=taskgen.get("changes"),
    )

    tool = contract.get("tool")
    if not isinstance(tool, dict) or set(tool) != _TOOL_KEYS:
        raise CapabilityAdapterError(f"tool fields must be exactly {sorted(_TOOL_KEYS)}")
    _text(tool.get("request_factory_id"), field="tool.request_factory_id")
    _text(tool.get("metric"), field="tool.metric")

    vqa = contract.get("vqa")
    if not isinstance(vqa, dict) or set(vqa) != _VQA_KEYS:
        raise CapabilityAdapterError(f"vqa fields must be exactly {sorted(_VQA_KEYS)}")
    phenomenon_ids = vqa.get("phenomenon_ids")
    if (
        not isinstance(phenomenon_ids, list)
        or not phenomenon_ids
        or any(not isinstance(item, str) or not item for item in phenomenon_ids)
        or len(phenomenon_ids) != len(set(phenomenon_ids))
    ):
        raise CapabilityAdapterError(
            "vqa.phenomenon_ids must be a non-empty unique string list"
        )

    gates = contract.get("required_gates")
    if (
        not isinstance(gates, list)
        or not gates
        or any(not isinstance(item, str) or not item for item in gates)
        or len(gates) != len(set(gates))
    ):
        raise CapabilityAdapterError(
            "required_gates must be a non-empty unique string list"
        )
    if operation == "official_passthrough" and "variant_spec" in gates:
        raise CapabilityAdapterError("official passthrough cannot require variant_spec")
    if operation != "official_passthrough" and "variant_spec" not in gates:
        raise CapabilityAdapterError("generated/reused variants require variant_spec")

    contract.update(
        {
            "task_name": task_name,
            "template_id": template_id,
            "aspect": aspect,
            "taskgen": taskgen,
        }
    )
    return contract


def validate_capability_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate structure, semantic scope, and exact trusted registry identity."""

    contract = _validate_structure(value)
    identity = (contract["task_name"], contract["template_id"])
    expected = _CONTRACTS.get(identity)
    if expected is None:
        raise CapabilityAdapterError(f"unknown capability adapter: {identity!r}")
    if contract != expected:
        raise CapabilityAdapterError(
            f"capability adapter contract changed for {identity!r}"
        )
    return deepcopy(contract)


def resolve_capability_contract(task_name: Any, template_id: Any) -> dict[str, Any]:
    """Resolve one task/template identity to its complete trusted contract."""

    identity = (
        _text(task_name, field="task_name"),
        _text(template_id, field="template_id"),
    )
    try:
        contract = _CONTRACTS[identity]
    except KeyError as exc:
        raise CapabilityAdapterError(f"unknown capability adapter: {identity!r}") from exc
    return validate_capability_contract(contract)


def _validate_task_adapter_structure(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TASK_ADAPTER_KEYS:
        raise CapabilityAdapterError(
            f"task adapter fields must be exactly {sorted(_TASK_ADAPTER_KEYS)}"
        )
    adapter = deepcopy(dict(value))
    if adapter.get("schema_version") != 1:
        raise CapabilityAdapterError("task adapter schema_version must be 1")
    task_name = _text(adapter.get("task_name"), field="task_adapter.task_name")
    control_template_id = _text(
        adapter.get("control_template_id"),
        field="task_adapter.control_template_id",
    )
    _text(adapter.get("task_profile"), field="task_adapter.task_profile")
    _text(adapter.get("planner_kind"), field="task_adapter.planner_kind")
    max_rounds = adapter.get("max_rounds")
    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or max_rounds <= 0
    ):
        raise CapabilityAdapterError("task_adapter.max_rounds must be positive")

    raw_contracts = adapter.get("capability_contracts")
    if not isinstance(raw_contracts, list) or not raw_contracts:
        raise CapabilityAdapterError(
            "task_adapter.capability_contracts must be non-empty"
        )
    contracts = [validate_capability_contract(item) for item in raw_contracts]
    if any(contract["task_name"] != task_name for contract in contracts):
        raise CapabilityAdapterError(
            "task adapter cannot contain another task's capability contract"
        )
    template_ids = [contract["template_id"] for contract in contracts]
    if template_ids != sorted(set(template_ids)):
        raise CapabilityAdapterError(
            "task adapter capability contracts must be unique and sorted"
        )
    if control_template_id not in template_ids:
        raise CapabilityAdapterError(
            "task adapter control_template_id must name a registered capability"
        )
    if max_rounds > len(template_ids):
        raise CapabilityAdapterError(
            "task_adapter.max_rounds exceeds its capability count"
        )

    raw_questions = adapter.get("vqa_questions")
    if not isinstance(raw_questions, Mapping):
        raise CapabilityAdapterError("task_adapter.vqa_questions must be an object")
    questions: dict[str, dict[str, Any]] = {}
    for raw_id, raw_spec in raw_questions.items():
        phenomenon_id = _text(
            raw_id, field="task_adapter.vqa_questions.phenomenon_id"
        )
        if not isinstance(raw_spec, Mapping) or set(raw_spec) != _VQA_QUESTION_KEYS:
            raise CapabilityAdapterError(
                f"VQA question {phenomenon_id!r} fields must be exactly "
                f"{sorted(_VQA_QUESTION_KEYS)}"
            )
        spec = deepcopy(dict(raw_spec))
        for field in sorted(_VQA_QUESTION_KEYS):
            _text(
                spec.get(field),
                field=f"task_adapter.vqa_questions.{phenomenon_id}.{field}",
            )
        questions[phenomenon_id] = spec

    raw_metric_rules = adapter.get("vqa_metric_rules")
    if not isinstance(raw_metric_rules, Mapping):
        raise CapabilityAdapterError(
            "task_adapter.vqa_metric_rules must be an object"
        )
    metric_rules: dict[str, list[str]] = {}
    for raw_metric, raw_ids in raw_metric_rules.items():
        metric = _text(raw_metric, field="task_adapter.vqa_metric_rules.metric")
        if (
            not isinstance(raw_ids, list)
            or not raw_ids
            or any(not isinstance(item, str) or not item for item in raw_ids)
            or len(raw_ids) != len(set(raw_ids))
        ):
            raise CapabilityAdapterError(
                f"task adapter VQA metric rule {metric!r} must be a "
                "non-empty unique string list"
            )
        unknown = sorted(set(raw_ids) - set(questions))
        if unknown:
            raise CapabilityAdapterError(
                f"task adapter VQA metric rule {metric!r} lacks question specs: "
                f"{unknown}"
            )
        metric_rules[metric] = list(raw_ids)

    adapter.update(
        {
            "task_name": task_name,
            "control_template_id": control_template_id,
            "capability_contracts": contracts,
            "vqa_questions": questions,
            "vqa_metric_rules": metric_rules,
        }
    )
    return adapter


def validate_task_adapter(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one complete task-level adapter against the trusted registry."""

    adapter = _validate_task_adapter_structure(value)
    task_name = adapter["task_name"]
    if task_name not in _TASK_ADAPTER_METADATA:
        raise CapabilityAdapterError(f"unknown task adapter: {task_name!r}")
    expected = _raw_task_adapter(task_name)
    if adapter != expected:
        raise CapabilityAdapterError(f"task adapter changed for {task_name!r}")
    return deepcopy(adapter)


def resolve_task_adapter(task_name: Any) -> dict[str, Any]:
    """Resolve the legacy complete compatibility view for one task.

    New production callers should normally use
    :func:`resolve_task_retrieval_index` and treat its entries as optional
    retrieval hints.  This complete view remains public because legacy paper
    protocols and audited VQA selection still have real callers.
    """

    normalized = _text(task_name, field="task_name")
    if normalized not in _TASK_ADAPTER_METADATA:
        raise CapabilityAdapterError(f"unknown task adapter: {normalized!r}")
    return validate_task_adapter(_raw_task_adapter(normalized))
