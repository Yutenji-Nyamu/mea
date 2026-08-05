"""Retrieval-only query API for reviewed Task, Tool, and VQA artifacts.

Runtime execution authority belongs to RuntimeTaskBinding. Records returned by
this module are optional immutable-by-copy hints and never constrain which
concerns the Plan Agent may propose.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from .artifact_retrieval_records import (
    ArtifactRetrievalIndexError,
    CapabilityAdapterError,
    OFFICIAL_CONTROL_TEMPLATE_ID,
    _CONTRACTS,
    _TASK_ADAPTER_METADATA,
    _text,
)
from .artifact_retrieval_schema import (
    _validate_change_roots,
    resolve_capability_contract,
    resolve_task_adapter,
    validate_capability_contract,
    validate_task_adapter,
)


def _known_artifact_adapter(task_name: str) -> dict[str, Any] | None:
    """Return a reviewed record view, or ``None`` without denying execution."""

    if task_name not in _TASK_ADAPTER_METADATA:
        return None
    return resolve_task_adapter(task_name)


def resolve_task_retrieval_index(
    task_name: Any,
    *,
    allow_unregistered: bool = False,
) -> dict[str, Any]:
    """Return optional reviewed Task/Tool/VQA hints for one task.

    Runtime authority comes from ``RuntimeTaskBinding``. An unregistered task
    may therefore have an empty retrieval index without becoming unsupported.
    """

    if not isinstance(allow_unregistered, bool):
        raise ArtifactRetrievalIndexError("allow_unregistered must be bool")
    normalized = _text(task_name, field="task_name")
    adapter = _known_artifact_adapter(normalized)
    if adapter is None:
        if not allow_unregistered:
            raise ArtifactRetrievalIndexError(
                f"no reviewed artifact index for task: {normalized!r}"
            )
        return {
            "schema_version": 1,
            "index_role": "retrieval_only",
            "execution_authority": False,
            "task_name": normalized,
            "control_template_id": OFFICIAL_CONTROL_TEMPLATE_ID,
            "entries": [],
            "vqa_questions": {},
            "vqa_metric_rules": {},
        }
    return {
        "schema_version": 1,
        "index_role": "retrieval_only",
        "execution_authority": False,
        "task_name": adapter["task_name"],
        "control_template_id": adapter["control_template_id"],
        "entries": deepcopy(adapter["capability_contracts"]),
        "vqa_questions": deepcopy(adapter["vqa_questions"]),
        "vqa_metric_rules": deepcopy(adapter["vqa_metric_rules"]),
    }


def registered_task_adapters() -> list[dict[str, Any]]:
    """Return every task adapter in deterministic registry order."""

    return [
        resolve_task_adapter(task_name)
        for task_name in _TASK_ADAPTER_METADATA
    ]


def registered_task_names() -> tuple[str, ...]:
    """Return the single public task membership list."""

    return tuple(adapter["task_name"] for adapter in registered_task_adapters())


def registered_retrieval_task_names() -> tuple[str, ...]:
    """Return reviewed task identities without granting runtime authority."""

    return registered_task_names()


def registered_task_vqa_questions() -> dict[str, dict[str, Any]]:
    """Return the union of task-owned audited VQA question definitions."""

    questions: dict[str, dict[str, Any]] = {}
    for adapter in registered_task_adapters():
        for phenomenon_id, spec in adapter["vqa_questions"].items():
            previous = questions.get(phenomenon_id)
            if previous is not None and previous != spec:
                raise CapabilityAdapterError(
                    f"conflicting task VQA question: {phenomenon_id!r}"
                )
            questions[phenomenon_id] = deepcopy(spec)
    return questions


def registered_vqa_questions() -> dict[str, dict[str, Any]]:
    """Return the exact union of reviewed retrieval-only VQA questions."""

    return registered_task_vqa_questions()


def task_vqa_metric_phenomena(task_name: Any, metric: Any) -> list[str]:
    """Resolve task-scoped VQA phenomena for a trusted metric."""

    adapter = resolve_task_adapter(task_name)
    normalized_metric = _text(metric, field="metric")
    return list(adapter["vqa_metric_rules"].get(normalized_metric, []))


def task_vqa_metric_questions(task_name: Any, metric: Any) -> list[str]:
    """Return task-scoped reviewed VQA question identifiers."""

    return task_vqa_metric_phenomena(task_name, metric)


def validate_contract_changes(
    contract: Mapping[str, Any], changes: Mapping[str, Any]
) -> dict[str, Any]:
    """Enforce the contract's object/scene roots on candidate TaskGen changes.

    Task-specific validators remain responsible for numeric ranges and exact
    nested fields.  This function prevents a capability from crossing the
    top-level object/scene authority boundary before those validators run.
    """

    trusted = validate_capability_contract(contract)
    return _validate_change_roots(
        change_scope=trusted["taskgen"]["change_scope"],
        allowed_roots=trusted["taskgen"]["allowed_change_roots"],
        changes=changes,
    )


def build_contract_tool_request(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize the trusted Tool request named by a capability contract.

    The registry remains declarative; executable factories are imported only
    when the runtime explicitly asks to materialize a request.
    """

    trusted = validate_capability_contract(contract)
    from .toolgen import (
        bell_active_tcp_min_xy_error_tool_request,
        contact_tool_request,
        bbh_distractor_success_tool_request,
        click_bell_distractor_success_tool_request,
        hammer_left_camera_contact_count_tool_request,
        official_success_tool_request,
        pickup_to_contact_tool_request,
        time_to_success_tool_request,
        validate_tool_request,
    )

    factory_id = trusted["tool"]["request_factory_id"]
    task_name = trusted["task_name"]
    if factory_id == "contact_tool_request":
        request = contact_tool_request()
    elif factory_id == "bbh_distractor_success_tool_request":
        request = bbh_distractor_success_tool_request()
    elif factory_id == "click_bell_distractor_success_tool_request":
        request = click_bell_distractor_success_tool_request()
    elif factory_id == "pickup_to_contact_tool_request":
        request = pickup_to_contact_tool_request()
    elif factory_id == "bell_active_tcp_min_xy_error_tool_request":
        request = bell_active_tcp_min_xy_error_tool_request()
    elif factory_id == "hammer_left_camera_contact_count_tool_request":
        request = hammer_left_camera_contact_count_tool_request()
    elif factory_id == "official_success_tool_request":
        request = official_success_tool_request(task_name)
    elif factory_id == "time_to_success_tool_request":
        request = time_to_success_tool_request(task_name)
    else:  # pragma: no cover - exact registry validation makes this defensive.
        raise CapabilityAdapterError(
            f"unknown Tool request factory: {factory_id!r}"
        )
    try:
        return validate_tool_request(
            request,
            expected_metric=trusted["tool"]["metric"],
        )
    except RuntimeError as exc:
        raise CapabilityAdapterError(
            f"Tool request does not match capability contract: {exc}"
        ) from exc


def registered_capability_contracts(
    task_name: str | None = None,
) -> list[dict[str, Any]]:
    """Return all contracts in deterministic task/template order."""

    normalized_task = None if task_name is None else _text(task_name, field="task_name")
    return [
        validate_capability_contract(contract)
        for (registered_task, _template), contract in sorted(_CONTRACTS.items())
        if normalized_task is None or registered_task == normalized_task
    ]


def resolve_artifact_contract(
    task_name: Any,
    template_id: Any,
) -> dict[str, Any]:
    """Retrieve one exact reviewed artifact contract."""

    return resolve_capability_contract(task_name, template_id)


__all__ = [
    "ArtifactRetrievalIndexError",
    "OFFICIAL_CONTROL_TEMPLATE_ID",
    "build_contract_tool_request",
    "registered_capability_contracts",
    "registered_retrieval_task_names",
    "registered_task_adapters",
    "registered_task_names",
    "registered_task_vqa_questions",
    "registered_vqa_questions",
    "resolve_artifact_contract",
    "resolve_capability_contract",
    "resolve_task_adapter",
    "resolve_task_retrieval_index",
    "task_vqa_metric_phenomena",
    "task_vqa_metric_questions",
    "validate_capability_contract",
    "validate_contract_changes",
    "validate_task_adapter",
]
