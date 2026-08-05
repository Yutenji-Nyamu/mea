"""Legacy catalog/capability transport for frozen paper protocols.

Reviewed records live in :mod:`mea.artifact_retrieval_index`; this module only
retains the historical TaskGen route projection and compatibility names.
Neither catalog membership nor these route strings authorize production
execution.
"""

from __future__ import annotations

from typing import Any, Mapping

from mea.artifact_retrieval_index import (
    ArtifactRetrievalIndexError,
    OFFICIAL_CONTROL_TEMPLATE_ID,
    build_contract_tool_request,
    registered_capability_contracts,
    registered_task_adapters,
    registered_task_names,
    registered_task_vqa_questions,
    resolve_capability_contract,
    resolve_task_adapter,
    resolve_task_retrieval_index,
    task_vqa_metric_phenomena,
    validate_capability_contract,
    validate_contract_changes,
    validate_task_adapter,
)


CapabilityAdapterError = ArtifactRetrievalIndexError


def taskgen_route(contract: Mapping[str, Any]) -> str:
    """Project a reviewed legacy operation onto the old TaskGen CLI route."""

    operation = validate_capability_contract(contract)["taskgen"]["operation"]
    return {
        "force_codegen": "force_codegen",
        "provider_scene_checker_codegen": "provider_scene_checker_codegen",
        "bounded_variant_overlay": "reuse",
        "reuse_variant": "reuse",
        "official_passthrough": "official",
    }[operation]


__all__ = [
    "CapabilityAdapterError",
    "OFFICIAL_CONTROL_TEMPLATE_ID",
    "build_contract_tool_request",
    "registered_capability_contracts",
    "registered_task_adapters",
    "registered_task_names",
    "registered_task_vqa_questions",
    "resolve_capability_contract",
    "resolve_task_adapter",
    "resolve_task_retrieval_index",
    "task_vqa_metric_phenomena",
    "taskgen_route",
    "validate_capability_contract",
    "validate_contract_changes",
    "validate_task_adapter",
]
