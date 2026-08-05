"""Backend task binding and unchanged official-candidate construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from mea.method_runtime import (
    BackendBindingRequest,
    BackendTaskBinding,
    MaterializedCandidate,
)
from mea.robotwin_task_context import (
    RoboTwinTaskContextError,
    resolve_robotwin_task_context,
)
from mea.taskgen.generic_backend import GenericRoboTwinTaskAdapter
from mea.visual_capture import EVENT_KEYFRAMES_PROFILE

from .runtime_contracts import _RoboTwinNativeCandidate, _required_text
from .task_identity import RoboTwinTaskIdentity


def bind_task(
    backend: Any,
    request: BackendBindingRequest,
) -> BackendTaskBinding:
    task_name = _required_text(
        request.task_reference.get("task_name"),
        "task_reference.task_name",
    )
    adapter = backend.task_adapter_factory(task_name)
    if (
        not isinstance(
            adapter,
            (GenericRoboTwinTaskAdapter, RoboTwinTaskIdentity),
        )
        or adapter.task_name != task_name
    ):
        raise TypeError(
            "task_adapter_factory must return a matching "
            "RoboTwin task identity"
        )
    policy = request.task_reference.get("policy", {})
    if not isinstance(policy, Mapping):
        raise TypeError("task_reference.policy must be an object")
    policy_contract = deepcopy(dict(policy))
    policy_name = str(policy_contract.get("name") or "bound_policy").strip()
    binding_id = str(
        request.task_reference.get("binding_id")
        or f"{task_name}/{policy_name}"
    ).strip()
    try:
        task_context = resolve_robotwin_task_context(
            backend.repo_root,
            task_name,
        )
    except RoboTwinTaskContextError as exc:
        raise ValueError(
            f"cannot bind RoboTwin TaskContext: {exc}"
        ) from exc
    # A Generic adapter loaded by the production discovery path already
    # carries this exact context.  Keep hand-constructed test/compat
    # adapters usable without treating their injected schema as source
    # authority.
    adapter_context = (
        deepcopy(dict(adapter.task_context))
        if isinstance(adapter, GenericRoboTwinTaskAdapter)
        and isinstance(adapter.task_context, Mapping)
        else task_context.to_dict()
    )
    return BackendTaskBinding(
        benchmark=backend.benchmark,
        binding_id=binding_id,
        task_contract={
            "schema_version": 1,
            "task_name": task_name,
            "official_source": adapter.official_source,
            "official_class": adapter.official_class,
            "task_schema": (
                deepcopy(dict(adapter.task_schema))
                if adapter.task_schema is not None
                else None
            ),
            "task_schema_available": adapter.task_schema is not None,
            "task_context": adapter_context,
            "policy": policy_contract,
            "visual_capture_profile_id": EVENT_KEYFRAMES_PROFILE,
        },
        native_task=adapter,
        artifacts={
            "official_source": adapter.official_source,
            **request.artifacts,
        },
        metadata={
            "task_name": task_name,
            "policy": policy_contract,
            **request.metadata,
        },
    )

def official_candidate(
    backend: Any,
    binding: BackendTaskBinding,
    *,
    source_query: str,
    seed: int,
    candidate_id: str = "official_control",
) -> MaterializedCandidate:
    """Bind an unchanged task after establishing execution context.

    A shared policy may execute an official task that has no reviewed
    ``TaskSchema``.  The unchanged control is the first real method round,
    so establish its actor/telemetry authority before policy inference
    rather than waiting for a later TaskGen or Tool-only Proposal.
    """

    adapter = binding.native_task
    if not isinstance(
        adapter,
        (GenericRoboTwinTaskAdapter, RoboTwinTaskIdentity),
    ):
        raise TypeError(
            "RoboTwin binding native_task has the wrong runtime type"
        )
    task_contract = deepcopy(dict(binding.task_contract))
    task_context_value = task_contract.get("task_context")
    runtime_probe_executed = False
    if not isinstance(task_contract.get("task_schema"), Mapping):
        policy = task_contract.get("policy")
        action_dimension = (
            policy.get("action_dimension", 0)
            if isinstance(policy, Mapping)
            else 0
        )
        if (
            isinstance(action_dimension, bool)
            or not isinstance(action_dimension, int)
            or action_dimension < 1
        ):
            raise ValueError(
                "schema-less official control requires the bound policy "
                "action_dimension"
            )
        try:
            runtime_probe = backend.task_context_probe_runner(
                repo_root=backend.repo_root,
                task_name=adapter.task_name,
                seed=int(seed),
                action_dimension=action_dimension,
            )
            task_context = resolve_robotwin_task_context(
                backend.repo_root,
                adapter.task_name,
                runtime_probe=runtime_probe,
            )
        except (RoboTwinTaskContextError, TypeError, ValueError) as exc:
            raise ValueError(
                "official control could not establish runtime "
                f"TaskContext authority: {exc}"
            ) from exc
        if task_context.task_schema is None:
            raise ValueError(
                "official control runtime TaskContext has no telemetry "
                "schema"
            )
        task_context_value = task_context.to_dict()
        task_contract.update(
            {
                "task_schema": deepcopy(dict(task_context.task_schema)),
                "task_schema_available": True,
                "task_context": task_context_value,
            }
        )
        runtime_probe_executed = True
    manifest = {
        "schema_version": 1,
        "status": "official",
        "task_name": adapter.task_name,
        "task_module": f"envs.{adapter.task_name}",
        "generation_kind": "official_passthrough",
    }
    native = _RoboTwinNativeCandidate(
        adapter=adapter,
        experiment_candidate={},
        taskgen_resolution={
            "schema_version": 1,
            "status": "bypassed",
            "route": "official_control",
        },
        rollout_manifest=manifest,
    )
    return MaterializedCandidate(
        benchmark=binding.benchmark,
        candidate_id=candidate_id,
        binding_id=binding.binding_id,
        source_query=source_query,
        task_contract={
            **task_contract,
            "task_module": manifest["task_module"],
        },
        native_task=native,
        artifacts={
            **binding.artifacts,
            "task_module": manifest["task_module"],
        },
        validation={
            "route": "official_control",
            "task_context": (
                {
                    "schema_origin": task_context_value.get(
                        "schema_origin"
                    ),
                    "runtime_probe_executed": runtime_probe_executed,
                }
                if isinstance(task_context_value, Mapping)
                else None
            ),
        },
        metadata={
            "official_control": True,
            "task_context_bound_before_rollout": isinstance(
                task_contract.get("task_schema"), Mapping
            ),
            "runtime_task_context_probe_executed": (
                runtime_probe_executed
            ),
        },
    )
