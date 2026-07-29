"""Frozen simulator and policy binding for one open-world evaluation.

The binding answers only *where* an experiment can run.  It deliberately has
no planner kind, task profile, aspect inventory, template itinerary, or round
budget.  Those are either retrieval hints or experiment-protocol state and
must not authorize a Query-derived candidate.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping


class PolicyTaskBindingError(ValueError):
    """Raised when a policy/task execution boundary is incomplete or changed."""


_TASK_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_DOTTED_IDENTIFIER = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_BINDING_KEYS = {
    "schema_version",
    "simulator",
    "task_name",
    "task_module",
    "task_schema",
    "policy",
    "checkpoint",
    "hooks",
}
_TASK_SCHEMA_KEYS = {"path", "task_family"}
_HOOK_KEYS = {"official_success", "render", "rollout"}
_OFFICIAL_SUCCESS_KEYS = {"kind", "module", "class_name", "method"}
_RENDER_KEYS = {"kind", "module"}
_ROLLOUT_KEYS = {"kind", "entrypoint", "task_name"}


def _text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PolicyTaskBindingError(f"{field} must be a non-empty string")
    return value.strip()


def _mapping(
    value: Any,
    *,
    field: str,
    keys: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PolicyTaskBindingError(f"{field} must be an object")
    result = deepcopy(dict(value))
    if keys is not None and set(result) != keys:
        raise PolicyTaskBindingError(
            f"{field} fields must be exactly {sorted(keys)}"
        )
    return result


def build_policy_task_binding(
    *,
    task_name: str,
    task_family: str,
    policy: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    task_module: str | None = None,
) -> dict[str, Any]:
    """Build the minimal immutable execution boundary for a RoboTwin task."""

    name = _text(task_name, field="task_name")
    if _TASK_NAME.fullmatch(name) is None:
        raise PolicyTaskBindingError("task_name is not a RoboTwin identifier")
    module = (
        _text(task_module, field="task_module")
        if task_module is not None
        else f"envs.{name}"
    )
    if _DOTTED_IDENTIFIER.fullmatch(module) is None:
        raise PolicyTaskBindingError("task_module must be a dotted identifier")
    family = _text(task_family, field="task_family")
    return validate_policy_task_binding(
        {
            "schema_version": 1,
            "simulator": "robotwin",
            "task_name": name,
            "task_module": module,
            "task_schema": {
                "path": f"mea/toolkit/schemas/{name}.json",
                "task_family": family,
            },
            "policy": deepcopy(dict(policy)),
            "checkpoint": deepcopy(dict(checkpoint)),
            "hooks": {
                "official_success": {
                    "kind": "task_method",
                    "module": module,
                    "class_name": name,
                    "method": "check_success",
                },
                "render": {
                    "kind": "robotwin_initial_frame",
                    "module": module,
                },
                "rollout": {
                    "kind": "act_eval_mea",
                    "entrypoint": "policy/ACT/eval_mea.sh",
                    "task_name": name,
                },
            },
        }
    )


def validate_policy_task_binding(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one JSON-serializable policy/task execution boundary."""

    binding = _mapping(
        value,
        field="PolicyTaskBinding",
        keys=_BINDING_KEYS,
    )
    if binding.get("schema_version") != 1:
        raise PolicyTaskBindingError(
            "PolicyTaskBinding.schema_version must be 1"
        )
    if binding.get("simulator") != "robotwin":
        raise PolicyTaskBindingError(
            "PolicyTaskBinding.simulator must be robotwin"
        )
    task_name = _text(binding.get("task_name"), field="binding.task_name")
    if _TASK_NAME.fullmatch(task_name) is None:
        raise PolicyTaskBindingError(
            "binding.task_name is not a RoboTwin identifier"
        )
    task_module = _text(
        binding.get("task_module"), field="binding.task_module"
    )
    if _DOTTED_IDENTIFIER.fullmatch(task_module) is None:
        raise PolicyTaskBindingError(
            "binding.task_module must be a dotted identifier"
        )

    task_schema = _mapping(
        binding.get("task_schema"),
        field="binding.task_schema",
        keys=_TASK_SCHEMA_KEYS,
    )
    expected_schema_path = f"mea/toolkit/schemas/{task_name}.json"
    if task_schema.get("path") != expected_schema_path:
        raise PolicyTaskBindingError(
            "binding.task_schema.path differs from the bound task"
        )
    task_schema["task_family"] = _text(
        task_schema.get("task_family"),
        field="binding.task_schema.task_family",
    )

    policy = _mapping(binding.get("policy"), field="binding.policy")
    checkpoint = _mapping(
        binding.get("checkpoint"), field="binding.checkpoint"
    )
    if checkpoint.get("ready") is not True:
        raise PolicyTaskBindingError("binding.checkpoint must be ready")
    checkpoint_id = _text(
        checkpoint.get("checkpoint_id"),
        field="binding.checkpoint.checkpoint_id",
    )
    if not checkpoint_id.startswith(f"act-{task_name}/"):
        raise PolicyTaskBindingError(
            "binding checkpoint_id differs from the bound task"
        )

    hooks = _mapping(
        binding.get("hooks"), field="binding.hooks", keys=_HOOK_KEYS
    )
    official_success = _mapping(
        hooks.get("official_success"),
        field="binding.hooks.official_success",
        keys=_OFFICIAL_SUCCESS_KEYS,
    )
    if official_success != {
        "kind": "task_method",
        "module": task_module,
        "class_name": task_name,
        "method": "check_success",
    }:
        raise PolicyTaskBindingError(
            "binding official-success hook differs from the task module"
        )
    render = _mapping(
        hooks.get("render"),
        field="binding.hooks.render",
        keys=_RENDER_KEYS,
    )
    if render != {
        "kind": "robotwin_initial_frame",
        "module": task_module,
    }:
        raise PolicyTaskBindingError(
            "binding render hook differs from the task module"
        )
    rollout = _mapping(
        hooks.get("rollout"),
        field="binding.hooks.rollout",
        keys=_ROLLOUT_KEYS,
    )
    if rollout != {
        "kind": "act_eval_mea",
        "entrypoint": "policy/ACT/eval_mea.sh",
        "task_name": task_name,
    }:
        raise PolicyTaskBindingError(
            "binding rollout hook differs from the ACT runtime"
        )

    binding.update(
        {
            "task_name": task_name,
            "task_module": task_module,
            "task_schema": task_schema,
            "policy": policy,
            "checkpoint": checkpoint,
            "hooks": {
                "official_success": official_success,
                "render": render,
                "rollout": rollout,
            },
        }
    )
    return binding


def policy_task_binding_from_target(
    target: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the canonical binding from a schema-v3 evaluation target."""

    if not isinstance(target, Mapping):
        raise PolicyTaskBindingError(
            "OpenWorldEvaluationTarget must be an object"
        )
    return validate_policy_task_binding(target.get("policy_task_binding"))


__all__ = [
    "PolicyTaskBindingError",
    "build_policy_task_binding",
    "policy_task_binding_from_target",
    "validate_policy_task_binding",
]
