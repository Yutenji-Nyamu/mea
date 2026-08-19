"""TaskGen standalone entrypoint with an explicit compatibility boundary.

The production Agent materializes Query-derived Task artifacts through
``mea.taskgen.runtime`` and ``MethodRuntime``.  This script remains the stable
standalone command used by historical paper protocols.  A generic provider
Proposal is identified explicitly; every task-specific, Table-3 or LIBERO
invocation is dispatched to the cold compatibility module.

Compatibility functions imported by older tests and paper scripts are exposed
lazily.  They are not imported while the production Agent imports TaskGen.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from experiments.paper.compat_taskgen import load_legacy_cli


_GENERIC_MODE = "generic_provider_scene_checker_codegen"
_COMPAT_ONLY_FLAGS = frozenset(
    {
        "--benchmark",
        "--capability-contract-json",
        "--task-proposal-json",
        "--taskgen-ablation-json",
        "--variant-hint-json",
        "--variant-id",
    }
)
_PATCH_BRIDGE_NAMES = (
    "OpenAICompatibleProvider",
    "require_task_artifact_act_runtime_eligible",
    "run_command",
    "run_probe",
)
_HISTORICAL_FUNCTIONS = (
    "_checker_fixture_failure_diagnosis",
    "_expert_terminal_authority_failure",
    "_tracked_actor_heights",
    "collect_click_bell_position_samples",
    "collect_position_samples",
    "create_provider_scene_checker_taskgen_run",
    "evaluate_run_telemetry",
    "newest_eval_dir",
    "prepare_planner_capability_binding",
    "run_act",
    "run_command",
    "run_official_expert_episodes",
    "run_probe",
    "run_visual_self_reflection",
    "task_artifact_summary",
    "update_manifest",
    "validate_planner_capability_binding",
)


def _option_value(argv: Sequence[str], option: str) -> str | None:
    for index, token in enumerate(argv):
        if token == option and index + 1 < len(argv):
            return argv[index + 1]
        prefix = option + "="
        if token.startswith(prefix):
            return token[len(prefix) :]
    return None


def _has_option(argv: Sequence[str], option: str) -> bool:
    prefix = option + "="
    return any(token == option or token.startswith(prefix) for token in argv)


def _is_generic_standalone(argv: Sequence[str]) -> bool:
    """Return whether argv selects only the generic Proposal CLI contract."""

    if _option_value(argv, "--mode") != _GENERIC_MODE:
        return False
    if _option_value(argv, "--benchmark") not in {None, "robotwin"}:
        return False
    return not any(
        _has_option(argv, flag)
        for flag in _COMPAT_ONLY_FLAGS - {"--benchmark"}
    )


def _is_bridge(value: Any) -> bool:
    return bool(getattr(value, "__taskgen_compat_bridge__", False))


def _call_legacy(name: str, /, *args: Any, **kwargs: Any) -> Any:
    """Call one frozen function while preserving historical patch points."""

    legacy = load_legacy_cli()
    restored: dict[str, Any] = {}
    patch_names = dict.fromkeys((*_PATCH_BRIDGE_NAMES, *_HISTORICAL_FUNCTIONS))
    for patch_name in patch_names:
        candidate = globals().get(patch_name)
        if candidate is None or _is_bridge(candidate):
            continue
        restored[patch_name] = getattr(legacy, patch_name)
        setattr(legacy, patch_name, candidate)
    try:
        return getattr(legacy, name)(*args, **kwargs)
    finally:
        for patch_name, original in restored.items():
            setattr(legacy, patch_name, original)


def _bridge(name: str) -> Callable[..., Any]:
    def invoke(*args: Any, **kwargs: Any) -> Any:
        return _call_legacy(name, *args, **kwargs)

    invoke.__name__ = name
    invoke.__qualname__ = name
    invoke.__doc__ = f"Lazy compatibility bridge for ``{name}``."
    invoke.__taskgen_compat_bridge__ = True  # type: ignore[attr-defined]
    return invoke


for _function_name in _HISTORICAL_FUNCTIONS:
    globals()[_function_name] = _bridge(_function_name)


def run_generic_standalone() -> None:
    """Run the unchanged generic Proposal path.

    The implementation remains frozen with the standalone compatibility CLI
    until its simulator/probe callbacks are fully owned by MethodRuntime.  The
    production Agent does not call this function.
    """

    _call_legacy("main")


def run_compat_standalone() -> None:
    """Run a task-specific or paper-protocol standalone invocation."""

    _call_legacy("main")


def main() -> None:
    if _is_generic_standalone(sys.argv[1:]):
        run_generic_standalone()
        return
    run_compat_standalone()


def __getattr__(name: str) -> Any:
    """Preserve uncommon historical imports without eager compatibility load."""

    return getattr(load_legacy_cli(), name)


__all__ = [
    *_HISTORICAL_FUNCTIONS,
    "main",
    "run_compat_standalone",
    "run_generic_standalone",
]


if __name__ == "__main__":
    main()
