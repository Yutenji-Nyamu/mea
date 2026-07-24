"""Paper-aligned production facade for catalog-bound evaluation planning.

Production callers should depend on :class:`CatalogPlanAgent` instead of
selecting one of the historical task-specific planners themselves.  The
facade keeps trusted template materialization runtime-owned while accepting
either a direct ClaimFirst control proposal or the validated wrapper emitted
by ``route_to_planner_proposal``.

The historical click-bell position and fixed-suite modes remain callable only
through explicit paper-ablation settings.  The production entry point is this
module.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from .click_bell import ClickBellAdaptivePlanAgent
from .official import OFFICIAL_TEMPLATE_ID, OfficialTaskPlanAgent
from .prototype import PlanAgentError, PlanAgentPrototype


CATALOG_PLAN_TASKS = (
    "beat_block_hammer",
    "click_bell",
    "adjust_bottle",
    "grab_roller",
)

# These modes are useful scientific controls, but they are not alternative
# production routers.  Keeping the names here makes the boundary discoverable
# without deleting the fixed/adaptive ablations required by the paper.
EXPERIMENT_ONLY_PLANNER_MODES = (
    "click_bell_position_lr",
    "click_bell_adaptive_catalog",
    "click_bell_fixed_suite",
)


class CatalogPlanError(PlanAgentError):
    """Raised when a request leaves the production task/template catalog."""


def _task_name(value: Any) -> str:
    normalized = str(value or "").strip()
    if normalized not in CATALOG_PLAN_TASKS:
        raise CatalogPlanError(
            f"unsupported catalog task {normalized!r}; "
            f"expected one of {list(CATALOG_PLAN_TASKS)}"
        )
    return normalized


def _validated_inner_proposal(
    value: Mapping[str, Any] | None,
    *,
    task_name: str,
) -> dict[str, Any] | None:
    """Normalize a global-route wrapper or direct ClaimFirst proposal.

    Schema-specific validation deliberately remains in the trusted delegate.
    This function only checks the cross-planner boundary and unwraps the
    public global-route envelope.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise CatalogPlanError("validated_proposal must be an object")
    outer = deepcopy(dict(value))
    if "proposal" in outer:
        if outer.get("task_name") != task_name:
            raise CatalogPlanError(
                "global-route proposal cannot switch the bound task"
            )
        inner = outer.get("proposal")
        if not isinstance(inner, Mapping):
            raise CatalogPlanError(
                "global-route proposal wrapper must contain a proposal object"
            )
        proposal = deepcopy(dict(inner))
    else:
        proposal = outer
    if proposal.get("task_name") != task_name:
        raise CatalogPlanError(
            "validated proposal cannot switch the bound task"
        )
    return proposal


class PlanMaterializer:
    """Materialize trusted catalog templates for one bound task.

    The task-specific delegates are an internal compatibility layer.  This
    class is the stable production surface while the legacy implementations
    are reduced or moved into paper experiments.
    """

    def __init__(
        self,
        repo_root: str | Path,
        *,
        task_name: str,
        provider: Any = None,
        model: str = "runtime-owned",
        task_module: str | None = None,
        start_seed: int | None = None,
        num_episodes: int = 1,
        telemetry_profile: str = "balanced_v1",
        max_rounds: int = 3,
        execution_backend: str = "act",
        task_profile: str = "adaptive_properties",
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.task_name = _task_name(task_name)
        self.provider = provider
        self.model = str(model)
        self.task_module = task_module
        self.start_seed = start_seed
        self.num_episodes = int(num_episodes)
        self.telemetry_profile = str(telemetry_profile)
        self.max_rounds = int(max_rounds)
        self.execution_backend = str(execution_backend).casefold()
        self.task_profile = str(task_profile).casefold()
        self._delegate = self._build_delegate()

    def _build_delegate(self) -> Any:
        if self.task_name == "beat_block_hammer":
            return PlanAgentPrototype(
                self.repo_root,
                self.provider,
                model=self.model,
                start_seed=self.start_seed,
                num_episodes=self.num_episodes,
            )
        if (
            self.task_name == "click_bell"
            and self.task_profile != "official"
        ):
            return ClickBellAdaptivePlanAgent(
                self.repo_root,
                self.provider,
                model=self.model,
                start_seed=(
                    self.start_seed if self.start_seed is not None else 100401
                ),
                num_episodes=self.num_episodes,
                telemetry_profile=self.telemetry_profile,
                max_rounds=self.max_rounds,
            )
        return OfficialTaskPlanAgent(
            self.repo_root,
            task_name=self.task_name,
            task_module=self.task_module,
            start_seed=(
                self.start_seed if self.start_seed is not None else 100000
            ),
            num_episodes=self.num_episodes,
            telemetry_profile=self.telemetry_profile,
            execution_backend=self.execution_backend,
        )

    def materialize_plan_step(
        self,
        template_id: str,
        round_number: int,
        user_request: str,
    ) -> dict[str, Any]:
        """Return one executable round from a trusted template identifier."""

        if (
            isinstance(round_number, bool)
            or not isinstance(round_number, int)
            or round_number <= 0
        ):
            raise CatalogPlanError("round_number must be a positive integer")
        request = str(user_request or "").strip()
        if not request:
            raise CatalogPlanError("user_request must be non-empty")
        materialize = getattr(self._delegate, "materialize_plan_step", None)
        if callable(materialize):
            result = deepcopy(
                materialize(str(template_id), round_number, request)
            )
            result.setdefault("task_name", self.task_name)
            return result
        if template_id != OFFICIAL_TEMPLATE_ID:
            raise CatalogPlanError(
                f"{self.task_name} supports only {OFFICIAL_TEMPLATE_ID!r}"
            )
        # OfficialTaskPlanAgent is a one-round deterministic compatibility
        # delegate.  Its private constructor is contained here so production
        # callers never depend on it.
        result = deepcopy(self._delegate._round(request))
        result["round_id"] = f"round_{round_number}"
        return result


class CatalogPlanAgent:
    """Single production planner interface for the four ACT catalog tasks."""

    planner_kind = "catalog_claim_first_v1"

    def __init__(self, repo_root: str | Path, **kwargs: Any):
        self.materializer = PlanMaterializer(repo_root, **kwargs)
        self.repo_root = self.materializer.repo_root
        self.task_name = self.materializer.task_name

    def plan(
        self,
        user_request: str,
        *,
        evaluation_id: str | None = None,
        history_context: list[dict[str, Any]] | None = None,
        history_metadata: dict[str, Any] | None = None,
        validated_proposal: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        proposal = _validated_inner_proposal(
            validated_proposal,
            task_name=self.task_name,
        )
        return self.materializer._delegate.plan(
            user_request,
            evaluation_id=evaluation_id,
            history_context=history_context,
            history_metadata=history_metadata,
            validated_proposal=proposal,
        )

    def materialize_plan_step(
        self,
        template_id: str,
        round_number: int,
        user_request: str,
    ) -> dict[str, Any]:
        return self.materializer.materialize_plan_step(
            template_id,
            round_number,
            user_request,
        )

    def decide_next_round(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.materializer._delegate.decide_next_round(**kwargs)


__all__ = [
    "CATALOG_PLAN_TASKS",
    "EXPERIMENT_ONLY_PLANNER_MODES",
    "CatalogPlanAgent",
    "CatalogPlanError",
    "PlanMaterializer",
]
