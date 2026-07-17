"""Deterministic two-round planner for bounded click_bell position variants."""

from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from mea.toolgen import official_success_tool_request
from mea.toolkit import load_task_schema

from .prototype import PlanAgentError, make_evaluation_id


CLICK_BELL_TEMPLATE_IDS = (
    "object_position.left_fixed",
    "object_position.right_fixed",
)
CLICK_BELL_POSITIONS = {
    "object_position.left_fixed": [-0.20, -0.08],
    "object_position.right_fixed": [0.20, -0.08],
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _git_head(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else None


class ClickBellPositionPlanAgent:
    """Run the same ACT seeds at safe fixed bell positions on both sides."""

    def __init__(
        self,
        repo_root: str | Path,
        *,
        start_seed: int = 100401,
        num_episodes: int = 1,
        telemetry_profile: str = "balanced_v1",
        max_rounds: int = 2,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.start_seed = int(start_seed)
        self.num_episodes = int(num_episodes)
        self.telemetry_profile = telemetry_profile
        self.max_rounds = int(max_rounds)
        if self.num_episodes <= 0:
            raise PlanAgentError("num_episodes must be positive")
        if self.max_rounds not in {1, 2}:
            raise PlanAgentError("click_bell max_rounds must be 1 or 2")
        load_task_schema(self.repo_root, "click_bell")

    def _round(self, template_id: str, round_number: int, request: str) -> dict[str, Any]:
        xy = CLICK_BELL_POSITIONS[template_id]
        side = "left" if xy[0] < 0 else "right"
        seeds = [self.start_seed + index for index in range(self.num_episodes)]
        return {
            "round_id": f"round_{round_number}",
            "template_id": template_id,
            "sub_aspect": template_id,
            "rationale": (
                f"Hold the bell at a safe fixed {side}-workspace position and "
                "evaluate ACT while preserving official task semantics."
            ),
            "task_instruction": (
                f"{request} This bounded round places the bell at fixed "
                f"workspace xy={xy}."
            ),
            "task_name": "click_bell",
            "task_module": "mea.tasks.click_bell",
            "telemetry_profile": self.telemetry_profile,
            "route": "reuse",
            "variant_hint": {
                "bell": {"position_mode": "fixed", "xy": list(xy)}
            },
            "execution": {
                "backend": "act",
                "seeds": seeds,
                "num_episodes": len(seeds),
                "gates": [
                    "variant_spec",
                    "render",
                    "rule",
                    "scene_position",
                    "vision",
                    "expert",
                    "act",
                    "toolkit",
                    "aggregate",
                ],
            },
            "observations": [
                "scene_alignment",
                "bell_position",
                "expert_solvable",
                "policy_success",
                "trusted_tools",
                "execution_vqa",
            ],
            "tool_request": official_success_tool_request("click_bell"),
        }

    def plan(
        self,
        user_request: str,
        *,
        evaluation_id: str | None = None,
        history_context: list[dict[str, Any]] | None = None,
        history_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = str(user_request).strip()
        if not request:
            raise PlanAgentError("user_request must be non-empty")
        resolved_id = evaluation_id or make_evaluation_id()
        if not re.fullmatch(r"eval_[A-Za-z0-9_]+", resolved_id):
            raise PlanAgentError("evaluation_id must begin with 'eval_'")
        evaluation_dir = self.repo_root / "mea/evaluation_runs" / resolved_id
        if evaluation_dir.exists():
            raise PlanAgentError(f"evaluation directory already exists: {evaluation_dir}")
        for child in ("plan", "execution", "summary"):
            (evaluation_dir / child).mkdir(parents=True, exist_ok=False)

        selected_templates = list(CLICK_BELL_TEMPLATE_IDS[: self.max_rounds])
        first_round = self._round(selected_templates[0], 1, request)
        history = {
            "schema_version": 1,
            "status": "passed" if history_context else "empty",
            "match_count": len(history_context or []),
            "matches": [
                {
                    key: item.get(key)
                    for key in ("evaluation_id", "user_request", "task_name", "similarity")
                }
                for item in (history_context or [])
                if isinstance(item, dict)
            ],
            **deepcopy(history_metadata or {}),
        }
        plan = {
            "schema_version": 5,
            "task_name": "click_bell",
            "policy": {
                "name": "ACT",
                "checkpoint_setting": "demo_clean",
                "expert_data_num": 50,
                "language_conditioned": False,
            },
            "evaluation_goal": "bounded_click_bell_left_right_position_generalization",
            "requested_template_ids": selected_templates,
            "rounds": [first_round],
            "round_decisions": [],
            "max_rounds": self.max_rounds,
            "planning_state": "awaiting_round_1_observation",
        }
        manifest = {
            "schema_version": 5,
            "evaluation_id": resolved_id,
            "status": "planned_round_1",
            "created_at": datetime.now().astimezone().isoformat(),
            "user_request": request,
            "base_commit": _git_head(self.repo_root),
            "planner": {
                "kind": "deterministic_click_bell_position_lr",
                "model_requested": None,
                "provider_called": False,
            },
            "plan_path": "plan/evaluation_plan.json",
            "history_retrieval_path": "plan/history_retrieval.json",
            "history_retrieval": history,
            "plan": plan,
        }
        _write_json(evaluation_dir / "request.json", {"user_request": request})
        _write_json(evaluation_dir / "plan/history_retrieval.json", history)
        _write_json(evaluation_dir / "plan/evaluation_plan.json", plan)
        _write_json(evaluation_dir / "manifest.json", manifest)
        return manifest

    def decide_next_round(
        self,
        *,
        evaluation_id: str,
        user_request: str,
        current_plan: dict[str, Any],
        observation_history: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        completed = len(observation_history)
        rounds = list(current_plan.get("rounds") or [])
        if completed != len(rounds) or completed < 1:
            raise PlanAgentError("click_bell planner expects one observation per round")
        latest = observation_history[-1]
        can_continue = bool(latest.get("pipeline_passed")) and completed < self.max_rounds
        remaining_templates = list(CLICK_BELL_TEMPLATE_IDS[completed : self.max_rounds])
        updated = deepcopy(current_plan)
        if can_continue:
            next_template = remaining_templates[0]
            next_round = self._round(next_template, completed + 1, user_request)
            updated["rounds"].append(next_round)
            updated["planning_state"] = f"awaiting_round_{completed + 1}_observation"
            action = "continue"
            reason = "The first position pipeline passed; evaluate the mirrored position."
        else:
            next_template = None
            next_round = None
            updated["planning_state"] = f"stopped_after_round_{completed}"
            action = "stop"
            reason = (
                "The bounded left/right position plan is complete."
                if latest.get("pipeline_passed")
                else "The latest pipeline failed, so the bounded plan stops early."
            )
        assessment = {
            "schema_version": 1,
            "state": "sufficient" if latest.get("pipeline_passed") else "pipeline_failure",
            "required_action": action,
            "pipeline_passed": bool(latest.get("pipeline_passed")),
            "reason": reason,
        }
        decision = {
            "schema_version": 2,
            "action": action,
            "observation_summary": (
                "The bounded click_bell position round produced auditable pipeline evidence."
            ),
            "decision_reason": reason,
            "next_template_id": next_template,
            "remaining_template_ids_before_decision": remaining_templates,
            "round_budget_before_decision": self.max_rounds - completed,
            "evidence_assessment": assessment,
            "next_round": next_round,
        }
        updated.setdefault("round_decisions", []).append(decision)
        evaluation_dir = self.repo_root / "mea/evaluation_runs" / evaluation_id
        _write_json(
            evaluation_dir / f"plan/evidence_after_round_{completed}.json", assessment
        )
        _write_json(
            evaluation_dir / f"plan/decision_after_round_{completed}.json", decision
        )
        _write_json(evaluation_dir / "plan/evaluation_plan.json", updated)
        manifest_path = evaluation_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"status": updated["planning_state"], "plan": updated})
        _write_json(manifest_path, manifest)
        return updated, decision
