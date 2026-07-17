"""Legacy and model-driven planners for bounded click_bell variants."""

from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from mea.taskgen import TaskGenError, extract_json_response
from mea.toolgen import (
    bell_active_tcp_min_xy_error_tool_request,
    official_success_tool_request,
)
from mea.toolkit import load_task_schema

from .prototype import PlanAgentError, make_evaluation_id
from .evidence_policy import assess_conditional_transition


CLICK_BELL_TEMPLATE_IDS = (
    "object_position.left_fixed",
    "object_position.right_fixed",
)
CLICK_BELL_POSITIONS = {
    "object_position.left_fixed": [-0.20, -0.08],
    "object_position.right_fixed": [0.20, -0.08],
}

CLICK_BELL_ADAPTIVE_ASPECTS = {
    "object_position": {
        "description": (
            "Generalization across safe left/right workspace positions while "
            "holding the official randomly sampled bell instance constant by seed."
        ),
        "template_ids": [
            "object_position.left_fixed",
            "object_position.right_fixed",
        ],
    },
    "object_instance": {
        "description": (
            "Generalization across official bell base0/base1 instances while "
            "holding the official randomly sampled pose constant by seed."
        ),
        "template_ids": [
            "object_instance.base0",
            "object_instance.base1",
        ],
    },
    "robustness.scene_clutter": {
        "description": (
            "Robustness to RoboTwin's simulator-native tabletop clutter while "
            "preserving the official bell pose, instance sampling, task logic, "
            "and ACT checkpoint."
        ),
        "template_ids": ["robustness.scene_clutter.official_table"],
    },
}

CLICK_BELL_ADAPTIVE_TEMPLATES = {
    "object_position.left_fixed": {
        "aspect_id": "object_position",
        "probe_role": "sentinel",
        "description": "Safe fixed left-workspace position.",
        "variant_hint": {"bell": {"position_mode": "fixed", "xy": [-0.20, -0.08]}},
    },
    "object_position.right_fixed": {
        "aspect_id": "object_position",
        "probe_role": "counterfactual",
        "description": "Mirrored safe right-workspace position.",
        "variant_hint": {"bell": {"position_mode": "fixed", "xy": [0.20, -0.08]}},
    },
    "object_instance.base0": {
        "aspect_id": "object_instance",
        "probe_role": "sentinel",
        "description": "Official larger white/black base0 bell instance.",
        "variant_hint": {
            "bell": {
                "position_mode": "official_random",
                "instance_mode": "fixed",
                "bell_id": 0,
            }
        },
    },
    "object_instance.base1": {
        "aspect_id": "object_instance",
        "probe_role": "counterfactual",
        "description": "Official smaller blue/brown base1 bell instance.",
        "variant_hint": {
            "bell": {
                "position_mode": "official_random",
                "instance_mode": "fixed",
                "bell_id": 1,
            }
        },
    },
    "robustness.scene_clutter.official_table": {
        "aspect_id": "robustness.scene_clutter",
        "probe_role": "sentinel",
        "description": (
            "Official click_bell scene plus simulator-generated physical "
            "tabletop distractors."
        ),
        "variant_hint": {
            "domain_randomization": {
                "cluttered_table": True,
                "clean_background_rate": 0.0,
            }
        },
    },
}

CLICK_BELL_POLICY = {
    "name": "ACT",
    "checkpoint_setting": "demo_clean",
    "expert_data_num": 50,
    "language_conditioned": False,
}

_ADAPTIVE_PROPOSAL_KEYS = {
    "schema_version",
    "task_name",
    "evaluation_goal",
    "requested_aspect_ids",
    "first_aspect_id",
}
_ADAPTIVE_DECISION_KEYS = {
    "schema_version",
    "action",
    "transition",
    "observation_summary",
    "decision_reason",
    "next_aspect_id",
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

    def _round(
        self, template_id: str, round_number: int, request: str
    ) -> dict[str, Any]:
        xy = CLICK_BELL_POSITIONS[template_id]
        side = "left" if xy[0] < 0 else "right"
        seeds = [self.start_seed + index for index in range(self.num_episodes)]
        return {
            "round_id": f"round_{round_number}",
            "template_id": template_id,
            "capability_id": "object_position.fixed_xy",
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
            "variant_hint": {"bell": {"position_mode": "fixed", "xy": list(xy)}},
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
            "tool_request": bell_active_tcp_min_xy_error_tool_request(),
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
            raise PlanAgentError(
                f"evaluation directory already exists: {evaluation_dir}"
            )
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
                    for key in (
                        "evaluation_id",
                        "user_request",
                        "task_name",
                        "similarity",
                    )
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
        can_continue = (
            bool(latest.get("pipeline_passed")) and completed < self.max_rounds
        )
        remaining_templates = list(CLICK_BELL_TEMPLATE_IDS[completed : self.max_rounds])
        updated = deepcopy(current_plan)
        if can_continue:
            next_template = remaining_templates[0]
            next_round = self._round(next_template, completed + 1, user_request)
            updated["rounds"].append(next_round)
            updated["planning_state"] = f"awaiting_round_{completed + 1}_observation"
            action = "continue"
            reason = (
                "The first position pipeline passed; evaluate the mirrored position."
            )
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
            "state": "sufficient"
            if latest.get("pipeline_passed")
            else "pipeline_failure",
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


class ClickBellAdaptivePlanAgent:
    """Select bounded click_bell aspects and adapt from real round evidence."""

    def __init__(
        self,
        repo_root: str | Path,
        provider: Any,
        *,
        model: str,
        start_seed: int = 100401,
        num_episodes: int = 1,
        telemetry_profile: str = "balanced_v1",
        max_rounds: int = 3,
    ):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.provider = provider
        self.model = str(model)
        self.start_seed = int(start_seed)
        self.num_episodes = int(num_episodes)
        self.telemetry_profile = telemetry_profile
        self.max_rounds = int(max_rounds)
        if self.num_episodes <= 0:
            raise PlanAgentError("num_episodes must be positive")
        if self.max_rounds not in {1, 2, 3, 4, 5}:
            raise PlanAgentError(
                "adaptive click_bell max_rounds must be between 1 and 5"
            )
        load_task_schema(self.repo_root, "click_bell")

    @staticmethod
    def _require_text(value: Any, field: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise PlanAgentError(f"{field} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _require_exact_keys(
        value: dict[str, Any], expected: set[str], name: str
    ) -> None:
        actual = set(value)
        if actual != expected:
            raise PlanAgentError(
                f"{name} fields mismatch: missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )

    @staticmethod
    def _templates_for_aspects(aspect_ids: list[str]) -> list[str]:
        return [
            template_id
            for aspect_id in aspect_ids
            for template_id in CLICK_BELL_ADAPTIVE_ASPECTS[aspect_id]["template_ids"]
        ]

    def _materialize_round(
        self,
        template_id: str,
        round_number: int,
        user_request: str,
    ) -> dict[str, Any]:
        if template_id not in CLICK_BELL_ADAPTIVE_TEMPLATES:
            raise PlanAgentError(f"unknown click_bell template: {template_id}")
        template = CLICK_BELL_ADAPTIVE_TEMPLATES[template_id]
        seeds = [self.start_seed + index for index in range(self.num_episodes)]
        capability_id = {
            "object_position": "object_position.fixed_xy",
            "object_instance": "object_instance.official_id",
            "robustness.scene_clutter": "robustness.scene_clutter",
        }[template["aspect_id"]]
        return {
            "round_id": f"round_{round_number}",
            "template_id": template_id,
            "capability_id": capability_id,
            "sub_aspect": template["aspect_id"],
            "aspect_id": template["aspect_id"],
            "probe_role": template["probe_role"],
            "rationale": template["description"],
            "task_instruction": (
                f"{user_request} Trusted bounded variant: " f"{template['description']}"
            ),
            "task_name": "click_bell",
            "task_module": "mea.tasks.click_bell",
            "telemetry_profile": self.telemetry_profile,
            "route": "reuse",
            "variant_hint": deepcopy(template["variant_hint"]),
            "execution": {
                "backend": "act",
                "seeds": seeds,
                "num_episodes": len(seeds),
                "gates": [
                    "variant_spec",
                    "render",
                    "rule",
                    "scene_variant",
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
                "bell_instance_id",
                "scene_clutter",
                "expert_solvable",
                "policy_success",
                "trusted_tools",
                "execution_vqa",
            ],
            "tool_request": (
                bell_active_tcp_min_xy_error_tool_request()
                if template["aspect_id"] == "object_position"
                else official_success_tool_request("click_bell")
            ),
        }

    def _validate_proposal(
        self,
        value: Any,
        *,
        user_request: str,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise PlanAgentError("ClickBellEvaluationProposal must be an object")
        self._require_exact_keys(
            value, _ADAPTIVE_PROPOSAL_KEYS, "ClickBellEvaluationProposal"
        )
        if value.get("schema_version") != 1:
            raise PlanAgentError("proposal.schema_version must be 1")
        if value.get("task_name") != "click_bell":
            raise PlanAgentError("proposal.task_name must be click_bell")
        aspect_ids = value.get("requested_aspect_ids")
        if (
            not isinstance(aspect_ids, list)
            or not aspect_ids
            or any(not isinstance(item, str) for item in aspect_ids)
            or len(aspect_ids) != len(set(aspect_ids))
        ):
            raise PlanAgentError(
                "requested_aspect_ids must be a non-empty unique string list"
            )
        unknown = [
            item for item in aspect_ids if item not in CLICK_BELL_ADAPTIVE_ASPECTS
        ]
        if unknown:
            raise PlanAgentError(f"unknown click_bell aspects: {unknown}")
        first_aspect = value.get("first_aspect_id")
        if first_aspect not in aspect_ids:
            raise PlanAgentError("first_aspect_id must be requested")
        requested_templates = self._templates_for_aspects(aspect_ids)
        first_template = CLICK_BELL_ADAPTIVE_ASPECTS[first_aspect]["template_ids"][0]
        return {
            "schema_version": 6,
            "task_name": "click_bell",
            "policy": dict(CLICK_BELL_POLICY),
            "evaluation_goal": self._require_text(
                value.get("evaluation_goal"), "evaluation_goal"
            ),
            "requested_aspect_ids": list(aspect_ids),
            "requested_template_ids": requested_templates,
            "rounds": [self._materialize_round(first_template, 1, user_request)],
            "round_decisions": [],
            "max_rounds": self.max_rounds,
            "planning_state": "awaiting_round_1_observation",
        }

    @staticmethod
    def _validate_observations(
        current_plan: dict[str, Any], observation_history: Any
    ) -> list[dict[str, Any]]:
        if not isinstance(observation_history, list) or not observation_history:
            raise PlanAgentError("observation_history must be non-empty")
        rounds = current_plan.get("rounds") or []
        if len(observation_history) != len(rounds):
            raise PlanAgentError("each planned round needs exactly one observation")
        normalized: list[dict[str, Any]] = []
        for round_plan, observation in zip(rounds, observation_history):
            if not isinstance(observation, dict):
                raise PlanAgentError("each observation must be an object")
            if observation.get("round_id") != round_plan.get("round_id"):
                raise PlanAgentError("observation.round_id does not match plan")
            if not isinstance(observation.get("pipeline_passed"), bool):
                raise PlanAgentError("observation.pipeline_passed must be boolean")
            normalized.append(deepcopy(observation))
        return normalized

    def _assess_evidence(
        self,
        current_plan: dict[str, Any],
        observation_history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        history = self._validate_observations(current_plan, observation_history)
        return assess_conditional_transition(
            current_plan,
            history,
            aspect_catalog=CLICK_BELL_ADAPTIVE_ASPECTS,
        )

    def _validate_decision(
        self,
        value: Any,
        *,
        current_plan: dict[str, Any],
        observation_history: list[dict[str, Any]],
        user_request: str,
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise PlanAgentError("ClickBellNextRoundDecision must be an object")
        self._require_exact_keys(
            value, _ADAPTIVE_DECISION_KEYS, "ClickBellNextRoundDecision"
        )
        if value.get("schema_version") != 1:
            raise PlanAgentError("decision.schema_version must be 1")
        assessment = self._assess_evidence(current_plan, observation_history)
        action = value.get("action")
        if action != assessment["required_action"]:
            raise PlanAgentError(
                f"action {action!r} conflicts with required evidence action "
                f"{assessment['required_action']!r}"
            )
        summary = self._require_text(
            value.get("observation_summary"), "observation_summary"
        )
        reason = self._require_text(value.get("decision_reason"), "decision_reason")
        transition = value.get("transition")
        next_aspect = value.get("next_aspect_id")
        if transition != assessment["required_transition"]:
            raise PlanAgentError(
                f"transition {transition!r} conflicts with required evidence "
                f"transition {assessment['required_transition']!r}"
            )
        if next_aspect != assessment["required_next_aspect_id"]:
            raise PlanAgentError(
                f"next_aspect_id {next_aspect!r} conflicts with required evidence "
                f"aspect {assessment['required_next_aspect_id']!r}"
            )
        if action == "stop":
            if transition != "stop" or next_aspect is not None:
                raise PlanAgentError(
                    "stop requires transition=stop and next_aspect_id=null"
                )
            next_template = None
            next_round = None
        else:
            if transition not in {"drill_down", "switch_aspect"}:
                raise PlanAgentError(
                    "continue transition must be drill_down or switch_aspect"
                )
            allowed_aspects = assessment["available_transitions"][transition]
            if next_aspect not in allowed_aspects:
                raise PlanAgentError(
                    f"next_aspect_id is not available for {transition}: "
                    f"{allowed_aspects}"
                )
            next_template = assessment["remaining_template_ids_by_aspect"][next_aspect][
                0
            ]
            next_round = self._materialize_round(
                next_template,
                len(current_plan["rounds"]) + 1,
                user_request,
            )
        return {
            "schema_version": 1,
            "action": action,
            "transition": transition,
            "observation_summary": summary,
            "decision_reason": reason,
            "next_aspect_id": next_aspect,
            "next_template_id": next_template,
            "evidence_assessment": assessment,
            "next_round": next_round,
        }

    @staticmethod
    def _history_summary(
        history_context: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        return [
            {
                key: item.get(key)
                for key in ("evaluation_id", "user_request", "task_name", "similarity")
            }
            for item in (history_context or [])
            if isinstance(item, dict)
        ]

    def _initial_prompt(
        self,
        user_request: str,
        history_context: list[dict[str, Any]] | None,
    ) -> str:
        example = {
            "schema_version": 1,
            "task_name": "click_bell",
            "evaluation_goal": "evaluate_position_and_instance_generalization",
            "requested_aspect_ids": ["object_position", "object_instance"],
            "first_aspect_id": "object_position",
        }
        capability_card = {
            aspect_id: {
                "description": aspect["description"],
                "trusted_variants": [
                    {
                        "template_id": template_id,
                        "description": CLICK_BELL_ADAPTIVE_TEMPLATES[template_id][
                            "description"
                        ],
                    }
                    for template_id in aspect["template_ids"]
                ],
            }
            for aspect_id, aspect in CLICK_BELL_ADAPTIVE_ASPECTS.items()
        }
        return f"""You are the bounded MEA Plan Agent for RoboTwin click_bell.
Decompose the open user query into relevant orthogonal evaluation aspects and
choose the first aspect.  Do not output Python, variants, seeds, gates, tools,
or execution parameters; the trusted runtime injects them.

USER QUERY:
{user_request}

POLICY:
{json.dumps(CLICK_BELL_POLICY, ensure_ascii=False, indent=2)}

TRUSTED CAPABILITY CARD:
{json.dumps(capability_card, ensure_ascii=False, indent=2)}

SIMILAR PLAN HISTORY (planning prior only; never execution evidence):
{json.dumps(self._history_summary(history_context), ensure_ascii=False, indent=2)}

The two bell instances are official base0/base1 assets with appearance, size,
and contact-height differences.  They are an instance axis, not pure texture
or a new OOD asset.  Select only aspects relevant to the query.  Return strict
JSON with exactly this shape:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""

    def _decision_prompt(
        self,
        user_request: str,
        current_plan: dict[str, Any],
        observation_history: list[dict[str, Any]],
    ) -> str:
        assessment = self._assess_evidence(current_plan, observation_history)
        can_continue = "continue" in assessment["allowed_actions"]
        if can_continue and assessment["available_transitions"]["drill_down"]:
            transition = "drill_down"
            next_aspect = assessment["available_transitions"][transition][0]
            action = "continue"
        elif can_continue:
            transition = "switch_aspect"
            next_aspect = assessment["available_transitions"][transition][0]
            action = "continue"
        else:
            transition = "stop"
            next_aspect = None
            action = "stop"
        example = {
            "schema_version": 1,
            "action": action,
            "transition": transition,
            "observation_summary": (
                "Summarize policy, aggregate, and VQA evidence without "
                "confusing policy failure with pipeline failure."
            ),
            "decision_reason": "Explain why evidence supports this direction.",
            "next_aspect_id": next_aspect,
        }
        return f"""You are the bounded adaptive MEA Plan Agent for click_bell.
Read the complete evidence from all completed rounds and explain the trusted
evidence policy's required drill-down, aspect switch, or stop transition.
Policy failure is valid evaluation evidence; pipeline failure is not policy
failure.  Use drill_down when a same-aspect counterfactual would clarify a
boundary, switch_aspect when the current aspect is sufficiently characterized,
and stop only when continuation is unsafe or no trusted target remains.  You
must copy required_action, required_transition, and required_next_aspect_id
from the trusted assessment into the corresponding output fields.

USER QUERY:
{user_request}

CURRENT PLAN:
{json.dumps(current_plan, ensure_ascii=False, indent=2)}

REAL OBSERVATION HISTORY:
{json.dumps(observation_history, ensure_ascii=False, indent=2)}

TRUSTED EVIDENCE ASSESSMENT AND AVAILABLE TRANSITIONS:
{json.dumps(assessment, ensure_ascii=False, indent=2)}

The runtime validates the action/transition/aspect and injects the exact next
variant.  Return strict JSON with exactly this shape:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""

    def plan(
        self,
        user_request: str,
        *,
        evaluation_id: str | None = None,
        history_context: list[dict[str, Any]] | None = None,
        history_metadata: dict[str, Any] | None = None,
        validated_proposal: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request = self._require_text(user_request, "user_request")
        resolved_id = evaluation_id or make_evaluation_id()
        if not re.fullmatch(r"eval_[A-Za-z0-9_]+", resolved_id):
            raise PlanAgentError("evaluation_id must begin with 'eval_'")
        evaluation_dir = self.repo_root / "mea/evaluation_runs" / resolved_id
        if evaluation_dir.exists():
            raise PlanAgentError(
                f"evaluation directory already exists: {evaluation_dir}"
            )
        for child in ("plan", "execution", "summary"):
            (evaluation_dir / child).mkdir(parents=True, exist_ok=False)

        history = {
            "schema_version": 1,
            "status": "passed" if history_context else "empty",
            "match_count": len(history_context or []),
            "matches": self._history_summary(history_context),
            **deepcopy(history_metadata or {}),
        }
        manifest = {
            "schema_version": 6,
            "evaluation_id": resolved_id,
            "status": "planning_round_1",
            "created_at": datetime.now().astimezone().isoformat(),
            "user_request": request,
            "base_commit": _git_head(self.repo_root),
            "planner": {
                "kind": "model_click_bell_adaptive_v1",
                "model_requested": self.model,
                "provider_called": False,
                "round_1_validation_errors": [],
            },
            "plan_path": "plan/evaluation_plan.json",
            "history_retrieval_path": "plan/history_retrieval.json",
            "history_retrieval": history,
            "plan": None,
        }
        _write_json(evaluation_dir / "request.json", {"user_request": request})
        _write_json(evaluation_dir / "plan/history_retrieval.json", history)
        _write_json(evaluation_dir / "manifest.json", manifest)
        errors: list[str] = []
        provider_called = validated_proposal is None
        plan = (
            self._validate_proposal(deepcopy(validated_proposal), user_request=request)
            if validated_proposal is not None
            else None
        )
        if validated_proposal is not None:
            _write_json(
                evaluation_dir / "plan/global_route_proposal.json",
                validated_proposal,
            )
        else:
            prompt = self._initial_prompt(request, history_context)
            (evaluation_dir / "plan/round_1_prompt.md").write_text(
                prompt, encoding="utf-8"
            )
            for attempt in range(2):
                attempt_prompt = prompt
                if errors:
                    attempt_prompt += (
                        "\n\nPREVIOUS VALIDATION ERROR:\n"
                        + errors[-1]
                        + "\nReturn a complete corrected JSON object.\n"
                    )
                try:
                    response = self.provider.text(
                        attempt_prompt,
                        model=self.model,
                        system="Return only strict ClickBellEvaluationProposal JSON.",
                        max_tokens=700,
                        temperature=0.0,
                    )
                except Exception as exc:
                    errors.append(f"provider call failed: {exc}")
                    continue
                suffix = "" if attempt == 0 else f"_retry_{attempt}"
                (evaluation_dir / f"plan/round_1_response{suffix}.txt").write_text(
                    response + "\n", encoding="utf-8"
                )
                try:
                    plan = self._validate_proposal(
                        extract_json_response(response), user_request=request
                    )
                    break
                except (PlanAgentError, TaskGenError) as exc:
                    errors.append(str(exc))
        if plan is None:
            manifest["status"] = "planning_failed"
            manifest["planner"].update(
                {
                    "provider_called": True,
                    "round_1_metadata": dict(
                        getattr(self.provider, "last_metadata", {})
                    ),
                    "round_1_validation_errors": errors,
                }
            )
            _write_json(evaluation_dir / "manifest.json", manifest)
            raise PlanAgentError(f"adaptive proposal failed twice: {errors}")

        manifest.update({"status": "planned_round_1", "plan": plan})
        manifest["planner"].update(
            {
                "provider_called": provider_called,
                "initial_proposal_source": (
                    "global_query_route"
                    if validated_proposal is not None
                    else "task_specific_model"
                ),
                "round_1_metadata": dict(getattr(self.provider, "last_metadata", {}))
                if provider_called
                else {},
                "round_1_validation_errors": errors,
            }
        )
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
        if current_plan.get("schema_version") != 6:
            raise PlanAgentError("adaptive click_bell plan schema_version must be 6")
        history = self._validate_observations(current_plan, observation_history)
        completed = len(history)
        evaluation_dir = self.repo_root / "mea/evaluation_runs" / evaluation_id
        if not evaluation_dir.is_dir():
            raise PlanAgentError(
                f"evaluation directory does not exist: {evaluation_dir}"
            )
        assessment = self._assess_evidence(current_plan, history)
        _write_json(
            evaluation_dir / f"plan/evidence_after_round_{completed}.json",
            assessment,
        )
        prompt = self._decision_prompt(user_request, current_plan, history)
        stem = f"decision_after_round_{completed}"
        (evaluation_dir / f"plan/{stem}_prompt.md").write_text(prompt, encoding="utf-8")
        errors: list[str] = []
        decision = None
        for attempt in range(2):
            attempt_prompt = prompt
            if errors:
                attempt_prompt += (
                    "\n\nPREVIOUS VALIDATION ERROR:\n"
                    + errors[-1]
                    + "\nReturn a complete corrected JSON object.\n"
                )
            try:
                response = self.provider.text(
                    attempt_prompt,
                    model=self.model,
                    system="Return only strict ClickBellNextRoundDecision JSON.",
                    max_tokens=700,
                    temperature=0.0,
                )
            except Exception as exc:
                errors.append(f"provider call failed: {exc}")
                continue
            suffix = "" if attempt == 0 else f"_retry_{attempt}"
            (evaluation_dir / f"plan/{stem}_response{suffix}.txt").write_text(
                response + "\n", encoding="utf-8"
            )
            try:
                decision = self._validate_decision(
                    extract_json_response(response),
                    current_plan=current_plan,
                    observation_history=history,
                    user_request=user_request,
                )
                break
            except (PlanAgentError, TaskGenError) as exc:
                errors.append(str(exc))
        if decision is None:
            manifest_path = evaluation_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = f"decision_failed_after_round_{completed}"
            planner = manifest.setdefault("planner", {})
            planner[f"{stem}_metadata"] = dict(
                getattr(self.provider, "last_metadata", {})
            )
            planner[f"{stem}_validation_errors"] = errors
            _write_json(manifest_path, manifest)
            raise PlanAgentError(f"adaptive decision failed twice: {errors}")

        updated = deepcopy(current_plan)
        updated.setdefault("round_decisions", []).append(decision)
        if decision["action"] == "continue":
            updated["rounds"].append(decision["next_round"])
            updated[
                "planning_state"
            ] = f"awaiting_round_{len(updated['rounds'])}_observation"
        else:
            updated["planning_state"] = f"stopped_after_round_{completed}"
        _write_json(evaluation_dir / f"plan/{stem}.json", decision)
        _write_json(evaluation_dir / "plan/evaluation_plan.json", updated)
        manifest_path = evaluation_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"status": updated["planning_state"], "plan": updated})
        planner = manifest.setdefault("planner", {})
        planner[f"{stem}_metadata"] = dict(getattr(self.provider, "last_metadata", {}))
        planner[f"{stem}_validation_errors"] = errors
        _write_json(manifest_path, manifest)
        return updated, decision


class ClickBellFixedSuitePlanAgent(ClickBellAdaptivePlanAgent):
    """Execute a frozen trusted template suite without evidence-driven routing.

    The global model may still decompose the open query once.  After that, the
    schedule is fixed before the first rollout: policy success and VQA evidence
    are reported but never used to choose the next template.  Pipeline failure
    still stops execution because continuing would not yield comparable data.
    """

    def plan(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        manifest = super().plan(*args, **kwargs)
        plan = manifest["plan"]
        plan["planning_policy"] = "fixed_predeclared_v1"
        plan["frozen_before_first_rollout"] = True
        manifest["plan"] = plan
        manifest["planning_policy"] = "fixed_predeclared_v1"
        manifest["planner"]["kind"] = "fixed_predeclared_click_bell_v1"
        evaluation_dir = (
            self.repo_root / "mea/evaluation_runs" / manifest["evaluation_id"]
        )
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
        history = self._validate_observations(current_plan, observation_history)
        completed = len(history)
        latest = history[-1]
        executed = {
            str(round_plan.get("template_id"))
            for round_plan in current_plan.get("rounds", [])
        }
        remaining = [
            template_id
            for template_id in current_plan.get("requested_template_ids", [])
            if template_id not in executed
        ]
        budget_remaining = max(int(current_plan.get("max_rounds") or 0) - completed, 0)
        pipeline_passed = bool(latest.get("pipeline_passed"))
        can_continue = pipeline_passed and bool(remaining) and budget_remaining > 0
        next_template = remaining[0] if can_continue else None
        next_round = (
            self._materialize_round(next_template, completed + 1, user_request)
            if next_template is not None
            else None
        )
        updated = deepcopy(current_plan)
        if next_round is not None:
            updated["rounds"].append(next_round)
            updated["planning_state"] = f"awaiting_round_{completed + 1}_observation"
            action = "continue"
            reason = "advance_to_next_predeclared_template"
        else:
            updated["planning_state"] = f"stopped_after_round_{completed}"
            action = "stop"
            reason = (
                "pipeline_failure_forces_stop"
                if not pipeline_passed
                else "fixed_schedule_complete"
                if not remaining
                else "fixed_round_budget_exhausted"
            )
        assessment = {
            "schema_version": 1,
            "planning_policy": "fixed_predeclared_v1",
            "pipeline_passed": pipeline_passed,
            "policy_evidence_used_for_routing": False,
            "vqa_evidence_used_for_routing": False,
            "remaining_template_ids": remaining,
            "round_budget_remaining": budget_remaining,
            "required_action": action,
            "reason": reason,
        }
        decision = {
            "schema_version": 1,
            "action": action,
            "transition": "fixed_advance" if can_continue else "stop",
            "observation_summary": (
                "Evidence was recorded but did not alter the frozen schedule."
            ),
            "decision_reason": reason,
            "next_aspect_id": (
                CLICK_BELL_ADAPTIVE_TEMPLATES[next_template]["aspect_id"]
                if next_template is not None
                else None
            ),
            "next_template_id": next_template,
            "evidence_assessment": assessment,
            "next_round": next_round,
        }
        updated.setdefault("round_decisions", []).append(decision)
        evaluation_dir = self.repo_root / "mea/evaluation_runs" / evaluation_id
        stem = f"decision_after_round_{completed}"
        _write_json(
            evaluation_dir / f"plan/evidence_after_round_{completed}.json", assessment
        )
        _write_json(evaluation_dir / f"plan/{stem}.json", decision)
        _write_json(evaluation_dir / "plan/evaluation_plan.json", updated)
        manifest_path = evaluation_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"status": updated["planning_state"], "plan": updated})
        _write_json(manifest_path, manifest)
        return updated, decision
