"""Deterministic LIBERO predicate MetricSpec adapter with registry reuse."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Mapping

from mea.toolgen.registry import (
    find_run_local_registration,
    public_registration_summary,
    register_run_local_tool,
)

from .benchmark import EpisodeRecord


_SOURCE = '''"""Deterministically compiled LIBERO predicate MetricSpec adapter."""

def evaluate_episode(record):
    if "goal_predicate_satisfied" not in record:
        raise ValueError("missing goal_predicate_satisfied")
    value = record["goal_predicate_satisfied"]
    if not isinstance(value, bool):
        raise TypeError("goal_predicate_satisfied must be boolean")
    return {
        "tool": "libero_goal_predicate_tool",
        "value": value,
        "unit": None,
        "passed": value,
        "evidence_steps": [max(0, int(record.get("executed_steps", 1)) - 1)],
        "details": {"predicate": record.get("goal_predicates", [])},
    }
'''


def _load_tool(path: Path):
    spec = importlib.util.spec_from_file_location("mea_compiled_libero_predicate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load compiled LIBERO predicate adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_episode


class LiberoPredicateToolBackend:
    """Compile/validate/register a bounded MetricSpec, then resolve exact reuse."""

    metric = "libero_goal_predicate_satisfied"

    def __init__(self, *, registry_dir: str | Path):
        self.registry_dir = Path(registry_dir).expanduser().resolve()

    @staticmethod
    def tool_spec(task_id: str, goal_predicates: list[list[str]]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task_name": task_id,
            "tool_id": "libero_goal_predicate_tool",
            "metric": LiberoPredicateToolBackend.metric,
            "question": "Was the LIBERO task goal predicate satisfied by the live rollout?",
            "metric_spec": {
                "operator": "episode_boolean_field",
                "field": "goal_predicate_satisfied",
                "goal_predicates": goal_predicates,
            },
            "required_signals": ["goal_predicate_satisfied", "executed_steps"],
            "output": {"kind": "boolean", "unit": None},
        }

    @staticmethod
    def _write_episode_projection(
        episode_dir: Path,
        record: EpisodeRecord,
        goal_predicates: list[list[str]],
    ) -> None:
        schema = {
            "schema_version": 1,
            "benchmark": "libero",
            "task_name": record.task_id,
            "signals": {
                "goal_predicate_satisfied": "boolean",
                "executed_steps": "integer",
            },
        }
        (episode_dir / "schema.json").write_text(
            json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        payload = record.to_dict()
        payload["goal_predicates"] = goal_predicates
        (episode_dir / "episode.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def compile_validate_register(
        self,
        *,
        output_dir: str | Path,
        episode_record: EpisodeRecord,
        goal_predicates: list[list[str]],
        source_query: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        episode_dir = Path(episode_record.actions_path).parent
        self._write_episode_projection(episode_dir, episode_record, goal_predicates)
        source_path = output / "compiled_metric_adapter.py"
        source_path.write_text(_SOURCE, encoding="utf-8")
        evaluate = _load_tool(source_path)
        fixtures = [
            {"name": "positive", "input": {"goal_predicate_satisfied": True, "executed_steps": 9}},
            {"name": "negative", "input": {"goal_predicate_satisfied": False, "executed_steps": 100}},
        ]
        validations = []
        for fixture in fixtures:
            result = evaluate(fixture["input"])
            validations.append(
                {
                    "fixture": fixture["name"],
                    "passed": result["value"] is fixture["input"]["goal_predicate_satisfied"],
                }
            )
        try:
            evaluate({"executed_steps": 1})
            missing_rejected = False
        except ValueError:
            missing_rejected = True
        if not all(item["passed"] for item in validations) or not missing_rejected:
            raise RuntimeError("LIBERO predicate Tool oracle fixtures failed")

        spec = self.tool_spec(episode_record.task_id, goal_predicates)
        generation_registration = {
            "tool": "libero_goal_predicate_tool",
            "validated_episode_count": 1,
            "validated_property_scenario_count": len(validations) + 1,
            "oracle_kind": "positive_negative_missing_fixture",
        }
        generation_manifest = {
            "generation_mode": "deterministic_metric_spec_adapter",
            "model_generated": False,
            "model_requested": None,
            "successful_attempt": None,
            "generator_source_sha256": None,
            "contract_sha256": None,
            "example_validation": validations,
            "source_query": source_query,
        }
        match = register_run_local_tool(
            self.registry_dir,
            tool_spec=spec,
            episode_dirs=[episode_dir],
            source_path=source_path,
            generation_registration=generation_registration,
            generation_manifest=generation_manifest,
            validation_episodes=validations,
        )
        live = evaluate(
            {
                **episode_record.to_dict(),
                "goal_predicates": goal_predicates,
            }
        )
        execution = self._execution(
            episode_record=episode_record,
            episode_dir=episode_dir,
            result=live,
            tool_spec=spec,
            route="predicate_metric_compile_validate_register",
            registration=public_registration_summary(match),
        )
        result = {
            "schema_version": 1,
            "status": "passed",
            "artifact_kind": "deterministic_predicate_metric_adapter",
            "model_generated": False,
            "source_query": source_query,
            "route": "predicate_metric_compile_validate_register",
            "oracle_fixtures": validations
            + [{"fixture": "missing", "passed": missing_rejected}],
            "registration": public_registration_summary(match),
            "live_value_non_null": live["value"] is not None,
            "tool_execution": execution,
        }
        (output / "toolgen_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result, execution

    def exact_reuse(
        self,
        *,
        output_dir: str | Path,
        episode_record: EpisodeRecord,
        goal_predicates: list[list[str]],
        source_query: str,
    ) -> dict[str, Any]:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        episode_dir = Path(episode_record.actions_path).parent
        spec = self.tool_spec(episode_record.task_id, goal_predicates)
        match = find_run_local_registration(
            self.registry_dir,
            tool_spec=spec,
            episode_dirs=[episode_dir],
        )
        if match is None:
            raise RuntimeError("expected exact run-local LIBERO Tool registration")
        evaluate = _load_tool(Path(match["source_path"]))
        live = evaluate(
            {
                **episode_record.to_dict(),
                "goal_predicates": goal_predicates,
            }
        )
        execution = self._execution(
            episode_record=episode_record,
            episode_dir=episode_dir,
            result=live,
            tool_spec=spec,
            route="exact_registry_reuse",
            registration=public_registration_summary(match),
        )
        result = {
            "schema_version": 1,
            "status": "passed",
            "source_query": source_query,
            "route": "exact_registry_reuse",
            "additional_rollouts": 0,
            "registration": public_registration_summary(match),
            "live_value_non_null": live["value"] is not None,
            "tool_execution": execution,
        }
        (output / "tool_reuse_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return result

    @classmethod
    def _execution(
        cls,
        *,
        episode_record: EpisodeRecord,
        episode_dir: Path,
        result: Mapping[str, Any],
        tool_spec: Mapping[str, Any],
        route: str,
        registration: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "status": "passed",
            "tool_request": {"metric": cls.metric},
            "tool_spec": dict(tool_spec),
            "route_decision": {
                "metric": cls.metric,
                "route": route,
                "registration": dict(registration),
            },
            "episodes": [
                {
                    "episode_dir": str(episode_dir),
                    "seed": episode_record.seed,
                    "policy_name": episode_record.policy_name,
                    "role": "policy_under_evaluation",
                    "variant": episode_record.task_id,
                    "result": dict(result),
                }
            ],
        }
