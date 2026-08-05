"""Provider generation and bounded repair for open Tool requests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.providers.json_response import extract_json_response

from .open_request_context import tool_generation_context
from .open_request_contract import (
    OpenToolRequestError,
    OpenToolRequestUnsupported,
    _text,
)
from .open_request_oracle import _unsupported_tool_response
from .open_request_validation import validate_open_tool_request


class OpenToolRequestAgent:
    """Generate one executable Tool request from a semantic evidence need."""

    def __init__(
        self,
        repo_root: str | Path,
        provider: Any,
        *,
        model: str,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.provider = provider
        self.model = _text(model, "model")
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []

    @staticmethod
    def _prompt(
        *,
        source_query: str,
        semantic_concern: str,
        tool_need: str,
        context: Mapping[str, Any],
    ) -> str:
        derived_available = bool(
            context.get("derived_observable_validation_available")
        )
        # Lead with an independently interpreted operator.  Earlier prompts
        # showed a derived-observable example first, which made the model skip
        # a simpler typed contract and weakened numeric validation.
        output_example = {
            "schema_version": 2,
            "task_name": context["task_name"],
            "metric": "query_minimum_distance",
            "question": "How close did the robot get to the target?",
            "metric_spec": {
                "schema_version": 1,
                "operation": "minimum_distance",
                "left_signal": "advertised_robot_position",
                "right_signal": "advertised_target_position",
                "dimensions": ["x", "y", "z"],
                "unit": "m",
                "null_semantics": "null_if_no_finite_sample",
            },
        }
        derived_contract = {
            "schema_version": 2,
            "operation": "derived_observable",
            "observable_id": "query_specific_observable",
            "description": "Exact trajectory reduction required by the Query.",
            "required_signals": ["advertised_semantic_field_name"],
            "unit": "physical_unit",
            "null_semantics": "null_if_no_finite_sample",
        }
        novel_rule = (
            "Only when no advertised typed operator exactly expresses the "
            "need, propose a schema_version=2 derived_observable semantic "
            "contract with metric_spec.operation exactly "
            '"derived_observable". Its description MUST contain 1-240 '
            "characters, "
            "required_signals MUST contain 1-8 advertised signal names, and "
            "null_semantics MUST be exactly null_if_no_finite_sample. Include "
            "all three fields in every complete response, including repairs. "
            "Also provide a precise physical unit. ToolGen will run a separate "
            "development-agent semantic review against this contract, "
            "restrict the implementation to those "
            "signals, and run deterministic finite-output and artifact-"
            "immutability checks on recorded telemetry. This is measurement "
            "evidence only, never success or reward authority."
            if derived_available
            else (
                "An independent oracle broker is unavailable in this run. "
                "Do not return derived_observable and do not approximate the "
                "need with a semantically different metric. If no exact "
                "registered or typed operator expresses the need, return "
                "exactly this unsupported shape: "
                '{"schema_version":1,"status":"unsupported",'
                '"reason_code":"independent_oracle_broker_unavailable",'
                '"reason":"The requested derived observable has no independent '
                'oracle broker."}.'
            )
        )
        return (
            "You are ToolGen in ManipEvalAgent. Derive the smallest executable "
            "measurement needed by the open Query. First inspect both the "
            "trusted static registry and validated_generated_tools. For an "
            "exact static match, return schema_version=1 with its metric id. "
            "For an exact generated match, copy that entry's schema_version=2 "
            "request and MetricSpec exactly. For a new measurement, return a "
            "schema_version=2 request with a schema_version=1 MetricSpec when "
            "one existing operator exactly expresses the need. "
            + novel_rule
            + " Provider-written Python for a typed operator is checked by a "
            "separate trusted numeric interpreter on live telemetry before "
            "exact registration/reuse. A derived observable is instead bound "
            "to its declared-signal validation authority. "
            "Replace every placeholder signal with a real advertised name. "
            "A registered "
            "composite target is an exact static match and may be selected by "
            "its schema_version=1 metric id; it will be generated and validated "
            "when no compatible registration exists. A fixed left/right signal "
            "does not satisfy an active-arm or active-gripper need when both "
            "sides are advertised. Do not invent an unavailable signal, task "
            "name, template, or aspect. Never select a metric listed in "
            "forbidden_metric_ids: those values are already present in the "
            "base Toolkit evidence or are semantically incompatible, so this "
            "Tool must add the missing Query-specific measurement. A click-point, "
            "TCP, end-effector, or gripper accuracy need must compare one "
            "advertised robot signal with one target signal; target-target "
            "distance is not robot accuracy. When "
            "MEASUREMENT NEED explicitly asks "
            "for a final/terminal x, y, z, height, or absolute component of an "
            "advertised semantic field, use terminal_signal_component with "
            "that exact signal and component. When it asks for the final or "
            "terminal difference, delta, or relative height/component between "
            "two advertised semantic fields, use terminal_signal_difference "
            "with those exact left/right signals and component. Do not replace "
            "a terminal two-signal difference with minimum_distance or a "
            "single terminal_signal_component. Treat an unqualified lift "
            "height difference between two objects as their terminal z "
            "difference; an event metric is not aligned. "
            "When the need asks for the terminal minimum distance from two or "
            "more candidate robot TCP/gripper fields to one target field, use "
            "terminal_minimum_distance with all candidate fields in "
            "left_signals and the target in right_signal. "
            "The structured ToolArtifactContext contains the exact Proposal, "
            "TaskArtifact authority summary, executed runtime schema, reusable "
            "artifacts, and oracle availability. Honor its typed need; do not "
            "invent a missing scene, checker, or authority. "
            "Return strict JSON only.\n\n"
            f"ORIGINAL QUERY:\n{source_query}\n\n"
            f"SEMANTIC CONCERN:\n{semantic_concern}\n\n"
            f"MEASUREMENT NEED:\n{tool_need}\n\n"
            "TELEMETRY AND TOOL CONTEXT:\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n\nOUTPUT EXAMPLE "
            "(replace every advertised_* placeholder):\n"
            + json.dumps(output_example, ensure_ascii=False, indent=2)
            + (
                "\n\nDERIVED FALLBACK METRIC_SPEC SHAPE (use only when no typed "
                "operator is exact):\n"
                + json.dumps(derived_contract, ensure_ascii=False, indent=2)
                if derived_available
                else ""
            )
        )

    def propose(
        self,
        *,
        source_query: str,
        semantic_concern: str,
        tool_need: str,
        task_name: str,
        generated_checker_semantics: bool = False,
        runtime_schema: Mapping[str, Any] | None = None,
        reusable_tool_requests: list[Mapping[str, Any]] | None = None,
        reusable_vqa_questions: list[Mapping[str, Any]] | None = None,
        forbidden_metric_ids: set[str] | None = None,
        derived_observable_oracle_available: bool = False,
        derived_observable_oracle_broker: Mapping[str, Any] | None = None,
        proposal: Mapping[str, Any] | None = None,
        task_artifact_summary: Mapping[str, Any] | None = None,
        allow_unsupported: bool = False,
    ) -> dict[str, Any]:
        context = tool_generation_context(
            self.repo_root,
            task_name=task_name,
            generated_checker_semantics=generated_checker_semantics,
            runtime_schema=runtime_schema,
            reusable_tool_requests=reusable_tool_requests,
            reusable_vqa_questions=reusable_vqa_questions,
            forbidden_metric_ids=forbidden_metric_ids,
            derived_observable_oracle_available=(
                derived_observable_oracle_available
            ),
            derived_observable_oracle_broker=(
                derived_observable_oracle_broker
            ),
            proposal=proposal,
            task_artifact_summary=task_artifact_summary,
        )
        prompt = self._prompt(
            source_query=_text(source_query, "source_query"),
            semantic_concern=_text(semantic_concern, "semantic_concern"),
            tool_need=_text(tool_need, "tool_need"),
            context=context,
        )
        self.last_prompt = prompt
        self.last_responses = []
        self.last_errors = []
        filled_bound_fields: list[str] = []
        request: dict[str, Any] | None = None
        for _attempt in range(2):
            attempt_prompt = prompt
            if self.last_errors:
                attempt_prompt += (
                    "\n\nPREVIOUS VALIDATION ERROR:\n"
                    + self.last_errors[-1]
                    + "\nReturn one corrected complete JSON object. Preserve "
                    "every required key from the output example. For a "
                    "derived_observable, description must contain 1-240 "
                    "characters and null_semantics must be exactly "
                    "null_if_no_finite_sample."
                )
            try:
                response = self.provider.text(
                    attempt_prompt,
                    model=self.model,
                    system="Return only strict ToolRequest JSON.",
                    max_tokens=900,
                    temperature=0.0,
                )
                self.last_responses.append(response)
                raw_request = extract_json_response(response)
                if not isinstance(raw_request, dict):
                    raise OpenToolRequestError(
                        "provider Tool request must be a JSON object"
                    )
                unsupported = _unsupported_tool_response(
                    raw_request,
                    context=context,
                    requested_need=tool_need,
                    provider={
                        "model_requested": self.model,
                        "called": True,
                        "attempt_count": len(self.last_responses),
                        "errors": list(self.last_errors),
                        "bound_fields_filled": list(filled_bound_fields),
                        "last_metadata": deepcopy(
                            dict(
                                getattr(
                                    self.provider, "last_metadata", {}
                                )
                                or {}
                            )
                        ),
                    },
                )
                if unsupported is not None:
                    if allow_unsupported:
                        return unsupported
                    raise OpenToolRequestUnsupported(unsupported)
                for field, value in (
                    ("task_name", str(context["task_name"])),
                    ("question", _text(tool_need, "tool_need")),
                ):
                    if field not in raw_request:
                        raw_request[field] = value
                        if field not in filled_bound_fields:
                            filled_bound_fields.append(field)
                request = validate_open_tool_request(
                    raw_request,
                    task_name=str(context["task_name"]),
                    available_signal_names={
                        str(item["name"])
                        for item in context["telemetry_schema"]["semantic_fields"]
                    },
                    available_signal_sides={
                        str(item["name"]): str(item["side"])
                        for item in context["telemetry_schema"]["semantic_fields"]
                        if isinstance(item.get("side"), str)
                        and str(item["side"]).strip()
                    },
                    available_actor_ids={
                        str(item["id"])
                        for item in context["telemetry_schema"]["tracked_actors"]
                    },
                    forbidden_metric_ids=set(
                        context["forbidden_metric_ids"]
                    ),
                    measurement_need=tool_need,
                    derived_observable_oracle_available=bool(
                        context["derived_observable_validation_available"]
                    ),
                )
                break
            except OpenToolRequestUnsupported:
                raise
            except Exception as exc:
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
        if request is None:
            raise OpenToolRequestError(
                "provider failed two open Tool request attempts: "
                + " | ".join(self.last_errors)
            )
        return {
            "schema_version": 1,
            "status": "selected",
            "artifact_kind": "rule_tool",
            "source": "provider_query_induced_tool_request",
            "tool_request": request,
            "context": context,
            "provider": {
                "model_requested": self.model,
                "called": True,
                "attempt_count": len(self.last_responses),
                "errors": list(self.last_errors),
                "bound_fields_filled": filled_bound_fields,
                "last_metadata": deepcopy(
                    dict(getattr(self.provider, "last_metadata", {}) or {})
                ),
            },
        }
