"""Query-first task retrieval for open manipulation evaluation.

The Plan Agent should interpret the Query and propose a semantic sub-aspect
before it sees executable
task/aspect choices.  This module keeps that ordering explicit:

1. validate a provider-authored initial sub-aspect proposal;
2. discover the public RoboTwin task library from official environment and
   instruction files;
3. retrieve the nearest base task using only the concern's task semantics;
4. apply the evaluated policy's task-scope contract.

The legacy planner catalog is accepted only as execution-capability metadata.
It never enters the concern prompt and never changes semantic retrieval scores.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from mea.taskgen.preservation_facts import (
    PreservationFactError,
    normalize_preservation_conditions,
)
from mea.toolkit.schema import (
    TaskSchemaError,
    load_task_schema,
    task_schema_path,
)


class OpenTaskResolutionError(ValueError):
    """Raised when an open concern or task-resolution contract is invalid."""


_LEGACY_CONCERN_KEYS = {
    "schema_version",
    "source_query",
    "sub_aspect",
    "hypothesis",
    "task_intent",
    "requested_variation",
    "measurement_need",
}
_CONCERN_KEYS = _LEGACY_CONCERN_KEYS | {"preserved_conditions"}
_EXPERIMENT_NEED_FIELDS = (
    "scene_need",
    "checker_need",
    "rule_tool_need",
    "vqa_tool_need",
)
_NEED_KEYS = {"required", "description"}
_TOOL_NEED_KEYS = _NEED_KEYS | {"reuse_first"}
_INVENTORY_KEYS = {
    "schema_version",
    "task_name",
    "description",
    "execution_status",
    "capability_aspects",
}
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "arm",
    "by",
    "for",
    "from",
    "in",
    "it",
    "of",
    "on",
    "one",
    "robot",
    "the",
    "then",
    "to",
    "two",
    "use",
    "using",
    "with",
}
EXPERIMENTAL_SUCCESS_CHECKER_GUIDANCE = (
    "If a Query calls an episode successful only when the official goal and "
    "any additional experimental condition both hold, request checker_need. A numeric "
    "difference Tool reports magnitude but cannot supply that pass/fail "
    "predicate. Mentioning the official goal or official predicate as one "
    "component of a combined condition does not make the Query official-only; "
    "record that invariant as a typed fact with property=official_goal and "
    "relation=required_conjunct, never as full official-success equivalence, "
    "and preserve every "
    "additional condition from the original Query. When both "
    "checker_need and rule_tool_need are required, keep their roles distinct: "
    "checker_need must describe a boolean conjunction such as 'official goal "
    "AND distractor remains uncontacted', while rule_tool_need describes the "
    "scalar or boolean observation used to diagnose it. Never copy a raw "
    "numeric measurement into checker_need as though it were a pass/fail "
    "predicate. If the checker applies a terminal-state distance threshold, "
    "the same-round Rule Tool must report the terminal value of that same "
    "distance. A trajectory peak or maximum is a separate trajectory weakness, "
    "not a scalar for setting the terminal threshold; later evidence refinement "
    "must not use its scale to relax, replace, or calibrate the terminal "
    "predicate. check_success is evaluated from simulator state, not from a "
    "whole-trajectory derived metric: smoothness, deviation, jerk, path length, "
    "or trajectory clearance belongs in rule_tool_need, never behind an "
    "invented checker helper."
)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenTaskResolutionError(f"{field} must be a non-empty string")
    return value.strip()


def _text_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise OpenTaskResolutionError(f"{field} must be a list")
    result = [_text(item, f"{field}[]") for item in value]
    if len(result) != len(set(result)):
        raise OpenTaskResolutionError(f"{field} must not contain duplicates")
    return result


def _experiment_need(
    value: Any,
    *,
    field: str,
    tool: bool,
) -> dict[str, Any]:
    keys = _TOOL_NEED_KEYS if tool else _NEED_KEYS
    if not isinstance(value, Mapping) or set(value) != keys:
        received = (
            sorted(str(item) for item in value)
            if isinstance(value, Mapping)
            else type(value).__name__
        )
        raise OpenTaskResolutionError(
            f"FreeConcern.{field} fields must be exactly {sorted(keys)}; "
            f"received {received}"
        )
    required = value.get("required")
    if not isinstance(required, bool):
        raise OpenTaskResolutionError(
            f"FreeConcern.{field}.required must be bool"
        )
    description = value.get("description")
    if description is not None:
        description = _text(
            description,
            f"FreeConcern.{field}.description",
        )
    if required != (description is not None):
        raise OpenTaskResolutionError(
            f"FreeConcern.{field}.description must be present exactly when "
            "required=true"
        )
    result = {
        "required": required,
        "description": description,
    }
    if tool:
        if value.get("reuse_first") is not True:
            raise OpenTaskResolutionError(
                f"FreeConcern.{field}.reuse_first must be true"
            )
        result["reuse_first"] = True
    return result


def validate_free_concern_experiment_needs(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the independent work requested by the first Plan decision."""

    if not isinstance(value, Mapping) or set(value) != set(
        _EXPERIMENT_NEED_FIELDS
    ):
        raise OpenTaskResolutionError(
            "FreeConcern experiment needs must contain exactly "
            f"{sorted(_EXPERIMENT_NEED_FIELDS)}"
        )
    result = {
        field: _experiment_need(
            value.get(field),
            field=field,
            tool=field in {"rule_tool_need", "vqa_tool_need"},
        )
        for field in _EXPERIMENT_NEED_FIELDS
    }
    if not any(need["required"] for need in result.values()):
        raise OpenTaskResolutionError(
            "FreeConcern must request at least one explicit evidence need"
        )
    return result


def _split_free_concern_response(
    value: Mapping[str, Any],
    *,
    expected_query: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Accept legacy concern-only fixtures and the production typed response."""

    supplied = set(value) if isinstance(value, Mapping) else set()
    if supplied == _CONCERN_KEYS or supplied == _LEGACY_CONCERN_KEYS:
        return (
            validate_free_concern(value, expected_query=expected_query),
            None,
        )
    expected = _CONCERN_KEYS | set(_EXPERIMENT_NEED_FIELDS)
    legacy_expected = _LEGACY_CONCERN_KEYS | set(_EXPERIMENT_NEED_FIELDS)
    if supplied != expected and supplied != legacy_expected:
        raise OpenTaskResolutionError(
            "FreeConcern response fields must match the concern-only or typed "
            f"shape; received {sorted(supplied)}"
        )
    concern = {
        field: value[field]
        for field in (
            _CONCERN_KEYS
            if "preserved_conditions" in value
            else _LEGACY_CONCERN_KEYS
        )
    }
    needs = {
        field: value[field]
        for field in _EXPERIMENT_NEED_FIELDS
    }
    return (
        validate_free_concern(concern, expected_query=expected_query),
        validate_free_concern_experiment_needs(needs),
    )


def _extract_json_response(response: Any) -> dict[str, Any]:
    source = str(response).strip()
    start = source.find("{")
    end = source.rfind("}")
    if start < 0 or end < start:
        raise OpenTaskResolutionError("FreeConcern response contains no JSON object")
    try:
        value = json.loads(source[start : end + 1])
    except json.JSONDecodeError as exc:
        raise OpenTaskResolutionError("FreeConcern response is invalid JSON") from exc
    if not isinstance(value, dict):
        raise OpenTaskResolutionError("FreeConcern response must be an object")
    return value


def build_free_concern_prompt(user_query: str, policy_card: Mapping[str, Any]) -> str:
    """Build the catalog-free first-stage prompt.

    The signature intentionally accepts no task inventory or aspect catalog.
    A single-task checkpoint name may be visible as policy metadata, but it is
    not an instruction to choose a predeclared concern.
    """

    query = _text(user_query, "user_query")
    scope = policy_task_scope_from_card(policy_card)
    example = {
        "schema_version": 1,
        "source_query": query,
        "sub_aspect": "a precise concern discovered from the Query",
        "hypothesis": "one falsifiable policy-behavior hypothesis",
        "task_intent": "invariant base manipulation action and goal in English",
        "requested_variation": "one bounded diagnostic change",
        "preserved_conditions": [],
        "measurement_need": "the observation needed to decide the hypothesis",
        "scene_need": {
            "required": True,
            "description": "the scene change needed to realize requested_variation",
        },
        "checker_need": {
            "required": False,
            "description": None,
        },
        "rule_tool_need": {
            "required": True,
            "description": "one primary numeric or symbolic observation needed",
            "reuse_first": True,
        },
        "vqa_tool_need": {
            "required": False,
            "description": None,
            "reuse_first": True,
        },
    }
    visible_scope = {
        "policy_name": scope["policy_name"],
        "single_task_checkpoint": scope["single_task_checkpoint"],
        "training_tasks": scope["training_tasks"],
        "language_conditioned": scope["language_conditioned"],
    }
    return f"""You are the Plan Agent in ManipEvalAgent.
Read the original Query and first discover the single most informative
sub-aspect and falsifiable hypothesis.  Describe the manipulation semantics
needed for that test in concise English in task_intent, even when the Query is
in another language.  task_intent must state the invariant base action and
goal, never the requested scene/appearance variation.  For a single-task
checkpoint, preserve its training-task semantics unless the Query explicitly
asks to evaluate a different manipulation task.  Put distractors and all other
diagnostic changes only in requested_variation.  Do not select from task names, task
templates, aspect identifiers, or a capability catalog: those are deliberately
not available until a later retrieval stage.
When requested_variation changes a scene, put each unchanged condition in
preserved_conditions as one object with exactly actor, property, axis, and
relation; do not encode preservation as prose in requested_variation and do
not use catch-all claims such as "all other conditions unchanged", "all other
object poses", or "the rest of the scene".
Use schema literals, not synonyms: property=position requires
relation="preserve" and axis="x", "y", "z", or "all"; property=orientation,
appearance, geometry, or model_identity requires relation="preserve" and
axis=null. property=contact_point requires axis=null and relation either
"preserve_local_offsets" or "preserve_world_position". Task-wide
property=official_goal or checker_semantics requires actor=null and axis=null;
official_goal allows relation="preserve" or "required_conjunct", while
checker_semantics requires relation="preserve".
Preservation is an authority claim. Task identity and policy checkpoint are
already frozen by the outer runtime binding, so do not repeat them as scene
preservation facts. At this pre-retrieval stage the default preservation list
is empty. A field listed as observable in policy or simulator metadata is a
measurement capability, not a preservation authority. Add an invariant only
after the current input names an authority that can compare it, such as exact
method reuse, same-seed simulator state, a checker fixture, or a visual
comparison for visible appearance. In particular, do not add actor identity, physics
timestep, or object-to-target binding without such authority. Never emit vague
preserve entries such as "target configuration", "intended goal", or "task
semantics". When a new checker adds a condition to the official task goal,
write property=official_goal and relation=required_conjunct; do not claim that
the extended checker preserves full official success semantics.
The requested change and preserved conditions must be jointly realizable:
never request a size/shape/pose/contact change while also declaring that same
quantity invariant. Prefer a bounded experiment whose invariants can be checked
from simulator state, checker fixtures, or exact method reuse; RGB is only
authority for visibly decidable appearance and plausibility.
At this pre-retrieval stage, workspace and camera bounds are not available.
Do not invent an absolute perturbation magnitude.  Specify the diagnostic
direction and let TaskGen choose the smallest measurable change after it
retrieves the official source and validates the first render.
If the hypothesis says a metric is larger/smaller than an undisturbed,
baseline, control, or official scene, that comparison requires a separate
control rollout.  Otherwise formulate a one-episode hypothesis with an
observable condition that the generated experiment can decide directly.

Independently declare the work needed to execute this first experiment.
Request a scene only when requested_variation changes the simulator scene;
request a checker only when the Query needs success semantics beyond the
official task; request a Rule Tool for numeric or symbolic evidence; request a
VQA Tool only for a visual judgment.  A Tool-only Query must not invent a scene
or checker.  An official-task-only Query must request Rule Tool reuse of the
official check_success() result while leaving scene, checker, and VQA needs
false.  Each Rule/VQA need must name one primary scalar or boolean observation;
leave independent measurements for an evidence-conditioned later round.
scene_need and checker_need must each contain exactly required and description;
never add reuse_first to either. Only rule_tool_need and vqa_tool_need contain
reuse_first, which must always be true because every Tool retrieves before
generating.
{EXPERIMENTAL_SUCCESS_CHECKER_GUIDANCE}

ORIGINAL QUERY:
{query}

EVALUATED POLICY SCOPE (metadata, not a concern menu):
{json.dumps(visible_scope, ensure_ascii=False, indent=2)}

Return strict JSON with exactly these fields:
{json.dumps(example, ensure_ascii=False, indent=2)}
"""


def validate_free_concern(
    value: Mapping[str, Any], *, expected_query: str | None = None
) -> dict[str, Any]:
    """Validate one unconstrained semantic concern before task retrieval."""

    supplied = set(value) if isinstance(value, Mapping) else set()
    if (
        not isinstance(value, Mapping)
        or (
            supplied != _CONCERN_KEYS
            and supplied != _LEGACY_CONCERN_KEYS
        )
    ):
        raise OpenTaskResolutionError(
            f"FreeConcern fields must be exactly {sorted(_CONCERN_KEYS)}"
        )
    result = deepcopy(dict(value))
    if result.get("schema_version") != 1:
        raise OpenTaskResolutionError("FreeConcern.schema_version must be 1")
    for field in sorted(_LEGACY_CONCERN_KEYS - {"schema_version"}):
        result[field] = _text(result.get(field), f"FreeConcern.{field}")
    try:
        result["preserved_conditions"] = normalize_preservation_conditions(
            result.get("preserved_conditions", ())
        )
    except PreservationFactError as exc:
        raise OpenTaskResolutionError(
            f"FreeConcern.preserved_conditions: {exc}"
        ) from exc
    if expected_query is not None and result["source_query"] != _text(
        expected_query, "expected_query"
    ):
        raise OpenTaskResolutionError(
            "FreeConcern.source_query differs from the original Query"
        )
    return result


class PlanAgentQueryInterpreter:
    """Create the Plan Agent's initial proposal without exposing task candidates."""

    def __init__(self, provider: Any, *, model: str, max_attempts: int = 2):
        self.provider = provider
        self.model = _text(model, "model")
        if (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or max_attempts < 1
        ):
            raise OpenTaskResolutionError("max_attempts must be positive")
        self.max_attempts = max_attempts
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []

    def propose(
        self, user_query: str, *, policy_card: Mapping[str, Any]
    ) -> dict[str, Any]:
        prompt = build_free_concern_prompt(user_query, policy_card)
        self.last_prompt = prompt
        self.last_responses = []
        self.last_errors = []
        concern: dict[str, Any] | None = None
        experiment_needs: dict[str, Any] | None = None
        for _attempt in range(self.max_attempts):
            attempt_prompt = prompt
            if self.last_errors:
                attempt_prompt += (
                    "\nPREVIOUS VALIDATION ERROR:\n"
                    + self.last_errors[-1]
                    + "\nRe-check every preserved_conditions object against "
                    "all schema-literal rules above, not only the first "
                    "reported error.\nReturn one corrected JSON object.\n"
                )
            try:
                response = self.provider.text(
                    attempt_prompt,
                    model=self.model,
                    system="Return only strict InitialSubAspectProposal JSON.",
                    max_tokens=700,
                    temperature=0.0,
                )
                self.last_responses.append(response)
                candidate_concern, candidate_needs = _split_free_concern_response(
                    _extract_json_response(response),
                    expected_query=user_query,
                )
                concern = candidate_concern
                experiment_needs = candidate_needs
                break
            except Exception as exc:
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
        if concern is None:
            raise OpenTaskResolutionError(
                f"provider failed {self.max_attempts} FreeConcern attempt(s): "
                + " | ".join(self.last_errors)
            )
        return {
            "schema_version": 1,
            "source": "provider_plan_agent_query_interpretation",
            "concern": concern,
            "experiment_needs": experiment_needs,
            "provider": {
                "model_requested": self.model,
                "called": True,
                "attempt_count": len(self.last_responses),
                "errors": list(self.last_errors),
                "last_metadata": deepcopy(
                    dict(getattr(self.provider, "last_metadata", {}))
                ),
            },
        }


def policy_task_scope_from_card(policy_card: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize existing PlanningContext policy metadata into a task gate."""

    if not isinstance(policy_card, Mapping):
        raise OpenTaskResolutionError("policy_card must be an object")
    policy_name = _text(policy_card.get("policy_name"), "policy_card.policy_name")
    checkpoint_id = _text(
        policy_card.get("checkpoint_id"), "policy_card.checkpoint_id"
    )
    single_task = policy_card.get("single_task_checkpoint")
    language_conditioned = policy_card.get("language_conditioned")
    checkpoint_ready = policy_card.get("checkpoint_ready")
    if not isinstance(single_task, bool):
        raise OpenTaskResolutionError(
            "policy_card.single_task_checkpoint must be bool"
        )
    if not isinstance(language_conditioned, bool):
        raise OpenTaskResolutionError("policy_card.language_conditioned must be bool")
    if checkpoint_ready is not True:
        raise OpenTaskResolutionError("task resolution requires a ready checkpoint")

    if single_task:
        training_tasks = [_text(policy_card.get("task_name"), "policy_card.task_name")]
    else:
        raw_training = policy_card.get("training_tasks")
        training_tasks = _text_list(
            raw_training, "policy_card.training_tasks"
        )
        if not training_tasks:
            raise OpenTaskResolutionError(
                "a multi-task checkpoint must declare training_tasks"
            )

    supports_unseen = policy_card.get("supports_unseen_tasks", False)
    if not isinstance(supports_unseen, bool):
        raise OpenTaskResolutionError(
            "policy_card.supports_unseen_tasks must be bool when present"
        )
    if single_task and supports_unseen:
        raise OpenTaskResolutionError(
            "a single-task checkpoint cannot claim unseen-task support"
        )
    if supports_unseen and not language_conditioned:
        raise OpenTaskResolutionError(
            "unseen-task support requires a language-conditioned policy"
        )
    return {
        "schema_version": 1,
        "policy_name": policy_name,
        "checkpoint_id": checkpoint_id,
        "single_task_checkpoint": single_task,
        "training_tasks": training_tasks,
        "language_conditioned": language_conditioned,
        "supports_unseen_tasks": supports_unseen,
    }


def _capability_index(catalog: Mapping[str, Any] | None) -> dict[str, list[str]]:
    if catalog is None:
        return {}
    raw_tasks = catalog.get("tasks")
    if not isinstance(raw_tasks, list):
        raise OpenTaskResolutionError("capability_catalog.tasks must be a list")
    result: dict[str, list[str]] = {}
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise OpenTaskResolutionError(
                f"capability_catalog.tasks[{index}] must be an object"
            )
        task_name = _text(
            raw.get("task_name"), f"capability_catalog.tasks[{index}].task_name"
        )
        raw_aspects = raw.get("aspects")
        if not isinstance(raw_aspects, list):
            raise OpenTaskResolutionError(
                f"capability_catalog.tasks[{index}].aspects must be a list"
            )
        aspects: list[str] = []
        for aspect_index, aspect in enumerate(raw_aspects):
            if not isinstance(aspect, Mapping):
                raise OpenTaskResolutionError(
                    "capability catalog aspects must be objects"
                )
            aspects.append(
                _text(
                    aspect.get("aspect_id"),
                    f"capability_catalog.tasks[{index}].aspects"
                    f"[{aspect_index}].aspect_id",
                )
            )
        result[task_name] = list(dict.fromkeys(aspects))
    return result


def validate_task_inventory(
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate the compact public RoboTwin task inventory."""

    if not isinstance(inventory, Sequence) or isinstance(
        inventory, (str, bytes, bytearray)
    ):
        raise OpenTaskResolutionError("task inventory must be a sequence")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(inventory):
        if not isinstance(raw, Mapping) or set(raw) != _INVENTORY_KEYS:
            raise OpenTaskResolutionError(
                f"task inventory entry {index} fields must be exactly "
                f"{sorted(_INVENTORY_KEYS)}"
            )
        item = deepcopy(dict(raw))
        if item.get("schema_version") != 1:
            raise OpenTaskResolutionError("task inventory schema_version must be 1")
        item["task_name"] = _text(
            item.get("task_name"), f"task_inventory[{index}].task_name"
        )
        item["description"] = _text(
            item.get("description"), f"task_inventory[{index}].description"
        )
        if item["task_name"] in seen:
            raise OpenTaskResolutionError("task inventory contains duplicate tasks")
        seen.add(item["task_name"])
        if item.get("execution_status") not in {
            "capability_registered",
            "official_base_only",
        }:
            raise OpenTaskResolutionError("invalid task inventory execution_status")
        item["capability_aspects"] = _text_list(
            item.get("capability_aspects"),
            f"task_inventory[{index}].capability_aspects",
        )
        if (
            item["execution_status"] == "capability_registered"
            and not item["capability_aspects"]
        ):
            raise OpenTaskResolutionError(
                "registered inventory task must expose at least one capability"
            )
        if (
            item["execution_status"] == "official_base_only"
            and item["capability_aspects"]
        ):
            raise OpenTaskResolutionError(
                "official-only task cannot expose registered capabilities"
            )
        result.append(item)
    return sorted(result, key=lambda item: item["task_name"])


def discover_robotwin_task_inventory(
    repo_root: str | Path,
    *,
    capability_catalog: Mapping[str, Any] | None = None,
    schema_backed_only: bool = False,
) -> list[dict[str, Any]]:
    """Discover official task bases without turning them into a concern menu.

    The default preserves the public 50-task discovery behavior.  Runtime
    callers may request ``schema_backed_only`` so every returned task already
    has the validated telemetry/execution schema required by generic TaskGen.
    Capability catalog membership only enriches the returned retrieval hints.
    """

    root = Path(repo_root).expanduser().resolve()
    if not isinstance(schema_backed_only, bool):
        raise OpenTaskResolutionError("schema_backed_only must be bool")
    env_root = root / "envs"
    instruction_root = root / "description" / "task_instruction"
    capabilities = _capability_index(capability_catalog)
    entries: list[dict[str, Any]] = []
    if not env_root.is_dir() or (
        not schema_backed_only and not instruction_root.is_dir()
    ):
        raise OpenTaskResolutionError(
            "RoboTwin envs and description/task_instruction directories are required"
        )
    for env_path in sorted(env_root.glob("*.py")):
        task_name = env_path.stem
        if task_name.startswith("_"):
            continue
        instruction_path = instruction_root / f"{task_name}.json"
        schema: Mapping[str, Any] | None = None
        if schema_backed_only:
            schema_path = task_schema_path(root, task_name)
            if not schema_path.is_file():
                continue
            try:
                schema = load_task_schema(root, task_name)
            except TaskSchemaError as exc:
                raise OpenTaskResolutionError(
                    f"invalid runtime TaskSchema: {task_name}: {exc}"
                ) from exc
        if instruction_path.is_file():
            try:
                instruction = json.loads(
                    instruction_path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise OpenTaskResolutionError(
                    f"invalid official task instruction: {task_name}"
                ) from exc
            if not isinstance(instruction, Mapping):
                raise OpenTaskResolutionError(
                    f"official task instruction must be an object: {task_name}"
                )
            description = _text(
                instruction.get("full_description"),
                f"{task_name}.full_description",
            )
        elif schema_backed_only:
            family = (
                str(schema.get("task_family")).strip()
                if isinstance(schema, Mapping)
                and isinstance(schema.get("task_family"), str)
                else "manipulation"
            )
            description = (
                f"RoboTwin {family} task "
                f"{task_name.replace('_', ' ')}."
            )
        else:
            continue
        aspects = capabilities.get(task_name, [])
        entries.append(
            {
                "schema_version": 1,
                "task_name": task_name,
                "description": description,
                "execution_status": (
                    "capability_registered" if aspects else "official_base_only"
                ),
                "capability_aspects": list(aspects),
            }
        )
    if not entries:
        raise OpenTaskResolutionError("no official RoboTwin tasks were discovered")
    return validate_task_inventory(entries)


def discover_robotwin_runtime_task_inventory(
    repo_root: str | Path,
    *,
    capability_catalog: Mapping[str, Any] | None = None,
    schema_backed_only: bool = True,
) -> list[dict[str, Any]]:
    """Discover official tasks eligible for the selected policy runtime.

    ACT keeps the historical schema-backed boundary.  A shared policy may
    attempt every official task; downstream TaskGen and Rule Tool stages still
    require a TaskSchema and must gate on that capability separately.
    """

    if not isinstance(schema_backed_only, bool):
        raise OpenTaskResolutionError(
            "schema_backed_only must be boolean"
        )

    return discover_robotwin_task_inventory(
        repo_root,
        capability_catalog=capability_catalog,
        schema_backed_only=schema_backed_only,
    )


def _tokens(value: str) -> list[str]:
    return [
        token
        for token in _TOKEN.findall(value.lower().replace("_", " "))
        if token not in _STOPWORDS
    ]


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(count * right.get(token, 0) for token, count in left.items())
    left_norm = math.sqrt(sum(count * count for count in left.values()))
    right_norm = math.sqrt(sum(count * count for count in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def rank_official_tasks(
    concern: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Rank base tasks from task intent only, independent of concern novelty."""

    trusted_concern = validate_free_concern(concern)
    trusted_inventory = validate_task_inventory(inventory)
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise OpenTaskResolutionError("top_k must be a positive integer")
    intent = Counter(_tokens(trusted_concern["task_intent"]))
    ranked: list[dict[str, Any]] = []
    for task in trusted_inventory:
        name_tokens = _tokens(task["task_name"])
        document = Counter(
            _tokens(task["description"]) + name_tokens + name_tokens
        )
        semantic = _cosine(intent, document)
        name_coverage = (
            len(set(name_tokens) & set(intent)) / len(set(name_tokens))
            if name_tokens
            else 0.0
        )
        score = round(0.75 * semantic + 0.25 * name_coverage, 6)
        ranked.append(
            {
                "task_name": task["task_name"],
                "score": score,
                "execution_status": task["execution_status"],
                "capability_aspects": list(task["capability_aspects"]),
            }
        )
    ranked.sort(key=lambda item: (-item["score"], item["task_name"]))
    return ranked[: min(top_k, len(ranked))]


def resolve_open_task(
    concern: Mapping[str, Any],
    *,
    policy_card: Mapping[str, Any],
    inventory: Sequence[Mapping[str, Any]],
    top_k: int = 3,
    semantic_threshold: float = 0.2,
    near_tie_margin: float = 0.03,
    can_generate_new_task: bool = False,
) -> dict[str, Any]:
    """Resolve an open concern to retrieval, generation, or explicit refusal."""

    trusted_concern = validate_free_concern(concern)
    scope = policy_task_scope_from_card(policy_card)
    trusted_inventory = validate_task_inventory(inventory)
    if (
        isinstance(semantic_threshold, bool)
        or not isinstance(semantic_threshold, (int, float))
        or not 0.0 < float(semantic_threshold) <= 1.0
    ):
        raise OpenTaskResolutionError(
            "semantic_threshold must be in the interval (0, 1]"
        )
    if not isinstance(can_generate_new_task, bool):
        raise OpenTaskResolutionError("can_generate_new_task must be bool")
    if (
        isinstance(near_tie_margin, bool)
        or not isinstance(near_tie_margin, (int, float))
        or not 0.0 <= float(near_tie_margin) < 1.0
    ):
        raise OpenTaskResolutionError(
            "near_tie_margin must be in the interval [0, 1)"
        )

    ranked_all = rank_official_tasks(
        trusted_concern, trusted_inventory, top_k=len(trusted_inventory)
    )
    ranked = ranked_all[: min(top_k, len(ranked_all))]
    best = ranked_all[0]
    matched = best["score"] >= float(semantic_threshold)
    plausible = [
        item
        for item in ranked_all
        if item["score"] >= float(semantic_threshold)
        and best["score"] - item["score"] <= float(near_tie_margin)
    ]
    training_tasks = set(scope["training_tasks"])

    decision = "unsupported"
    reason_code = "no_semantic_task_match"
    selected: dict[str, Any] | None = None
    if matched:
        if scope["single_task_checkpoint"]:
            anchor = scope["training_tasks"][0]
            compatible = next(
                (item for item in plausible if item["task_name"] == anchor),
                None,
            )
            if compatible is not None:
                decision = "retrieve_and_adapt"
                reason_code = (
                    "nearest_training_task"
                    if best["task_name"] == anchor
                    else "policy_compatible_semantic_near_tie"
                )
                selected = compatible
            else:
                reason_code = "policy_task_mismatch"
        else:
            compatible = next(
                (
                    item
                    for item in plausible
                    if item["task_name"] in training_tasks
                    or scope["supports_unseen_tasks"]
                ),
                None,
            )
        if not scope["single_task_checkpoint"] and compatible is not None:
            decision = "retrieve_and_adapt"
            reason_code = (
                "nearest_training_task"
                if compatible["task_name"] in training_tasks
                else "nearest_official_open_task"
            )
            selected = compatible
        elif not scope["single_task_checkpoint"]:
            reason_code = "policy_task_mismatch"
    elif (
        can_generate_new_task
        and scope["language_conditioned"]
        and scope["supports_unseen_tasks"]
        and not scope["single_task_checkpoint"]
    ):
        decision = "generate_new"
        reason_code = "no_near_official_base"
    elif can_generate_new_task and not scope["supports_unseen_tasks"]:
        reason_code = "policy_not_open_task_capable"
    elif scope["supports_unseen_tasks"] and not can_generate_new_task:
        reason_code = "task_generation_unavailable"

    return {
        "schema_version": 1,
        "decision": decision,
        "reason_code": reason_code,
        "query_interpretation": trusted_concern,
        "policy_scope": scope,
        "selected_base_task": deepcopy(selected),
        "ranked_candidates": deepcopy(ranked),
        "resolution_contract": {
            "concern_created_before_inventory": True,
            "catalog_role": "execution_capability_inventory_only",
            "retrieval_field": "QueryInterpretation.task_intent",
            "semantic_threshold": float(semantic_threshold),
            "semantic_near_tie_margin": float(near_tie_margin),
            "plausible_candidate_names": [
                item["task_name"] for item in plausible
            ],
            "preserve_base_task_semantics": decision == "retrieve_and_adapt",
        },
    }


# Compatibility class name for historical callers and immutable artifacts.
FreeConcernAgent = PlanAgentQueryInterpreter


__all__ = [
    "PlanAgentQueryInterpreter",
    "EXPERIMENTAL_SUCCESS_CHECKER_GUIDANCE",
    "FreeConcernAgent",
    "OpenTaskResolutionError",
    "build_free_concern_prompt",
    "discover_robotwin_runtime_task_inventory",
    "discover_robotwin_task_inventory",
    "policy_task_scope_from_card",
    "rank_official_tasks",
    "resolve_open_task",
    "validate_free_concern",
    "validate_free_concern_experiment_needs",
    "validate_task_inventory",
]
