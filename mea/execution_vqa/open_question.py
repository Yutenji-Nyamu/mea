"""Query-induced VQA question generation over a shared ToolArtifactContext."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from mea.providers.json_response import extract_json_response
from mea.toolgen.artifact_context import validate_tool_artifact_context

from .query import (
    QUESTION_CATALOG,
    RUN_LOCAL_QUESTION_TYPES,
    RUN_LOCAL_TARGET_ROLES,
    RUN_LOCAL_VISUAL_SCOPES,
    ExecutionVQAQueryError,
    build_execution_vqa_query,
    validate_run_local_question_spec,
    vqa_need_semantic_key,
)
from .reviewed_generated_questions import find_reviewed_generated_vqa_question
from .reviewed_registry import ReviewedVQAQuerySpecError


class OpenVQAQuestionError(ValueError):
    """Raised when a Query-induced VQA question cannot be materialized."""


_PROVIDER_RESPONSE_KEYS = {"schema_version", "question_spec"}
_FALLBACK_REASON_PREFIXES = ("task_owned_fallback:", "generic_fallback:")
_REGISTRY_KIND = "evaluation_local_vqa_question_registry"
_REGISTRY_ENTRY_KEYS = {
    "schema_version",
    "kind",
    "semantic_key",
    "task_name",
    "vqa_need",
    "question_spec",
    "source_query",
    "semantic_concern",
    "artifact_path",
    "reuse_count",
}


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OpenVQAQuestionError(f"{field} must be a non-empty string")
    return value.strip()


def _normalized_semantic_text(value: Any, field: str) -> str:
    return " ".join(_text(value, field).casefold().split())


def _need_description(
    explicit_need: str | Mapping[str, Any] | None,
    *,
    context: Mapping[str, Any],
) -> str | None:
    need = explicit_need
    if need is None:
        need = context["proposal"].get("vqa_tool_need")
    if need is None:
        return None
    if isinstance(need, str):
        return _text(need, "vqa_need")
    if not isinstance(need, Mapping):
        raise OpenVQAQuestionError("vqa_need must be a string, mapping, or null")
    return _text(need.get("description"), "vqa_need.description")


def _is_semantic_miss(query: Mapping[str, Any]) -> bool:
    return any(
        str(reason).startswith(_FALLBACK_REASON_PREFIXES)
        for reason in query["selection_reasons"]
    )


def _catalog_query_exact_for_need(
    query: Mapping[str, Any],
    *,
    vqa_need: str,
) -> bool:
    """Do not call a catalog hit exact unless its question text is exact."""

    questions = query.get("questions")
    if not isinstance(questions, list) or len(questions) != 1:
        return False
    question = questions[0]
    return bool(
        isinstance(question, Mapping)
        and _normalized_semantic_text(
            question.get("question"),
            "catalog question",
        )
        == _normalized_semantic_text(vqa_need, "vqa_need")
    )


def _validate_registry_entry(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _REGISTRY_ENTRY_KEYS:
        raise OpenVQAQuestionError(
            "run-local VQA registry entry fields changed"
        )
    entry = deepcopy(dict(value))
    if entry.get("schema_version") != 1 or entry.get("kind") != "vqa_question":
        raise OpenVQAQuestionError(
            "run-local VQA registry entry version/kind is invalid"
        )
    task_name = _text(entry.get("task_name"), "registry task_name")
    vqa_need = _text(entry.get("vqa_need"), "registry vqa_need")
    expected_key = vqa_need_semantic_key(
        task_name=task_name,
        vqa_need=vqa_need,
    )
    if entry.get("semantic_key") != expected_key:
        raise OpenVQAQuestionError(
            "run-local VQA registry semantic key is invalid"
        )
    try:
        entry["question_spec"] = validate_run_local_question_spec(
            entry.get("question_spec")
        )
    except ExecutionVQAQueryError as exc:
        raise OpenVQAQuestionError(str(exc)) from exc
    for field in ("source_query", "semantic_concern"):
        entry[field] = _text(entry.get(field), f"registry {field}")
    artifact_path = entry.get("artifact_path")
    if artifact_path is not None:
        entry["artifact_path"] = _text(
            artifact_path,
            "registry artifact_path",
        )
    reuse_count = entry.get("reuse_count")
    if (
        isinstance(reuse_count, bool)
        or not isinstance(reuse_count, int)
        or reuse_count < 0
    ):
        raise OpenVQAQuestionError(
            "run-local VQA registry reuse_count must be non-negative"
        )
    return entry


def load_run_local_vqa_questions(
    registry_dir: str | Path,
) -> list[dict[str, Any]]:
    """Load validated questions generated earlier in this evaluation."""

    path = Path(registry_dir).expanduser().resolve() / "index.json"
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenVQAQuestionError(
            "run-local VQA registry index is invalid"
        ) from exc
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != 1
        or value.get("kind") != _REGISTRY_KIND
        or set(value) != {"schema_version", "kind", "entries"}
        or not isinstance(value.get("entries"), list)
    ):
        raise OpenVQAQuestionError(
            "run-local VQA registry envelope is invalid"
        )
    entries = [_validate_registry_entry(item) for item in value["entries"]]
    keys = [entry["semantic_key"] for entry in entries]
    if len(keys) != len(set(keys)):
        raise OpenVQAQuestionError(
            "run-local VQA registry contains duplicate semantic keys"
        )
    return entries


def register_run_local_vqa_question(
    registry_dir: str | Path,
    bundle: Mapping[str, Any],
    *,
    artifact_path: str | None,
) -> dict[str, Any]:
    """Register one generated question or record one exact run-local reuse."""

    context = validate_tool_artifact_context(bundle.get("artifact_context"))
    task_name = context["task_name"]
    vqa_need = _text(bundle.get("vqa_need"), "bundle vqa_need")
    semantic_key = vqa_need_semantic_key(
        task_name=task_name,
        vqa_need=vqa_need,
    )
    if bundle.get("semantic_key") != semantic_key:
        raise OpenVQAQuestionError(
            "VQA bundle semantic key differs from its typed need"
        )
    try:
        question_spec = validate_run_local_question_spec(
            bundle.get("question_spec")
        )
    except ExecutionVQAQueryError as exc:
        raise OpenVQAQuestionError(str(exc)) from exc
    proposal = context["proposal"]
    entry = {
        "schema_version": 1,
        "kind": "vqa_question",
        "semantic_key": semantic_key,
        "task_name": task_name,
        "vqa_need": vqa_need,
        "question_spec": question_spec,
        "source_query": _text(
            proposal.get("source_query"),
            "proposal source_query",
        ),
        "semantic_concern": _text(
            proposal.get("semantic_concern"),
            "proposal semantic_concern",
        ),
        "artifact_path": artifact_path,
        "reuse_count": 0,
    }
    registry = Path(registry_dir).expanduser().resolve()
    existing = load_run_local_vqa_questions(registry)
    match = next(
        (
            item
            for item in existing
            if item["semantic_key"] == semantic_key
        ),
        None,
    )
    if match is None:
        existing.append(_validate_registry_entry(entry))
        registered = existing[-1]
    else:
        if match["question_spec"] != question_spec:
            raise OpenVQAQuestionError(
                "exact VQA semantic key maps to conflicting question specs"
            )
        match["reuse_count"] += 1
        if artifact_path is not None:
            match["artifact_path"] = artifact_path
        registered = match
    registry.mkdir(parents=True, exist_ok=True)
    (registry / "index.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": _REGISTRY_KIND,
                "entries": existing,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return deepcopy(registered)


def _exact_reusable_question(
    context: Mapping[str, Any],
    *,
    vqa_need: str,
) -> dict[str, Any] | None:
    semantic_key = vqa_need_semantic_key(
        task_name=context["task_name"],
        vqa_need=vqa_need,
    )
    for raw_entry in context["reusable_artifacts"]["vqa_questions"]:
        entry = _validate_registry_entry(raw_entry)
        if entry["semantic_key"] == semantic_key:
            return entry
    return None


class OpenVQAQuestionAgent:
    """Retrieve an exact VQA contract or generate one strict question spec."""

    def __init__(self, provider: Any, *, model: str) -> None:
        self.provider = provider
        self.model = _text(model, "model")
        self.last_prompt: str | None = None
        self.last_responses: list[str] = []
        self.last_errors: list[str] = []

    @staticmethod
    def _prompt(
        *,
        context: Mapping[str, Any],
        vqa_need: str,
    ) -> str:
        example = {
            "schema_version": 1,
            "question_spec": {
                "id": "run_local.target_pose_changed",
                "question_type": "visible_state_change",
                "target_role": "target_object",
                "question": (
                    "Does the intended target visibly change pose during the "
                    "rollout?"
                ),
                "visual_scope": "rollout_change",
                "numeric_authority": "no_numeric_oracle",
            },
        }
        return (
            "You are the VQA Tool generator in ManipEvalAgent. Retrieval found "
            "no exact visual question for the Proposal. Generate one binary, "
            "directly observable question that addresses only the requested "
            "visual evidence need. Consume the exact Proposal, TaskArtifact "
            "authority summary, and executed runtime schema below. Do not "
            "invent a task template, actor, scene change, checker result, "
            "numeric threshold, or success authority. The VQA result is "
            "auxiliary evidence, so numeric_authority must be "
            "no_numeric_oracle. Return one strict JSON object with exactly "
            "schema_version and question_spec; question_spec must contain "
            "exactly id, question_type, target_role, question, visual_scope, "
            "and numeric_authority. The id must start with run_local.; the "
            "question must be one line, end with ?, and be answerable from "
            "rendered frames or the rollout montage. No prose or markdown.\n\n"
            f"VQA EVIDENCE NEED:\n{vqa_need}\n\n"
            "ALLOWED question_type VALUES:\n"
            + json.dumps(sorted(RUN_LOCAL_QUESTION_TYPES))
            + "\nALLOWED target_role VALUES:\n"
            + json.dumps(sorted(RUN_LOCAL_TARGET_ROLES))
            + "\nALLOWED visual_scope VALUES:\n"
            + json.dumps(sorted(RUN_LOCAL_VISUAL_SCOPES))
            + "\n\nTOOL ARTIFACT CONTEXT:\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n\nOUTPUT EXAMPLE:\n"
            + json.dumps(example, ensure_ascii=False, indent=2)
        )

    @staticmethod
    def _validate_response(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping) or set(value) != _PROVIDER_RESPONSE_KEYS:
            raise OpenVQAQuestionError(
                "provider VQA response fields must be exactly "
                f"{sorted(_PROVIDER_RESPONSE_KEYS)}"
            )
        if value.get("schema_version") != 1:
            raise OpenVQAQuestionError(
                "provider VQA response schema_version must be 1"
            )
        try:
            spec = validate_run_local_question_spec(value.get("question_spec"))
        except ExecutionVQAQueryError as exc:
            raise OpenVQAQuestionError(str(exc)) from exc
        if spec["id"] in QUESTION_CATALOG:
            raise OpenVQAQuestionError(
                "provider VQA question id collides with the trusted catalog"
            )
        if spec["numeric_authority"] != "no_numeric_oracle":
            raise OpenVQAQuestionError(
                "provider VQA question must declare no_numeric_oracle"
            )
        return spec

    def propose(
        self,
        *,
        artifact_context: Mapping[str, Any],
        vqa_need: str | Mapping[str, Any] | None = None,
        template_id: str | None = None,
        tool_contract: Mapping[str, Any] | None = None,
        reviewed_registry_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        context = validate_tool_artifact_context(artifact_context)
        need = _need_description(vqa_need, context=context)
        if need is None:
            return {
                "schema_version": 1,
                "status": "not_requested",
                "artifact_kind": "vqa_question",
                "source": "proposal_typed_need",
                "semantic_key": None,
                "vqa_need": None,
                "query": None,
                "question_spec": None,
                "artifact_context": context,
                "provider": {
                    "model_requested": self.model,
                    "called": False,
                    "attempt_count": 0,
                    "errors": [],
                    "last_metadata": {},
                },
            }

        proposal = context["proposal"]
        semantic_key = vqa_need_semantic_key(
            task_name=context["task_name"],
            vqa_need=need,
        )
        exact_reuse = _exact_reusable_question(
            context,
            vqa_need=need,
        )
        if exact_reuse is not None:
            try:
                query = build_execution_vqa_query(
                    task_name=context["task_name"],
                    template_id=template_id,
                    sub_aspect=proposal.get("semantic_concern"),
                    tool_contract=tool_contract,
                    proposed_phenomenon_ids=[
                        exact_reuse["question_spec"]["id"]
                    ],
                    proposed_question_specs=[
                        exact_reuse["question_spec"]
                    ],
                )
            except ExecutionVQAQueryError as exc:
                raise OpenVQAQuestionError(str(exc)) from exc
            return {
                "schema_version": 1,
                "status": "reused",
                "artifact_kind": "vqa_question",
                "source": "evaluation_local_exact_vqa_need_reuse",
                "semantic_key": semantic_key,
                "vqa_need": need,
                "query": query,
                "question_spec": deepcopy(exact_reuse["question_spec"]),
                "artifact_context": context,
                "matched_registration": exact_reuse,
                "provider": {
                    "model_requested": self.model,
                    "called": False,
                    "attempt_count": 0,
                    "errors": [],
                    "last_metadata": {},
                },
            }
        reviewed_reuse = None
        if reviewed_registry_dir is not None:
            try:
                reviewed_reuse = find_reviewed_generated_vqa_question(
                    reviewed_registry_dir,
                    task_name=context["task_name"],
                    vqa_need=need,
                )
            except ReviewedVQAQuerySpecError as exc:
                raise OpenVQAQuestionError(
                    f"invalid reviewed generated VQA registry: {exc}"
                ) from exc
        if reviewed_reuse is not None:
            question_spec = reviewed_reuse["question_spec"]
            try:
                query = build_execution_vqa_query(
                    task_name=context["task_name"],
                    template_id=template_id,
                    sub_aspect=proposal.get("semantic_concern"),
                    tool_contract=tool_contract,
                    proposed_phenomenon_ids=[question_spec["id"]],
                    proposed_question_specs=[question_spec],
                )
            except ExecutionVQAQueryError as exc:
                raise OpenVQAQuestionError(str(exc)) from exc
            return {
                "schema_version": 1,
                "status": "reused",
                "artifact_kind": "vqa_question",
                "source": "reviewed_persistent_exact_vqa_need_reuse",
                "semantic_key": semantic_key,
                "vqa_need": need,
                "query": query,
                "question_spec": deepcopy(question_spec),
                "artifact_context": context,
                "matched_registration": deepcopy(
                    reviewed_reuse["registration"]
                ),
                "validation": {
                    "scope": "reviewed_persistent",
                    "exact_semantic_need_match": True,
                    "question_spec_sha256": reviewed_reuse[
                        "question_spec_sha256"
                    ],
                    "review_sha256": reviewed_reuse["review_sha256"],
                    "current_rollout_vqa_execution_required": True,
                },
                "provider": {
                    "model_requested": self.model,
                    "called": False,
                    "attempt_count": 0,
                    "errors": [],
                    "last_metadata": {},
                },
            }
        try:
            retrieved = build_execution_vqa_query(
                task_name=context["task_name"],
                template_id=template_id,
                sub_aspect=proposal.get("semantic_concern"),
                tool_contract=tool_contract,
                reviewed_registry_dir=reviewed_registry_dir,
            )
        except ExecutionVQAQueryError as exc:
            raise OpenVQAQuestionError(str(exc)) from exc
        if (
            not _is_semantic_miss(retrieved)
            and _catalog_query_exact_for_need(
                retrieved,
                vqa_need=need,
            )
        ):
            return {
                "schema_version": 1,
                "status": "reused",
                "artifact_kind": "vqa_question",
                "source": "execution_vqa_exact_retrieval",
                "semantic_key": semantic_key,
                "vqa_need": need,
                "query": retrieved,
                "question_spec": None,
                "artifact_context": context,
                "provider": {
                    "model_requested": self.model,
                    "called": False,
                    "attempt_count": 0,
                    "errors": [],
                    "last_metadata": {},
                },
            }

        prompt = self._prompt(context=context, vqa_need=need)
        self.last_prompt = prompt
        self.last_responses = []
        self.last_errors = []
        question_spec: dict[str, Any] | None = None
        for _attempt in range(2):
            attempt_prompt = prompt
            if self.last_errors:
                attempt_prompt += (
                    "\n\nPREVIOUS VALIDATION ERROR:\n"
                    + self.last_errors[-1]
                    + "\nReturn one corrected complete JSON object. This is "
                    "the only repair attempt."
                )
            try:
                response = self.provider.text(
                    attempt_prompt,
                    model=self.model,
                    system="Return only strict VQA question-spec JSON.",
                    max_tokens=600,
                    temperature=0.0,
                )
                self.last_responses.append(response)
                question_spec = self._validate_response(
                    extract_json_response(response)
                )
                break
            except Exception as exc:
                self.last_errors.append(f"{type(exc).__name__}: {exc}")
        if question_spec is None:
            raise OpenVQAQuestionError(
                "provider failed the initial VQA question attempt and one "
                "repair: " + " | ".join(self.last_errors)
            )
        try:
            query = build_execution_vqa_query(
                task_name=context["task_name"],
                template_id=template_id,
                sub_aspect=proposal.get("semantic_concern"),
                tool_contract=tool_contract,
                proposed_phenomenon_ids=[question_spec["id"]],
                proposed_question_specs=[question_spec],
            )
        except ExecutionVQAQueryError as exc:
            raise OpenVQAQuestionError(str(exc)) from exc
        return {
            "schema_version": 1,
            "status": "generated",
            "artifact_kind": "vqa_question",
            "source": "provider_query_induced_vqa_question",
            "semantic_key": semantic_key,
            "vqa_need": need,
            "query": query,
            "question_spec": question_spec,
            "artifact_context": context,
            "provider": {
                "model_requested": self.model,
                "called": True,
                "attempt_count": len(self.last_responses),
                "errors": list(self.last_errors),
                "last_metadata": deepcopy(
                    dict(getattr(self.provider, "last_metadata", {}) or {})
                ),
            },
        }


def materialize_open_execution_vqa_query(
    *,
    provider: Any,
    model: str,
    artifact_context: Mapping[str, Any],
    vqa_need: str | Mapping[str, Any] | None = None,
    template_id: str | None = None,
    tool_contract: Mapping[str, Any] | None = None,
    reviewed_registry_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Functional API for the production RoundExecutor."""

    return OpenVQAQuestionAgent(provider, model=model).propose(
        artifact_context=artifact_context,
        vqa_need=vqa_need,
        template_id=template_id,
        tool_contract=tool_contract,
        reviewed_registry_dir=reviewed_registry_dir,
    )


__all__ = [
    "OpenVQAQuestionAgent",
    "OpenVQAQuestionError",
    "load_run_local_vqa_questions",
    "materialize_open_execution_vqa_query",
    "register_run_local_vqa_question",
    "vqa_need_semantic_key",
]
