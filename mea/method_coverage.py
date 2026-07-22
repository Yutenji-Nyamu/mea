"""Executable top-down audit of the paper-method reproduction boundary.

The audit distinguishes source support from runtime evidence.  Claim status is
derived from AST/source checks and validated JSON artifacts; it is never a
manually asserted success label.
"""

from __future__ import annotations

import ast
import binascii
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


STATUS_IMPLEMENTED = "implemented"
STATUS_PARTIAL = "partial"
STATUS_EVIDENCE_PENDING = "evidence_pending"
VALID_STATUSES = {
    STATUS_IMPLEMENTED,
    STATUS_PARTIAL,
    STATUS_EVIDENCE_PENDING,
}


@dataclass(frozen=True)
class CodeRequirement:
    check_id: str
    kind: str
    path: str | None = None
    symbol: str | None = None
    value: str | None = None
    values: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceRequirement:
    check_id: str
    patterns: tuple[str, ...]
    validator: str


@dataclass(frozen=True)
class ClaimSpec:
    rank: int
    claim_id: str
    title: str
    paper_anchor: str
    code: tuple[CodeRequirement, ...]
    evidence: tuple[EvidenceRequirement, ...] = ()


QUERY_GOLD_UNSUPPORTED_AXES = frozenset(
    {
        "camera_viewpoint",
        "conclusion.multi_task_consistency",
        "language.paraphrase_consistency",
        "object_appearance.material_gloss",
        "object_appearance.texture",
        "object_physics.mass",
        "occlusion.target_contact",
        "performance.motion_smoothness",
        "performance.path_efficiency",
        "safety.boundary_clearance",
        "safety.unintended_contact",
    }
)


CLAIMS: tuple[ClaimSpec, ...] = (
    ClaimSpec(
        1,
        "plan_evidence_loop",
        "Evidence-driven dynamic sub-aspect Plan Agent",
        "Sec. 3.2; Figs. 2 and 5",
        (
            CodeRequirement(
                "adaptive_step_agent",
                "symbol",
                "mea/planner/adaptive_step.py",
                "AdaptivePlanStepAgent.propose",
            ),
            CodeRequirement(
                "session_navigation_options",
                "symbol",
                "mea/planner/session.py",
                "BoundTaskPlanSession.navigation_options",
            ),
            CodeRequirement(
                "session_applies_model_step",
                "symbol",
                "mea/planner/session.py",
                "BoundTaskPlanSession.apply_plan_step",
            ),
            CodeRequirement(
                "adaptive_step_inside_runtime_loop",
                "call_inside_loop",
                "scripts/manipeval_agent.py",
                value="apply_plan_step",
            ),
        ),
    ),
    ClaimSpec(
        2,
        "proposal_every_round",
        "Runtime supports observation-conditioned Task/Tool Proposal each round",
        "Figs. 2-4",
        (
            CodeRequirement(
                "bounded_proposal_agent",
                "symbol",
                "mea/proposal_agent.py",
                "BoundedProposalAgent.propose",
            ),
            CodeRequirement(
                "common_proposal_transition_inside_runtime_loop",
                "call_inside_loop",
                "scripts/manipeval_agent.py",
                value="apply_bounded_round_proposal",
            ),
        ),
    ),
    ClaimSpec(
        3,
        "task_agnostic_adapter",
        "Declarative task capability adapter",
        "Fig. 2; Sec. 3.3",
        (
            CodeRequirement(
                "resolve_capability",
                "symbol",
                "mea/capability_adapter.py",
                "resolve_capability_contract",
            ),
            CodeRequirement(
                "validate_capability",
                "symbol",
                "mea/capability_adapter.py",
                "validate_capability_contract",
            ),
            CodeRequirement(
                "registered_templates",
                "symbol",
                "mea/capability_adapter.py",
                "registered_templates",
            ),
        ),
    ),
    ClaimSpec(
        4,
        "complete_task_codegen",
        "Complete executable task scene and success code generation",
        "Sec. 3.3.1; Fig. 3",
        (
            CodeRequirement(
                "generated_module_builder",
                "symbol",
                "mea/taskgen/prototype.py",
                "build_generated_module",
            ),
            CodeRequirement(
                "generated_scene_and_success",
                "function_literals",
                "mea/taskgen/prototype.py",
                "build_generated_module",
                values=("load_actors", "check_success"),
            ),
            CodeRequirement(
                "success_spec_validation",
                "symbol",
                "mea/taskgen/success_spec.py",
                "validate_success_spec",
            ),
            CodeRequirement(
                "success_spec_compiler",
                "symbol",
                "mea/taskgen/success_spec.py",
                "compile_success_spec",
            ),
            CodeRequirement(
                "compiled_success_method",
                "function_literals",
                "mea/taskgen/success_spec.py",
                "_render_bbh_success_method",
                values=("check_success", "check_actors_contact"),
            ),
        ),
    ),
    ClaimSpec(
        5,
        "generated_metric_toolgen",
        "Proposal-conditioned generated metric Tool",
        "Sec. 3.3.2; Fig. 4",
        (
            CodeRequirement(
                "toolgen_generate",
                "symbol",
                "mea/toolgen/prototype.py",
                "ToolGenPrototype.generate",
            ),
            CodeRequirement(
                "tool_static_validation",
                "symbol",
                "mea/toolgen/prototype.py",
                "validate_generated_tool",
            ),
            CodeRequirement(
                "tool_execution",
                "symbol",
                "mea/toolgen/prototype.py",
                "execute_generated_tool",
            ),
        ),
    ),
    ClaimSpec(
        6,
        "policy_simulator_cards",
        "Structured policy and simulator context cards",
        "Sec. 3.2",
        (
            CodeRequirement(
                "planning_context_builder",
                "symbol",
                "mea/planner/context.py",
                "build_planning_context",
            ),
            CodeRequirement(
                "planning_context_validator",
                "symbol",
                "mea/planner/context.py",
                "validate_planning_context",
            ),
            CodeRequirement(
                "planning_context_consumed_by_initial_router",
                "text",
                "scripts/manipeval_agent.py",
                value="planning_contexts=global_planning_contexts",
            ),
        ),
    ),
    ClaimSpec(
        7,
        "typed_evidence_packet",
        "Typed scalar and visual evidence packet",
        "Sec. 3.2; App. A.3.5",
        (
            CodeRequirement(
                "evidence_packet_builder",
                "symbol",
                "mea/planner/evidence_policy.py",
                "build_evidence_packet",
            ),
            CodeRequirement(
                "evidence_packet_validator",
                "symbol",
                "mea/planner/evidence_policy.py",
                "validate_evidence_packet",
            ),
        ),
    ),
    ClaimSpec(
        8,
        "task_asset_document_rag",
        "Reuse-first task and documentation retrieval",
        "Sec. 3.3.1; Fig. 3",
        (
            CodeRequirement(
                "task_catalog",
                "symbol",
                "mea/retrieval/task_library.py",
                "discover_task_catalog",
            ),
            CodeRequirement(
                "task_retriever",
                "symbol",
                "mea/retrieval/task_library.py",
                "TaskRetriever",
            ),
            CodeRequirement(
                "tool_examples",
                "symbol",
                "mea/toolgen/prototype.py",
                "retrieve_examples",
            ),
        ),
    ),
    ClaimSpec(
        9,
        "visual_diagnosis_repair",
        "Rendered scene visual diagnosis and bounded repair",
        "Sec. 3.3.1; Fig. 3; App. A.3.4",
        (
            CodeRequirement(
                "reflection_loop",
                "symbol",
                "mea/taskgen/reflection.py",
                "execute_reflection_loop",
            ),
            CodeRequirement(
                "repair_generated_method",
                "symbol",
                "mea/taskgen/reflection.py",
                "repair_generated_method",
            ),
        ),
    ),
    ClaimSpec(
        10,
        "semantic_history_reuse",
        "Canonical aspect-aware historical plan reuse",
        "App. A.3.3",
        (
            CodeRequirement(
                "history_requested_aspects",
                "parameter",
                "mea/history/database.py",
                "EvaluationHistoryDB.retrieve_similar",
                value="requested_aspect_ids",
            ),
            CodeRequirement(
                "history_aspect_similarity",
                "symbol",
                "mea/history/database.py",
                "_aspect_similarity",
            ),
        ),
    ),
    ClaimSpec(
        11,
        "persistent_tool_reuse",
        "Reviewed Tool persistence across evaluations",
        "Sec. 3.3.2; App. A.3.3",
        (
            CodeRequirement(
                "install_reviewed_tool",
                "symbol",
                "mea/toolgen/reviewed_registry.py",
                "install_reviewed_registration",
            ),
            CodeRequirement(
                "find_reviewed_tool",
                "symbol",
                "mea/toolgen/reviewed_registry.py",
                "find_reviewed_registration",
            ),
        ),
    ),
    ClaimSpec(
        12,
        "taxonomy_unsupported_boundary",
        "Canonical aspect taxonomy and explicit unsupported boundary",
        "Sec. 3.2; App. A.4",
        (
            CodeRequirement(
                "query_gold_taxonomy",
                "mapping_keys",
                "mea/aspects.py",
                "_ASPECT_ONTOLOGY",
                values=tuple(sorted(QUERY_GOLD_UNSUPPORTED_AXES)),
            ),
            CodeRequirement(
                "route_validation",
                "symbol",
                "mea/planner/global_query.py",
                "validate_route_selection",
            ),
        ),
    ),
    ClaimSpec(
        13,
        "readme_agent_freshness",
        "README.Agent knowledge extraction and freshness hashes",
        "Sec. 3.3.1; Fig. 3",
        (
            CodeRequirement(
                "knowledge_index",
                "symbol",
                "mea/knowledge/extractor.py",
                "build_knowledge_index_data",
            ),
            CodeRequirement(
                "symbol_source",
                "symbol",
                "mea/knowledge/extractor.py",
                "source_symbol_text",
            ),
        ),
    ),
    ClaimSpec(
        14,
        "stage_recovery_resume",
        "Stage-aware recovery and resumable evaluation protocol",
        "App. A.3.4",
        (
            CodeRequirement(
                "stage_recovery",
                "symbol",
                "mea/round_recovery.py",
                "run_stage_aware_round_recovery",
            ),
            CodeRequirement(
                "protocol_runner",
                "symbol",
                "scripts/manipeval_protocol.py",
                "run_protocol",
            ),
            CodeRequirement(
                "resume_cli",
                "text",
                "scripts/manipeval_protocol.py",
                value="--resume-run",
            ),
        ),
    ),
    ClaimSpec(
        15,
        "live_run_local_vqa",
        "Run-local Dynamic Execution VQA on rollout keyframes",
        "App. A.3.5",
        (
            CodeRequirement(
                "run_local_question",
                "symbol",
                "mea/execution_vqa/query.py",
                "validate_run_local_question_spec",
            ),
            CodeRequirement(
                "execution_vqa",
                "symbol",
                "mea/execution_vqa/prototype.py",
                "run_execution_vqa",
            ),
        ),
        (
            EvidenceRequirement(
                "live_run_local_vqa_artifact",
                (
                    "mea/evaluation_runs/**/execution_vqa/execution_vqa.json",
                    "mea/validation_runs/**/execution_vqa.json",
                ),
                "run_local_vqa",
            ),
        ),
    ),
    ClaimSpec(
        16,
        "matched_fixed_adaptive_protocol",
        "Matched ACT fixed versus adaptive protocol",
        "Tables 1-2 mechanism evidence",
        (
            CodeRequirement(
                "strategy_comparator",
                "symbol",
                "mea/strategy_comparison.py",
                "compare_fixed_dynamic",
            ),
        ),
        (
            EvidenceRequirement(
                "matched_strategy_artifact",
                (
                    "mea/validation_runs/**/comparison/summary.json",
                    "mea/validation_runs/**/*strategy*comparison*.json",
                ),
                "matched_strategy",
            ),
        ),
    ),
)


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path escapes repository root: {relative}")
    return path


def _read_tree(root: Path, relative: str) -> tuple[ast.Module | None, str | None]:
    path = _safe_path(root, relative)
    if not path.is_file():
        return None, "source file is missing"
    try:
        return ast.parse(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, SyntaxError) as exc:
        return None, f"source is unreadable: {type(exc).__name__}: {exc}"


def _resolve_symbol(tree: ast.AST, dotted: str) -> ast.AST | None:
    parts = dotted.split(".")
    scope = list(getattr(tree, "body", []))
    current: ast.AST | None = None
    for part in parts:
        current = next(
            (
                node
                for node in scope
                if isinstance(
                    node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                )
                and node.name == part
            ),
            None,
        )
        if current is None:
            return None
        scope = list(getattr(current, "body", []))
    return current


def _call_tail(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _check_code(root: Path, requirement: CodeRequirement) -> dict[str, Any]:
    base = {
        "check_id": requirement.check_id,
        "kind": requirement.kind,
        "path": requirement.path,
        "symbol": requirement.symbol,
    }
    if requirement.kind == "repo_symbol":
        matches: list[str] = []
        for path in sorted((root / "mea").rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, SyntaxError):
                continue
            if any(
                isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == requirement.symbol
                for node in ast.walk(tree)
            ):
                matches.append(path.relative_to(root).as_posix())
        return {
            **base,
            "passed": bool(matches),
            "detail": (
                f"found in {matches[0]}" if matches else "repository symbol is missing"
            ),
        }

    if requirement.path is None:
        return {**base, "passed": False, "detail": "check has no source path"}
    path = _safe_path(root, requirement.path)
    if requirement.kind == "text":
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return {
                **base,
                "passed": False,
                "detail": f"source is unreadable: {type(exc).__name__}: {exc}",
            }
        passed = bool(requirement.value and requirement.value in source)
        return {
            **base,
            "passed": passed,
            "detail": "required CLI/source marker found"
            if passed
            else "marker missing",
        }

    tree, error = _read_tree(root, requirement.path)
    if tree is None:
        return {**base, "passed": False, "detail": error}
    if requirement.kind == "symbol":
        node = _resolve_symbol(tree, str(requirement.symbol))
        return {
            **base,
            "passed": node is not None,
            "detail": "AST symbol found" if node is not None else "AST symbol missing",
        }
    if requirement.kind == "parameter":
        node = _resolve_symbol(tree, str(requirement.symbol))
        arguments = (
            [argument.arg for argument in node.args.args + node.args.kwonlyargs]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            else []
        )
        passed = requirement.value in arguments
        return {
            **base,
            "passed": passed,
            "detail": (
                f"parameter {requirement.value!r} found"
                if passed
                else f"parameter {requirement.value!r} missing"
            ),
        }
    if requirement.kind == "call_inside_loop":
        matches = 0
        for loop in (
            node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))
        ):
            matches += sum(
                isinstance(node, ast.Call)
                and _call_tail(node.func) == requirement.value
                for node in ast.walk(loop)
            )
        return {
            **base,
            "passed": matches > 0,
            "detail": f"matching calls inside loops: {matches}",
        }
    if requirement.kind == "function_literals":
        node = _resolve_symbol(tree, str(requirement.symbol))
        constants = (
            "\n".join(
                str(item.value)
                for item in ast.walk(node)
                if isinstance(item, ast.Constant) and isinstance(item.value, str)
            )
            if node is not None
            else ""
        )
        missing = [value for value in requirement.values if value not in constants]
        return {
            **base,
            "passed": not missing,
            "detail": (
                "required generated methods are present"
                if not missing
                else f"generated method markers missing: {missing}"
            ),
        }
    if requirement.kind == "mapping_keys":
        mapping: Mapping[Any, Any] | None = None
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                target = (
                    node.target
                    if isinstance(node, ast.AnnAssign)
                    else node.targets[0]
                    if len(node.targets) == 1
                    else None
                )
                if isinstance(target, ast.Name) and target.id == requirement.symbol:
                    try:
                        value = ast.literal_eval(node.value)
                    except (TypeError, ValueError):
                        value = None
                    if isinstance(value, Mapping):
                        mapping = value
                    break
        missing = sorted(set(requirement.values) - set(mapping or {}))
        return {
            **base,
            "passed": not missing,
            "detail": "required ontology ids found"
            if not missing
            else f"missing ids: {missing}",
        }
    return {**base, "passed": False, "detail": "unknown code check kind"}


def _artifact_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    return path if path.is_relative_to(root) else None


def _read_json_object(path: Path) -> Mapping[str, Any] | None:
    """Read a bounded JSON object used only as evidence provenance."""

    if not path.is_file() or path.stat().st_size > 5 * 1024 * 1024:
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, Mapping) else None


def _validated_png_dimensions(path: Path) -> tuple[int, int] | None:
    """Validate PNG chunk framing/CRCs and return the IHDR dimensions.

    The method audit should not accept a few bytes named ``*.png`` as visual
    evidence.  This deliberately stays dependency-free so the 0-ACT audit can
    run outside the RoboTwin image environment.
    """

    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 57 or data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    offset = 8
    dimensions: tuple[int, int] | None = None
    saw_idat = False
    saw_iend = False
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        end = offset + 12 + length
        if end > len(data):
            return None
        chunk_type = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length : end])[0]
        if binascii.crc32(chunk_type + payload) & 0xFFFFFFFF != expected_crc:
            return None
        if chunk_type == b"IHDR":
            if dimensions is not None or length != 13:
                return None
            width, height = struct.unpack(">II", payload[:8])
            dimensions = (width, height)
        elif chunk_type == b"IDAT":
            saw_idat = saw_idat or bool(payload)
        elif chunk_type == b"IEND":
            saw_iend = length == 0
            offset = end
            break
        offset = end
    if offset != len(data) or not (dimensions and saw_idat and saw_iend):
        return None
    return dimensions


def validate_run_local_vqa_evidence(
    repo_root: str | Path, artifact_path: str | Path, value: Mapping[str, Any]
) -> tuple[bool, str]:
    root = Path(repo_root).expanduser().resolve()
    if value.get("status") != "passed":
        return False, "VQA status is not passed"
    model = value.get("model_requested")
    if not isinstance(model, str) or not model.strip():
        return False, "VQA has no live model identity"
    provider = value.get("provider_metadata")
    if not isinstance(provider, Mapping):
        return False, "VQA has no provider response metadata"
    provider_id = provider.get("id")
    served_model = provider.get("model")
    usage = provider.get("usage")
    if not isinstance(provider_id, str) or not provider_id.strip():
        return False, "VQA provider response id is missing"
    if not isinstance(served_model, str) or not served_model.strip():
        return False, "VQA served-model identity is missing"
    if not isinstance(usage, Mapping):
        return False, "VQA provider token usage is missing"
    total_tokens = usage.get("total_tokens")
    if (
        isinstance(total_tokens, bool)
        or not isinstance(total_tokens, int)
        or total_tokens <= 0
    ):
        return False, "VQA provider token usage is invalid"
    query = value.get("query")
    if not isinstance(query, Mapping):
        return False, "VQA query is missing"
    questions = query.get("questions")
    if not isinstance(questions, list):
        return False, "VQA questions are missing"
    local_ids = {
        item.get("id")
        for item in questions
        if isinstance(item, Mapping)
        and isinstance(item.get("id"), str)
        and item["id"].startswith("run_local.")
    }
    if not local_ids:
        return False, "query contains no run-local question"
    observation = value.get("observation")
    phenomena = (
        observation.get("phenomena") if isinstance(observation, Mapping) else None
    )
    if not isinstance(phenomena, list):
        return False, "VQA observation phenomena are missing"
    observed = {
        item.get("id"): item.get("observed")
        for item in phenomena
        if isinstance(item, Mapping)
    }
    if not local_ids.issubset(observed):
        return False, "run-local questions lack returned phenomena"
    if any(
        observed[item] is not None and not isinstance(observed[item], bool)
        for item in local_ids
    ):
        return False, "run-local observed values are not boolean-or-null"

    representative = _artifact_path(root, value.get("representative_episode"))
    if representative is None or not representative.is_dir():
        return False, "representative rollout episode is missing"
    episode = _read_json_object(representative / "episode.json")
    if episode is None:
        return False, "representative episode manifest is missing or invalid"
    if str(episode.get("policy_name", "")).casefold() not in {"act", "expert"}:
        return False, "representative episode has no policy provenance"
    if not isinstance(episode.get("success"), bool):
        return False, "representative episode success is not explicit"
    video_path = representative / "video.mp4"
    if not video_path.is_file() or video_path.stat().st_size <= 0:
        return False, "representative rollout video is missing or empty"

    selection = value.get("selection")
    if not isinstance(selection, Mapping):
        return False, "VQA keyframe selection is missing"
    selected_video = _artifact_path(root, selection.get("video_path"))
    if selected_video is None or selected_video != video_path.resolve():
        return False, "keyframes are not bound to the representative rollout video"
    frame_count = selection.get("frame_count")
    fps = selection.get("fps")
    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count <= 0
        or isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or fps <= 0
    ):
        return False, "keyframe selection has invalid video metadata"
    selected_frames = selection.get("selected_frames")
    if not isinstance(selected_frames, list) or not 2 <= len(selected_frames) <= 8:
        return False, "VQA selected frames are missing or invalid"
    frame_ids: set[str] = set()
    frame_indices: set[int] = set()
    for frame in selected_frames:
        if not isinstance(frame, Mapping):
            return False, "VQA selected frame is not an object"
        frame_id = frame.get("frame_id")
        frame_index = frame.get("frame_index")
        source = frame.get("source")
        if (
            not isinstance(frame_id, str)
            or not frame_id
            or frame_id in frame_ids
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not 0 <= frame_index < frame_count
            or frame_index in frame_indices
            or not isinstance(source, str)
            or not source
        ):
            return False, "VQA selected frame provenance is invalid"
        frame_ids.add(frame_id)
        frame_indices.add(frame_index)
    observation_frame_ids = observation.get("frame_ids")
    if not isinstance(observation_frame_ids, list) or any(
        not isinstance(item, str) or item not in frame_ids
        for item in observation_frame_ids
    ):
        return False, "VQA observation references unknown keyframes"

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return False, "VQA audit artifacts are missing"
    montage = artifacts.get("montage")
    montage_path = _artifact_path(root, montage)
    dimensions = (
        _validated_png_dimensions(montage_path) if montage_path is not None else None
    )
    if dimensions is None or min(dimensions) < 16:
        return False, "VQA montage is not a valid non-trivial PNG"
    selected_montage = _artifact_path(root, selection.get("montage_path"))
    if selected_montage != montage_path:
        return False, "VQA montage and selection provenance disagree"
    selection_path = _artifact_path(root, artifacts.get("selection"))
    if selection_path is None or _read_json_object(selection_path) != selection:
        return False, "saved keyframe selection is missing or differs"
    query_path = _artifact_path(root, artifacts.get("query"))
    if query_path is None or _read_json_object(query_path) != query:
        return False, "saved VQA query is missing or differs"
    result_path = _artifact_path(root, artifacts.get("result"))
    if result_path is None or result_path != Path(artifact_path).resolve():
        return False, "VQA result artifact does not identify itself"
    for name in ("prompt", "response"):
        path = _artifact_path(root, artifacts.get(name))
        if path is None or not path.is_file() or path.stat().st_size <= 0:
            return False, f"VQA {name} artifact is missing or empty"
    return True, (
        f"validated provider-backed run-local VQA for {sorted(local_ids)} "
        f"on {representative.relative_to(root).as_posix()}"
    )


def validate_matched_strategy_evidence(
    repo_root: str | Path, artifact_path: str | Path, value: Mapping[str, Any]
) -> tuple[bool, str]:
    root = Path(repo_root).expanduser().resolve()
    if value.get("status") != "passed":
        return False, "registered comparison status is not passed"
    if value.get("registered_identity_match") is not True:
        return False, "registered identity was not positively validated"
    registered_evidence = value.get("evidence")
    if (
        not isinstance(registered_evidence, Mapping)
        or not isinstance(registered_evidence.get("registration_id"), str)
        or not registered_evidence["registration_id"].strip()
    ):
        return False, "registered evidence identity is missing"
    registration_id = registered_evidence["registration_id"]
    comparison = value.get("comparison")
    if not isinstance(comparison, Mapping):
        return False, "registered comparison payload is missing"
    protocol = comparison.get("protocol")
    if not isinstance(protocol, str) or "fixed_vs_dynamic" not in protocol:
        return False, "artifact is not a fixed-versus-dynamic protocol"
    identity = comparison.get("identity")
    policy = identity.get("policy") if isinstance(identity, Mapping) else None
    if (
        not isinstance(policy, Mapping)
        or str(policy.get("name", "")).casefold() != "act"
    ):
        return False, "comparison is not ACT-only"
    strategies = comparison.get("strategies")
    if not isinstance(strategies, Mapping):
        return False, "comparison strategies are missing"
    required = ("fixed_predeclared_v1", "dynamic_evidence_v1")
    sample_maps: dict[str, set[tuple[str, int]]] = {}
    for name in required:
        run = strategies.get(name)
        if not isinstance(run, Mapping):
            return False, f"strategy {name} is missing"
        samples = run.get("samples")
        totals = run.get("totals")
        if (
            not isinstance(samples, list)
            or not samples
            or not isinstance(totals, Mapping)
        ):
            return False, f"strategy {name} has no completed samples"
        if totals.get("act_rollouts") != len(samples):
            return False, f"strategy {name} ACT count does not match samples"

        evaluation_dir = _artifact_path(root, run.get("evaluation_dir"))
        if evaluation_dir is None:
            return False, f"strategy {name} evaluation directory is invalid"
        manifest = _read_json_object(evaluation_dir / "manifest.json")
        summary = _read_json_object(evaluation_dir / "summary/summary.json")
        if manifest is None or summary is None:
            return False, f"strategy {name} evaluation manifest/summary is missing"
        evaluation_id = run.get("evaluation_id")
        if (
            not isinstance(evaluation_id, str)
            or not evaluation_id
            or manifest.get("evaluation_id") != evaluation_id
            or evaluation_dir.name != evaluation_id
        ):
            return False, f"strategy {name} evaluation identity is invalid"
        if manifest.get("lifecycle_status") != "completed" or manifest.get(
            "status"
        ) not in {"completed", "completed_with_pipeline_failure"}:
            return False, f"strategy {name} evaluation is not complete"
        if summary.get("status") != "completed":
            return False, f"strategy {name} summary is not complete"
        registration = run.get("registration_identity")
        if (
            not isinstance(registration, Mapping)
            or not registration
            or manifest.get("registration_identity") != registration
            or registration.get("registration_id") != registration_id
            or registration.get("strategy") != name
            or registration.get("expected_evaluation_id") != evaluation_id
        ):
            return False, f"strategy {name} registration identity is invalid"
        manifest_plan = manifest.get("plan")
        manifest_policy = (
            manifest_plan.get("policy") if isinstance(manifest_plan, Mapping) else None
        )
        manifest_planner = manifest.get("planner")
        planning_policy = manifest.get("planning_policy") or (
            manifest_planner.get("planning_policy")
            if isinstance(manifest_planner, Mapping)
            else None
        )
        if (
            not isinstance(manifest_policy, Mapping)
            or str(manifest_policy.get("name", "")).casefold() != "act"
            or manifest_policy != policy
            or planning_policy != name
        ):
            return False, f"strategy {name} manifest is not the registered ACT run"
        if (
            manifest.get("task_name") != identity.get("task_name")
            or run.get("task_name") != identity.get("task_name")
        ):
            return False, f"strategy {name} task identity is invalid"
        rounds = summary.get("rounds")
        if not isinstance(rounds, list) or not rounds:
            return False, f"strategy {name} has no completed round provenance"

        identities: set[tuple[str, int]] = set()
        for sample in samples:
            if not isinstance(sample, Mapping):
                return False, f"strategy {name} contains an invalid sample"
            variant = sample.get("variant_id")
            seed = sample.get("seed")
            if (
                not isinstance(variant, str)
                or isinstance(seed, bool)
                or not isinstance(seed, int)
            ):
                return False, f"strategy {name} sample identity is invalid"
            if not isinstance(sample.get("success"), bool):
                return False, f"strategy {name} sample success is not boolean"

            episode_path = _artifact_path(root, sample.get("episode"))
            if episode_path is None or episode_path.name != "episode.json":
                return False, f"strategy {name} sample has no ACT artifact path"
            try:
                relative_episode = episode_path.relative_to(root)
            except ValueError:
                return False, f"strategy {name} sample ACT path escapes repository"
            parts = relative_episode.parts
            if (
                len(parts) < 7
                or parts[0:2] != ("mea", "generated_tasks")
                or parts[3:5] != ("evaluation", "telemetry")
            ):
                return False, f"strategy {name} sample ACT path has invalid provenance"
            child_id = parts[2]
            expected_prefix = registration.get("expected_child_run_prefix")
            if (
                not isinstance(expected_prefix, str)
                or not expected_prefix
                or not child_id.startswith(expected_prefix)
            ):
                return False, f"strategy {name} child registration prefix is invalid"
            child_dir = root / "mea/generated_tasks" / child_id
            child_manifest = _read_json_object(child_dir / "manifest.json")
            episode = _read_json_object(episode_path)
            if (
                child_manifest is None
                or child_manifest.get("run_id") != child_id
                or child_manifest.get("status") != "completed"
                or child_manifest.get("registration_identity") != registration
            ):
                return False, f"strategy {name} child run provenance is invalid"
            if (
                episode is None
                or str(episode.get("policy_name", "")).casefold() != "act"
                or episode.get("registration_identity") != registration
                or episode.get("task_name") != identity.get("task_name")
                or episode.get("seed") != seed
                or episode.get("success") != sample.get("success")
            ):
                return False, f"strategy {name} ACT episode provenance is invalid"
            episode_dir = "/".join(parts[5:-1])
            trusted = child_manifest.get("trusted_tool_evaluation")
            trusted_episodes = (
                trusted.get("episodes") if isinstance(trusted, Mapping) else None
            )
            if not isinstance(trusted_episodes, list) or not any(
                isinstance(item, Mapping)
                and item.get("episode_dir") == episode_dir
                and str(item.get("policy_name", "")).casefold() == "act"
                and item.get("seed") == seed
                for item in trusted_episodes
            ):
                return False, f"strategy {name} child does not cite the ACT episode"
            if not any(
                isinstance(item, Mapping)
                and item.get("pipeline_passed") is True
                and item.get("variant_id") == variant
                and item.get("taskgen_run_id") == child_id
                for item in rounds
            ):
                return False, f"strategy {name} sample is not traced by a round"
            identities.add((variant, seed))
        if len(identities) != len(samples):
            return False, f"strategy {name} contains duplicate samples"
        sample_maps[name] = identities
    if not sample_maps["dynamic_evidence_v1"].issubset(
        sample_maps["fixed_predeclared_v1"]
    ):
        return False, "dynamic samples are outside the fixed candidate suite"
    overlap = comparison.get("overlap")
    if not isinstance(overlap, list) or len(overlap) != len(
        sample_maps["dynamic_evidence_v1"]
    ):
        return False, "overlap rows do not cover dynamic samples"
    return True, "validated matched ACT fixed-versus-dynamic evidence"


_EVIDENCE_VALIDATORS: dict[
    str, Callable[[str | Path, str | Path, Mapping[str, Any]], tuple[bool, str]]
] = {
    "run_local_vqa": validate_run_local_vqa_evidence,
    "matched_strategy": validate_matched_strategy_evidence,
}


def _check_evidence(root: Path, requirement: EvidenceRequirement) -> dict[str, Any]:
    candidates = sorted(
        {
            path.resolve()
            for pattern in requirement.patterns
            for path in root.glob(pattern)
            if path.is_file() and path.resolve().is_relative_to(root)
        }
    )
    validator = _EVIDENCE_VALIDATORS[requirement.validator]
    issues: list[str] = []
    for path in candidates:
        if path.stat().st_size > 5 * 1024 * 1024:
            issues.append(f"{path.relative_to(root)}: JSON artifact exceeds 5 MiB")
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            issues.append(f"{path.relative_to(root)}: {type(exc).__name__}")
            continue
        if not isinstance(value, Mapping):
            issues.append(f"{path.relative_to(root)}: root is not an object")
            continue
        passed, detail = validator(root, path, value)
        if passed:
            return {
                "check_id": requirement.check_id,
                "validator": requirement.validator,
                "passed": True,
                "artifact": path.relative_to(root).as_posix(),
                "candidate_count": len(candidates),
                "detail": detail,
            }
        issues.append(f"{path.relative_to(root)}: {detail}")
    return {
        "check_id": requirement.check_id,
        "validator": requirement.validator,
        "passed": False,
        "artifact": None,
        "candidate_count": len(candidates),
        "detail": issues[:5] or ["no candidate artifact found"],
    }


def build_method_coverage_report(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"repository root is missing: {root}")
    claims: list[dict[str, Any]] = []
    for spec in CLAIMS:
        code_checks = [_check_code(root, item) for item in spec.code]
        evidence_checks = [_check_evidence(root, item) for item in spec.evidence]
        code_ready = bool(code_checks) and all(item["passed"] for item in code_checks)
        evidence_ready = all(item["passed"] for item in evidence_checks)
        status = (
            STATUS_PARTIAL
            if not code_ready
            else STATUS_EVIDENCE_PENDING
            if evidence_checks and not evidence_ready
            else STATUS_IMPLEMENTED
        )
        claims.append(
            {
                "rank": spec.rank,
                "claim_id": spec.claim_id,
                "title": spec.title,
                "paper_anchor": spec.paper_anchor,
                "status": status,
                "code_ready": code_ready,
                "evidence_required": bool(evidence_checks),
                "evidence_ready": evidence_ready,
                "code_checks": code_checks,
                "evidence_checks": evidence_checks,
            }
        )
    counts = {
        status: sum(item["status"] == status for item in claims)
        for status in sorted(VALID_STATUSES)
    }
    return {
        "schema_version": 1,
        "protocol": "mea_paper_method_coverage_v1",
        "scope": "paper_method_functionality_not_full_statistical_reproduction",
        "claim_count": len(claims),
        "counts": counts,
        "claims": claims,
        "limitations": [
            "Static source checks demonstrate interfaces, not semantic correctness.",
            "Only claims with explicit evidence requirements may become evidence_pending.",
            "A passed N=1 artifact is mechanism evidence, not a paper-scale result.",
            "The audit reads existing artifacts and starts no provider, simulator, or ACT rollout.",
        ],
    }


def render_method_coverage_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# MEA Paper-Method Coverage",
        "",
        f"- scope: `{report.get('scope')}`",
        f"- claims: `{report.get('claim_count')}`",
        "",
        "| Rank | Claim | Paper anchor | Derived status |",
        "|---:|---|---|---|",
    ]
    for item in report.get("claims") or []:
        lines.append(
            f"| {item.get('rank')} | {item.get('title')} | "
            f"{item.get('paper_anchor')} | `{item.get('status')}` |"
        )
    lines.extend(["", "Statuses are derived from the attached checks.", ""])
    return "\n".join(lines)


__all__ = [
    "CLAIMS",
    "QUERY_GOLD_UNSUPPORTED_AXES",
    "STATUS_EVIDENCE_PENDING",
    "STATUS_IMPLEMENTED",
    "STATUS_PARTIAL",
    "VALID_STATUSES",
    "build_method_coverage_report",
    "render_method_coverage_markdown",
    "validate_matched_strategy_evidence",
    "validate_run_local_vqa_evidence",
]
