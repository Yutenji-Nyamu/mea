"""Bounded visual self-reflection for generated RoboTwin task scenes."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from .prototype import (
    PROTECTED_PATHS,
    TaskGenError,
    build_generated_module,
    extract_load_actors,
    validate_load_actors,
)


class VisualReflectionError(RuntimeError):
    """Raised when visual self-reflection exhausts its repair budget."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def protected_hashes(repo_root: Path) -> dict[str, str]:
    return {
        relative: _sha256_bytes((repo_root / relative).read_bytes())
        for relative in PROTECTED_PATHS
    }


def expected_color_name(spec: dict[str, Any]) -> str:
    red, green, blue = spec["changes"]["block"]["color"]
    if blue >= 0.5 and blue > red and blue > green:
        return "blue"
    if red >= 0.5 and red > green and red > blue:
        return "red"
    if green >= 0.5 and green > red and green > blue:
        return "green"
    return "custom_rgb"


def validate_vision_observation(
    value: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VisualReflectionError("Vision observation 必须是 JSON object")
    observed = str(value.get("observed_color") or "unknown").strip().lower()
    expected = expected_color_name(spec)
    aliases = {
        "blue": {"blue", "蓝色", "蓝"},
        "red": {"red", "红色", "红"},
        "green": {"green", "绿色", "绿"},
    }
    color_matches = expected == "custom_rgb" or observed in aliases.get(expected, set())
    unexpected = value.get("unexpected_changes")
    if not isinstance(unexpected, list):
        unexpected = [str(unexpected)] if unexpected else []
    suggestions = value.get("suggestions")
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)] if suggestions else []
    confidence = value.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    result = {
        "aligned": bool(value.get("aligned")),
        "target_actor": str(value.get("target_actor") or "block"),
        "expected_color": expected,
        "observed_color": observed,
        "color_matches": color_matches,
        "unexpected_changes": [str(item) for item in unexpected],
        "diagnosis": str(value.get("diagnosis") or "").strip(),
        "suggestions": [str(item) for item in suggestions],
        "confidence": max(0.0, min(1.0, confidence)),
    }
    result["passed"] = bool(
        result["aligned"] and color_matches and not result["unexpected_changes"]
    )
    if not result["diagnosis"] and not result["passed"]:
        result["diagnosis"] = (
            f"Expected {expected} block alignment, observed color={observed}, "
            f"unexpected_changes={result['unexpected_changes']}."
        )
    return result


def validate_click_bell_vision_observation(value: Any) -> dict[str, Any]:
    """Validate a visual plausibility check; simulator state owns exact XY."""

    if not isinstance(value, dict):
        raise VisualReflectionError("click_bell Vision observation must be an object")
    if value.get("target_actor") != "bell":
        raise VisualReflectionError("click_bell target_actor must be 'bell'")
    for field in ("aligned", "bell_visible"):
        if not isinstance(value.get(field), bool):
            raise VisualReflectionError(f"click_bell {field} must be a JSON boolean")
    unexpected = value.get("unexpected_changes")
    if not isinstance(unexpected, list):
        raise VisualReflectionError("click_bell unexpected_changes must be a list")
    suggestions = value.get("suggestions")
    if not isinstance(suggestions, list):
        raise VisualReflectionError("click_bell suggestions must be a list")
    if isinstance(value.get("confidence"), bool):
        raise VisualReflectionError("click_bell confidence must be numeric")
    try:
        confidence = float(value.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise VisualReflectionError("click_bell confidence must be numeric") from exc
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise VisualReflectionError("click_bell confidence must be in [0, 1]")
    result = {
        "aligned": value["aligned"],
        "target_actor": "bell",
        "bell_visible": value["bell_visible"],
        "unexpected_changes": [str(item) for item in unexpected],
        "diagnosis": str(value.get("diagnosis") or "").strip(),
        "suggestions": [str(item) for item in suggestions],
        "confidence": confidence,
        "position_authority": "simulator_tracked_actor_xy",
    }
    result["passed"] = bool(
        result["aligned"]
        and result["bell_visible"]
        and not result["unexpected_changes"]
    )
    if not result["diagnosis"] and not result["passed"]:
        result["diagnosis"] = "Bell visibility or scene plausibility check failed."
    return result


def validate_bbh_distractor_vision_observation(
    value: Any,
) -> dict[str, Any]:
    """Validate only proposal-derived visual facts for the distractor scene.

    Actor identity, exact offset, contacts, and success remain simulator/fixture
    authorities.  The vision model only establishes that both intended objects
    are visible and that the initial scene is visually plausible.
    """

    if not isinstance(value, dict):
        raise VisualReflectionError(
            "BBH distractor vision observation must be an object"
        )
    for field in (
        "aligned",
        "target_visible",
        "lookalike_distractor_visible",
        "scene_physically_plausible",
    ):
        if not isinstance(value.get(field), bool):
            raise VisualReflectionError(
                f"BBH distractor vision {field} must be a JSON boolean"
            )
    unexpected = value.get("unexpected_changes")
    if not isinstance(unexpected, list):
        unexpected = [str(unexpected)] if unexpected else []
    suggestions = value.get("suggestions")
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)] if suggestions else []
    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise VisualReflectionError(
            "BBH distractor vision confidence must be finite and within [0, 1]"
        )
    result = {
        "aligned": value["aligned"],
        "target_actor": str(value.get("target_actor") or "block"),
        "target_visible": value["target_visible"],
        "lookalike_distractor_visible": value[
            "lookalike_distractor_visible"
        ],
        "scene_physically_plausible": value["scene_physically_plausible"],
        "unexpected_changes": [str(item) for item in unexpected],
        "diagnosis": str(value.get("diagnosis") or "").strip(),
        "suggestions": [str(item) for item in suggestions],
        "confidence": float(confidence),
        "authority_boundary": {
            "visual": [
                "target_visible",
                "lookalike_distractor_visible",
                "scene_physically_plausible",
            ],
            "simulator_or_fixture": [
                "actor_identity",
                "distractor_offset",
                "contact_latches",
                "success",
            ],
        },
    }
    result["passed"] = bool(
        result["aligned"]
        and result["target_visible"]
        and result["lookalike_distractor_visible"]
        and result["scene_physically_plausible"]
        and not result["unexpected_changes"]
    )
    if not result["diagnosis"] and not result["passed"]:
        result["diagnosis"] = (
            "The proposal-derived target/distractor scene failed one or more "
            "visual visibility or plausibility checks."
        )
    return result


def execute_reflection_loop(
    *,
    max_repairs: int,
    observe: Callable[[int], dict[str, Any]],
    repair: Callable[[int, dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Run observe/repair rounds without coupling the state machine to RoboTwin."""

    if max_repairs < 0 or max_repairs > 5:
        raise ValueError("max_repairs 必须在 0 到 5 之间")
    attempts: list[dict[str, Any]] = []
    for attempt_index in range(max_repairs + 1):
        observation = observe(attempt_index)
        attempt = {
            "attempt_index": attempt_index,
            "observation": observation,
        }
        attempts.append(attempt)
        if observation.get("passed"):
            return {
                "passed": True,
                "max_repairs": max_repairs,
                "repairs_used": attempt_index,
                "final_attempt": attempt_index,
                "attempts": attempts,
            }
        if attempt_index == max_repairs:
            break
        attempt["repair"] = repair(attempt_index + 1, observation)
    return {
        "passed": False,
        "max_repairs": max_repairs,
        "repairs_used": max_repairs,
        "final_attempt": max_repairs,
        "attempts": attempts,
        "failure_reason": "visual self-reflection repair budget exhausted",
    }


def _repair_prompt(
    user_request: str,
    spec: dict[str, Any],
    current_method: str,
    observation: dict[str, Any],
) -> str:
    return f"""You are the Visual Self-Reflection repair agent for RoboTwin TaskGen.

The current task rendered successfully or returned a structured probe error, but
the observation below did not align with the intended scene. Revise the complete
load_actors() method using only the evidence provided.

ORIGINAL USER REQUEST:
{user_request}

VALIDATED VARIANT SPEC:
{json.dumps(spec, ensure_ascii=False, indent=2)}

FAILED OBSERVATION AND DIAGNOSIS:
{json.dumps(observation, ensure_ascii=False, indent=2)}

CURRENT COMPLETE METHOD:
```python
{current_method}
```

OUTPUT CONTRACT:
1. Return exactly one Python code fence containing the complete
   def load_actors(self): method and nothing else.
2. Correct only diagnosed scene mismatches. Preserve official position/yaw
   sampling, hammer creation, actor names, mass, prohibited areas, and inheritance.
   A single sampled render cannot invalidate an official_random position/yaw
   distribution, so never change its sampling bounds from visual placement alone.
3. For scale=1.0 the BeatBlockHammer block must use half_size
   (0.025, 0.025, 0.025). For another scale, precompute the product and emit a
   literal numeric triple such as (0.02, 0.02, 0.02); arithmetic expressions in
   create_box(half_size=...) are forbidden by the static contract.
4. Use the literal RGB tuple from VariantSpec for create_box(color=...).
5. Do not call super(), import modules, access files/network/processes, or add
   any new task behavior.
"""


def install_repaired_method(
    repo_root: Path,
    run_dir: Path,
    method_source: str,
    spec: dict[str, Any],
    protected_before: dict[str, str],
) -> dict[str, Any]:
    ast_result = validate_load_actors(method_source, spec)
    hashes_after = protected_hashes(repo_root)
    if hashes_after != protected_before:
        raise TaskGenError("Visual repair 修改了受保护的官方文件")
    module_source = build_generated_module(method_source)
    compile(module_source, str(run_dir / "task.py"), "exec")

    temporary_task = run_dir / "task.py.repairing"
    temporary_method = run_dir / "generation/load_actors.py.repairing"
    temporary_task.write_text(module_source, encoding="utf-8")
    temporary_method.write_text(method_source, encoding="utf-8")
    temporary_task.replace(run_dir / "task.py")
    temporary_method.replace(run_dir / "generation/load_actors.py.txt")

    static_validation = {
        "variant_spec": {"valid": True},
        "load_actors_ast": ast_result,
        "protected_diff": {
            "valid": True,
            "hashes_after": hashes_after,
        },
    }
    (run_dir / "validation/static.json").write_text(
        json.dumps(static_validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return static_validation


def repair_generated_method(
    repo_root: Path,
    run_dir: Path,
    provider: Any,
    *,
    model: str,
    spec: dict[str, Any],
    observation: dict[str, Any],
    repair_index: int,
    protected_before: dict[str, str],
) -> dict[str, Any]:
    attempt_dir = run_dir / "reflection" / f"attempt_{repair_index - 1:02d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    current_method = (run_dir / "generation/load_actors.py.txt").read_text(
        encoding="utf-8"
    )
    (attempt_dir / "method_before_repair.py").write_text(
        current_method, encoding="utf-8"
    )
    request = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))[
        "user_request"
    ]
    prompt = _repair_prompt(request, spec, current_method, observation)
    (attempt_dir / "repair_prompt.md").write_text(prompt, encoding="utf-8")
    response = provider.text(
        prompt,
        model=model,
        system=(
            "Return exactly one Python code fence containing the corrected complete "
            "load_actors(self) method."
        ),
        max_tokens=4096,
        temperature=0.0,
    )
    (attempt_dir / "repair_response.txt").write_text(
        response + "\n", encoding="utf-8"
    )
    revised_method = extract_load_actors(response)
    (attempt_dir / "candidate_load_actors.py").write_text(
        revised_method, encoding="utf-8"
    )
    static_validation = install_repaired_method(
        repo_root,
        run_dir,
        revised_method,
        spec,
        protected_before,
    )
    result = {
        "repair_index": repair_index,
        "method_sha256_before": _sha256_bytes(current_method.encode("utf-8")),
        "method_sha256_after": _sha256_bytes(revised_method.encode("utf-8")),
        "static_validation": static_validation,
        "provider_metadata": dict(getattr(provider, "last_metadata", {})),
        "installed": True,
    }
    (attempt_dir / "repair.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def inject_oversized_block_fixture(
    repo_root: Path,
    run_dir: Path,
    spec: dict[str, Any],
    protected_before: dict[str, str],
) -> dict[str, Any]:
    """Inject a test-only visual mismatch that the existing AST gate permits."""

    source = (run_dir / "generation/load_actors.py.txt").read_text(encoding="utf-8")
    tree = ast.parse(source)
    changed = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name != "create_box":
            continue
        for keyword in node.keywords:
            if keyword.arg == "half_size":
                keyword.value = ast.Tuple(
                    elts=[ast.Constant(0.06), ast.Constant(0.06), ast.Constant(0.06)],
                    ctx=ast.Load(),
                )
                changed = True
    if not changed:
        raise TaskGenError("oversized_block fixture 找不到 create_box half_size")
    ast.fix_missing_locations(tree)
    injected = ast.unparse(tree).strip() + "\n"

    fixture_dir = run_dir / "reflection/fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "method_before_fixture.py").write_text(source, encoding="utf-8")
    (fixture_dir / "oversized_load_actors.py").write_text(injected, encoding="utf-8")
    # As with the wrong-color fixture below, validate the deliberately injected
    # method against a fixture-only contract.  The committed VariantSpec remains
    # unchanged, so the visual gate still observes and repairs the mismatch.
    fixture_spec = copy.deepcopy(spec)
    fixture_spec["changes"]["block"]["scale"] = 2.4
    static_validation = install_repaired_method(
        repo_root,
        run_dir,
        injected,
        fixture_spec,
        protected_before,
    )
    result = {
        "fixture": "oversized_block",
        "test_only": True,
        "injected_half_size": [0.06, 0.06, 0.06],
        "expected_half_size": [0.025, 0.025, 0.025],
        "static_gate_still_passes": bool(
            static_validation["load_actors_ast"]["valid"]
        ),
    }
    (fixture_dir / "fixture.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def inject_wrong_color_fixture(
    repo_root: Path,
    run_dir: Path,
    spec: dict[str, Any],
    protected_before: dict[str, str],
) -> dict[str, Any]:
    """Inject a test-only red/blue mismatch after the normal static gate."""

    source = (run_dir / "generation/load_actors.py.txt").read_text(encoding="utf-8")
    tree = ast.parse(source)
    changed = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else None
        if name != "create_box":
            continue
        for keyword in node.keywords:
            if keyword.arg == "color":
                keyword.value = ast.Tuple(
                    elts=[ast.Constant(1.0), ast.Constant(0.0), ast.Constant(0.0)],
                    ctx=ast.Load(),
                )
                changed = True
    if not changed:
        raise TaskGenError("wrong_color fixture 找不到 create_box color")
    ast.fix_missing_locations(tree)
    injected = ast.unparse(tree).strip() + "\n"

    fixture_spec = copy.deepcopy(spec)
    fixture_spec["changes"]["block"]["color"] = [1.0, 0.0, 0.0]
    fixture_static = validate_load_actors(injected, fixture_spec)
    hashes_after = protected_hashes(repo_root)
    if hashes_after != protected_before:
        raise TaskGenError("wrong_color fixture 检测到 protected-file 变化")
    module_source = build_generated_module(injected)
    compile(module_source, str(run_dir / "task.py"), "exec")

    fixture_dir = run_dir / "reflection/fixture"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    (fixture_dir / "method_before_fixture.py").write_text(source, encoding="utf-8")
    (fixture_dir / "wrong_color_load_actors.py").write_text(
        injected, encoding="utf-8"
    )
    (run_dir / "task.py").write_text(module_source, encoding="utf-8")
    (run_dir / "generation/load_actors.py.txt").write_text(
        injected, encoding="utf-8"
    )
    result = {
        "fixture": "wrong_color",
        "test_only": True,
        "injected_color": [1.0, 0.0, 0.0],
        "expected_color": spec["changes"]["block"]["color"],
        "injected_method_structurally_valid": bool(fixture_static["valid"]),
        "injected_after_normal_static_gate": True,
    }
    (fixture_dir / "fixture.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result
