"""Independent semantic review for provider-generated TaskGen checkers."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Protocol

from mea.providers.json_response import extract_json_response


class CheckerSemanticReviewError(RuntimeError):
    """Raised when a generated checker fails its Proposal-level review."""

    def __init__(self, message: str, *, provider_calls: int = 0) -> None:
        super().__init__(message)
        self.provider_calls = provider_calls


class CheckerSemanticReviewUnavailableError(CheckerSemanticReviewError):
    """Raised when the independent review itself is unavailable or malformed."""


class TextProvider(Protocol):
    def text(self, prompt: str, **kwargs: Any) -> str:
        ...


_CHECKS = {
    "implements_every_checker_requirement",
    "preserves_quantifiers_and_temporal_relations",
    "uses_direct_current_simulator_observables",
    "does_not_substitute_correlated_proxy",
}
_RAW_FIELDS = {"schema_version", "status", "checks", "reason"}
_BINDING_FIELDS = {
    "authority",
    "proposal_semantics_sha256",
    "checker_sha256",
    "official_checker_sha256",
}
_ROBOTWIN_RUNTIME_API_AUTHORITY = """\
- `self.scene.get_contacts()` returns current PhysX contacts.
- Each contact body exposes its scene entity as `.entity`.
- A RoboTwin Actor wrapper exposes that same scene entity as `.actor`.
- `self.robot.left_gripper` and `self.robot.right_gripper` contain gripper
  joints; each joint tuple's `[0].child_link` is the corresponding gripper
  link entity.
These are established read-only RoboTwin runtime APIs. Do not reject their
use as unsupported; review whether the code uses them with the exact actor,
side, simultaneity, and object identities required by the Proposal."""


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CheckerSemanticReviewError(
            f"semantic-review identity is not canonical JSON: {exc}"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def checker_review_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Keep TaskGen semantics while excluding Query wording and Tool needs."""

    intent = candidate.get("evaluation_intent")
    return {
        "base_task": candidate.get("base_task"),
        "semantic_concern": candidate.get("semantic_concern"),
        "scene_need": deepcopy(candidate.get("scene_need")),
        "checker_need": deepcopy(candidate.get("checker_need")),
        "evaluation_intent": (
            {
                field: deepcopy(intent.get(field))
                for field in (
                    "original_concern",
                    "hypothesis",
                    "requested_change",
                    "preserved_conditions",
                    "required_observation",
                )
            }
            if isinstance(intent, Mapping)
            else None
        ),
    }


def checker_review_identity_sha256(
    candidate: Mapping[str, Any],
) -> str:
    return _canonical_sha256(checker_review_identity(candidate))


def _validate_response(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckerSemanticReviewError(
            "TaskGen checker semantic review must be an object"
        )
    review = deepcopy(dict(value))
    checks = review.get("checks")
    if (
        set(review) != _RAW_FIELDS
        or review.get("schema_version") != 1
        or review.get("status") not in {"approved", "rejected"}
        or not isinstance(checks, Mapping)
        or set(checks) != _CHECKS
        or any(type(item) is not bool for item in checks.values())
        or not isinstance(review.get("reason"), str)
        or not review["reason"].strip()
    ):
        raise CheckerSemanticReviewError(
            "TaskGen checker semantic review contract is invalid"
        )
    review["checks"] = dict(checks)
    review["reason"] = review["reason"].strip()
    return review


def validate_checker_semantic_review_binding(
    value: Any,
    *,
    candidate: Mapping[str, Any],
    checker_sha256: str,
) -> dict[str, Any]:
    """Fail closed when an approved review is stale or not Proposal-bound."""

    if not isinstance(value, Mapping):
        raise CheckerSemanticReviewError(
            "generated checker lacks a semantic review"
        )
    review = deepcopy(dict(value))
    if set(review) != _RAW_FIELDS | _BINDING_FIELDS:
        raise CheckerSemanticReviewError(
            "generated checker semantic review binding is malformed"
        )
    normalized = _validate_response(
        {field: review[field] for field in _RAW_FIELDS}
    )
    if (
        normalized["status"] != "approved"
        or not all(normalized["checks"].values())
        or review.get("authority") != "development_agent_proxy"
        or review.get("proposal_semantics_sha256")
        != checker_review_identity_sha256(candidate)
        or review.get("checker_sha256") != checker_sha256
        or not isinstance(review.get("official_checker_sha256"), str)
        or len(review["official_checker_sha256"]) != 64
    ):
        raise CheckerSemanticReviewError(
            "generated checker semantic review is rejected or stale"
        )
    return review


def _prompt(
    *,
    candidate: Mapping[str, Any],
    task_context: Any,
    method_provenance: Mapping[str, str],
    generated_scene: str,
    official_checker: str,
    generated_checker: str,
) -> str:
    return f"""You are TaskGen's separate checker semantic-review pass.
Review the generated check_success method against the frozen Proposal.
You are a development-agent proxy, not independent human gold.

Approve only if every checker_need requirement is directly implemented from
current simulator state. Preserve all quantifiers, simultaneity, temporal
relations, object identities, and corresponding-pair relations literally.
Do not accept a correlated proxy: for example, a closed gripper is not proof
of target contact, height is not proof of placement, and eventual contacts are
not simultaneous contacts. The unchanged official checker may be composed as
a required conjunct, but it cannot stand in for the added condition.
If the exact requested predicate is unavailable from the supplied runtime API,
reject it; do not weaken or reinterpret the Proposal. Do not rewrite code.

FROZEN PROPOSAL:
{json.dumps(candidate, ensure_ascii=False, sort_keys=True, indent=2)}

TASK CONTEXT:
{json.dumps(task_context, ensure_ascii=False, sort_keys=True, indent=2)}

SUPPORTED ROBOTWIN CHECKER API:
{_ROBOTWIN_RUNTIME_API_AUTHORITY}

METHOD PROVENANCE:
{json.dumps(method_provenance, ensure_ascii=False, sort_keys=True, indent=2)}

EFFECTIVE LOAD_ACTORS:
```python
{generated_scene.rstrip()}
```

OFFICIAL CHECK_SUCCESS:
```python
{official_checker.rstrip()}
```

GENERATED CHECK_SUCCESS:
```python
{generated_checker.rstrip()}
```

Return strict JSON with exactly:
{{
  "schema_version": 1,
  "status": "approved" or "rejected",
  "checks": {{
    "implements_every_checker_requirement": true or false,
    "preserves_quantifiers_and_temporal_relations": true or false,
    "uses_direct_current_simulator_observables": true or false,
    "does_not_substitute_correlated_proxy": true or false
  }},
  "reason": "one concise sentence"
}}
"""


def review_generated_checker(
    *,
    provider: TextProvider,
    model: str,
    candidate: Mapping[str, Any],
    task_context: Any,
    method_provenance: Mapping[str, str],
    generated_scene: str,
    official_checker: str,
    generated_checker: str,
    attempt_dir: Path,
) -> dict[str, Any]:
    """Run and persist one independent review before simulator preflight."""

    prompt = _prompt(
        candidate=candidate,
        task_context=task_context,
        method_provenance=method_provenance,
        generated_scene=generated_scene,
        official_checker=official_checker,
        generated_checker=generated_checker,
    )
    (attempt_dir / "checker_semantic_review_prompt.md").write_text(
        prompt,
        encoding="utf-8",
    )
    try:
        response = provider.text(
            prompt,
            model=model,
            system="Return only strict TaskGen checker semantic-review JSON.",
            max_tokens=600,
            temperature=0.0,
        )
        (attempt_dir / "checker_semantic_review_response.txt").write_text(
            str(response) + "\n",
            encoding="utf-8",
        )
        raw_review = _validate_response(
            extract_json_response(str(response))
        )
    except Exception as exc:
        raise CheckerSemanticReviewUnavailableError(
            "TaskGen checker semantic review unavailable: "
            f"{type(exc).__name__}: {exc}",
            provider_calls=1,
        ) from exc
    review = {
        **raw_review,
        "authority": "development_agent_proxy",
        "proposal_semantics_sha256": (
            checker_review_identity_sha256(candidate)
        ),
        "checker_sha256": _text_sha256(generated_checker),
        "official_checker_sha256": _text_sha256(official_checker),
    }
    (attempt_dir / "checker_semantic_review.json").write_text(
        json.dumps(
            review,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if review["status"] != "approved" or not all(
        review["checks"].values()
    ):
        raise CheckerSemanticReviewError(
            "generated checker failed semantic review: "
            + review["reason"],
            provider_calls=1,
        )
    return review


__all__ = [
    "CheckerSemanticReviewError",
    "CheckerSemanticReviewUnavailableError",
    "checker_review_identity",
    "checker_review_identity_sha256",
    "review_generated_checker",
    "validate_checker_semantic_review_binding",
]
