"""Paper-aligned paths for Plan Agent artifacts.

New evaluations write only the canonical names below.  Readers may use
``resolve_plan_artifact`` with a historical relative path so immutable
ClaimFirst/FreeConcern bundles remain readable without being rewritten.
"""

from __future__ import annotations

from pathlib import Path


QUERY_INTERPRETATION = Path("plan/query_interpretation.json")
QUERY_INTERPRETATION_PROMPT = Path("plan/query_interpretation_prompt.md")
QUERY_INTERPRETATION_RESPONSE_PREFIX = "query_interpretation_response_"
PLAN_AGENT_CAPABILITIES = Path("plan/plan_agent_capabilities.json")
PLAN_AGENT_SESSION = Path("plan/plan_agent_session")
PLAN_AGENT_STEPS = Path("plan/plan_agent_steps")
INITIAL_SUB_ASPECT_PROPOSAL = Path("plan/initial_sub_aspect_proposal")
PROPOSAL_MATERIALIZATION = Path("plan/proposal_materialization")
PROPOSAL_FILENAME = "proposal.json"


def resolve_plan_artifact(
    evaluation_dir: str | Path,
    canonical: str | Path,
    *historical: str | Path,
) -> Path:
    """Return the first existing canonical or immutable historical artifact."""

    root = Path(evaluation_dir)
    candidates = [
        root / Path(canonical),
        *(root / Path(item) for item in historical),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


__all__ = [
    "INITIAL_SUB_ASPECT_PROPOSAL",
    "PLAN_AGENT_CAPABILITIES",
    "PLAN_AGENT_SESSION",
    "PLAN_AGENT_STEPS",
    "PROPOSAL_FILENAME",
    "PROPOSAL_MATERIALIZATION",
    "QUERY_INTERPRETATION",
    "QUERY_INTERPRETATION_PROMPT",
    "QUERY_INTERPRETATION_RESPONSE_PREFIX",
    "resolve_plan_artifact",
]
