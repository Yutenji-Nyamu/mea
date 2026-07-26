"""Provider-written, state-compatible LIBERO BDDL generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from mea.taskgen import extract_json_response

from .benchmark import (
    LiberoBenchmarkAdapter,
    LiberoContractError,
    TaskContract,
    build_task_contract,
    validate_phase1_bddl,
)


class LiberoTaskGenBackend:
    """Generate one BDDL edit, validate it, and permit at most one repair."""

    def __init__(self, provider: Any, *, model: str):
        self.provider = provider
        self.model = model

    @staticmethod
    def prompt(
        *,
        user_query: str,
        proposal: Mapping[str, Any],
        base_bddl: str,
    ) -> str:
        return f"""You are TaskGen for a minimal LIBERO experiment.
Write a complete BDDL problem derived from the official source below.

The planner proposal was produced only after an official control rollout:
{json.dumps(dict(proposal), ensure_ascii=False, indent=2)}

Phase-1 compatibility contract:
- Keep the problem name/domain, fixtures, regions, objects, and every :init
  predicate exactly unchanged.
- Change only :language, :obj_of_interest, and :goal.
- Select exactly one existing non-basket object other than alphabet_soup_1.
- The sole goal must be `(In <selected_object> basket_1_contain_region)`.
- The language and obj_of_interest must name the same selected object.
- Do not add objects, regions, comments, code, or Markdown.

Original Query:
{user_query}

OFFICIAL BDDL:
{base_bddl}

Return strict JSON with exactly:
{{"bddl_text":"the complete BDDL text","selected_object":"object_id","rationale":"short reason"}}
"""

    def generate(
        self,
        *,
        user_query: str,
        proposal_bundle: Mapping[str, Any],
        output_dir: str | Path,
        seed: int,
    ) -> tuple[TaskContract, dict[str, Any]]:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        base_path, init_path = LiberoBenchmarkAdapter.official_paths()
        base_text = base_path.read_text(encoding="utf-8")
        proposal = proposal_bundle.get("proposal", proposal_bundle)
        perturbation = proposal.get("requested_perturbation")
        controlled_changes = (
            perturbation.get("controlled_changes")
            if isinstance(perturbation, Mapping)
            else None
        )
        if not isinstance(controlled_changes, list) or not any(
            isinstance(item, str)
            and ("object" in item.casefold() or "goal" in item.casefold())
            for item in controlled_changes
        ):
            raise LiberoContractError(
                "Phase-1 LIBERO TaskGen supports an explicit object/goal identity "
                "change; the Planner requested a different controlled change"
            )
        prompt = self.prompt(
            user_query=user_query,
            proposal=proposal,
            base_bddl=base_text,
        )
        (output / "prompt.md").write_text(prompt, encoding="utf-8")

        responses: list[str] = []
        errors: list[str] = []
        candidate_path = output / "generated_task.bddl"
        parsed_candidate: dict[str, Any] | None = None
        checks: dict[str, Any] | None = None
        selected_object: str | None = None
        for attempt in range(2):
            attempt_prompt = prompt
            if errors:
                attempt_prompt += (
                    "\nThe previous output failed validation:\n"
                    + errors[-1]
                    + "\nReturn one corrected complete JSON object."
                )
            response = self.provider.text(
                attempt_prompt,
                model=self.model,
                system="Return only strict JSON containing a complete LIBERO BDDL file.",
                max_tokens=2600,
                temperature=0.0,
            )
            responses.append(response)
            (output / f"response_attempt_{attempt + 1}.txt").write_text(
                response, encoding="utf-8"
            )
            try:
                payload = extract_json_response(response)
                if set(payload) != {"bddl_text", "selected_object", "rationale"}:
                    raise LiberoContractError("TaskGen response fields are not exact")
                if not all(isinstance(payload[key], str) and payload[key].strip() for key in payload):
                    raise LiberoContractError("TaskGen response values must be non-empty strings")
                candidate_path.write_text(payload["bddl_text"].strip() + "\n", encoding="utf-8")
                _base, parsed_candidate, checks = validate_phase1_bddl(
                    base_path=base_path,
                    candidate_path=candidate_path,
                )
                goal_object = parsed_candidate["goal_state"][0][1]
                if payload["selected_object"] != goal_object:
                    raise LiberoContractError("selected_object disagrees with parsed goal")
                selected_object = goal_object
                break
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if parsed_candidate is None or checks is None or selected_object is None:
            raise LiberoContractError(
                "provider BDDL failed initial attempt plus one repair: " + " | ".join(errors)
            )

        proposal_path = output / "planner_proposal.json"
        proposal_path.write_text(
            json.dumps(dict(proposal_bundle), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        contract = build_task_contract(
            candidate_path=candidate_path,
            candidate=parsed_candidate,
            checks=checks,
            official_init_state_path=init_path,
            source_query=user_query,
            proposal_artifact=str(proposal_path),
        )
        contract_path = output / "task_contract.json"
        contract_path.write_text(
            json.dumps(contract.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {
            "schema_version": 1,
            "status": "passed",
            "experiment_seed": int(seed),
            "generation": {
                "source": "provider_written_bddl",
                "model_requested": self.model,
                "attempt_count": len(responses),
                "repair_count": max(0, len(responses) - 1),
                "provider_metadata": dict(getattr(self.provider, "last_metadata", {})),
            },
            "selected_object": selected_object,
            "planner_taskgen_alignment": True,
            "checks": checks,
            "artifacts": {
                "prompt": str(output / "prompt.md"),
                "responses": [
                    str(output / f"response_attempt_{index + 1}.txt")
                    for index in range(len(responses))
                ],
                "bddl": str(candidate_path),
                "task_contract": str(contract_path),
                "planner_proposal": str(proposal_path),
            },
        }
        (output / "taskgen_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return contract, result


def run_libero_taskgen_cli(args: Any) -> None:
    """Early CLI dispatch used by scripts/manipeval_taskgen.py."""

    from mea.providers import OpenAICompatibleProvider

    if not args.request:
        raise SystemExit("--request is required for --benchmark libero")
    run_id = args.run_id or "libero_taskgen"
    output = (
        Path(args.repo_root).expanduser().resolve()
        / "mea"
        / "evaluation_runs"
        / run_id
        / "libero"
        / "taskgen"
    )
    if not args.task_proposal_json:
        raise SystemExit(
            "--benchmark libero requires a planner-owned --task-proposal-json"
        )
    proposal: Mapping[str, Any] = json.loads(args.task_proposal_json)
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        text_model=args.text_model,
        max_retries=1,
    )
    _contract, result = LiberoTaskGenBackend(
        provider, model=args.text_model
    ).generate(
        user_query=args.request,
        proposal_bundle=proposal,
        output_dir=output,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
