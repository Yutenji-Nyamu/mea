# README.Agent: bounded adaptive evaluation planning

This document describes the capabilities and constraints visible to the outer
Plan Agent. The model selects semantic template identifiers only. Trusted
runtime code injects task instructions, seeds, gates, TaskGen routes, and Tool
requests; the model must never invent these fields.

## Policy and simulator

- Policy: ACT; canonical task: `beat_block_hammer`.
- Checkpoint: `demo_clean`, trained with 50 expert demonstrations.
- The policy is not language-conditioned in this evaluation.
- Official block position samples `x` from `[-0.25, 0.25]` and `y` from
  `[-0.05, 0.15]`, rejecting `abs(x) < 0.05`; official yaw is approximately
  `[-0.5, 0.5]` radians.
- Pipeline completion and policy success are separate. `policy_success=0` is
  valid evidence and does not mean that generation or execution failed.

## Trusted sub-aspect templates

The current prototype exposes exactly three template ids:

1. `object_appearance.color_blue`: one episode at validated seed `100000`;
   isolates the blue block appearance.
2. `object_position.official_random`: two episodes at validated seeds `100002`
   and `100003`; preserves blue appearance and evaluates official position/yaw
   samples.
3. `performance.pickup_to_contact_timing`: one episode at validated seed
   `100000`; measures elapsed simulator time from first hammer pickup threshold
   crossing to first strict hammer-block physical contact.

The model initially emits only `requested_template_ids` and
`first_template_id`. It must select only aspects explicitly requested by the
user. The runtime materializes the complete first round from the catalog.

## Tool planning boundary

- A template contains a semantic `tool_request`, never a Tool route.
- Runtime resolves exact Trusted Tool matches to `reuse`, exact registered
  composite targets to `force_codegen`, and all other metrics to `unsupported`.
- `hammer_block_contact_ever` is a Trusted Tool.
- `pickup_to_first_contact_time` is a composition-validated generated target;
  missing pickup/contact yields `null`.
- Expert telemetry validates instrumentation and scene solvability. It is not
  evidence that ACT achieved the same result.

## Adaptive protocol

- `max_rounds` is exactly 3, but the evaluation may stop earlier.
- After every executed round, the Plan Agent receives the complete observation
  history and chooses either `continue` with one unexecuted requested template,
  or `stop`.
- A template can run at most once. The model cannot select a template omitted
  from `requested_template_ids`.
- Latest `pipeline_passed=false`, an exhausted round budget, or no remaining
  requested template forces `stop`.
- Otherwise the model must continue to one remaining user-requested template;
  observations may determine the order, but cannot erase an explicit request.
- Every executable round runs ordered gates `ast`, `render`, `rule`, `vision`,
  `expert`, and `act`.

Return only the strict JSON object requested by the active prompt.
