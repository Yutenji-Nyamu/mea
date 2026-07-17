# README.Agent: bounded adaptive evaluation planning

This document describes the capabilities and constraints visible to the outer
Plan Agent. The model selects semantic template identifiers only. Trusted
runtime code injects task instructions, seeds, gates, TaskGen routes, and Tool
requests; the model must never invent these fields.

## Global query boundary

`scripts/manipeval_agent.py --auto-route` first builds a trusted ACT catalog.
A task is model-visible only when its TaskSchema, `dataset_stats.pkl`, and
`policy_last.ckpt` are present.  The global model may select only a catalog
task/profile/aspect and may return explicit `unsupported`; it never outputs a
path, module, checkpoint payload, seed, gate, Tool route, or variant body.

The current global catalog has two task adapters:

- `beat_block_hammer / generated`: the three BBH templates below;
- `click_bell / adaptive_properties`: `object_position` and
  `object_instance`, each backed by two trusted variants.

The validated global selection is translated directly into the existing task
planner proposal, so the task planner does not call a second initial model.
After execution, all adaptive task planners obey the shared deterministic
conditional-transition contract: pipeline failure stops; valid policy failure
drills into the same aspect; valid success switches to an uncovered requested
aspect; no target or exhausted budget stops.

## Policy and simulator

- Policy: ACT; this task-adapter appendix describes `beat_block_hammer`.
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

Up to three similar completed evaluations may be included as planning priors.
They preserve prior query-to-template decompositions and policy labels. They
are not current-run evidence and must never be merged into the current
Aggregate or used to claim current policy success.

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
  history plus a deterministic `EvidenceAssessment` and explains one of
  `continue`, `verify`, or `stop`.
- `evidence_conflict` or incomplete/invalid Aggregate coverage forces one
  same-template `verify` round with a fresh trusted seed when budget remains.
  A template may be verified at most once; arbitrary duplicate execution is
  forbidden.
- Clear positive and clear negative Tool results are both sufficient evidence.
  A documented semantic absence such as `contact_not_observed_after_pickup`
  is not treated as missing telemetry.
- Pipeline failure, an unresolved second conflict/uncertainty, or exhausted
  budget forces `stop`. Sufficient evidence continues only to an unexecuted
  user-requested template; after all requested aspects are covered it stops.
- The model cannot override the deterministic required action, invent a
  verification seed, or select a template omitted from the user request.
- Every executable round runs ordered gates `ast`, `render`, `rule`, `vision`,
  `expert`, and `act`.

Return only the strict JSON object requested by the active prompt.
