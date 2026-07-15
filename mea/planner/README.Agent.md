# README.Agent: bounded multi-round evaluation planning

This document describes the capabilities and constraints visible to the outer
Plan Agent in the current MEA prototype.

## Policy and simulator

- Policy: ACT; canonical task: `beat_block_hammer`.
- Checkpoint: `demo_clean`, trained with 50 expert demonstrations.
- The policy is not language-conditioned in this evaluation.
- The official block position samples `x` from `[-0.25, 0.25]` and `y` from
  `[-0.05, 0.15]`, rejecting `abs(x) < 0.05`.
- Official yaw is sampled within approximately `[-0.5, 0.5]` radians.
- `num_episodes` is an evaluation count, not the per-episode step limit.

## Available validated routes

1. `object_appearance.color`: use `force_codegen` so TaskGen generates the
   complete `load_actors()` implementation. The validated blue color is
   `[0.0, 0.2, 1.0]`.
2. `object_position`: use the trusted `reuse` route, keep the block blue, and
   use official position/yaw sampling. Simulator probes return exact
   `block_pose` values for every seed.

Every round runs the ordered gates `ast`, `render`, `rule`, `vision`, `expert`,
and `act`. Pipeline completion and policy task success are separate signals.
A `policy_success=0` result is still a valid observation and does not imply a
generation-pipeline failure.

## Offline Tool planning

- Round 1 proposes `pickup_to_first_contact_time` with `force_codegen`. Pickup
  means the first 250 Hz sample where hammer Z has risen by the schema's
  `0.03 m` threshold; the target is elapsed simulator time to first strict
  physical hammer-block contact. Missing pickup/contact yields `null`.
- This timing metric is intentionally absent from the Trusted Tool catalog.
  Runtime validates generated code against a private composition of the
  verified first-pickup and first-contact primitives on ACT/expert telemetry.
- Round 2 proposes `hammer_block_contact_ever` with `reuse` and therefore calls
  the verified Trusted Tool without invoking GPT.
- The Plan Agent declares semantics only. Runtime code resolves telemetry paths,
  policy/expert roles, reference values, generated filenames, and artifacts.
- Expert contact validates instrumentation and scene solvability; it is never
  evidence that ACT made contact.

## Multi-round protocol

- Initial planning proposes Round 1 only: blue block, seed `100000`, 1 episode.
- After Round 1, the Plan Agent receives actual scene, VQA, expert, ACT, and
  policy observations.
- If the pipeline passed, propose Round 2: preserve blue, official position
  randomization, expert-solvable seeds `100002` and `100003`, 2 episodes.
- Every evaluation seed must independently pass the expert gate. Seed `100001`
  is excluded because its sampled edge pose failed the official expert planner.
- If the pipeline failed, stop and explain the failure rather than executing a
  second round blindly.
- After execution, aggregate both round-level results and produce one final
  evidence-grounded response.

Return only the strict JSON object requested by the active prompt.
