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

- Each round must also propose the exact `hammer_block_contact_ever` ToolSpec
  shown in the active prompt. It asks whether hammer and block had strict
  physical contact in the recorded trajectory.
- Round 1 uses `tool_spec.route=force_codegen` to exercise bounded ToolGen over
  the ACT trajectory and expert validation control.
- Round 2 uses `tool_spec.route=reuse` to select the verified Trusted Tool.
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
