# README.Agent: single-round evaluation planning

This document describes the capabilities and constraints visible to the outer
Plan Agent in the first MEA prototype.

## Policy under evaluation

- Policy: ACT.
- Canonical task: `beat_block_hammer`.
- Checkpoint setting: `demo_clean`, trained with 50 expert demonstrations.
- The policy is not language-conditioned during this evaluation.
- `num_episodes` controls the number of evaluation episodes; it is separate
  from the task-specific per-episode step limit.

## Canonical scene

- The scene contains a hammer and one static red block.
- The official block position is sampled with `x` in `[-0.25, 0.25]` and `y`
  in `[-0.05, 0.15]`, rejecting the central strip `abs(x) < 0.05`.
- The official yaw is sampled within approximately `[-0.5, 0.5]` radians.
- The sign of the block's `x` position determines whether the expert uses the
  left or right arm.
- The first prototype must preserve official position and yaw sampling,
  block scale, task logic, success criterion, checkpoint, and random-call
  order. It changes only the block color.

## Available execution route

- Select sub-aspect `object_appearance.color`.
- Select route `force_codegen`, which asks the inner TaskGen agent to generate
  the complete `load_actors()` method.
- Use seed `100000` and exactly one episode.
- Required gates are `ast`, `render`, `rule`, `vision`, `expert`, and `act`.
- Required observations are `scene_alignment`, `observed_color`,
  `expert_solvable`, `act_pipeline_status`, and `policy_success`.

## Validated example

User query:

```text
评估 ACT 在蓝色方块场景中的表现。
```

Canonical inner instruction:

```text
把 beat_block_hammer 任务中的红色方块改成蓝色，其他行为保持不变。
```

Validated variant hint:

```json
{
  "block": {
    "position_mode": "official_random",
    "yaw_mode": "official_random",
    "scale": 1.0,
    "color": [0.0, 0.2, 1.0]
  }
}
```

The previous TaskGen run confirmed a blue rendered block, passed the expert
gate, and completed the ACT evaluation pipeline. The policy itself scored
`0/1`; pipeline completion and policy success must remain separate signals.

## Output rule

Return one strict JSON object matching the schema shown in the prompt. Do not
return Markdown, Python, multiple rounds, or additional variants.
