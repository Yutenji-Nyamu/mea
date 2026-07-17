# Offline ToolGen contract

Generate exactly one complete function:

```python
def generated_tool(trajectory):
    ...
```

The function runs after rollout and receives a fresh, read-only-style
`TrajectoryView`. It must return exactly:

```python
{
    "value": JSON-compatible value,
    "unit": str or None,
    "passed": bool or None,
    "evidence_steps": [physics_step, ...],
    "details": {"short_key": JSON-compatible value},
}
```

Available trajectory data is declared by the task schema. The current bounded
prototypes expose:

- `trajectory.trace`: 250 Hz NumPy arrays. Common fields include
  `physics_step`, `policy_step`, `simulation_time_seconds`, `success`,
  `left_tcp_position`, and `right_tcp_position`. BBH additionally declares
  hammer/block pose and functional-point arrays. `click_bell` additionally
  declares `bell_position` and `bell_contact_position`.
- `trajectory.events`: contact intervals, success transitions, and errors.
- `trajectory.hammer_block_contacts()`: hammer-block contact intervals.
- `trajectory.metadata`: episode identity, seed, policy, success, and counts.
- `trajectory.schema`: task thresholds, actor identities, and physics timestep.
  The relevant exact keys are `pickup_height_threshold_m` and
  `physics_timestep_seconds`; there is no `physics_timestep` key.
- `trajectory.policy_states`: policy-boundary action/robot/actor CSV rows.
- `np` is injected; do not import NumPy. Only allowlisted pure numeric
  attribute chains are accepted.

Rules:

- Do not import anything or access files, network, processes, environment, or
  Python introspection.
- Do not mutate trajectory data.
- Use physical contact only when `physical_contact` is true. A reported contact
  interval alone is not sufficient.
- `evidence_steps` contains physics steps, not policy steps or video frames.
- No type annotations, decorators, helper functions, or top-level statements.
- Prefer simulator values over visual inference.
- Only access arrays as `trajectory.trace["field_name"]`;
  `trajectory.semantic_trace` does not exist.

For `pickup_to_first_contact_time`, pickup is the first trace sample whose
hammer center Z rise from the initial sample is at least
`schema.pickup_height_threshold_m`; it is not the maximum-height sample and is
not claimed to be the first stable gripper grasp. Contact must be strict
physical contact. Return `value=None` when pickup/contact is missing or contact
precedes pickup, and explain the case in `details.reason`.

For `bell_active_tcp_min_xy_error`, choose the active arm from the initial bell
X coordinate (negative is left, otherwise right), compute finite XY distances
to `bell_contact_position`, and return the minimum in metres with
`passed=None`. The evidence step is the physics step at that minimum. This is a
diagnostic for the requested position aspect; it does not replace the official
task success check.

## Registry scopes

- `run_local`: automatically registered only after static, schema,
  determinism, and private-oracle validation; executable only inside the same
  evaluation.
- `reviewed_persistent`: installed only from an explicit `approved` review
  manifest pinned to the source registration, code, ToolSpec, full contract,
  and telemetry-schema hashes.  It is still generated code, not a Trusted
  Tool.

Persistent lookup requires an exact task/metric/ToolSpec/contract/schema
match.  Every reuse executes the reviewed source twice on the current
trajectories and checks the private oracle again; provider calls remain zero.
Pending reviews, candidate promotion, tampered artifacts, path escape, and
symlinks are never executable.  If a reviewed lookup misses, normal codegen
may run only when a provider was explicitly supplied.
