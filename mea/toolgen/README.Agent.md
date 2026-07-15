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

Available trajectory data:

- `trajectory.trace`: 250 Hz NumPy arrays: `physics_step`, `policy_step`,
  `simulation_time_seconds`, `success`, `hammer_position`, `block_position`,
  `hammer_functional_position`, `block_functional_position`,
  `left_tcp_position`, `right_tcp_position`.
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

For `pickup_to_first_contact_time`, pickup is the first trace sample whose
hammer center Z rise from the initial sample is at least
`schema.pickup_height_threshold_m`; it is not the maximum-height sample and is
not claimed to be the first stable gripper grasp. Contact must be strict
physical contact. Return `value=None` when pickup/contact is missing or contact
precedes pickup, and explain the case in `details.reason`.
