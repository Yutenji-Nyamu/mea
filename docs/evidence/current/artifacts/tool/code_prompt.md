You are the ToolGen code agent for an offline RoboTwin trajectory.

USER REQUEST:
What is the minimum XY distance between the official active-arm TCP and the bell contact point during the episode?

TARGET ORACLE:
- target metric: bell_active_tcp_min_xy_error
- exact reference tool: none; validated by a private composition oracle
- contract: {
  "metric": "bell_active_tcp_min_xy_error",
  "description": "Minimum XY distance between the official active-arm TCP and the bell contact point over the recorded trajectory.",
  "oracle_kind": "private_semantic_trace_oracle",
  "supported_task_names": [
    "click_bell"
  ],
  "aspect_ids": [
    "object_position"
  ],
  "unit": "m",
  "available_schema_keys": [
    "physics_timestep_seconds"
  ],
  "required_signals": [
    "semantic_trace.bell_position",
    "semantic_trace.bell_contact_position",
    "semantic_trace.left_tcp_position",
    "semantic_trace.right_tcp_position",
    "semantic_trace.physics_step",
    "semantic_trace.simulation_time_seconds"
  ],
  "output_contract": {
    "value_type": "number",
    "unit": "m",
    "passed_rule": "always_null",
    "evidence_rule": "minimum_error_physics_step",
    "details_keys": [
      "active_arm",
      "min_error_physics_step",
      "simulation_time_seconds"
    ]
  },
  "validation_requirements": {
    "min_episodes": 2,
    "distinct_reference_values": true,
    "required_reference_values": []
  }
}
- generate this target directly; do not call a Trusted Tool and do not choose reuse.
- select the active arm from initial bell_position X:
  negative selects left_tcp_position, otherwise right_tcp_position.
- access every recorded array only as `trajectory.trace["field_name"]`;
  `trajectory.semantic_trace` does not exist and must never be used.
- compute Euclidean XY distance to bell_contact_position at every trace row.
- prefer the supported finite reduction `np.argmin(np.where(np.isfinite(d), d, np.inf))`.
- return the finite minimum in meters with passed=None.
- evidence is the physics step at the minimum; details must contain exactly
  active_arm, min_error_physics_step, and simulation_time_seconds.

OUTPUT CONTRACT AND AVAILABLE DATA:
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


RETRIEVED VERIFIED EXAMPLES:
VERIFIED EXAMPLE time_to_success:
```python
def time_to_success_example(trajectory):
    first = trajectory.success_events[0] if trajectory.success_events else None
    return {
        "value": float(first["simulation_time_seconds"]) if first else None,
        "unit": "s",
        "passed": None,
        "evidence_steps": (
            [int(first["physics_step"])] if first else []
        ),
        "details": {
            "physics_step": first.get("physics_step") if first else None
        },
    }
```

Output exactly one Python fenced block containing the complete
`def generated_tool(trajectory):` function and nothing else.
