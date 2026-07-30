# ToolGen Rule Tool contract

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

The active prompt declares the exact `TrajectoryView` surface: available
`trace` keys and shapes, events, metadata, schema, policy states, and any
allowlisted helper. Use only that surface. Access recorded arrays as
`trajectory.trace["field_name"]`; `trajectory.semantic_trace` does not exist.
`np` is injected when declared, so do not import NumPy.

Task- and metric-specific recipes do not live in this common contract. They
must arrive through the current Query's MetricSpec, retrieved target guidance,
or validated examples. A retrieval miss means implementing the requested
observable over the declared telemetry, not substituting a familiar task or
older metric.

Rules:

- Do not import anything or access files, network, processes, environment, or
  Python introspection.
- Do not mutate trajectory data.
- Use physical contact only when the declared contact record says
  `physical_contact=true`; an interval alone is insufficient.
- `evidence_steps` contains physics steps, not policy steps or video frames.
- Convert every returned NumPy scalar to a plain Python `float`, `int`, or
  `bool`; JSON-compatible means no `np.float*`, `np.int*`, or `np.bool*`.
- A diagnostic MetricSpec without a pass threshold returns `passed=None`; do
  not import success thresholds from the Query, checker, or another Tool.
- Copy the requested operation into `details.operation`. Use
  `details.reason="measured"` for a valid measurement and only a null reason
  allowed by the active prompt when measurement is unavailable.
- No type annotations, decorators, helper functions, or top-level statements.
- Finite `for` loops over recorded arrays, `range`, `zip`, or `enumerate` are
  allowed; unbounded `while`, async, recursion, imports, and external I/O are
  not.
- Prefer simulator values over visual inference.
- Do not call ndarray methods such as `.all()` or `.tolist()`; use the
  allowlisted `np` functions and convert only the final scalar outputs.
- Implement a Query-induced typed metric directly in Python. MetricSpec is the
  semantic contract, not source to copy or a preselected Tool ID.
- A `derived_observable` is a new Query-induced Rule Tool. Implement only its
  described observable over declared signals; never replace it with the
  nearest older operator or invent an undeclared signal.
- A derived Tool is admitted only after a separate development-agent semantic
  review plus AST, declared-signal, deterministic execution, finite
  scalar/unit/evidence, and artifact-immutability checks. This is not
  independent human/model validation. The gates authorize trajectory
  measurement only; generated code never defines task success or reward.

## Registry scopes

- `run_local`: automatically registered only after static, schema,
  determinism, and private-oracle validation; executable only inside the same
  evaluation.
- `reviewed_persistent`: installed only from an explicit `approved` review
  manifest pinned to the source registration, code, ToolSpec, full contract,
  and telemetry-schema hashes. It remains generated code and is not
  automatically authoritative.

Persistent lookup requires an exact task/metric/ToolSpec/contract/schema
match. Every reuse executes the reviewed source twice on current trajectories
and reapplies its stored semantic-validation contract; provider calls remain
zero.
Pending reviews, candidate promotion, tampered artifacts, path escape, and
symlinks are never executable.  If a reviewed lookup misses, normal codegen
may run only when a provider was explicitly supplied.
