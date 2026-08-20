# TaskGen output rules

- Generation starts from the retrieved official source. Implement every
  retained and new scene delta explicitly stated by the current Proposal;
  never infer or silently inherit an earlier round's generated code or state.
- Return complete methods, never patches or prose.
- Change only fields authorized by the validated Proposal.
- Use only actors, poses, telemetry, thresholds, and assets declared by the
  supplied TaskContext/TaskSchema or retrieved official source. If that context
  is insufficient, do not fabricate the missing field.
- Preserve actor identity, collision behavior, random-call order, and required
  telemetry names.
- For a pose change, reuse the official pose construction and alter only the
  Proposal-named component. If the Proposal gives only a direction, derive the
  smallest measurable change from the retrieved spawn/workspace range, stay
  away from its boundary, and keep every key actor fully in the unchanged
  camera view.
- Do not import or access files, network, environment variables, processes,
  dynamic execution, dunder attributes, or `super()`.
- Use wrapper-provided `np`, `sapien`, `create_actor`, `create_box`, and
  `rand_pose`.
- For a fixed-angle rotation, write the numeric quaternion explicitly; generated
  methods must not call `np.sin`, `np.cos`, or `np.deg2rad`.
- The ordinary scene-only route must not replace `check_success()`.
- Actors already listed in the task telemetry/execution schema remain tracked
  when a generated scene moves or replaces the same public actor. Do not
  redeclare them in `self.mea_telemetry_tracked_actors`.
- Every newly added actor must have a stable runtime name distinct from all
  base-scene actor names. When reusing an existing asset model, pass a unique
  `runtime_name` to `create_actor()` and make telemetry `scene_name` exactly
  equal to it; the asset `modelname` is not the actor identity. Declare only
  new actors, with exactly `id`, `task_attribute`, `scene_name`,
  `functional_points`, `contact_points`, and boolean `contact_focus`.
- When a Proposal requests both scene and checker, generate `load_actors()` and
  `check_success()` together. The checker is experimental and must never be
  relabeled as official success.
- When the Proposal requires the official goal as a conjunct, call
  `self.mea_official_check_success()` and combine its boolean result with the
  added directly observable condition. Never replace that official conjunct
  with a correlated proxy.
- Implement every `checker_need` requirement directly in `check_success()`.
- Preserve all quantifiers, simultaneity, temporal relations, object identities,
  and corresponding-pair relations literally.
- Derive every checker condition from direct current simulator observables
  supplied by the TaskContext, TaskSchema, or retrieved official source.
- Never substitute a correlated proxy. If the exact requested predicate is not
  available from current simulator state, leave the candidate unsupported
  instead of weakening or reinterpreting the Proposal.
- Every added checker conjunct must be false initially and true in the supplied
  official-expert terminal state. Do not require instantaneous PhysX contact
  when the official terminal state may release contact; use the repair
  evidence's current actor state or another Proposal-consistent predicate.
- Gripper closure is not target contact, height is not placement, and sequential
  events are not simultaneous events.
- `check_success()` reads current simulator state only. Trajectory deviation,
  smoothness, jerk, path length, and rollout clearance belong to ToolGen.
- A generated challenge must preserve at least one feasible official action
  path to every required contact or functional point. A fixed center offset is
  not proof of approach clearance, and `add_prohibit_area()` is not a robot
  reachability check. Never relax an official success threshold to compensate
  for an expert-solvability failure.
- For robot-contact checks, `PhysxContact` uses `bodies[*].entity`; the
  RoboTwin Robot wrapper has no `get_links()`, so use
  `left_entity.get_links()` plus `right_entity.get_links()`.
- Keep checker-local collections immutable: build link/entity collections with
  `+` or tuple literals rather than `.append()`/`.extend()`. Those mutators are
  outside the generated-checker API contract and can waste the one repair
  attempt before simulator validation.
- Before adding an obstacle or distractor, use the retrieved asset scale and
  collision geometry to keep it initially disjoint from the target and from
  the official expert contact path. Reusing the target asset at a small center
  offset is invalid when their collision extents overlap.
- For lift checkers, derive bounds from TaskSchema or initial actor state:
  near-zero world z does not mean an actor stayed unlifted on raised support.
- A paper-claim run requires compile/semantic fixtures, render, expert
  solvability, and the generated scene/checker to remain bound to one artifact.
