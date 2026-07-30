# TaskGen output rules

- Return complete methods, never patches or prose.
- Change only fields authorized by the validated Proposal.
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
- A generated challenge must preserve at least one feasible official action
  path to every required contact or functional point. A fixed center offset is
  not proof of approach clearance, and `add_prohibit_area()` is not a robot
  reachability check. Never relax an official success threshold to compensate
  for an expert-solvability failure.
- For robot-contact checks, `PhysxContact` uses `bodies[*].entity`; the
  RoboTwin Robot wrapper has no `get_links()`, so use
  `left_entity.get_links()` plus `right_entity.get_links()`.
- For lift checkers, derive bounds from TaskSchema or initial actor state:
  near-zero world z does not mean an actor stayed unlifted on raised support.
- A paper-claim run requires compile/semantic fixtures, render, expert
  solvability, and the generated scene/checker to remain bound to one artifact.
