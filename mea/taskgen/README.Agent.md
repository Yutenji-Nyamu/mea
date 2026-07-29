# TaskGen output rules

- Return complete methods, never patches or prose.
- Change only fields authorized by the validated Proposal.
- Preserve actor identity, collision behavior, random-call order, and required
  telemetry names.
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
- The explicit `provider_scene_checker_codegen` route must generate both
  `load_actors()` and `check_success()` from the same Proposal. Its checker is
  experimental and must never be relabeled as official success.
- A paper-claim run requires compile/semantic fixtures, render, expert
  solvability, and the generated scene/checker to remain bound to one artifact.
