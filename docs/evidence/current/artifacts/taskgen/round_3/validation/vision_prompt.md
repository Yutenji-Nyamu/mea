You are the visual self-reflection stage of RoboTwin TaskGen.
The image shows the same-seed official scene on the left and the generated
Query-derived scene on the right.

SEMANTIC CONCERN:
scene_robustness.lateral_shift_terminal_left_tcp_proximity: After translating the stapler initial position by +0.03 m along the world x-axis, the policy will fail the satisfiable combined terminal criterion of official press_stapler success and left-TCP-to-stapler distance no greater than 0.080 m, exposing a spatial-retargeting weakness.

REQUESTED SCENE NEED:
Construct or adapt the official press_stapler scene by translating the stapler initial position by +0.03 m along the world x-axis. Preserve unchanged: task identity; policy checkpoint; official core predicate as a required conjunct.

CHECKER NEED (context only; RGB cannot validate success logic):
Boolean terminal predicate: official core predicate as a required conjunct AND Euclidean distance between left_tcp_position and stapler_position is less than or equal to 0.080 m.

DECLARED CONDITIONS TO PRESERVE:
- task identity
- policy checkpoint
- official core predicate as a required conjunct

Judge only visible facts: render usability, whether key task actors are visible,
whether the requested visible change is consistent or contradicted, obvious
physical implausibility, and visible unintended changes. Report every
visible preservation violation of a declared condition in unexpected_changes. Use
not_visually_decidable for mass, friction, identity, exact coordinates,
contacts, predicates, or other facts that RGB cannot establish. Do not infer
checker correctness or task success from the initial frame.
requested_change_assessment must be exactly one of: consistent, contradicted,
not_visually_decidable. Never substitute synonyms such as inconsistent.
visual_physical_plausibility must be exactly one of: plausible, implausible,
uncertain. Never substitute synonyms such as realistic or good.

Return strict JSON with exactly these fields:
{
  "schema_version": 1,
  "render_usable": true,
  "key_task_actors_visible": true,
  "requested_change_assessment": "consistent",
  "visual_physical_plausibility": "plausible",
  "unexpected_changes": [],
  "diagnosis": "The generated scene is visible and physically plausible.",
  "repair_instructions": [],
  "confidence": 0.8
}
