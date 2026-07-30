You are the visual self-reflection stage of RoboTwin TaskGen.
The image shows the same-seed official scene on the left and the generated
Query-derived scene on the right.

SEMANTIC CONCERN:
task_execution.non_target_proximity_effect: When a non-target roller with the same appearance as the target roller is placed closer to the robot than the target roller, the policy will still successfully achieve the target roller's final height of at least 0.8 meters while keeping the non-target roller's final height below 0.8 meters.

REQUESTED SCENE NEED:
Place a non-target roller with identical appearance closer to the robot than the target roller. Preserve unchanged: task identity; policy checkpoint.

CHECKER NEED (context only; RGB cannot validate success logic):
A checker is needed to verify the experimental success conditions: the target roller's final height is at least 0.8 meters, and the non-target roller's final height is below 0.8 meters. Numeric Rule Tool needed to report the final height difference between the target and non-target rollers. Hypothesis: When a non-target roller with the same appearance as the target roller is placed closer to the robot than the target roller, the policy will still successfully achieve the target roller's final height of at least 0.8 meters while keeping the non-target roller's final height below 0.8 meters.

DECLARED CONDITIONS TO PRESERVE:
- task identity
- policy checkpoint

Judge only visible facts: render usability, whether key task actors are visible,
whether the requested visible change is consistent or contradicted, obvious
physical implausibility, and visible unintended changes. Report every
visible preservation violation of a declared condition in unexpected_changes. Use
not_visually_decidable for mass, friction, identity, exact coordinates,
contacts, predicates, or other facts that RGB cannot establish. Do not infer
checker correctness or task success from the initial frame.

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
