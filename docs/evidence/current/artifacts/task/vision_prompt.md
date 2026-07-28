You are the visual self-reflection stage of RoboTwin TaskGen.
The image shows the same-seed official scene on the left and the generated
Query-derived scene on the right.

SEMANTIC CONCERN:
object_generalization.position_translation: The ACT policy's first weakness in manipulated-object generalization is spatial: translating the bell within a bounded reachable workspace will reduce official press success or increase final bell-contact distance relative to the unchanged control.

REQUESTED SCENE NEED:
TaskGen must create a bounded object_position variant overlay for the bell and preserve the official click_bell success contract.

CHECKER NEED (context only; RGB cannot validate success logic):
Generate an experimental check_success predicate that decides: The ACT policy's first weakness in manipulated-object generalization is spatial: translating the bell within a bounded reachable workspace will reduce official press success or increase final bell-contact distance relative to the unchanged control.

Judge only visible facts: render usability, whether key task actors are visible,
whether the requested visible change is consistent or contradicted, obvious
physical implausibility, and visible unintended changes. Use
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
