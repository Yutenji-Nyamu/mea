You are the visual self-reflection stage of RoboTwin TaskGen.
The image shows the same-seed official scene on the left and the generated
Query-derived scene on the right.

SEMANTIC CONCERN:
robustness.object_instance.bell_size_scaling: ACT will be less likely to complete the click when the bell is uniformly reduced in size by 20%, even though the bell contact position, scene, instruction, and policy checkpoint remain unchanged.

REQUESTED SCENE NEED:
Create a bounded object-instance variant by uniformly scaling the bell's visual and collision geometry to 80% of its reference dimensions while preserving its contact-point world position. Preserve unchanged: bell contact-point world position; bell center position; bell material and color; scene layout and camera viewpoint; task identity; task instruction; policy checkpoint; official success semantics.

CHECKER NEED (context only; RGB cannot validate success logic):
No change requested; preserve the official implementation.

DECLARED CONDITIONS TO PRESERVE:
- bell contact-point world position
- bell center position
- bell material and color
- scene layout and camera viewpoint
- task identity
- task instruction
- policy checkpoint
- official success semantics

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
