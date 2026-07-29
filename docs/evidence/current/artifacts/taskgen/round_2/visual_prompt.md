You are the visual self-reflection stage of RoboTwin TaskGen.
The image shows the same-seed official scene on the left and the generated
Query-derived scene on the right.

SEMANTIC CONCERN:
Robustness to a bounded change in the bell's color while its task-relevant geometry and location remain fixed: ACT will be less likely to click the bell successfully when the bell color is changed to a bounded alternate color, despite identical bell position, shape, size, material, scene layout, camera viewpoint, and click_bell instruction.

REQUESTED SCENE NEED:
Change only the bell color from the checkpoint's reference color to one bounded alternate color; preserve the bell center position, shape, size, material, scene layout, camera viewpoint, task instruction, policy checkpoint, and official success semantics. Preserve unchanged: the bell center position; shape; size; material; scene layout; camera viewpoint; task instruction; policy checkpoint; official success semantics.

CHECKER NEED (context only; RGB cannot validate success logic):
No change requested; preserve the official implementation.

DECLARED CONDITIONS TO PRESERVE:
- the bell center position
- shape
- size
- material
- scene layout
- camera viewpoint
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
