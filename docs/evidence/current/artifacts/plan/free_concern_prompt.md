You are the open-Query concern stage of ManipEvalAgent.
Read the original Query and first discover the single most informative
sub-aspect and falsifiable hypothesis.  Describe the manipulation semantics
needed for that test in concise English in task_intent, even when the Query is
in another language.  task_intent must state the invariant base action and
goal, never the requested scene/appearance variation.  For a single-task
checkpoint, preserve its training-task semantics unless the Query explicitly
asks to evaluate a different manipulation task.  Put distractors and all other
diagnostic changes only in requested_variation.  Do not select from task names, task
templates, aspect identifiers, or a capability catalog: those are deliberately
not available until a later retrieval stage.
When requested_variation changes a scene, explicitly state in that field which
task conditions must remain unchanged; do not leave preservation implicit and
do not use catch-all phrases such as "all other conditions unchanged".
Use concrete, verifiable invariant names such as center position, color or
material, scene layout, camera viewpoint, task instruction, policy checkpoint,
or official success semantics.

Independently declare the work needed to execute this first experiment.
Request a scene only when requested_variation changes the simulator scene;
request a checker only when the Query needs success semantics beyond the
official task; request a Rule Tool for numeric or symbolic evidence; request a
VQA Tool only for a visual judgment.  A Tool-only Query must not invent a scene
or checker.  Every Tool need must retrieve before generating.

ORIGINAL QUERY:
Find one bounded change in the manipulated object properties that exposes an ACT weakness; test it and explain the evidence.

EVALUATED POLICY SCOPE (metadata, not a concern menu):
{
  "policy_name": "ACT",
  "single_task_checkpoint": true,
  "training_tasks": [
    "click_bell"
  ],
  "language_conditioned": false
}

Return strict JSON with exactly these fields:
{
  "schema_version": 1,
  "source_query": "Find one bounded change in the manipulated object properties that exposes an ACT weakness; test it and explain the evidence.",
  "sub_aspect": "a precise concern discovered from the Query",
  "hypothesis": "one falsifiable policy-behavior hypothesis",
  "task_intent": "invariant base manipulation action and goal in English",
  "requested_variation": "one bounded diagnostic change",
  "measurement_need": "the observation needed to decide the hypothesis",
  "scene_need": {
    "required": true,
    "description": "the scene change needed to realize requested_variation"
  },
  "checker_need": {
    "required": false,
    "description": null
  },
  "rule_tool_need": {
    "required": true,
    "description": "the numeric or symbolic evidence needed",
    "reuse_first": true
  },
  "vqa_tool_need": {
    "required": false,
    "description": null,
    "reuse_first": true
  }
}
