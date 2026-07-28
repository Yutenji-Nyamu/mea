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

ORIGINAL QUERY:
这个 ACT 策略执行调整瓶子任务时，对未见对象属性的泛化能力如何，最先在哪里暴露弱点？

EVALUATED POLICY SCOPE (metadata, not a concern menu):
{
  "policy_name": "ACT task-specific checkpoint portfolio",
  "single_task_checkpoint": false,
  "training_tasks": [
    "withheld_until_semantic_task_retrieval"
  ],
  "language_conditioned": false
}

Return strict JSON with exactly these fields:
{
  "schema_version": 1,
  "source_query": "这个 ACT 策略执行调整瓶子任务时，对未见对象属性的泛化能力如何，最先在哪里暴露弱点？",
  "sub_aspect": "a precise concern discovered from the Query",
  "hypothesis": "one falsifiable policy-behavior hypothesis",
  "task_intent": "invariant base manipulation action and goal in English",
  "requested_variation": "one bounded diagnostic change",
  "measurement_need": "the observation needed to decide the hypothesis"
}
