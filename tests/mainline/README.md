# Mainline suite

This is the default, high-information regression suite for the paper method:

- open Query interpretation and Plan Agent sessions;
- evidence-conditioned round planning and stopping;
- generic TaskGen plus visual diagnosis;
- open ToolGen orchestration;
- RoboTwin round execution and final AnswerScope;
- the production CLI boundary and compact evidence bundle.

It intentionally excludes paper-result protocols, model/checkpoint deployment,
legacy task-specific planners, and exhaustive invalid-input matrices. Those
tests remain available explicitly; moving a test out of the default suite does
not delete it.

During development, run the exact failing node and then this default suite;
do not maintain a second, largely overlapping broad "focused" list. Mainline
tests assert observable behavior and semantic boundaries, not source-text
layout or the continued absence of names already removed from production.
