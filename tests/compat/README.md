# Compatibility tests

This directory is the destination for retained legacy interfaces: catalog and
task-specific planners, historical TaskGen dialects, old registry readers, and
deprecated CLI spellings.

Compatibility coverage is run explicitly on the server and does not belong to
the default paper-method regression suite. Existing cases remain in
`tests/manipeval` until their production callers have been migrated or removed.

`test_multi_round_runtime.py` covers the historical `execute_round` wrapper now
owned by `experiments.paper.compat_agent_runner`.

`test_plan_agent_prototype.py` and `test_bound_task_plan_session.py` retain the
pre-production catalog planner and bound-task session behavior. The production
Plan Agent session remains covered in `tests/mainline/test_claim_first_runtime.py`.

Repeated prompt-wording, operator-schema, old alias, and ablation cases that
still share mainline fixtures are collected explicitly by
`tests/manipeval/test_cold_method_matrices.py`.
