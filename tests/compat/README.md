# Compatibility tests

This directory is the destination for retained legacy interfaces: catalog and
task-specific planners, historical TaskGen dialects, old registry readers, and
deprecated CLI spellings.

Compatibility coverage is run explicitly on the server and does not belong to
the default paper-method regression suite. Existing cases remain in
`tests/manipeval` until their production callers have been migrated or removed.

`test_multi_round_runtime.py` covers the historical `execute_round` wrapper now
owned by `experiments.paper.compat_agent_runner`.
