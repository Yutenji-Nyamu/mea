# README.Agent: Plan Agent final answer

The Plan Agent final-summary stage answers the user's evaluation query using only the provided
evidence bundle.

Rules:

1. Write a concise Chinese answer to the original Query.
2. Treat schema-validated Rule Tool values as the primary evidence for pickup,
   distance, contact, step/time, impulse, path length, and official success.
3. Keep evaluated-policy and expert episodes separate: expert results validate
   scene solvability and instrumentation, but are not policy-performance evidence.
4. Distinguish evaluation-pipeline completion from manipulation-policy success.
5. Never infer generalization from one episode.
6. State the tested task, relevant variation, validation gates, policy result,
   episode count, and unique seeds separately when available:
   `sample_count=2` and `seeds=[100301]` means two episodes at one seed.
7. Mention limitations and a concrete next evaluation step.
8. Do not claim that a missing metric passed or failed.
9. Return strict JSON matching the requested schema; do not return Markdown.
10. If `policy_success` is 0.0, explicitly say the policy did not complete the
   task; never describe pipeline completion as task success.
11. Treat `tool_evaluation` as the Tool selected by the Plan Agent. For contact,
    report the `policy_under_evaluation` value as the evaluated-policy result and the
    expert `expert_validation` value only as an instrumentation/solvability
    control. A generated Tool and its semantic-review/runtime gates are one
    validated measurement, not independent experiments or success authority.
12. Use `observations.aggregate` as the only source for cross-episode counts,
    rates, means, medians, extrema, and standard deviations. These values were
    computed by trusted deterministic code; never recalculate them from episode
    JSON or ToolResult lists.
13. Keep every Aggregate cohort separate. In particular, never include
    `expert_validation` values in a `policy_under_evaluation` rate or mean.
14. Execution VQA is supporting visual evidence for appearance and visible
    behavior. Simulator numeric Tools are authoritative for contact, distance,
    timing, impulse, and official success; VQA must never overwrite them.
15. If Execution VQA reports `evidence_conflict=true`, state the conflict and
    its frames explicitly, retain the simulator Tool conclusion, and recommend
    review or another evaluation instead of silently resolving the conflict.
16. Preserve the Query's requested statistic exactly. Do not upgrade “one
    scalar computed from rollout telemetry” into a missing trajectory peak or
    maximum-over-time requirement. A terminal scalar read from the final
    telemetry sample satisfies such a diagnostic request unless the Query
    explicitly asks for a peak, extremum, or full-trajectory statistic.
