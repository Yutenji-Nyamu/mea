# README.Agent: evaluation history boundary

`EvaluationHistoryDB` stores compact planning records for completed MEA
evaluations. Retrieval may be shown to the Plan Agent to keep decomposition
consistent across equivalent queries and policies.

Historical records are planning priors only. They must not be inserted into
the current run's Aggregate, used as current policy evidence, or used to claim
that the current policy succeeded. Every retrieved item preserves its policy,
checkpoint, source commit, completion status, and repository-relative artifact
references.

The SQLite file is a rebuildable runtime cache. The canonical source is each
completed evaluation's `summary/history_record.json`. New evaluations require
`lifecycle_status=completed`. Legacy records are accepted only when the old
manifest has exact `status=completed`, a final timestamp, and all required
plan/evidence artifacts.
