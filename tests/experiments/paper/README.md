# Paper experiment tests

This directory is the destination for tests of result protocols rather than the
MEA production method: dense/adaptive comparisons, policy ranking, proxy human
and VQA studies, checkpoint deployment, and paper-table reproduction.

Such tests are explicit server-side suites and are not collected by the default
`pytest` command. Existing cases remain in `tests/manipeval` until a mechanical
move can preserve their imports and fixtures without adding compatibility code.

Current cold ownership includes the evidence-bundle publishing/report protocol
in `test_evidence_report.py`.
