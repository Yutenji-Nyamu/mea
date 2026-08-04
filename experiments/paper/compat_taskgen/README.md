# TaskGen compatibility layer

This directory preserves the former standalone `manipeval_taskgen.py`
implementation for frozen paper protocols and historical imports. It contains
the BBH/ClickBell dialects, registered/reviewed execution, Table-3 switches and
the legacy simulator/probe orchestration.

Production Agent rounds do not import this module. They use:

```text
Plan Agent Proposal
  -> mea.taskgen.runtime / GenericTaskGenBackend
  -> MethodRuntime
  -> shared RoundExecutor
```

[`scripts/manipeval_taskgen.py`](../../../scripts/manipeval_taskgen.py) is a
small lazy dispatcher. The generic standalone mode remains behavior-compatible;
all other modes enter this compatibility layer explicitly. Historical function
imports are bridged only for paper tests and scripts.

Delete this directory only after the paper/compat callers and their tests have
been removed. Do not add new production behavior here.
