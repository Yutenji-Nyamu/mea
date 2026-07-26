# Batch 24 LIBERO / SmolVLA method-chain smoke

This bundle is a compact, selected view of the raw run
`mea/evaluation_runs/eval_batch24_libero_method_chain_live_v2`.

## Outcome

- The official `libero_object/task0` control and the generated custom task each
  executed one SmolVLA rollout at seed `100800`, relative control, with an
  explicit 100-step horizon. Both failed within that horizon.
- ClaimFirst ran only after the control evidence. TaskGen produced a complete
  BDDL file; parser, registered problem class, unchanged object/region/init
  state, official init-state compatibility and first-frame render gates passed.
- A query-routed, deterministic goal-predicate MetricSpec adapter returned the
  non-null live value `false`, entered the existing Aggregate path, and was
  retrieved by exact registry reuse for a second Query with zero additional
  rollout. This adapter was compiled from a bounded schema; it was not written
  by a model.
- This run is **not scientific claim evidence**. The control failed, and
  ClaimFirst requested a language-only semantics-preserving change while
  TaskGen generated a new goal object. `protocol_audit.json` records the
  Planner/TaskGen mismatch and AnswerScope terminates as `pipeline_invalid`.

The runtime now rejects this mismatch before TaskGen or a custom rollout.

## Selected data flow

1. `planner/`: exact prompts, raw provider responses and validated proposals.
2. `taskgen/`: exact prompt/response, generated BDDL, TaskContract and simulator
   compatibility probe.
3. `render/` and `rollout/`: custom first frame, two short videos, actions and
   compact EpisodeRecords.
4. `tool/`: compiled predicate MetricSpec adapter, validation/registration
   result, Aggregate output and zero-rollout exact-reuse result. The historical
   source-artifact filename is `generated_tool.py`; it does not imply model
   generation.
5. `evidence/`: final EvidencePacket, AnswerScope and protocol audit.

`compact_result.json` is the machine-readable entry point. Raw duplicate frame
arrays and Python bytecode caches are intentionally excluded.
