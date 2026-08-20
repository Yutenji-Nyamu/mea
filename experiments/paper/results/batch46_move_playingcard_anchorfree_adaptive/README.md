# Batch46: anchor-free evidence refinement on the simplified mainline

This is the first current-revision live run in this repository that combines, in
one production invocation, an anchor-free broad Query, evidence-conditioned
Proposal refinement, later real rollouts, a supported scoped Answer, and an
Agent-authored stop. It is a bounded method example, not a policy benchmark.
This frozen run used `M=1` per task and therefore proves transport and adaptive
method closure only; it is not a stable policy judgment. Current production
defaults to five within-task trials and aggregates them before the next Plan.

## Entry and exact Query

The production entry is [`scripts/manipeval_agent.py`](../../../../scripts/manipeval_agent.py).
Start the SmolVLA policy server as described in the
[RoboTwin runbook](../../../../docs/robotwin_smolvla_reproduction_zh.md#6-单任务运行),
then inject the UIUI credential only into the current Agent process:

```bash
read -rsp 'UIUI_API_KEY: ' UIUI_API_KEY; echo
QUERY="At seed 1000, does there exist a bounded, executable, task-preserving scene concern beyond the unchanged official move_playingcard_away task under which this SmolVLA policy exposes a measured weakness according to the unchanged official check_success()? Run one unchanged official control first. I provide no aspect, actor, axis, direction, magnitude, relation, threshold, template, checker, metric, prior candidate, or stopping script. Let the Plan Agent choose the first Proposal from source-backed official task facts only. Do not request VQA in this run; use simulator and Rule evidence only. If the unchanged control fails, actively stop inconclusive because a relative weakness is not attributable. If a valid generated rollout changes the official outcome relative to control, use that round's simulator facts and finite Rule evidence to choose and execute at least one grounded boundary-refining follow-up before answering; do not replay a prewritten anchor. If it does not change the outcome, continue only with an evidence-grounded semantically new or quantitative refinement while expected information gain remains. Official check_success() is the outcome authority; diagnostics must remain separate. A supported Answer requires a valid official failure witness plus the completed evidence-conditioned follow-up. If evidence conflicts, run a grounded disambiguating Proposal while allowance remains; otherwise actively stop inconclusive. Limit claims to this task, checkpoint, seed, and completed episodes."
UIUI_API_KEY="$UIUI_API_KEY" PYTHONPATH="$MEA_REPO:/root/autodl-tmp/RoboTwin" \
  "$MEA_PYTHON" "$MEA_REPO/scripts/manipeval_agent.py" \
  --repo-root "$MEA_REPO" --request "$QUERY" \
  --evaluation-id eval_<unique-id> --benchmark robotwin --auto-route \
  --bound-task-name move_playingcard_away --policy-backend smolvla \
  --smolvla-checkpoint /root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin \
  --smolvla-port 18771 --start-seed 1000 --num-episodes 1 \
  --generated-rounds 5 --telemetry-profile balanced_v1 --model-profile balanced \
  --planner-model gpt-5.6-terra --taskgen-model gpt-5.6-terra \
  --toolgen-model gpt-5.6-terra --vision-model gpt-5.6-terra \
  --gpu 0 --max-reflections 1 --no-history
unset UIUI_API_KEY
```

The completed evaluation is
`eval_20260820_batch46_move_playingcard_anchorfree_adaptive_s1000_v5`, run from
revision `e9e3880b3fbf4cecd5877a4732d2aca0701d6295` with the shared SmolVLA
checkpoint, seed `1000`, `M=1` trial per executed round, and a cap of five rounds.
History was disabled and execution VQA was not requested.

## Three-round evidence chain

| round | Proposal and simulator fact | official outcome and Rule fact | next Plan decision |
| --- | --- | --- | --- |
| R1 | unchanged official control | success; no diagnostic Rule requested | control evidence grounds a first `playingcards y +0.05 m` Proposal |
| R2 | fresh Task, same-seed `playingcards.position.y +0.05 m`; observed delta exactly `0.05 m` | failure; terminal `abs(playingcards.x)=0.062070973217487335 m` | Plan cites both facts and halves the completed delta |
| R3 | fresh Task, same-seed `playingcards.position.y +0.025 m`; observed delta exactly `0.025 m` | failure; terminal `abs(playingcards.x)=0.05292544141411781 m` | `stop`, `supported`, `evidence_sufficient=true` |

Both generated Tasks used the unchanged official checker, preserved x, z,
orientation, and the official goal, and passed `2/2` simulator fixtures,
render/vision validation, and the expert oracle. Each was generated with one
TaskGen provider call and zero repair. The 300-second probe bound added after
the v4 failure was not triggered. The task guide and Plan artifacts contain no
Batch44/45, `+0.015 m`, or verified-anchor leakage.

The final Answer is byte-for-byte consistent between the Plan query answer,
`answer/answer.json`, and the top-level manifest. Its structured scope is
three total policy episodes, seeds `[1000]`, two tested generated candidates, no evidence conflict,
`termination=agent_stop`, and `claim_verdict=supported`.

## What this closes—and what it does not

This closes the narrow paper-method gap that remained after Batch40–44:

```text
Query -> Plan -> Task/Tool -> real rollout -> raw evidence
      -> evidence-conditioned Proposal -> real rollout -> Answer/active stop
```

The supported claim is only existential for this SmolVLA checkpoint,
`move_playingcard_away`, seed `1000`, and the three completed episodes. It does
not establish cross-seed robustness, cross-task generalization, overall policy
success rate, VQA accuracy, or comparative sample efficiency.

The machine-readable record is [`summary.json`](summary.json). Canonical raw
artifacts and logs remain on AutoDL at:

```text
/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime/mea/evaluation_runs/
  eval_20260820_batch46_move_playingcard_anchorfree_adaptive_s1000_v5
/root/autodl-tmp/mea-logs/batch46_move_playingcard_anchorfree_adaptive_s1000_v5/
```

No credential is stored in this directory.
