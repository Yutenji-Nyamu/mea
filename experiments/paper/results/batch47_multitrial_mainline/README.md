# Batch47: five-trial anchor-free adaptive mainline

This is the first completed current-mainline run that combines an anchor-free
broad Query, five paired policy trials per executed task, one aggregate result
per Plan round, evidence-conditioned Proposal refinement, later real rollouts,
a supported scoped Answer, and an Agent-authored stop. It is one bounded method
example, not a policy benchmark or a cross-task result.

## Entry and Query

Production entry: [scripts/manipeval_agent.py](../../../../scripts/manipeval_agent.py).
Start the SmolVLA policy server from the
[RoboTwin runbook](../../../../docs/robotwin_smolvla_reproduction_zh.md#6-单任务运行).
The UIUI key belongs only to the current Agent process:

~~~bash
read -rsp 'UIUI_API_KEY: ' UIUI_API_KEY; echo
QUERY="Using the same five policy-trial seeds 1000 through 1004 for the unchanged control and every candidate, does there exist a bounded, executable, task-preserving scene concern beyond the unchanged official move_playingcard_away task under which this SmolVLA policy exposes a measured weakness according to the unchanged official check_success()? Run one unchanged official control first and compare aggregate outcomes. I provide no aspect, actor, axis, direction, magnitude, relation, threshold, template, checker, metric, prior candidate, or stopping script. Let the Plan Agent choose the first Proposal from source-backed official task facts only. Do not request VQA in this run; use simulator and Rule evidence only. If a valid generated task changes the aggregate official outcome relative to control, use that round's simulator facts and finite aggregate Rule evidence to choose and execute at least one grounded boundary-refining follow-up before answering; do not replay a prewritten anchor. Otherwise continue only with an evidence-grounded semantically new or quantitative refinement while expected information gain remains. Official check_success() is the outcome authority; diagnostics must remain separate. If evidence conflicts, run a grounded disambiguating Proposal while allowance remains; otherwise actively stop inconclusive. Limit claims to this task, checkpoint, seed group, and completed episodes."
UIUI_API_KEY="$UIUI_API_KEY" PYTHONPATH="$MEA_REPO:/root/autodl-tmp/RoboTwin" \
  "$MEA_PYTHON" "$MEA_REPO/scripts/manipeval_agent.py" \
  --repo-root "$MEA_REPO" --request "$QUERY" --benchmark robotwin --auto-route \
  --bound-task-name move_playingcard_away --policy-backend smolvla \
  --smolvla-checkpoint /root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin \
  --smolvla-port 18771 --start-seed 1000 --num-episodes 5 \
  --generated-rounds 5 --telemetry-profile balanced_v1 --model-profile balanced \
  --planner-model gpt-5.6-terra --taskgen-model gpt-5.6-terra \
  --toolgen-model gpt-5.6-terra --vision-model gpt-5.6-terra \
  --gpu 0 --max-reflections 1 --no-history --evaluation-id eval_<unique-id>
unset UIUI_API_KEY
~~~

The completed evaluation is
eval_20260820_batch47_move_playingcard_anchorfree_m5_s1000_v5, run from
revision f03a93a19d0d561c21eac4284f1a7a851b5fc485. It used the shared SmolVLA
checkpoint revision 967623a0f38c7e1236c66b3893c830398d793ff7, seeds
1000 through 1004 in every round, M=5, no history, no execution VQA, and a
five-round allowance.

## Three-round evidence chain

| round | Proposal and simulator fact | aggregate official outcome and Rule fact | next Plan decision |
| --- | --- | --- | --- |
| R1 | unchanged official control | 5/5 success; no Rule requested | source facts plus the valid control ground playingcards x +0.05 m |
| R2 | fresh Task; same-seed playingcards x +0.05 m; observed anchor delta 0.05 m | 4/5 success; five finite minimum-terminal-abs-x values, range 0.004444–0.392452 m | Plan cites the 5/5→4/5 change and Rule range, then proposes total x +0.10 m |
| R3 | fresh Task; evidence-conditioned same-seed playingcards x +0.10 m; observed anchor delta 0.10 m | 4/5 success; five finite terminal-abs-x values, range 0.036089–0.336521 m | stop, supported, evidence_sufficient=true |

Each round used requested and actual seeds [1000,1001,1002,1003,1004].
Each generated Task used one logical TaskGen code-generation call, zero local
repair, the unchanged official checker, verified preservation, 2/2 simulator
fixtures, visual validation, and the expert oracle before policy execution.
Task materialization and these pre-policy validation gates were anchored at
seed 1000; the accepted Task then executed once on all five exact seeds.

The ordinary provider retry window was exercised rather than enlarged: the Plan
call after R1 and the R3 TaskGen call each recovered on retry 1. History was
disabled, and the task guide and Plan artifacts have no Batch44/45, +0.015 m, or
verified-anchor leakage. R2 generated a new Rule Tool; R3 exactly reused an
existing semantic-key Rule Tool on the new five-episode telemetry.

The final Answer is identical in the Plan query answer, evidence bundle,
answer/answer.json, and top-level manifest. Its structured scope is 15 policy
episodes, five unique seeds, two tested generated candidates, no evidence
conflict, termination=agent_stop, and claim_verdict=supported.

## Boundary and accounting

The supported claim is limited to this task, checkpoint, one predeclared seed
group, and the 15 completed episodes. It does not establish a confidence
interval, another seed group, cross-task transfer, VQA accuracy, or another
policy backend. In particular, TaskGen's typed scene/preservation/vision/expert
preflight is still anchored at the first seed even though policy execution and
official outcomes cover all five exact seeds.

The final successful evaluation consumed 15 episodes. Failed attempts are
reported separately: v1=0, v2=5, v3=5, and v4=15, for 25 diagnostic episodes
that are not part of the final scientific sample.

The machine-readable record is [summary.json](summary.json), and the full
implementation/test chronology is
[implementation_and_test_ledger.md](implementation_and_test_ledger.md).
Canonical raw artifacts remain on AutoDL at:

~~~text
/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime/mea/evaluation_runs/
  eval_20260820_batch47_move_playingcard_anchorfree_m5_s1000_v5
/root/autodl-tmp/mea-logs/batch47_move_playingcard_anchorfree_m5_s1000_v5/
~~~

No credential is stored in this directory.
