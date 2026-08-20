# MEA

MEA is a compact reproduction of the ManipEvalAgent paper method.
Ordinary production uses `scripts/manipeval_agent.py` and follows
`Query → Plan/Proposal → Task/Tool → rollout → Rule/VQA → evidence → Answer`.

## Minimal live example

Start the [RoboTwin SmolVLA policy server](docs/robotwin_smolvla_reproduction_zh.md#6-单任务运行),
then run on AutoDL with `MEA_REPO` and `MEA_PYTHON` configured as in the
[running guide](docs/running_guide_zh.md). The UIUI key is passed only to the
current Agent process:

The adaptive number of Plan rounds and the number of policy trials inside one
executed task are separate. Ordinary paper-aligned runs use five trials per
round and give Plan one aggregate; `--num-episodes 1` is only a mechanism debug.

```bash
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
```

Outputs are written to `mea/evaluation_runs/<evaluation-id>`. Verified AutoDL
examples demonstrate two bounded outcomes of the method loop:

- Historical `M=1` mechanism proof: [`move_playingcard_away` Batch46](experiments/paper/results/batch46_move_playingcard_anchorfree_adaptive/README.md):
  anchor-free Query → successful control → `y +0.05 m` failure →
  evidence-conditioned `y +0.025 m` failure → live Rule → supported Answer and
  Agent stop in three policy episodes. It does not establish a stable rate.
- Historical `M=1` exploration: [`press_stapler` Batch37](docs/evidence/current/README.md): control plus nine
  generated rollouts and Rule evidence, followed by an active, scoped
  inconclusive Answer after information saturation (ten episodes, one seed).

Start from the [Chinese documentation index](docs/index_zh.md); use the
[running guide](docs/running_guide_zh.md) for commands and outputs.
Historical protocols, deployment ledgers, and frozen artifacts live under
[`experiments/paper/`](experiments/paper/README.md) and are not default context.
