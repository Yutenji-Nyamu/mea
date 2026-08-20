# MEA

MEA is a compact reproduction of the ManipEvalAgent paper method.
Ordinary production uses `scripts/manipeval_agent.py` and follows
`Query → Plan/Proposal → Task/Tool → rollout → Rule/VQA → evidence → Answer`.

## Minimal live example

Start the [RoboTwin SmolVLA policy server](docs/robotwin_smolvla_reproduction_zh.md#6-单任务运行),
then run on AutoDL with `MEA_REPO` and `MEA_PYTHON` configured as in the
[running guide](docs/running_guide_zh.md). The UIUI key is passed only to the
current Agent process:

```bash
read -rsp 'UIUI_API_KEY: ' UIUI_API_KEY; echo
QUERY='Starting from one unchanged official episode as control, is there at least one generated, executable, task-preserving scene concern on which this policy succeeds according to official_check_success and for which a Rule Tool returns a finite live measurement from that same episode? Let the control evidence and retrieved task knowledge determine the candidate. I supply no aspect, object, axis, template, scene edit, checker, metric, or stopping script. The unchanged official control alone is not the witness. Generate or reuse only the artifacts required by the Proposal, and actively stop and answer yes as soon as one generated candidate satisfies both conditions; otherwise report only the tested scope.'
UIUI_API_KEY="$UIUI_API_KEY" PYTHONPATH="$MEA_REPO:/root/autodl-tmp/RoboTwin" \
  "$MEA_PYTHON" "$MEA_REPO/scripts/manipeval_agent.py" \
  --repo-root "$MEA_REPO" --request "$QUERY" --auto-route \
  --bound-task-name grab_roller --policy-backend smolvla \
  --smolvla-checkpoint /root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin \
  --smolvla-port 18771 --start-seed 1000 --num-episodes 1 \
  --generated-rounds 6 --no-history --evaluation-id eval_<unique-id>
unset UIUI_API_KEY
```

Outputs are written to `mea/evaluation_runs/<evaluation-id>`. Frozen AutoDL
examples from earlier revisions demonstrate the bounded method loop:

- [`grab_roller` Batch40](experiments/paper/results/batch40_paper_mainline_cleanup/README.md):
  control → generated Task → rollout → live Rule → Aggregate → supported
  Answer in two policy episodes; a verdict-hidden replay separately revalidated the stop.
- [`press_stapler` Batch37](docs/evidence/current/README.md): control plus nine
  generated rollouts and Rule evidence, followed by an active, scoped
  inconclusive Answer after information saturation (ten episodes, one seed).

Start from the [Chinese documentation index](docs/index_zh.md); use the
[running guide](docs/running_guide_zh.md) for commands and outputs.
Historical protocols, deployment ledgers, and frozen artifacts live under
[`experiments/paper/`](experiments/paper/README.md) and are not default context.
