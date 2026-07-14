# ManipEval extensions

This fork keeps the upstream RoboTwin task identity and checkpoint layout while
adding opt-in evaluation controls. The upstream `policy/ACT/eval.sh` remains
unchanged and is the regression baseline.

## Parameterized ACT evaluation

`policy/ACT/eval_mea.sh` accepts the six upstream positional arguments followed
by four optional values:

```text
TASK CONFIG CKPT EXPERT_NUM SEED GPU [NUM_EPISODES] [TASK_MODULE] [TASK_OVERLAY] [START_SEED]
```

Run one official episode:

```bash
policy/ACT/eval_mea.sh \
  beat_block_hammer demo_clean demo_clean 50 0 0 1
```

Run one blue-block variant at a fixed evaluation seed:

```bash
policy/ACT/eval_mea.sh \
  beat_block_hammer demo_clean demo_clean 50 0 0 1 \
  mea.tasks.beat_block_hammer \
  configs/manipeval/beat_block_hammer_blue.yml \
  100000
```

When omitted, `NUM_EPISODES` remains 100 and `START_SEED` retains the upstream
formula `100000 * (1 + SEED)`. A task overlay is recursively merged into the
selected task YAML, but cannot replace the canonical task name, task config, or
checkpoint setting supplied on the command line.

## BeatBlockHammer variant protocol

The custom task is loaded only when `TASK_MODULE=mea.tasks.beat_block_hammer`.
Without `mea.enabled: true`, it delegates to the upstream implementation.

Supported block controls are:

```yaml
mea:
  enabled: true
  block:
    position_mode: fixed       # fixed | official_random
    xy: [0.15, 0.05]
    yaw_mode: fixed            # fixed | official_random
    yaw: 0.0
    scale: 1.0
    color: [1.0, 0.0, 0.0]
```

The repository includes an isolated blue appearance overlay and a fixed-pose
overlay under `configs/manipeval/`.

## UIUI-compatible provider

The provider is intentionally separate from the rollout loop. It reads the API
key from the environment and supports OpenAI-compatible text and local-image
chat requests.

```bash
export UIUI_API_KEY='...'
python scripts/uiui_smoke.py --mode text
python scripts/uiui_smoke.py --mode vision --image /path/to/frame.png
```

Credentials, checkpoints, generated evaluations, and operation logs are not
tracked by Git.
