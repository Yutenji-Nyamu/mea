# ManipEval 扩展说明

本仓库是 RoboTwin 的完整 fork。在保留上游任务身份（task identity）和
checkpoint 目录结构的同时，增加一组可选的评估扩展。上游
`policy/ACT/eval.sh` 保持不变，继续作为 regression baseline。

## 参数化 ACT 评估

`policy/ACT/eval_mea.sh` 先接收上游的六个位置参数，之后增加四个可选参数：

```text
TASK CONFIG CKPT EXPERT_NUM SEED GPU [NUM_EPISODES] [TASK_MODULE] [TASK_OVERLAY] [START_SEED]
```

运行一个官方场景 episode：

```bash
policy/ACT/eval_mea.sh \
  beat_block_hammer demo_clean demo_clean 50 0 0 1
```

使用固定 evaluation seed 运行一个蓝色方块变式：

```bash
policy/ACT/eval_mea.sh \
  beat_block_hammer demo_clean demo_clean 50 0 0 1 \
  mea.tasks.beat_block_hammer \
  configs/manipeval/beat_block_hammer_blue.yml \
  100000
```

省略 `NUM_EPISODES` 时仍默认为 100；省略 `START_SEED` 时仍使用上游公式
`100000 * (1 + SEED)`。`task_overlay` 会递归合并到所选 task YAML 中，
但不能覆盖由命令行提供的规范任务名、task config 或 checkpoint setting。

这里的 `NUM_EPISODES` 控制评估 episode 数量，不修改单个 episode 的
rollout step limit。

## BeatBlockHammer 变式协议

只有显式传入 `TASK_MODULE=mea.tasks.beat_block_hammer` 时，才会加载自定义
任务实现。若配置中没有 `mea.enabled: true`，该实现会直接委托给上游任务，
保持官方行为。

目前支持以下方块控制项：

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

仓库在 `configs/manipeval/` 下提供了相互独立的蓝色外观 overlay 和固定姿态
overlay。它们不会修改上游 `envs/` 中的官方任务文件。

## UIUI-compatible provider

UIUI provider 与 rollout loop 保持解耦。它从环境变量读取 API key，并通过
OpenAI-compatible Chat Completions 接口支持文本请求和本地图像请求。

```bash
export UIUI_API_KEY='...'
python scripts/uiui_smoke.py --mode text
python scripts/uiui_smoke.py --mode vision --image /path/to/frame.png
```

API key、SSH 凭据、checkpoint、运行时资源、生成的评估结果和原始操作日志
均不会被 Git 跟踪。

## 文档语言约定

项目新增的开发说明、实验记录和提交交接默认使用中文；函数名、参数名、
路径、协议名、模型名等关键 technical terms 保留英文，便于与代码和论文对应。
