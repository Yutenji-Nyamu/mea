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

## TaskGen 原型

仓库现已提供 `scripts/manipeval_taskgen.py`，支持把自然语言场景修改请求转换为
独立 `run_id` 下的薄 task subclass。对于 BeatBlockHammer，UIUI GPT 会生成完整
`load_actors()`；生成结果通过 AST、import、protected-file、setup/render、rule、
vision 和 expert gates 后，才进入 ACT evaluation。

首次真实蓝色方块验证使用 `force_codegen`，完成了从自然语言到 ACT 1 episode 的
全链路。TaskGen pipeline 全部 gate 通过；ACT policy 在该 episode 为 0/1 success。
详细设计、命令、验证边界和证据见
[`docs/taskgen_prototype.md`](taskgen_prototype.md)。

## 单轮 Plan Agent

`scripts/manipeval_agent.py` 是当前端到端入口。外层 GPT 读取 policy metadata、
RoboTwin 场景约束、执行预算和已验证蓝色方块示例，输出严格的单轮
`EvaluationPlan`；validator 通过后，再调用现有 TaskGen 完成完整代码生成和评估。

第一版只支持 ACT + BeatBlockHammer 蓝色方块、一个 round、seed 100000 和一个
episode。外层 summary 分别记录 `pipeline_passed` 和 `policy_success`。实现、schema、
命令与首次真实验证见
[`docs/plan_agent_prototype.md`](plan_agent_prototype.md)。

## Task Retrieval 与用户反馈

`force_codegen` 在生成 Python 前，会让独立的 `TaskRetriever` 查看 50 个 RoboTwin
task name，并选择 1–3 份相关源码。规范 task 始终排在第一位；GPT 只会看到被选中
的完整源码。本轮蓝色方块评估选择了 `beat_block_hammer` 和
`blocks_ranking_rgb`，分别作为行为权威实现和颜色构造参考。

评估完成后，独立 `FeedbackAgent` 读取结构化 evidence bundle，区分 pipeline 是否
完成与 ACT policy 是否成功，并生成中文反馈。每次外层 evaluation 的首要阅读入口为
`mea/evaluation_runs/<evaluation_id>/evaluation_report.md`；其中索引了 plan、检索、
生成代码、场景图像、Vision 结果、视频与 policy result。设计和真实验证见
[`docs/evaluation_feedback_and_retrieval.md`](evaluation_feedback_and_retrieval.md)。

## Visual Self-Reflection

TaskGen 默认允许最多 2 次视觉诊断驱动的 CodeGen repair。每个 attempt 都执行
setup-only render 和 UIUI Vision；失败时将原始请求、`VariantSpec`、当前完整
`load_actors()`、diagnosis 与 suggestions 交给 repair agent。repair 仍必须生成完整
method，并重新通过 AST、protected-file、render、rule 和 Vision gates；只有最终通过
后才运行 expert 与 ACT。

```bash
python scripts/manipeval_agent.py \
  --request '评估 ACT 在蓝色方块场景中的表现。' \
  --max-reflections 2
```

每次 attempt 的首帧、Vision 输入输出、repair prompt/response 和验证结果保存在
`mea/generated_tasks/<run_id>/reflection/`。实现、测试 fixture、已发现的 VLM 尺度判断
限制和后续路线见 [`docs/visual_self_reflection.md`](visual_self_reflection.md)。

## 文档语言约定

项目新增的开发说明、实验记录和提交交接默认使用中文；函数名、参数名、
路径、协议名、模型名等关键 technical terms 保留英文，便于与代码和论文对应。
