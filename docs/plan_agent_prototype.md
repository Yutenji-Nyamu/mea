# 单轮 Plan Agent 原型

## 目标与边界

本原型在现有 TaskGen 外增加一个外层 `Plan Agent`，把开放的用户评估问题转换为
严格的 `EvaluationPlan`，再驱动内层 TaskGen、场景验证和 ACT evaluation。

第一版故意限制为：

- policy：ACT；
- task：`beat_block_hammer`；
- sub-aspect：`object_appearance.color`；
- 一个 round、一个 task、seed 100000、一个 episode；
- route：`force_codegen`；
- 只使用已验证的蓝色 `[0.0, 0.2, 1.0]`；
- 保持官方 position/yaw sampling、scale、task logic 和 checkpoint 不变。

外层 Agent 不生成 Python，只决定“评估什么”和“给内层什么规范指令”。内层 GPT
仍负责生成完整 `load_actors()`。

## 调用链

```text
用户评估问题
  -> scripts/manipeval_agent.py
  -> PlanAgentPrototype
  -> EvaluationPlan validator
  -> evaluation_plan.json
  -> scripts/manipeval_taskgen.py
  -> 内层 VariantSpec Agent
  -> 内层 GPT 生成完整 load_actors()
  -> AST / protected-file checks
  -> setup + render + rule check
  -> UIUI Vision check
  -> expert gate
  -> ACT 1 episode
  -> 外层 deterministic summary
```

`pipeline_passed` 与 `policy_success` 是两个独立结果。场景生成和执行链路正常并不
代表 ACT 一定完成任务；同样，ACT policy 失败也不能被误判为 Plan Agent 或
TaskGen failure。

## 主要文件

- `mea/planner/prototype.py`：生成、校验并保存单轮 `EvaluationPlan`。
- `mea/planner/README.Agent.md`：向外层 GPT 提供 policy metadata、场景约束、可用
  route、评估预算和已验证示例。
- `scripts/manipeval_agent.py`：外层 orchestration CLI，创建 child TaskGen run 并汇总
  observations。
- `scripts/manipeval_taskgen.py`：新增可选 `--run-id`，使父 evaluation 能稳定绑定
  child run。
- `tests/manipeval/test_plan_agent.py`：覆盖正常蓝色计划、多 round 拒绝和非验证颜色
  拒绝。

## EvaluationPlan

外层 GPT 的核心输出如下：

```json
{
  "task_name": "beat_block_hammer",
  "policy": {
    "name": "ACT",
    "checkpoint_setting": "demo_clean",
    "expert_data_num": 50,
    "language_conditioned": false
  },
  "rounds": [
    {
      "round_id": "round_1",
      "sub_aspect": "object_appearance.color",
      "task_instruction": "把 beat_block_hammer 任务中的红色方块改成蓝色，其他行为保持不变。",
      "route": "force_codegen",
      "variant_hint": {
        "block": {
          "position_mode": "official_random",
          "yaw_mode": "official_random",
          "scale": 1.0,
          "color": [0.0, 0.2, 1.0]
        }
      },
      "execution": {
        "seeds": [100000],
        "num_episodes": 1,
        "gates": ["ast", "render", "rule", "vision", "expert", "act"]
      }
    }
  ],
  "stop_after_round": 1
}
```

Validator 会拒绝额外 round、额外颜色、position/yaw 变化、非 `force_codegen` route、
多 episode 或缺失 gate，防止外层和内层对评估语义产生漂移。

## 使用方式

执行完整单轮评估：

```bash
export UIUI_API_KEY='...'
python scripts/manipeval_agent.py \
  --request '评估 ACT 在蓝色方块场景中的表现。' \
  --gpu 0
```

只验证外层计划，不启动 TaskGen 或 simulator：

```bash
python scripts/manipeval_agent.py \
  --request '评估 ACT 在蓝色方块场景中的表现。' \
  --plan-only
```

每次外层运行保存在：

```text
mea/evaluation_runs/<evaluation_id>/
├── request.json
├── manifest.json
├── plan/
│   ├── prompt.md
│   ├── response.txt
│   └── evaluation_plan.json
├── execution/
│   ├── taskgen_command.json
│   ├── taskgen.log
│   └── child_run.json
└── summary/
    └── summary.json
```

外层目录通过 `child_run_id` 指向
`mea/generated_tasks/<run_id>/` 中的完整代码、验证、视频和 policy 结果。两类运行
目录均被 Git ignore。

## 首次真实验证

- plan-only evaluation：`eval_20260714_133645_planonly`；
- full evaluation：`eval_20260714_133759_blue_act`；
- child TaskGen run：`run_20260714_133759_blue_act_round_1`；
- planner model：`gpt-4o-2024-11-20`；
- planner usage：997 prompt、333 completion、1330 total tokens；
- 内层 codegen usage：4542 prompt、420 completion、4962 total tokens；
- 生成的完整 `load_actors()` 与第一次蓝色 run 的文件 hash 完全一致；
- AST、render、rule、Vision、expert 和 ACT gates 全部通过；
- Vision：`observed_color=blue`、confidence 1.0、无 unexpected changes；
- ACT 视频：H264、320x240、10 FPS、40 seconds；
- policy result：0/1；
- 外层 summary：`pipeline_passed=true`、`policy_success=0.0`。

该结果证明了“开放用户问题 → 外层计划 → 内层代码生成 → 执行 → 结构化
observations”的单轮闭环。它仍是 smoke test，不能凭一个 episode 得出颜色泛化
结论。
