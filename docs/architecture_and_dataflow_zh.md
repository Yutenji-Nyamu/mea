# 当前架构与数据流

## 1. 原始 RoboTwin ACT 调用链

```text
policy/ACT/eval.sh
→ script/eval_policy.py
   ├─ 读取 task_config 与六个官方参数
   ├─ import envs.<task_name>，构造同名 task class
   ├─ deploy_policy.get_model()，加载 dataset_stats.pkl 与 checkpoint
   └─ episode loop
      ├─ setup_demo(seed)
      ├─ get_obs()
      ├─ ACT eval / get_action()
      ├─ Base_Task.take_action()
      │  └─ 将一次 policy action 展开为多个 250 Hz physics steps
      ├─ check_success() / eval_success
      └─ 写 video.mp4 与 _result.txt
```

这条链负责真正的 policy inference、机器人控制、物理仿真和官方成功判定。
MEA 保留了这条执行主干，官方 `policy/ACT/eval.sh` 仍可作为 regression
baseline。

## 2. 当前 MEA 调用链

端到端入口是 `scripts/manipeval_agent.py`：

```text
自然语言 User Query
→ bounded Plan Agent
   └─ 选择 sub-aspect template，Runtime 注入 task、seed 与预算
→ Task / Documentation Retrieval
→ TaskGen
   ├─ 生成薄 task subclass / 完整 load_actors()
   ├─ AST、import 与 protected-file checks
   ├─ setup-only render 与 rule checks
   └─ scene Visual Self-Reflection：诊断、有限 repair、重新验证
→ expert gate
→ policy/ACT/eval_mea.sh
→ 原始 RoboTwin ACT 执行主干
→ Trajectory Recorder + Trusted Tools
→ Auto Tool Router
   ├─ Trusted catalog exact match：reuse
   ├─ evaluation-local exact match：run-local reuse
   ├─ registered composite target：force_codegen + validation
   └─ unsupported metric：拒绝执行并保留审计产物
→ deterministic Aggregate Toolkit
→ Execution VQA
→ round observation 回到 Plan Agent，决定继续或停止
→ evaluation-level aggregate + evidence bundle
→ Feedback Agent
→ evaluation_report.md
```

这里有两种不同的视觉检查：scene VQA 检查生成场景是否符合请求；Execution
VQA 检查 rollout 中实际出现的可见现象。接触、距离、冲量等精确量始终以
simulator Tool 为准，VQA 只能补充视觉证据或报告 `evidence_conflict`。

## 3. 当前数据流与主要产物

一次 evaluation 由一个或多个 round 组成，主要产物关系如下：

```text
mea/evaluation_runs/<evaluation_id>/
├── plan/                         # 初始 plan 与轮间决策
├── execution/<round_id>/
│   ├── planned_tool/             # ToolRequest、route、ToolSpec、execution
│   ├── aggregate_result.json     # 本轮多 episode 可信统计
│   └── execution_vqa/            # 关键帧、拼图、prompt、Vision observation
├── tool_registry/                # 本 evaluation 已验证 generated Tool
├── summary/
│   ├── aggregate_result.json     # evaluation-level 统计
│   ├── summary.json
│   └── evidence_bundle.json
├── feedback/
└── evaluation_report.md

mea/generated_tasks/<run_id>/
├── task.py                       # 生成的薄 task subclass
├── retrieval/ 与 reflection/     # RAG、render、Vision、repair 证据
├── evaluation/telemetry/
│   ├── act/episode_*/
│   └── expert/episode_*/
└── manifest.json
```

每个 telemetry episode 当前包含：

- `episode.json`：task、policy/expert、seed、成功、步数、耗时与 artifact 索引；
- `states.csv`：policy boundary 的 action、机器人状态与 tracked actor 刚体状态；
- `semantic_trace.npz`：250 Hz 任务语义、TCP、success 与时间索引；
- `events.jsonl`：contact interval、success transition 与 error；
- `schema.json`：本次 TaskSchema 快照；
- `video.mp4`：10 FPS RGB 视觉证据。

Tool 读取离线 trajectory 后输出带 `value`、`unit`、`evidence_steps` 和
`details` 的结构化结果。Aggregate Toolkit 负责跨 episode 的数学统计；GPT
只解释聚合结果，不从若干 JSON 中自行计算均值或成功率。

## 4. 相比原始 RoboTwin 增加了什么

- 参数化评估：`num_episodes`、`start_seed`、`task_module`、`task_overlay`，且
  保留官方六参数兼容性；
- 自然语言到结构化 Plan，再到受限 TaskGen 的 agent 编排；
- 不改变规范 `task_name` 和 checkpoint 路径的场景变式；
- Task/Documentation RAG、AST gate、render gate、Visual Self-Reflection 和
  expert solvability gate；
- 可审计的 trajectory telemetry、Trusted Tool、ToolGen 与 Auto Tool Router；
- 同一 evaluation 内的 generated Tool 复用；
- 显式、带正反例/oracle/determinism/code-integrity gate 的 candidate eligibility，
  但不自动提升为 Trusted Tool；
- 跨 episode 的 deterministic Aggregate Toolkit；
- rollout 关键帧 Execution VQA，以及数值证据冲突保护；
- 多轮 observation 回流、最终中文 Feedback 和统一 `evaluation_report.md`；
- planner、taskgen、toolgen、vision、feedback 的独立 model profile/override。

因此，MEA 的主要变化发生在官方执行主干之前的 Proposal/Generation，以及
执行之后的 Observation/Tool/Feedback；ACT 模型本身和 RoboTwin 的物理执行语义
没有被替换。
