# MEA 架构与数据流

本文说明当前实现，而不是目标蓝图。MEA 不替换 RoboTwin 的物理仿真、机器人控制、
官方任务成功判定或 ACT 推理；它在执行前增加受限规划/生成，在执行中记录可审计证据，
在执行后用确定性工具、视觉检查和反馈组织评估结果。

## 1. 分层与主要模块

| 层 | 主要位置 | 职责 |
| --- | --- | --- |
| 端到端编排 | `scripts/manipeval_agent.py` | 创建 evaluation、执行多轮 plan、汇总证据并生成反馈 |
| 规划 | `mea/planner/` | BBH 使用受限 Plan Agent；其他已注册任务使用确定性的 official planner |
| 检索与历史 | `mea/retrieval/`、`mea/history/`、`mea/knowledge/` | 检索任务/源码知识，复用历史评估上下文 |
| TaskGen | `scripts/manipeval_taskgen.py`、`mea/taskgen/` | 生成或复用受限 task overlay；也可创建不改官方源码的 passthrough run |
| RoboTwin 执行 | `mea/taskgen/probe.py`、`policy/ACT/eval_mea.sh` | setup/render、official expert `play_once()`、ACT rollout |
| Checkpoint 获取 | `scripts/download_act_checkpoint.py` | 按任务和固定 revision 只下载 ACT 所需的 policy/stats 文件 |
| 任务语义与记录 | `mea/toolkit/schema.py`、`mea/toolkit/recorder.py`、`mea/toolkit/schemas/` | 用 TaskSchema 跟踪 actor/语义，写 telemetry、事件和视觉证据 |
| 可信测量 | `mea/toolkit/tools.py`、`mea/toolgen/` | 复用 Trusted Tool，或生成并验证 evaluation-local Tool |
| 聚合 | `mea/toolkit/aggregate.py` | 跨 episode 做确定性统计，不让语言模型自行算成功率/均值 |
| Execution VQA | `mea/execution_vqa/` | 从受限问题目录选择问题，读取事件关键帧并检查可见现象 |
| 反馈 | `mea/feedback/` | 把结构化 observation 和证据索引整理为最终报告 |
| 模型适配 | `mea/providers/` | 各阶段模型 profile 与 OpenAI-compatible provider |

`scripts/manipeval_taskgen.py` 是内层入口；它也适合做 setup/expert/ACT 的单次调试。
`scripts/manipeval_agent.py` 是正常端到端入口，负责把多个内层 run 组织成一次 evaluation。

## 2. Route 与 execution backend

Route 决定任务定义从哪里来，execution backend 决定用什么执行：

- `generated` route：使用受限生成/修复的 task overlay；当前仅完整覆盖 BBH；
- `official` route：直接复用 RoboTwin 官方任务，不生成或改写任务源码；
- `expert|act|both` backend：分别表示只运行 expert、以 ACT 为被评 policy，或同时保留
  expert 验证与 ACT 评估。official route 可使用这三种 backend，不再与 expert 绑定。

### 2.1 Generated + ACT 路线

当前这条路线只完整覆盖 `beat_block_hammer`（BBH）：

```text
自然语言请求
→ bounded Plan Agent
→ 任务/文档/历史检索
→ TaskGen 生成或复用薄 task overlay
→ AST、import、protected-file、setup/render checks
→ Scene Visual Self-Reflection（有限诊断、修复、重验）
→ official expert solvability gate
→ policy/ACT/eval_mea.sh
→ RoboTwin ACT 推理、控制与物理仿真
→ Recorder / Trusted Tools / Aggregate / Execution VQA
→ 多轮 observation → Feedback → evaluation_report.md
```

ACT 主干仍沿用 RoboTwin 的语义：

```text
policy/ACT/eval_mea.sh
→ script/eval_policy.py
   → import envs.<task_name>
   → setup_demo(seed)
   → 加载 dataset_stats.pkl 与 policy checkpoint
   → get_obs() / ACT get_action() / Base_Task.take_action()
   → task.check_success()
   → 写连续 rollout video 与结果
```

这条路线需要相应 ACT checkpoint。`scripts/download_act_checkpoint.py` 可按任务选择性
下载官方 `policy_last.ckpt` 和 `dataset_stats.pkl`，避免拉取整个大数据仓库。TaskGen 的
ACT runner 会从当前 manifest 读取 task/module，并在启动仿真前检查该任务的 checkpoint；
当前 Agent contract 固定使用官方 `demo_clean-50` 布局。

### 2.2 Official passthrough 路线

`click_bell`、`adjust_bottle`、`grab_roller` 等已有 TaskSchema 的官方任务走确定性路线：

```text
自然语言请求 + 明确 task_name
→ OfficialTaskPlanAgent
→ official passthrough run（不生成 task 代码、不改官方任务）
→ setup_demo(seed) + schema/rule checks
→ execution backend：expert / ACT / ACT + expert
→ Recorder（expert 事件关键帧或 ACT 连续 rollout）
→ generic Trusted Tools / Aggregate / Execution VQA
→ Feedback → evaluation_report.md
```

`expert` 不需要 checkpoint，并把官方 expert 作为主执行证据。`act` 先做非 expert 的
setup/render/rule probe，随后由 RoboTwin ACT evaluator 执行其原生 expert eligibility
筛选；`both` 同时保留 expert 验证和 ACT 结果，并以 ACT 作为 VQA/报告的主 policy 证据。
初始化不稳定的 expert seed 会被记录为 rejected seed 并继续扫描候选 seed；`both` 会核对
两边最终实际 seed，不一致即失败，但仍不能替代带显式 seed manifest 和 Easy/Hard 统计的
exact-seed paired protocol。

## 3. TaskSchema 是跨任务边界

`mea/toolkit/schemas/<task_name>.json` 声明：

- 要跟踪的 actor id、task attribute 与 RoboTwin scene name；
- functional/contact point；
- semantic field 的名称、source 与单位语义；
- contact focus、success contract 和 Trusted Tool profile。

Recorder 只解释 schema 支持的通用 source，例如 actor position、functional/contact
point 和左右 TCP，不应为每个新任务增加 `if task_name == ...`。BBH 的少量兼容字段只为
读取历史 artifact 保留。

扩展一个官方 expert 任务的最小步骤是：

1. 增加并验证 TaskSchema；
2. 用 probe 检查 actor attribute、scene name、有限数值和官方 `check_success()`；
3. 用多个 stable seed 运行 expert telemetry 与 generic Tools；
4. 若需要视觉判断，在 `mea/execution_vqa/query.py` 增加受限任务问题映射；
5. 若需要任务专属 metric，再增加 Trusted Tool 或受验证的 ToolGen contract。

如果要生成该任务的场景变式，还需新增 planner template、TaskGen allowlist/repair contract
和检索知识；仅有 TaskSchema 并不授权模型任意改任务代码。

## 4. Telemetry 与证据流

每个完成的 episode 目录以 `episode.json` 为索引，主要 artifact 如下：

| Artifact | 采样/来源 | 用途 |
| --- | --- | --- |
| `states.csv` | policy boundary | action、机器人状态、tracked actor 完整快照 |
| `semantic_trace.npz` | 每个 250 Hz physics step | 任务关键位置、TCP、success 与时间/帧索引 |
| `dynamics_trace.npz` | `balanced_v1` 默认每 5 个 physics step（50 Hz） | qpos/qvel、EE/TCP、gripper、刚体 pose/velocity |
| `events.jsonl` | 250 Hz monitor，按 interval/transition 汇总 | contact、success transition、error |
| `schema.json` | episode 快照 | 固定本次字段和任务语义，避免仓库 schema 漂移 |
| `telemetry_profile.json` | episode 快照 | 固定采样 profile 与 hash |
| `visual_keyframes.json` | official expert 可选 | 稀疏帧、触发原因、physics step 和编码状态 |
| `video.mp4` | 路线相关 | ACT 为连续 rollout；official expert 为稀疏事件帧视频 |

两种 `video.mp4` 不能当成同一种时间序列：

- ACT 视频约 10 FPS，一帧对应一个 policy observation；当 backend 为 `act` 或 `both` 时，
  Execution VQA 选择这份连续视频作为主视觉证据；
- official expert 的 `event_keyframes_v1` 只抓 initial、首次新增 physical contact、
  success transition、final，并以 2 FPS 编码成兼容 VQA 的短视频。它是有序事件证据，
  不是连续运动录像；
- 稀疏帧索引写入 semantic/event evidence，Execution VQA 可精确选中 success 前后帧；
- 相机或 ffmpeg 失败不会丢弃数值 telemetry。此时 visual capture 标为 failed，official
  Execution VQA 明确 skipped，而不是伪造视觉结论。

一次端到端 evaluation 的主要目录关系是：

```text
mea/evaluation_runs/<evaluation_id>/
├── plan/                         # 初始 plan、历史检索、轮间决策
├── execution/<round_id>/
│   ├── taskgen_command.json
│   ├── planned_tool/             # request、route、ToolSpec、execution
│   ├── aggregate_result.json
│   └── execution_vqa/            # 选帧、montage、prompt、视觉 observation
├── tool_registry/                # 本 evaluation 内已验证的 generated Tool
├── summary/                      # round/evaluation aggregate 与 evidence bundle
├── feedback/
└── evaluation_report.md

mea/generated_tasks/<run_id>/
├── task.py                       # 仅 generated route
├── overlay.yml                   # generated 变式或 official 空 overlay
├── generation/official_source.json  # 仅 official passthrough
├── retrieval/、validation/、reflection/
├── evaluation/telemetry/
│   ├── act/episode_*/
│   └── expert/episode_*/
└── manifest.json
```

## 5. 可信边界

证据优先级是架构的一部分：

1. RoboTwin 官方 `check_success()` 是任务成功的权威来源；
2. simulator telemetry 与确定性 Tool 是距离、接触、时刻、位移等数值的权威来源；
3. Aggregate Toolkit 负责跨 episode 数学，语言模型只解释结果；
4. Execution VQA 只回答代码内登记的现象问题，不把用户文本或 ToolGen 自由文本直接
   拼入视觉 prompt；
5. VQA 与数值证据冲突时保存 `evidence_conflict`，不能覆盖可信数值；
6. generated Tool 只在当前 evaluation 内复用，经过 eligibility/oracle/determinism/
   code-integrity checks 也不会自动晋升为全局 Trusted Tool。

Scene VQA 与 Execution VQA 也应区分：前者检查生成场景是否符合请求，后者检查真实
rollout 中的可见现象。official passthrough 不生成场景，因此不需要 Scene VQA。

## 6. 扩展点与当前限制

| 需求 | 优先扩展位置 | 当前限制 |
| --- | --- | --- |
| 新 official expert 任务 | TaskSchema、任务 VQA 映射、必要的 Trusted Tool | 可复用 Recorder/聚合；需真实 seed 验收 |
| 新任务 ACT 评估 | TaskSchema、选择性 checkpoint 下载、通用 ACT backend、preflight | official passthrough 已支持；当前仅约定 `demo_clean-50` |
| 新 generated 任务族 | planner template、TaskGen contract、知识卡、repair gate | 当前生成/修复 contract 仍以 BBH 为主 |
| 新数值 metric | 已有 Trusted Tool 或 ToolGen target + required signals | Recorder 未记录的信号不能事后推断 |
| 更高频动力学 | 新显式 telemetry profile | 默认 `balanced_v1` 只保存 50 Hz dynamics |
| 更丰富视觉 | 新 visual capture profile | `event_keyframes_v1` 无 depth/segmentation，也不连续 |
| expert/ACT 严格对照 | 显式 seed manifest、paired runner、Easy/Hard 统计 | `both` 会拒绝实际 seed 不一致，但尚无预先锁定与 paired 统计 |

可读性维护约定：当入口、路线、artifact contract 或可信边界改变时，简要同步本文件；
当安装/命令改变时同步运行指引；真实实验结果放入 development log，不把易过期的单次
数值堆进架构文档。
