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
| 严格 paired 评估 | `scripts/manipeval_paired.py`、`mea/paired.py` | 冻结 exact seed，运行 Easy/Hard eligibility 与 ACT，并做确定性逐 seed 统计 |
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

### 2.3 Exact-seed Easy/Hard paired 路线

`scripts/manipeval_paired.py` 是与 Agent 解耦的确定性实验入口。它不调用 UIUI、planning、
Execution VQA 或反馈模型；其职责是让同一个官方任务、同一个 `demo_clean-50` ACT
checkpoint 在两个环境 condition 上使用相同的 numeric seed，并保留完整拒绝原因：

```text
显式 --seeds 或已有 seed manifest
→ 验证任务名、顺序、非负整数、非空且无重复
→ 冻结 seed_manifest.json（Easy=demo_clean，Hard=demo_randomized）
→ 以 ACT 的 eval mode 对每个请求 seed 分别做两边 exact expert eligibility probe
→ 固化两边都 eligible 的有序交集（不扫描、不替换）
→ 两个 condition 分别调用 ACT exact-seed evaluator
→ 核验 evaluator 实际 seed 与冻结交集完全一致
→ 读取 policy success 与 telemetry time-to-success
→ mea/paired.py 按 seed join 并写 paired summary
```

eligible 但另一边不 eligible 的 seed 也保留在结果里，只是不进入 paired policy denominator。
任何 evaluator 少跑、多跑、重复或替换 seed 都是 protocol violation，而不是可以忽略的失败
episode。summary 的成功率分母是两边均实际执行 policy 的 paired seed；同时保存请求数、
共同 eligibility、coverage 和四格结果（双成功、仅 Easy、仅 Hard、双失败）。
`time-to-success` 只在两边都成功且都有有效时间的 seed 上比较，因此是 survivor-conditional
辅助指标，不能代替全样本成功率。

严格之处是 seed 清单、顺序和“不替换”契约，不是 identical-scene 因果控制。Easy 与 Hard
的配置分支可能在 actor 放置前消费不同数量的随机数，因此相同 numeric seed 仍可能对应不同
的潜在几何。当前结果应称为 same-seed paired comparison；若论文需要“同一底层场景仅改变
随机化强度”，必须进一步拆分 RNG stream 或持久化并重放 scene specification。任何协议
违反都会令 `valid_for_comparison=false`，默认以非零状态退出。

这条路线面向可复现数值实验，不生成视觉解释；需要视觉证据和自然语言报告时，仍由 Agent
路径单独完成。两者的结果不能仅凭相同 `run_id` 自动视为同一实验，必须以 seed manifest、
condition、checkpoint 设置和 artifact 路径核对。

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

mea/paired_runs/<run_id>/
├── seed_manifest.json            # 请求 seed、condition 与 checkpoint contract
├── eligibility/                  # Easy/Hard 逐 seed probe 与冻结交集
├── conditions/                   # 两个 condition 的 exact-seed ACT 结果与 telemetry
└── paired_summary.json           # 确定性 paired join、coverage、成功率与时间指标
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
| Easy/Hard 严格对照 | `scripts/manipeval_paired.py`、`mea/paired.py` | 已支持 exact seed 与 paired 统计；当前独立于 Agent/VQA，正式论文结论仍需足量预注册 seed |

可读性维护约定：当入口、路线、artifact contract 或可信边界改变时，简要同步本文件；
当安装/命令改变时同步运行指引；真实实验结果放入 development log，不把易过期的单次
数值堆进架构文档。

## 7. 2026-07-17 新增的三条最小通路

### 7.1 ACT-only 完整 Agent 协议层

`scripts/manipeval_protocol.py` 与 `mea/protocol.py` 位于 Agent 之上，不替代 Agent、RoboTwin
或 paired runner。它按 repetition 调用完整 `manipeval_agent.py`，固定 `ACT + --no-history`，
并从 evaluation manifest、child manifest 与 ACT `episode.json` 回收真实分母、成功、步数和
wall-clock。预算限定为 1 / 3 / 5，默认 1；append-only attempt、chunk/resume、锁、PID 与
Git/config/schedule 校验用于避免中断后覆盖或跨代码版本混算。

```text
protocol config + seed schedule
→ complete Agent evaluation (official task, ACT)
→ child TaskGen/ACT/Tool/VQA/Feedback artifacts
→ validate ACT episode metadata and denominator
→ protocol_summary.json + protocol_report.md
```

这一层是低成本论文协议骨架，不是论文的正式 10-repeat 实验。N=1 明确标为 smoke；重复
实际 seed、缺 artifact 或 pipeline failure 会令结果不可比较。

### 7.2 第二个真正 generated family：click_bell position_lr

`ClickBellPositionPlanAgent` 生成 left-fixed 与 right-fixed 两个受信 template；
`mea/taskgen/click_bell.py` 只编译 `bell.xy` declarative overlay；运行时薄 subclass 位于
`mea/tasks/click_bell.py`。两轮共用同一组 seed，policy failure 是有效实验结果，不会阻止
第二轮；scene/pipeline failure 才提前停止。

```text
position_lr Plan (left, right; same seeds)
→ bounded VariantSpec + overlay.yml (no text codegen)
→ preserve official RNG consumption and bell semantics
→ simulator XY + rule + visual plausibility + expert gate per seed
→ ACT rollout
→ Trusted Tool + Aggregate + Dynamic Execution VQA
→ round evidence with declared and measured bell positions
```

Scene VQA 不拥有精确坐标判定权；`tracked_actors[id=bell]` 的 simulator pose 才是数值权威。
这证明系统已不再只有“BBH generated + 其他任务 passthrough”，但仍只是位置这一种受限变化，
也仍使用确定性 template planner，不能冒充论文中的开放式任务生成。

### 7.3 缓存式 Planner / VQA 验证层

`scripts/manipeval_validate.py` 与 `mea/validation.py` 读取 suite 中显式列出的既有 artifact。
它不创建 provider，也不启动仿真；每个 artifact 经过路径 containment、hash 和原 Planner /
Execution VQA contract 校验后才计分。确定性 planner 自动排除出模型 Planner 指标；VQA 的
human 与 simulator-proxy 标签分层汇总，单类别时 AUROC 为 unavailable，而不是伪造数值。

新增运行产物：

```text
mea/protocol_runs/<run_id>/
├── protocol_manifest.json
├── repetitions/rep_*/attempt_*/
├── summary/protocol_summary.json
└── protocol_report.md

mea/validation_runs/<run_id>/
├── validation_summary.json
└── validation_report.md
```

当前验证层完成的是可审计的 scorer 与 cached smoke；真正补齐论文 Table 6–8 仍需要独立、
人工标注且覆盖正负/困难扰动的 Planner 与 VQA 数据集。
