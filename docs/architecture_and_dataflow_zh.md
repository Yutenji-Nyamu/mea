# MEA 架构与数据流

本文说明当前实现，而不是目标蓝图。MEA 不替换 RoboTwin 的物理仿真、机器人控制、
官方任务成功判定或 ACT 推理；它在执行前增加受限规划/生成，在执行中记录可审计证据，
在执行后用确定性工具、视觉检查和反馈组织评估结果。项目长期目标、论文逐点映射、可复用
资产和跨对话开发规则见 [MEA 项目手册](project_playbook_zh.md)。

## 1. 分层与主要模块

| 层 | 主要位置 | 职责 |
| --- | --- | --- |
| 端到端编排 | `scripts/manipeval_agent.py` | 创建 evaluation、执行多轮 plan、汇总证据并生成反馈 |
| 规划 | `mea/planner/` | BBH 使用受限 Plan Agent；`click_bell` 可使用证据约束的属性 Planner；其他已注册任务使用确定性的 official planner |
| 检索与历史 | `mea/retrieval/`、`mea/history/`、`mea/knowledge/` | 检索任务/源码知识，复用历史评估上下文 |
| TaskGen | `scripts/manipeval_taskgen.py`、`mea/taskgen/` | 生成或复用受限 task overlay；也可创建不改官方源码的 passthrough run |
| TaskGen capability | `mea/taskgen/capabilities.py` | 用共享 capability catalog 和 `VariantSpec` v2 固定受控轴、生成模式与必须保留的官方语义 |
| RoboTwin 执行 | `mea/taskgen/probe.py`、`policy/ACT/eval_mea.sh` | setup/render、official expert `play_once()`、ACT rollout |
| 严格 paired 评估 | `scripts/manipeval_paired.py`、`mea/paired.py` | 冻结 exact seed，运行 Easy/Hard eligibility 与 ACT，并做确定性逐 seed 统计 |
| 完整 Agent 协议 | `scripts/manipeval_protocol.py`、`mea/protocol.py` | 用 1 / 3 / 5 预算重复完整 ACT Agent；generated 样本按 `(variant_id, seed)` 核验并逐变体统计 |
| 小型实验/验证 | `mea/benchmark_pilot.py`、`mea/query_dataset.py`、`mea/vqa_perturbations.py` | 三任务 N=1 聚合、20-query 草稿校验和缓存 montage 图像代理扰动 |
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

- `generated` route：使用受限生成/修复的 task overlay；当前覆盖 BBH，以及 declarative `click_bell` 属性变体；
- `official` route：直接复用 RoboTwin 官方任务，不生成或改写任务源码；
- `expert|act|both` backend：分别表示只运行 expert、以 ACT 为被评 policy，或同时保留
  expert 验证与 ACT 评估。official route 可使用这三种 backend，不再与 expert 绑定。

### 2.1 Generated + ACT 路线

当前 BBH 走受限代码生成/修复，`click_bell` 走不生成 Python 的 declarative 属性 overlay：

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

## 3. TaskSchema 与 TaskGen Capability 是两条跨任务边界

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

`mea/taskgen/capabilities.py` 进一步把“可生成什么”从 BBH 和 `click_bell` 的薄适配层中抽出。
共享 capability card 声明 `capability_id`、受控轴、允许的 generation mode、默认 metric，以及
必须保留的官方 pose/instance RNG、`play_once()`、`check_success()` 和 checkpoint 语义。
`VariantSpec` v2 的固定 envelope 为：

```text
task_name + variant_id + capability_id + intent
+ controlled_axis + generation_mode + changes + preserve
```

受信 catalog 注入 `controlled_axis`、`generation_mode` 和 `preserve`；任务适配层只能填写已经
过任务级验证的 `changes`。旧 BBH/`click_bell` spec 可读取并升级，但新 artifact 统一写 v2。
Capability 描述 TaskGen 权限，TaskSchema 描述 telemetry 语义；两者故意分离，增加 generated
能力不会静默改变 Recorder 或使已验证 Tool 缓存失效。这是论文 Sec. 3.3.1、Fig. 3 的共享
TaskGen 合同骨架，当前只覆盖两个任务族和少量受限轴，不是通用 3D task generator。

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

official profile 继续以 `seed` 作为样本身份；`click_bell position_lr` 的 v2 协议把样本身份
升级为 `(variant_id, seed)`。同一个 numeric seed 出现在 left/right 两个 variant 中是设计内
现象，只记为诊断；缺失、额外或重复的复合身份才是 protocol violation。manifest 冻结预期
variant 集，summary 同时输出整体和逐 variant 的 requested/observed、coverage、成功率、policy
step、physics step、simulation time 与 rollout wall-clock，避免把两轮错误折叠成一个 seed。

这一层是论文 Sec. 3.2、Fig. 5 和 Tables 1–2 所需的低成本实验协议骨架，不是论文的正式
10-repeat 实验。N=1 明确标为 smoke；缺 artifact、身份漂移或 pipeline failure 会令结果不可
比较。恢复仍以 repetition 为粒度，尚不能从某个 variant 中间继续。

### 7.2 第二个 generated family：click_bell 属性自适应

兼容 profile `position_lr` 仍由 `ClickBellPositionPlanAgent` 固定运行 left/right 两轮。
新增 `adaptive_properties` 由模型从开放查询选择 `object_position`、`object_instance` 及首个
方面；精确 variant、seed、gate 和 Tool 都由受信目录注入，模型不能任意生成执行参数。
位置轴使用 left/right fixed XY；实例轴使用官方 base0/base1，两者包含外观、大小和接触高度
差异，因此不把它描述成纯颜色或纯纹理变化。

```text
open query → model selects requested aspects + first aspect
→ trusted template catalog materializes one bounded round
→ bounded VariantSpec + overlay.yml (no text codegen)
→ preserve official pose/instance RNG consumption and bell semantics
→ simulator XY or task_attribute bell_id + rule + visual plausibility + expert gate
→ ACT rollout (same click_bell checkpoint)
→ Trusted Tool + Aggregate + Dynamic Execution VQA
→ hard evidence policy derives exactly one drill_down / switch_aspect / stop direction
→ model summarizes the real evidence under that constraint → next round or stop
```

位置 variant 固定 XY、保留官方随机实例；实例 variant 固定 `bell_id`、保留官方随机 pose。
两轴都先消费官方随机调用再覆盖受控值，因此同 seed 可形成正交对照。Scene VQA 不拥有精确
坐标或实例 ID 判定权：前者来自 `tracked_actors[id=bell]`，后者来自 simulator task attribute。

证据状态机把 policy failure 与 pipeline failure 分开：流水线失败只能停止；聚合不完整或 VQA
冲突只能在同一方面补反事实（无目标时带 unresolved 停止）；有效 policy failure 优先深挖
同方面；成功且还有未覆盖方面时切换。Provider 返回相反动作会被拒绝并重试。当前仍是两个
人工定义属性、四个受信 template 的最小闭环，不等同于论文中的通用开放式 TaskGen。
一次真实 N=1 smoke 见
[2026-07-17 开发记录](development_log_20260717_adaptive_click_bell_zh.md)。

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

## 8. 2026-07-17 新增的论文对齐最小通路

### 8.1 `click_bell` aspect-driven ToolGen

`object_position.fixed_xy` capability 会请求新 target
`bell_active_tcp_min_xy_error`：根据初始 bell x 选择官方 active arm，在完整 semantic trace 上
计算对应 TCP 到 bell contact point 的最小 XY 距离，并保存最小值所在 physics step、仿真时间
和 arm。它是连续数值，不强行设 pass threshold，因此 `passed=null`。私有 oracle 只用于生成
时的独立校验，模型生成的工具不能读取 oracle 输出。

```text
aspect + required telemetry signals
→ generate candidate Tool
→ static safety / schema / oracle / determinism validation
→ register in evaluation-local registry
→ later round resolves the same request and reuses registered Tool
```

这条 `generate → validate → register → reuse` 通路对应论文 Sec. 3.3.2、Fig. 4。它已证明
`click_bell` 不只使用固定 Trusted Tool；但 registry 仍限定在单次 evaluation，且一个 metric
不能证明论文中的开放式 ToolGen 覆盖率。

### 8.2 ACT 三任务 N=1 聚合

`scripts/manipeval_benchmark_pilot.py` 读取配置中明确列出的 `adjust_bottle`、`grab_roller`、
`click_bell` 同 task/seed direct official ACT artifact 与完整 Agent protocol artifact。聚合器先
验证 direct paired summary 和 Agent protocol 都可比较，再记录 binary success、policy steps、
physics steps、rollout/process wall-clock 和 exact binary agreement。

该入口为 Tables 1–2 的计量基础做 instrumentation smoke。N=1 没有方差，direct route 也没有
可比的外层进程墙钟，因此 `table2_consistency=null`、`paper_table_eligible=false`；它不能被称为
论文效率表或结论一致性复现。

### 8.3 20-query 草稿与缓存 montage 扰动

`configs/manipeval_validation/query_aspects_draft_v1.json` 固定 20 条开放 query/aspect 草稿，
其中当前 capability 支持 5 条、未支持 15 条。每条都强制保存
`source=model_draft`、`review_status=unreviewed`、空 `human_votes`；校验器不会输出人机一致性。
它只是论文 Table 6 的人工 review/import 前置格式，不是 human gold。

`scripts/manipeval_vqa_perturb.py` 对缓存的真实 rollout montage 确定性生成 clean、scene-clutter
image proxy、background-texture image proxy 和 lighting image proxy，保持原 query 与数值证据
hash，再调用现有 Execution VQA。结果按 perturbation 与 label source 聚合；clean 图做字节级
核对，派生图记录 transform/hash。这对应 Tables 7–8 的低成本接口验证，但扰动发生在缓存图像
而非 RoboTwin simulator，标签来自 simulator proxy 而非人工标注，所以
`paper_table_eligible=false`，也不能把其 accuracy/AUROC 写成论文指标。

本批实现、真实 smoke 和限制集中记录在
[2026-07-17 generated protocol / capability / ToolGen / pilot 开发记录](development_log_20260717_generated_protocol_capability_toolgen_pilot_zh.md)。
