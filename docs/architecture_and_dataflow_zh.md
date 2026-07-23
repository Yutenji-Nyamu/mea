# MEA 架构与数据流

本文说明当前实现，而不是目标蓝图。MEA 不替换 RoboTwin 的物理仿真、机器人控制、
官方任务成功判定或 ACT 推理；它在执行前增加受限规划/生成，在执行中记录可审计证据，
在执行后用确定性工具、视觉检查和反馈组织评估结果。当前实现只覆盖两个任务族和少量受信
capability，是受限功能原型，不是开放世界 TaskGen 或论文规模结果。项目长期目标、论文逐点映射、
可复用资产和跨对话开发规则见 [MEA 项目手册](project_playbook_zh.md)。

## 1. 分层与主要模块

| 层 | 主要位置 | 职责 |
| --- | --- | --- |
| 端到端编排 | `scripts/manipeval_agent.py` | 创建 evaluation、执行多轮 plan、汇总证据并生成反馈 |
| 规划 | `mea/planner/` | 全局可信 ACT catalog 把开放 Query 路由到 BBH/click_bell；任务 Planner 再物化受信轮次，通用证据合同决定继续方向 |
| Bound PlanSession | `mea/planner/session.py` | 一次 evaluation 冻结一个 RoboTwin task、一个 ACT checkpoint 和轮数上限；公共 evidence policy 决定 action/aspect/template，并对 task adapter 物化的候选轮次做最终裁决 |
| 语义 Proposal | `mea/proposals.py`、`mea/proposal_agent.py`、`scripts/manipeval_proposal.py` | 用受限 `TaskProposal`/`ToolProposal` 描述“测什么”；主 Agent 可用 `novel_first_round` 生成未精确登记的新变式，再投影到可信 capability envelope；路径、seed、checkpoint 和 gate 仍由 runtime 注入 |
| 检索与历史 | `mea/retrieval/`、`mea/history/`、`mea/knowledge/` | 检索任务/源码知识，复用历史评估上下文 |
| TaskGen | `scripts/manipeval_taskgen.py`、`mea/taskgen/` | 生成或复用受限 task overlay；也可创建不改官方源码的 passthrough run |
| TaskGen resolution | `mea/taskgen/resolver.py` | 在上游可信 capability 已选 route 后、provider 创建前计算 exact executable semantic key；记录 official/内置 materializer，并可从显式审核的 task registry 做 exact semantic lookup |
| TaskGen capability | `mea/taskgen/capabilities.py` | 用共享 capability catalog 和 `VariantSpec` v2 固定受控轴、生成模式与必须保留的官方语义 |
| reviewed Task 与生产验收 | `mea/taskgen/reviewed_registry.py`、`mea/taskgen/production_acceptance.py` | 安装经显式审核的 generated Task；新 run 复核 immutable provenance、当前 runtime dependencies、artifact contract 与 ACT 前置条件，derived bundle 在 run-local 重建 |
| TaskGen 局部恢复 | `mea/taskgen/attempts.py` | 在 policy 启动前对 SuccessSpec、code/static、render/vision、expert failure 做最多一次 typed repair/regenerate；policy failure 不重试 |
| RoboTwin 执行 | `mea/taskgen/probe.py`、`policy/ACT/eval_mea.sh` | setup/render、official expert `play_once()`、ACT rollout |
| 严格 paired 评估 | `scripts/manipeval_paired.py`、`mea/paired.py` | 冻结 exact seed，运行 Easy/Hard eligibility 与 ACT，并做确定性逐 seed 统计 |
| 完整 Agent 协议 | `scripts/manipeval_protocol.py`、`mea/protocol.py` | 用 1 / 3 / 5 预算重复完整 ACT Agent；generated 样本按 `(variant_id, seed)` 核验并逐变体统计 |
| 小型实验/验证 | `mea/benchmark_pilot.py`、`mea/query_dataset.py`、`mea/vqa_perturbations.py` | 三任务 N=1 聚合、20-query 草稿校验和缓存 montage 图像代理扰动 |
| Checkpoint 获取 | `scripts/download_act_checkpoint.py` | 按任务和固定 revision 只下载 ACT 所需的 policy/stats 文件 |
| 任务语义与记录 | `mea/toolkit/schema.py`、`mea/toolkit/recorder.py`、`mea/toolkit/schemas/` | 用 TaskSchema 跟踪 actor/语义，写 telemetry、事件和视觉证据 |
| 可信测量 | `mea/toolkit/tools.py`、`mea/toolgen/` | 复用 Trusted Tool，或生成并验证 Tool；run-local 直接复用，显式审核后可跨 evaluation 精确复用 |
| 聚合 | `mea/toolkit/aggregate.py` | 跨 episode 做确定性统计，不让语言模型自行算成功率/均值 |
| Execution VQA | `mea/execution_vqa/` | 从受限问题目录选择问题，也接受经严格 schema 校验的 run-local ToolProposal 问题；读取事件关键帧并检查可见现象，不能覆盖 simulator 数值权威 |
| 反馈 | `mea/feedback/` | 把结构化 observation 和证据索引整理为最终报告 |
| 可读证据报告 | `mea/feedback/evidence_report.py`、`scripts/manipeval_evidence_report.py` | 从真实 evaluation 生成含 proposal、代码/overlay、render、ACT 视频、Tool/VQA/Aggregate/decision 的紧凑报告，并可发布小型 GitHub bundle |
| 模型适配 | `mea/providers/` | 各阶段模型 profile 与 OpenAI-compatible provider |
| 运行时审计 | `mea/runtime_ledger.py`、`mea/round_provenance.py` | 在 provider/ACT 调用前持久化开始记录，并把每轮计划、summary 与实际 child/Tool/VQA/ledger artifact 做 hash 绑定 |
| 跨任务父层 | `mea/portfolio.py`、`mea/evaluation_graph.py`、对应 CLI | portfolio 审核并汇总 completed child；EvaluationGraph 用 child outcome 条件决定是否启动下一 checkpoint-bound child |

`scripts/manipeval_taskgen.py` 是内层入口；它也适合做 setup/expert/ACT 的单次调试。
`scripts/manipeval_agent.py` 是正常端到端入口，负责把多个内层 run 组织成一次 evaluation。

### 1.1 全局开放 Query 入口

`scripts/manipeval_agent.py --auto-route` 是 Fig. 2 顶层入口。它先建立只读、受信的 ACT
evaluation catalog；只有同时存在 TaskSchema、`dataset_stats.pkl` 和 `policy_last.ckpt` 的
任务才会暴露给模型。当前 allowlist 只有 BBH 与 `click_bell`，并固定各自 profile、aspect、
template、metric 和最大轮数；checkpoint 路径、Python module、seed、gate 和 variant 内容从不
进入模型输出。

```text
open query
→ completed-only global planning history（trusted task allowlist）
→ build_act_catalog() + catalog SHA-256
→ GlobalQueryRouter：route supported subset 并保留 gaps，或在没有可回答子集时显式 unsupported
→ strict validator：task/profile/aspect 必须来自 catalog
→ route_to_planner_proposal()：只在 evaluation 开始时选一次 task/checkpoint
→ EvaluationTarget + BoundTaskPlanSession
→ catalog round，或 novel_first_round 生成 TaskProposal + ToolProposal v2
→ capability envelope 校验 changes/metric/gates，task adapter 只物化轮次
→ capability-bound task resolution → TaskGen → ACT → Tool/VQA/Aggregate
→ BoundTaskPlanSession.directive() + adjudicate()
→ drill_down / switch_aspect / stop
→ Feedback 回答原始 query
```

默认 catalog 模式下，全局 Router 已调用模型后，任务 Planner 不再为首轮重复调用模型；显式
`--proposal-mode novel_first_round` 会增加一次受限 Proposal 调用，为首轮产生 catalog 中没有精确登记、
但仍处于 capability 范围内的 Task/Tool/VQA 请求。后续 Planner 只能解释公共 PlanSession 确定的
动作、转移与目标；candidate plan 还必须通过 `adjudicate()`，不能覆盖 task/checkpoint、预算、
历史轮次或 evidence-selected template。若宽 Query 同时包含受支持与未支持轴，Router 可执行同一
task 中的 supported subset，并把其余 task-qualified capabilities 留在 route/最终 limitations；只有
没有任何 material answer 时才生成 `status=unsupported` 的无执行 evaluation。两种情况都不会偷偷
降级到无关任务或把 gap 伪装成已支持能力。

一次 evaluation 只绑定一个 task 和它的 ACT checkpoint。Planner 可以在该 task 的受信
sub-aspect/variant 间继续、切换或停止，但不能切换 task/checkpoint；跨任务回答由 portfolio
创建多个独立 child evaluation。`BoundTaskPlanSession` 的实现本身对 task 无关，当前则仍由
BBH/click_bell adapter 在其后提供 materialization 细节。

关键 artifact 位于 `plan/global_act_catalog.json`、`plan/global_query_route.json`、
`plan/global_query_prompt.md`、`plan/global_route_proposal.json` 和
`plan/evidence_after_round_*.json`。novel 模式还会保存
`plan/bounded_proposal/{prompt,response_*,proposal_bundle,execution_vqa_query_smoke}`；真实执行时公共
裁决另写 `plan/runtime_directive_after_*.json`。历史只消费已完成 evaluation；plan-only 不会反向
写成执行证据。

TaskProposal 与 capability 同时存在时，TaskGen 在 provider 创建前计算 resolution，并把成功物化后的结果
保存为 `generation/task_resolution.json`。resolver 只允许 exact executable semantic match；Query/intent
改写不改变 executable key，而 changes、capability、SuccessSpec 或 contract 改变都会产生新 key。official
与内置 overlay 继续执行上游可信 capability 已选择的 materializer；force-codegen 可在正常 runtime 中查询
显式配置的 reviewed-task registry。命中时由 registry 的 VariantSpec 与 immutable artifacts 掌握 authority，
当前请求的 run-local variant id 可以不同，但语义字段、contract 与 provenance 必须匹配；未命中则诚实转入
codegen。当前真实 registry 只覆盖一个受限 BBH variant，仍不是论文 Fig. 3 面向任意资产/文档的全局
retrieve-first selector。

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
6. 未审核的 generated Tool 只在当前 evaluation 内自动复用；经显式人工审核并精确固定
   registration/code/ToolSpec/contract/schema hashes 后，才可进入 reviewed persistent registry
   跨 evaluation 复用，但始终不会自动晋升为全局 Trusted Tool；
7. reviewed generated Task 同样不能自动晋升：当前 registry 固定 task.py、VariantSpec、
   overlay/load_actors、可选 SuccessSpec 与 validation/static 等 immutable inputs，以及登记的 5 个
   Python runtime dependencies。`TaskArtifactBundle` 与 `SceneCheckSpec` 是 run-local derived rebuild，
   production acceptance 必须在 ACT 前重新核验。development-agent review 不是独立人工审核，也不使
   artifact 具备 paper eligibility。

Scene VQA 与 Execution VQA 也应区分：前者检查生成场景是否符合请求，后者检查真实
rollout 中的可见现象。official passthrough 不生成场景，因此不需要 Scene VQA。

## 6. 扩展点与当前限制

| 需求 | 优先扩展位置 | 当前限制 |
| --- | --- | --- |
| 新 official expert 任务 | TaskSchema、任务 VQA 映射、必要的 Trusted Tool | 可复用 Recorder/聚合；需真实 seed 验收 |
| 新任务 ACT 评估 | TaskSchema、选择性 checkpoint 下载、通用 ACT backend、preflight | official passthrough 已支持；当前仅约定 `demo_clean-50` |
| 新 generated 任务族 | planner template、TaskGen contract、知识卡、repair gate | 当前生成/修复 contract 仍以 BBH 为主 |
| 新成功语义 | 公共 Proposal v2、oracle-bounded SuccessSpec、正负 fixture 与 production acceptance | compiler/envelope 已有；公共 Proposal Agent 仍主要产生 v1/official-preserving 语义，尚无 ACT live 证据 |
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
→ explicit source/evidence review pins code + ToolSpec + contract + schema hashes
→ install into reviewed persistent registry
→ new process/evaluation exact-match lookup
→ current trajectories re-run determinism + private-oracle gates, provider=false
```

这条 `generate → validate → register → reuse` 通路对应论文 Sec. 3.3.2、Fig. 4。它已证明
`click_bell` 不只使用固定 Trusted Tool。`mea/toolgen/reviewed_registry.py` 增加跨 evaluation
复用，但不会自动提升任何生成代码：pending/candidate 不可执行，只有包含 reviewer、时区时间、
四项人工检查和精确 registration/code/ToolSpec/contract/schema hashes 的 `approved` manifest 才能
安装；路径越界、symlink、内容篡改、当前 schema/contract 变化都会拒绝复用或回退 codegen。
reviewed Tool 仍是 generated Tool，不进入全局 Trusted Tool catalog。当前只有一个 metric 的
缓存跨进程 smoke，不能证明论文中的开放式 ToolGen 覆盖率。

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

### 8.4 TaskGen 功能验收切片

`mea/taskgen/acceptance.py` 与 `scripts/manipeval_taskgen_acceptance.py` 只读核验四类既有真实
artifact：official reuse、`click_bell` bounded overlay、BBH 真 codegen + Task/Knowledge
retrieval provenance，以及 `wrong_color` 错误注入形成的
`static pass → visual reject → diagnosis → repair → static revalidate → visual pass`。验收进程本身
不调用 provider、simulator 或 ACT，并给每个源文件保存 SHA-256。

它证明 TaskGen 的复用、受限生成、检索 provenance 和视觉修复接口曾由真实 artifact 贯穿；
因为本次只是缓存只读复核，顶层固定 `cached_artifact=true`、`no_ACT=true`、
`paper_table_eligible=false`。`oversized_block` 历史 fixture 曾被 VQA 漏检，不纳入通过证据。

本批实现、真实 smoke 和限制集中记录在
[2026-07-17 generated protocol / capability / ToolGen / pilot 开发记录](development_log_20260717_generated_protocol_capability_toolgen_pilot_zh.md)。

全局开放 Query、通用 evidence transition、reviewed persistent Tool registry、TaskGen acceptance
以及 click_bell/BBH 真实 Stage 1 证据见
[2026-07-17 开放 Query 核心闭环 Stage 1 开发记录](development_log_20260717_stage1_open_query_loop_zh.md)。

## 9. 论文协议补齐：proxy 标注、真实 clutter 与恢复

### 9.1 Table 6-facing Planner 验证

`configs/manipeval_validation/query_aspects_development_agent_proxy_v1.json` 保存 20 条经过
开发代理复核的 query/aspect 标签，并按 object、scene、performance、safety、language/multi-task
分类。它与被测 `GlobalQueryRouter` 分离；runner 会实时调用 Planner，而不是复用预写答案。
unsupported 能力使用 `(task_name, aspect_id)`，避免“某 aspect 在任务 A 支持，就被错误当成在
所有任务都支持”的全局 union 漏洞。

```text
20-query proxy labels + current trusted ACT catalog
→ stale-label fail-fast
→ live GlobalQueryRouter call (budget 1/3/5/20)
→ strict schema/task/capability/aspect/first-aspect scoring
→ dataset/catalog SHA-256 + JSON trace + Markdown report
```

这仍不是论文 Table 6：`human_reviewer_count=0`、`paper_table_eligible=false`，后续必须用独立
多人 majority 标注替换 development-agent proxy。

### 9.2 simulator-native clean / scene-clutter VQA

`click_bell` 的 `robustness.scene_clutter` capability 编译为 RoboTwin 原生
`domain_randomization.cluttered_table=true` 且 `clean_background_rate=0`。场景合同从
`task.info.cluttered_table_info` 读取实际 clutter 对象与数量；这不是 RGB 后处理或 image proxy。
同 seed 的 clean 与 clutter rollout 继续使用同一 ACT checkpoint，验证器还要求 bell pose、
quaternion 与 instance 一致，确保比较只改变场景 clutter。

```text
open query → robustness.scene_clutter
→ bounded overlay → RoboTwin native clutter generation
→ simulator-state + render + rule + expert gates
→ ACT N=1 → Trusted Tool/Aggregate
→ reviewed VQAQuerySpec selects allowlisted questions
→ Dynamic Execution VQA
→ proxy labels audit clean vs clutter completed artifacts
```

VQAQuerySpec registry 只能选择代码中已有的可信视觉现象，不能注入自由 prompt；spec、review 与
index 均由精确 hash 固定，路径越界、symlink、篡改或多重匹配一律 fail closed。当前 reviewer
是 `development_agent_proxy`，因此 N=1 汇总的 AUROC 固定为 `null`，仍不具论文 Tables 7–8
资格。

### 9.3 fixed-suite 对照、微消融与有界恢复

`ClickBellFixedSuitePlanAgent` 在第一条 rollout 前冻结与动态 Planner 相同的候选 template suite；
后续 policy/VQA 证据被记录但不用于路由。`mea/strategy_comparison.py` 只接受同 task、ACT
逻辑配置、telemetry profile、base commit、开放 query、global route、candidate-suite hash 与
`(variant_id, seed)` 身份；fixed 还必须完整覆盖冻结 suite，缺失 success 或路径越界直接拒绝。
它只验证 Table 1-facing 的效率机制，不是论文用标准 benchmark 对 MEA 的原始 Table 1 对照；
N=1 也不计算 Table 2 一致性。checkpoint 文件内容 hash 尚未进入合同。

`mea/micro_ablation.py` 只读既有真实 TaskGen/ToolGen artifact，并用确定性 fault counterfactual
完成 4 个 functional gate check；第 5 行只证明 RAG provenance，不计入 functional summary，
也没有 no-RAG effect estimate。它启动 0 次 ACT，不能产生论文 Table 3 的生成成功率。

2026-07-17 的旧兼容实现只在 Tool orchestration 子阶段复用同一 telemetry，额外 ACT 为 0，
因此不同于论文 App. A.3.4 的整轮 restart。2026-07-19 起该兼容 budget 默认 0；当前默认行为与
artifact 见第 11.3 节。

本批实现、真实同 seed Clean/Clutter `N=1`、开发代理标注结果和明确的论文边界见
[2026-07-17 Planner/VQA 代理验证、真实 Clutter 与协议补齐](development_log_20260717_proxy_clutter_protocol_zh.md)。

## 10. 预注册执行身份、原生场景轴与性能路线

### 10.1 Evidence identity 从 parent 贯穿 child

`mea/evidence_manifest.py` 在 clean Git HEAD 上把 Query、候选 template suite、checkpoint 文件
内容、telemetry profile、`(strategy, variant_id, seed)` 样本表和源 artifact 固定为 SHA-256
identity。`mea/strategy_plan.py` 再从该 manifest 生成不可执行的 registered route、fixed/dynamic
精确 argv 和后处理配置；生成这两类文件本身均启动 `0` 次 ACT。

```text
prereg config
→ hash-pinned evidence manifest
→ registered route + exact command plan
→ Agent preflight validates manifest / plan / route / observed argv
→ parent registration_identity
→ TaskGen child --registration-identity-json
→ child manifest returns the same identity
→ parent rejects mismatch
→ completed strategy artifacts
→ registered comparator revalidates identity before comparison
```

registered execution 禁止 live `--auto-route`，避免运行时模型重新选择不同 task/suite。manifest、
route、plan、evaluation id 或实参缺失/漂移都会 fail closed；parent 还会拒绝 candidate suite 与
child identity 变化。canonical self-hash 只证明文件内容未变，不能单独证明命令已执行；真实
证据仍由 parent/child artifact 和 post-hoc comparator 共同形成。

当前注册协议只覆盖 `click_bell`、同一 seed、每个 candidate 一次。fixed 必须遍历完整冻结
suite，dynamic 可由证据提前停止；这验证 Tables 1-facing 的样本节省机制，不提供 N=1 下不存在
的 Table 2 consistency。

### 10.2 `click_bell` 的 simulator-native scene capabilities

`adaptive_properties` 现在除 position、instance、clutter 外，还支持：

| Capability / template | RoboTwin 原生变化 | simulator 数值权威 | VQA 角色 |
| --- | --- | --- | --- |
| `scene_background_texture` / `scene_background_texture.unseen` | `random_background=true`、`clean_background_rate=0`，eval mode 选择 unseen wall/table texture | `task.info.texture_info` | bell 在 unseen 背景下是否仍清晰可见 |
| `scene_lighting` / `scene_lighting.static_random` | `random_light=true`、`crazy_random_light_rate=0`，每 episode 静态随机光色 | simulator light configuration | bell 是否因曝光问题而不可见 |

两者都编译为 bounded overlay，保留官方 bell、pose/instance sampling、任务成功语义和 ACT
checkpoint，并继续经过 structure、simulator state、render/rule 与 expert gates。它们不同于
缓存 montage 的 image proxy；VQA 只能判断 allowlist 中的可见现象，不能覆盖 simulator state
或 `check_success()`。

### 10.3 Official performance route

`performance.completion_time_stability` 不生成任务变化，而选择
`performance.completion_time_stability.official`：

```text
open Query
→ global catalog: click_bell / adaptive_properties / performance
→ task_execution.official_passthrough（原官方场景与 ACT checkpoint）
→ trusted time_to_success
→ Aggregate：success-conditioned completion-time statistics
→ bell_visibly_pressed VQA
→ evidence transition / final feedback
```

该路线明确区分“任务生成能力”和“在官方任务上执行的性能测量”。`time_to_success` 读取记录器中
首次成功时间，不由模型估计；VQA 仅补充可见按铃证据。预算 1 只验证接线，3/5 才能形成小样本
稳定性描述。

完整实现边界与待回填服务器验证见
[2026-07-17 预注册、原生场景轴与性能能力](development_log_20260717_prereg_scene_performance_zh.md)。

## 11. 2026-07-19：声明式 capability、真实开关与整轮恢复

### 11.1 一份合同贯穿 Plan → TaskGen → Tool/VQA/gates

`mea/capability_adapter.py` 把 BBH 与 `click_bell` 的受信 template 统一成同一 JSON-compatible
合同：

```text
(task_name, template_id)
→ canonical aspect + object/scene/performance/execution scope
→ TaskGen operation + capability_id + task_variant_id + allowed change roots
→ Tool request factory + metric
→ allowlisted Execution VQA phenomena
→ required gates
```

Planner 只选择 template；adapter 注入可执行细节。BBH 的三个评估 template 明确复用同一个
`task_variant_id=object_appearance.color_blue`，不再把“评估方面”误写成“新生成任务身份”。
`mea/aspects.py` 只接受显式 alias，不做 fuzzy matching；object capability 只能改 `block/bell`，
scene capability 只能改 `domain_randomization`。该边界同时由 adapter 与 VariantSpec v2 校验。
Agent 在启动 TaskGen 前逐项核对 route、VariantSpec、Tool request、VQA phenomena 与 gates；BBH
的 planner-owned VariantSpec 直接成为生成输入，模型 proposal 不能改写它。TaskGen 物化后再做
一次 exact binding，official passthrough 同时核对 task module、空 overlay 与 official spec。

### 11.2 Table 3 开关现在真正执行

`mea/module_ablation_execution.py` 消费 hash-bound schedule，按论文的 TaskGen
`complete/no_rag/no_visual_self_check/no_readme_agent/base` 与 ToolGen `complete/no_rag` 走不同
代码分支；Tool validation 是两条 ToolGen 分支共享的不变量，仅切换 RAG。每个 item 保存模块
call count、candidate、deterministic contract judge、typed outcome 和
execution trace。内置 driver 是 0-provider、0-simulator、0-ACT 的开发 smoke，只证明控制流，
固定 `paper_table_eligible=false`；论文 Table 3 仍需真实生成和独立人工审核。

### 11.3 App. A.3.4 stage-aware whole-round recovery

Agent 默认关闭旧 same-telemetry Tool retry，启用最多一次整轮恢复。只有
`tool_execution/unexpected_exception` 会启动新 child run、新 execution 目录和新 ACT；失败 attempt
不会被覆盖。policy/simulator failure 不重试。每轮保存不可变 identity、attempt、总 ACT 和恢复
额外 ACT。planning/TaskGen/ToolGen 的局部重试仍分别留在所属 stage，中央 action table 禁止把
未知异常或语义失败升级成整轮重试。

### 11.4 Scene-shift completed-artifact collector

`mea/scene_shift_collector.py` 与 `scripts/manipeval_scene_shift_collect.py` 只扫描 completed passing Agent
run，解析 parent/child、ACT episode/video、Execution VQA/query/montage，并为七类来源写 SHA-256
清单与逐项缺失诊断。它不会启动 provider/simulator/ACT，也绝不从被测 VQA prediction 反推
gold label；只有调用方提供完整 development-agent proxy 标签时才输出 `unvalidated` suite draft。
因此 collector 能证明证据是否齐全，不能单独填论文 Tables 7–8。
当前 evidence bundle 仅作为 discovery index；正式 suite 由七类逐文件 hash、parent child membership
与 VQA/episode 自绑定负责验真，尚未把 `round_id + evidence_bundle hash` 纳入 suite schema。

collector 重验的是 VQA provider 的五字段响应合同；Agent 在响应通过后追加的
`evidence_conflict` 属于派生证据字段，不应再次送入严格 provider schema。scene state 同样按
capability 分开判定：unseen texture 必须来自 `eval_mode=true` 的 unseen split；static lighting
则以 `random_light=true`、`crazy_random_light_rate=0` 和 simulator light component 颜色为权威，
不要求 `eval_mode=true`。这两个边界由真实 completed-run 收集与回归测试共同覆盖。

本批代码、真实 N=1 fixed/dynamic pair、诚实结果和剩余论文 gap 见
[2026-07-19 capability / module switches / recovery / scene pair 开发记录](development_log_20260719_capability_recovery_scene_pair_zh.md)。

## 12. 2026-07-19：调用开始账本、轮级 provenance 与跨任务父层

### 12.1 crash-safe call-start ledger

`mea/runtime_ledger.py` 为每个外部调用保存 append-only JSONL。OpenAI-compatible provider 在
每次 `session.post` 前记录 `provider_transport_started`；同一逻辑调用的重试共享
`logical_call_id`，因此逻辑调用数与 transport attempt 数不会混在一起。TaskGen 在 ACT
subprocess 启动前记录 `act_batch_started`、seed 和声明的 rollout 数。每条记录都先写入并
`fsync`；账本不可写时外部调用不会开始，避免静默少计。

```text
(evaluation_id, logical_round_id, round_attempt_index, child_run_id)
→ provider logical call / transport attempt start
→ ACT batch start / declared rollout count
→ per-stage call_starts.jsonl
→ strict reader + runtime_totals
```

账本只保存身份、模型名、modality 和计数所需字段，不接收 prompt、图像、凭据、header、URL
或 checkpoint 内容。它证明“runner 已在外部调用前持久化开始记录”，不是 provider 已返回、
episode 已完成或 policy 成功；因此 manifest 同时保留 started 与 completed 统计。规划、全局路由、
每轮、轮间决策和最终反馈均使用同一合同。

### 12.2 round provenance sidecar

`mea/round_provenance.py` 在每轮完成后以 exclusive create 写
`summary/<round_id>.provenance.json`。sidecar 绑定 round plan、去除 provenance 指针后的
round summary、最终 attempt/child identity，以及实际存在的 child manifest、VariantSpec、
reflection、ACT、TaskGen command、Tool、Aggregate、Execution VQA、recovery 和 runtime ledger
文件的 SHA-256 与字节数。round summary 只保存 sidecar 指针和 binding hash，避免自哈希循环。

独立 verifier 会重算 plan、summary、sidecar 指针和所有可达文件；缺文件、路径越界、symlink、
hash 或 size 漂移都会 fail closed。这个 sidecar 解决的是“本轮结论究竟引用了哪组运行产物”，
不证明代码科学正确，也不把 provenance 变成 policy outcome 或论文指标。

### 12.3 真实 TaskGen scene repair 与单项验收

BBH 的 `wrong_color` fixture 在正常 static gate 通过后注入结构合法但语义错误的红色目标块；
真实 RoboTwin render 和 Scene VQA 先拒绝场景并给出 typed diagnosis，repair 随后改写受限方法，
再次通过 AST/protected-diff、真实 render、Scene VQA 和 official expert gate。该运行使用 0 ACT，
最终通过；`scripts/manipeval_taskgen_acceptance.py --only-reflection` 可只读核验这一个源 run，
无需依赖其他历史缓存。

这条证据对应 Fig. 3 与 App. A.3.4 的 TaskGen 内部 visual failure→repair，而不是整轮 ACT
recovery。验收命令本身是 post-hoc、0-provider/0-simulator/0-ACT；它验证的是源 run 留下的
真实 artifact，不能把验收时间写成新的 simulator 实验。

### 12.4 live-provider Table 3 micro 与独立 proxy review

`mea/module_ablation_live.py` 从 hash-bound schedule 选择 matched item，按真实开关调用 provider，
但 generation 阶段只写 candidate、call-start ledger 和 `success=null`。之后另一条 `review`
命令才能 append `development_agent_proxy` 标签；provider 输出不能给自己打分，标签也不能覆盖
原 candidate。

当前最小运行比较 TaskGen `complete/no_rag` 与 ToolGen `complete/no_rag`：4 次 provider、
0 simulator、0 ACT，代理审核通过 3/4。TaskGen 两项和 ToolGen complete 通过；ToolGen no-RAG
因输出没有遵守所需顶层 schema 被代理拒绝。这只证明 live matched generation→独立 review 的
数据通路和一次开发观察；`reviewer=development_agent_proxy`、没有独立人工 reviewer，且
`paper_table_eligible=false`，不能称为论文 Table 3 成功率、因果消融或 RAG 效果结论。

### 12.5 同一 Query 的两任务 portfolio

`scripts/manipeval_portfolio.py plan` 从 checkpoint-ready 可信 catalog 固定
`click_bell` 与 BBH，为同一 Query 生成两个精确 child argv；每个 child 强制一轮、一条 ACT、
`--max-agent-rounds 1` 且两种 recovery budget 均为 0，所以父计划的 ACT 上限是 2。plan 是 inert
artifact，本身启动 0 provider/0 simulator/0 ACT。

两个 child 完成后，`reuse` 模式只接受显式 evaluation id，重算每轮 ACT seed、episode 分母、
pipeline 与 policy outcome，并在新格式 child 上验证 call-start ledger 和 round provenance。最终
报告分列 strengths、weaknesses、recommendations 和 limitations，且绝不以 pipeline pass 代替
`policy_success`。

```text
one open Query
→ portfolio command plan (click_bell 1 ACT + BBH 1 ACT hard ceiling)
→ two ordinary Agent child evaluations
→ per-child ledger + round provenance + authoritative policy outcome
→ portfolio reuse verifier
→ strengths / weaknesses / recommendations / limitations
```

当前父层是两个受信任务的最小 adapter，不是任意任务图。`reuse` 汇总本身不启动新 runtime，
也不会反向证明这些 child 是由当前 Query 因果启动；在 command plan 与 completed child 增加
双向 execution binding 前，必须把“严格按计划执行”作为外部操作事实单独说明。

真实 `portfolio_batch10_cross_task_ecbf7b1` 严格按 plan 启动两个 child，provider call-start 合计
`14`，ACT started/completed 为 `2/2`。`click_bell/object_position.left_fixed` 与
`beat_block_hammer/object_appearance.color_blue` 的 pipeline 都通过，ACT policy 都是 `0/1`。父层
因此没有宣称 policy 强项，只把证据链完整列为 provenance strength，并把两个 policy outcome 列为
weakness；这证明最终 synthesis 尊重 Rule Tool outcome，不把 pipeline completion 伪装成成功。

### 12.6 当前真实证据边界

simulator-native unseen texture 与 static randomized lighting 现在各有 seed `100402`、`100403` 两条
独立 completed artifact。离线 collector 得到 candidate `4/4` ready、diagnostics `0`、每个 condition
`2` 个 unique seed；ACT 在 texture 为 `0/2`，在 lighting 为 `2/2`。四张 montage 由开发代理实际查看
后写入 proxy 标签，未从被测 VQA 输出推导标签。

该 suite 仍是 `emitted_unvalidated`：两个 condition 的 primary visibility 标签全为 `true`，严格
validator 因缺少正/负可见性平衡而拒绝。它证明两-seed 真实数据通路与 coverage，不证明跨 seed
稳健性、VQA accuracy/AUROC 或 Tables 7–8；上述新增能力仍处于 small-scale、ACT-only、部分
development-agent proxy 的功能复现层级。

本批实现、真实运行、失败补救和自顶向下剩余 gap 见
[2026-07-19 runtime provenance / repair / portfolio 开发记录](development_log_20260719_runtime_provenance_portfolio_zh.md)。

## 13. 2026-07-19：Bound PlanSession、语义 Proposal 与可读证据

### 13.1 一次 evaluation 的固定边界

`mea/planner/session.py` 新增 `BoundTaskPlanSession`。全局 Router 只在 evaluation 开始时选择一次
checkpoint-ready task；session 随即冻结：

```text
EvaluationTarget
= task_name + task_family + task_profile + planner_kind
+ ACT policy/checkpoint + allowed aspects + max_rounds
```

session 会拒绝 task、policy/checkpoint 合同、未知 aspect 和超预算 round 漂移，并把不同 task
adapter 的旧计划补齐成统一 round schema。`plan/bound_task_session.json` 保存 Query、固定 target、
已选 aspect、轮预算、每轮 proposal 和 decision。核心状态机因此可以跨 evaluation 复用；但一次
evaluation 内不会从 BBH 跳到 `click_bell`。跨任务 portfolio 仍由多个这种单任务 child 组成。

三个 0-ACT plan-only smoke 覆盖了边界：BBH 可把开放 Query 分成 appearance + timing，
`click_bell` 可分成 position + instance；把 `click_bell` 绑定后询问 unsupported friction 时，
系统显式记录 unsupported，而没有改路由到另一个 ready task。

### 13.2 `TaskProposal` 与 `ToolProposal`

`mea/proposals.py` 在 Plan 与 TaskGen/ToolGen 之间增加两个严格语义合同：

- `TaskProposal` 固定 task/aspect/capability、`reuse_first=true`、受 capability 根字段约束的
  `changes`，并强制保留官方 success semantics；它不接受 module path、seed、checkpoint 或 gate。
- `ToolProposal` 固定同一 task/aspect 的 evaluation goal、Rule metric、自然语言问题和受信 VQA
  phenomenon ids；随后投影为既有 route-free Tool request。

主 Agent 先把 task adapter 物化的轮次提升为这两个 proposal，再逐项核对 proposal 与实际
capability、TaskGen、Tool route 和 VQA assignment；proposal 不再只是报告字段。当前 runtime
仍由 BBH/`click_bell` adapter 产生首轮 materialization，这是尚未完全消除的 task-specific 层。

`BoundedProposalAgent` 与 `scripts/manipeval_proposal.py` 进一步演示模型面对固定
EvaluationTarget，只在 capability card 内提出一个不等于现有 template 的新变化，并同时分配
Tool/VQA。真实 smoke 为 `click_bell` 提出 `xy=[-0.14,-0.12]`，TaskGen 将其物化并完成真实
setup/render probe，ACT 为 0。该 CLI 证明“Query→新语义 proposal→真实 TaskGen materialization”，
不代表任意 3D task generation；proposal 仍受 catalog capability 限制，VQA 仍只能选 allowlist。

### 13.3 两轮 BBH live N=1

`eval_20260719_batch11_bbh_adaptive_n1_v3` 用一个开放 Query 完成两轮真实执行：

```text
Query
→ bound BBH / ACT demo_clean-50 session
→ round 1 appearance TaskProposal
→ true Python codegen + render/expert gates + ACT N=1
→ Rule/VQA/Aggregate evidence → continue timing
→ round 2 timing TaskProposal
→ reuse task + generated timing Tool + ACT N=1
→ VQA/Aggregate → final feedback
```

两轮 pipeline 均完成，两个 ACT `policy_success` 都为 `0`；相应 expert controls 成功，只证明场景
可解和计量链可用，不是 ACT 成功。总 wall-clock 为 `587.7 s`。这证明论文 Fig. 2–4 的 proposal、
reuse/codegen、Tool/VQA、evidence-driven next round 与 final answer 在一个 bound session 内真实贯通，
但每轮只有 N=1，不能推断均值、方差或泛化。

本批另有两次集成运行各启动 1 条 ACT 后失败，分别暴露并修复 round budget 被硬编码，以及可选
bound `task_name` 缺省值被错误当作不相等。失败 run 保留，不从预算中删除；本批 ACT started
总数是 `1 + 1 + 2 = 4`，最终 v3 结论只使用其中 2 条 completed evidence round。

### 13.4 illustrated evidence report 与 publish bundle

完整 Agent 成功结束后会自动写 `evidence_report.md`。独立 CLI 还能把同一真实 evaluation 发布到
`docs/evidence_runs/<evaluation_id>/`：

```text
Query + fixed task/checkpoint scope + paper data-flow diagram
→ initial aspect decomposition
→ each TaskProposal → task code/overlay + VariantSpec + real render
→ ACT result + small video
→ ToolProposal → Tool route/source/result
→ dynamic VQA montage + Aggregate + next decision
→ final answer + raw artifact index
```

publisher 只复制报告实际展示的小型真实文件；视频超过上限就保留 server-source 说明，缺失 artifact
显示 `N/A`，不造 proxy。v3 已发布到
`docs/evidence_runs/eval_20260719_batch11_bbh_adaptive_n1_v3/README.md`，整个 bundle 约 `1.1 MB`，
包含代码、render、视频、proposal 和结构化结果。它改善可读性和人工核验，不替代完整
evaluation 目录、provenance verifier 或论文统计。

本批实现和真实证据见
[2026-07-19 Bound PlanSession / Proposal / illustrated evidence 开发记录](development_log_20260719_bound_plan_proposals_evidence_zh.md)。

## 14. 2026-07-19：论文主体方法的公共上下文、typed evidence 与覆盖审计

### 14.1 当前主链

本批把此前分散在 task adapter 中的输入与输出提升为公共合同：

```text
open Query
→ 各 checkpoint-ready task 的 PlanningContext
   ├── PolicyCard：ACT checkpoint、输入/输出与预算边界
   ├── SimulatorCard：RoboTwin task/schema、可观测量与执行约束
   └── AdapterView：允许的 aspect/template/change roots/Tool/VQA/gates
→ GlobalQueryRouter：读取上述 context，固定一个 task，列出 unsupported aspects
→ BoundTaskPlanSession：冻结所选 task/checkpoint/预算
→ TaskProposal + ToolProposal
→ TaskGen：retrieve/reuse/generate
   → TaskArtifactBundle(scene method + official reuse 或 compiled SuccessSpec)
   → SceneCheckSpec → static/render/visual/expert gates → bounded repair
→ ACT rollout
→ trusted Rule Tool 或 typed MetricSpec + run-local Dynamic Execution VQA
→ Aggregate → EvidencePacket
→ evidence-driven transition → 下一轮 bounded Proposal 或 final feedback
```

`PlanningContext` 在初始 Query route 前由受信项目元数据构建，不由模型猜测；`EvidencePacket` 使用
`sufficient / uncertain / conflicting / pipeline_invalid` 等离散强度，并保留 pipeline、policy、
rule 与 VQA 原始字段，不伪造概率。请求了 VQA 而结果缺失、失败或跳过时不会被当成 sufficient。
一次 evaluation 仍固定一个 task 和一个 ACT checkpoint；跨任务
Query 由多个 child evaluation 与 portfolio 表达。

### 14.2 每轮 Proposal 与 TaskGen 产物

`--proposal-mode bounded_each_round` 在首轮生成受限 Proposal，并在真实 evidence policy 返回
`continue` 后为下一轮再次调用 Proposal Agent。候选只能在当前 `EvaluationTarget` 与 capability
envelope 内选择；公共 session 会拒绝 task、policy/checkpoint、预算和不受支持 aspect 漂移。
`plan-only` 没有前一轮 observation，因此只能验证首轮 Proposal 与上下文，不能作为逐轮行为证据。

所有 TaskGen 路由现在应保存同形的 `generation/task_artifact_bundle.json` 与 SceneCheckSpec。bundle
明确区分 `generated_code`、`bounded_overlay_wrapper` 和 `official_reuse` 的 scene 来源。official/overlay
路线复用 RoboTwin `check_success()`；当前 BBH codegen 路线则把封闭 `SuccessSpec v1` 编译为完整
`check_success(self)`，再以 AST 与可信展开做精确差分验证。v1 的 actor、阈值和逻辑仍固定为 official
等价语义，所以“完整 task artifact”表示 scene 与 success method 都可执行、可追踪，不表示模型已经能
任意生成正确的成功函数。

### 14.3 typed MetricSpec 与语义复用

`ToolProposal v3` 可携带 `MetricSpec v1`。当前 DSL 只允许三个受限 operator：
`minimum_distance` 声明左右 trace signal 与二维/三维投影；`event_count` 统计受限事件；
`time_between_events` 计算两个受限事件首次出现的时间差。事件 selector 只接受 recorder 原生的
`contact_interval` / `success_transition`，以及可选的精确 actor pair 和 physical-only 条件，不接受
任意表达式。编译结果必须通过 AST gate、至少两个 episode 的 determinism/oracle differential gate，
并证明核心 telemetry 未被修改，随后才可进入 run-local registry。复用 key 忽略自然语言
`question`，但仍精确绑定 task、metric、typed spec、signals、output 和验证合同；这允许问句改写复用
同一 executable Tool，而不会宽松匹配不同指标。

历史 plan retrieval 同样增加 canonical aspect overlap；文本相似只作为排序的一部分。扩展 ontology
中的 aspect 不会自动进入 ACT capability catalog，无法 materialize 的轴必须显式保持 unsupported。

### 14.4 16 项可执行覆盖审计

`mea/method_coverage.py` 与 `scripts/manipeval_method_coverage.py` 对论文主体 16 项做只读审计。
每项状态由 AST/source check 和声明的 artifact validator 推导：

- `implemented`：所需源码合同存在，且需要运行证据的项目也找到严格通过的 artifact；
- `evidence_pending`：源码就绪，但真实运行 artifact 缺失或未通过 validator；
- `partial`：源码合同仍缺。

审计启动 0 provider、0 simulator、0 ACT。静态通过不证明语义正确；N=1 artifact 只证明机制；缓存
VQA replay 必须保留 source evaluation、真实视频/montage、live model identity 和
`act_rollouts_started=0`，不能冒充新 rollout。完整映射和剩余 gap 见
[论文主体方法 16 项覆盖审计](development_log_20260719_paper_method_coverage_zh.md)。

## 15. 2026-07-22：公共 Proposal、受限 Task artifact 与条件 EvaluationGraph

### 15.1 从开放 Query 到逐轮证据的受限公共链路

`scripts/manipeval_agent.py` 现在让 BBH 与 `click_bell` 共享
`adjudicate_bounded_transition()`：task adapter 只提出候选，公共 `BoundTaskPlanSession` 根据上一轮
`EvidencePacket` 给出的唯一 directive 裁决是否接受。`--bound-requested-aspect-id` 可以重复传入，
把一次单任务 evaluation 精确限制到 Query 允许的 aspect 集。

`mea/evaluation_graph.py` 在更上一层把一个 Query 分成最多两个独立 child。每个 child 固定一个
RoboTwin task、一个对应 ACT checkpoint、一个 aspect 和一次 N=1 round；第二 child 可声明
`if_previous_failed_or_uncertain`。completed child 必须与 graph 派生 evaluation ID、原 Query、精确 aspect、
单轮预算和一次 ACT start 一致，才可转成 typed `ChildOutcome`。父层保存 plan、outcome、下一节点和最终
综合，但当前 CLI 只生成/裁决 inert child command，不自动启动它们。因此 live provider plan 与 synthetic replay 已验证，
真正的跨任务 live orchestration 仍是下一阶段工作，不能把 graph plan/replay 报成 ACT 结果。

```text
open Query
→ bounded EvaluationGraph plan
→ child A: fixed task/checkpoint/aspect → ACT N=1 → typed outcome
→ parent replay/adjudication
   ├── enough evidence → stop + synthesize
   └── failed/uncertain → child B: another fixed task/checkpoint/aspect
→ strengths / weaknesses / recommendation / limitations
```

### 15.2 本批真实机制证据

`eval_20260722_batch14_click_flagship_n1_v2` 在同一固定 `click_bell` ACT checkpoint 下运行两轮：
第一轮 query-generated 左侧位置成功后，旧 task-specific 路径从 `object_position` 切换到
`object_instance`；第二轮 official base0 也成功。两轮 ACT 为 `2/2`，Dynamic VQA 均为 passed 且
`evidence_conflict=false`，全部 required gates 通过。原始图文包仅保留在服务器侧忽略目录
`mea/evaluation_runs/eval_20260722_batch14_click_flagship_n1_v2/`，未随仓库发布；其可提交摘要见本节和
[2026-07-22 开发记录](development_log_20260722_minimal_paper_loop_zh.md)。该 run
早于公共 `AdaptivePlanStepAgent` 的最终接线，不能作为当前公共 step 的 clean-head live 证据。

这仍只是单 seed、每变体 N=1 的机制验收；没有覆盖右侧位置、base1、clutter、纹理、光照，也没有
形成论文 Tables 1--3、6--9 的统计结论。实现与失败修复细节见
[2026-07-22 最小论文闭环开发记录](development_log_20260722_minimal_paper_loop_zh.md)。

## 16. 2026-07-22：evidence-conditioned 动态 sub-aspect

正常 adaptive runtime 不再只消费 evaluation 开始时冻结的 aspect 列表。每轮真实执行结束后，
`AdaptivePlanStepAgent` 在固定 task/checkpoint 与 capability envelope 内读取 Query、coverage、Rule、
Dynamic VQA 和 `EvidencePacket`，再选择 `propose`、`refine` 或 `stop`：

```text
开放 Query
→ Global Router：固定 task + ACT checkpoint + initial required aspect
→ Round TaskProposal / ToolProposal
→ TaskGen：reuse / overlay / codegen
   → VariantSpec + generated code + SuccessSpec/repair report
   → static → render → vision → expert gate
→ ACT telemetry + events + video
→ trusted/generated/typed Rule Tool + Dynamic VQA
→ Aggregate → EvidencePacket
→ navigation_options(required / covered / discoverable)
→ AdaptivePlanStepAgent(Query, Y1:t)
   ├── refine 当前失败/不确定 aspect
   ├── propose 同 task 的另一受支持 aspect → 下一轮
   └── stop → strengths / weaknesses / recommendation / limitations
```

对应的人工可审计数据集中保存在 evaluation 下：

| 阶段 | 主要 artifact |
| --- | --- |
| Query / scope | `plan/global_query_route.json`、`plan/bound_task_session.json` |
| 每轮 Proposal | `plan/bounded_proposal/round_*/`、`plan/runtime_directive_after_round_*.json` |
| TaskGen | child 的 `generation/variant_spec.json`、task code/overlay、SuccessSpec 与 repair report |
| 视觉 gate | scene render、vision response、scene validation、expert telemetry |
| ACT | rollout video、`telemetry.json`、`events.jsonl`、episode manifest |
| Tool / VQA | Tool request/source/result、VQA prompt/keyframes/observation |
| 证据 / 决策 | `plan/evidence_after_round_*.json`、PlanStep prompt/responses/decision、Aggregate |
| 人工阅读 | `evidence_report.md` 或 `docs/evidence_runs/<evaluation_id>/` 小型发布包 |

`ToolProposal v3` 携带的 `MetricSpec v1` 现在可从正常 runtime 进入严格
compile/differential/register/reuse 通路。一条真实
episode 已足够做 deterministic rerun 与 trusted-interpreter equality；不再要求两个 episode 的 live
值不同。安全 concern 当前只有 canonical aspect `safety.hammer_left_camera_contact`：它统计
`020_hammer ↔ left_camera` 的精确 physical contact，是窄代理。通用
`safety.unintended_contact` 仍 unsupported，不代表所有意外接触、clearance 或完整安全性。

错误 SuccessSpec 的本批 recovery 也有明确边界：非法候选被结构化诊断后，最多一次替换为可信
official-equivalent spec；不合并非法字段，也不声称模型完成了 Proposal-derived semantic repair。

更上层 `EvaluationGraph` 的 scope 固定为 `cross_checkpoint_portfolio`。它可汇总多个任务专属 ACT
child 的 required/covered aspect 和最终答案；未触发的 conditional child 单列为
`conditional_not_activated`，不伪装成 required gap。父层不在单 child 中切 checkpoint，也不替代上述
单任务动态闭环。论文对应与剩余 gap 见 [自顶向下审查](paper_claim_gap_zh.md)。

## 17. 2026-07-23：partial route、reviewed Task 与当前 clean-head 边界

### 17.1 宽 Query 的 supported subset 与显式 gaps

全局 Router 不再把“部分可回答”的宽 Query 整体拒绝。它必须同时满足两条约束：

```text
broad Query
→ task-qualified requested aspects
   ├── supported subset 非空：route，并只执行该 subset
   │   └── unsupported axes 原样进入 gaps / final limitations
   └── supported subset 为空：status=unsupported，0 ACT
```

严格 validator 会拒绝虚构 gap、把另一个 task 的能力混入当前 task，或在调用方已经显式绑定父 aspect
时擅自扩成 partial route。当前 plan-only evidence 对“operated bell properties”宽 Query 选择了
`object_position` 作为首个方面，同时保留 appearance、mass、scale 等未支持轴；这只证明可信 catalog
内的自主选择，不表示系统发现或执行了 catalog 外的新属性。

### 17.2 reviewed generated Task 的复用与 acceptance

`scripts/manipeval_task_registry.py` 为 generated Task 提供 pending review template、显式 install 与 exact
find。正常 Agent/TaskGen 通过 `--reviewed-task-registry` 使用它；命中后 TaskGen 不再调用 codegen
provider，但仍在新 run 中物化、render/probe，并在 ACT 前运行 production acceptance。

注册边界故意区分两类文件：

- immutable copied inputs：task.py、VariantSpec、overlay/load_actors、可选 SuccessSpec、validation/static；
- run-local derived outputs：`TaskArtifactBundle` 与 `SceneCheckSpec`，每个新 run 重新生成并验证。

acceptance 还固定当前登记的 5 个 Python runtime dependency hashes。这个合同能检测 reviewed source、
contract 或 runtime 依赖篡改，但不是完整 Conda、driver、checkpoint 和 simulator environment 的 bitwise
snapshot。当前真实复用只覆盖一个 development-agent 审核的受限 BBH scale variant；expert 与 ACT N=1
均已走 TaskGen provider=0 路径，但 `paper_table_eligible=false`，也不能称为全局 retrieve-first RAG。

### 17.3 bounded Proposal 修复与 stage-specific recovery

Proposal Agent 对 v2/v3 VQA question binding 的修复只从当前 capability card 重建允许的结构字段；它不
修改 scene、SuccessSpec、metric 或其他可执行语义，并把 repair trace 写入 proposal artifact。该机制用于
消除 provider 输出中可确定修复的 schema 漂移，不证明模型进行了新的语义推理。

TaskGen/ToolGen/Planning 的恢复继续遵循论文 App. A.3.4 的 stage/action 边界：哪个 stage 失败，就在该
stage 内有界 repair/regenerate；policy/simulator outcome 不重试。论文并不要求把所有生产分支改写成同一个
中央 recovery controller，因此“统一 controller”不是剩余论文 gap；真实 visual repair 和正常路径覆盖面
仍然是实现限制。

### 17.4 clean-head live v4 的有限完成态证据

`eval_20260723_batch17_clean_head_click_live_n1_v4` 使用 task-only binding
`--bound-task-name click_bell`；Query 自身也点名该 task，但没有 bound aspect、history 或内部测试顺序
提示。所有模型角色显式覆盖为 `gpt-5.6-terra`。Router 对宽 Query 保留 supported
`object_position + object_instance`，同时把
color/gloss/texture/mass/scale 留作 partial gaps。完整运行是：

```text
Query
→ autonomous first aspect: object_position.left_fixed
→ query_generated xy=[-0.14,-0.12]
→ ACT seed=100502, success=1
→ Rule/VQA/Aggregate/Evidence
→ AdaptivePlanStep provider: propose/switch_aspect
→ object_instance.base0
→ ACT seed=100502, success=1
→ Rule/VQA/Aggregate/Evidence
→ hard cap=2 → completed
```

两轮 pipeline、Aggregate 和 Dynamic VQA 都为 passed，`evidence_conflict=false`；最终
`status/lifecycle=completed`。runtime ledger 为 11 个 logical provider calls、16 个 transport attempts、
2 个 ACT starts。该 run 没有触发 whole-round recovery/restart，因此这三个数是无 recovery 路径的精确
计数，不能拿来验证之后新增的 restart instrumentation。

仓库内的 [v4 compact evidence bundle](evidence_runs/eval_20260723_batch17_clean_head_click_live_n1_v4/)
包含 15 个小文件（内容合计 600,465 bytes，目录占用约 628K）的两轮视频、scene、VQA montage、代码
与紧凑数据；完整 machine audit
仍留在服务器 evaluation 目录。整个开发批次新增 ACT starts 为 4，而非成功 v4 中的 2：reviewed BBH
reuse=1、clean-head v2=1、v3=0、v4=2；clean-head 本身累计 3 ACT，左侧 position 在 v2/v4 重复。

失败历史继续保留：v2 在首轮成功后于下一 Proposal gateway 失败；v3 的 Luna vision 调用返回 403，
在 ACT 前以 pipeline failure 终止，ACT starts=0。v4 的成功说明全角色 Terra override 在这一次运行可用，
不证明 Luna 或默认混合 profile 已稳定。最终源码又补了失败/restart attempt ledger 与 feedback `finally`
收口，但 v4 artifact 产生于无 restart 路径，不能把源码合同自动提升成 live restart 证据。

### 17.5 下一方法 gap：Query-conditioned evidence sufficiency

v4 证明真实 Evidence 能改变下一 aspect，但 hard cap=2 仍在 `right_fixed`、`base1` 未测时结束。
架构上需要分开三个概念：

| 概念 | 含义 | 当前问题 |
| --- | --- | --- |
| `candidate_universe` | 当前 task/capability 下所有合法可选测试 | 不能自动等同于 Query 必须覆盖的集合 |
| `required_coverage` | 回答当前 Query 所需的最小证据，由量词和目标决定 | `all/across/some/worst-case/compare` 尚无显式 sufficiency contract |
| `budget_cap` | 用户允许的最大 rollout/round 成本 | cap 耗尽只能写 `budget_exhausted`，不能写 `evidence_sufficient` |

已有 0-ACT cached replay v2 在固定同一非 policy evidence 时，只修改 `policy_success`：`0` 产生
`drill_down position`，`1` 产生 `switch instance`。它与 v4 一起证明 branch sensitivity，但不证明停止
充分性。下一批应先用缓存 evidence 建立量词解析、required coverage 和 stop-reason 的确定性合同，再决定
是否支付新的 ACT。
