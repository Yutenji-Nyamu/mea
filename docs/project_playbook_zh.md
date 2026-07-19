# MEA 项目手册：目标、约束与跨对话协作

本文保存项目中不应随单次实验频繁变化的约定。它回答“为什么做、复现什么、哪些边界不能
混淆、开发时怎样取舍、哪些组件可以复用”。当前代码调用链见
[架构与数据流](architecture_and_dataflow_zh.md)，安装和命令见
[简明运行指引](running_guide_zh.md)，实时 commit、服务器状态、真实实验和下一步则只放在
外部 handoff 快照中。

## 1. 项目目的

MEA 是在 RoboTwin 2.0 上复现和逐步验证 ManipEvalAgent 的工程项目。目标不是另写一个
机器人仿真器，也不是只给 RoboTwin 包一层自然语言报告，而是跑通论文的核心闭环：

```text
开放式评估问题
→ Plan Agent 动态提出可执行的 sub-aspect
→ TaskGen / ToolGen 复用或生成任务与测量工具
→ 用少量 rollout 执行被评 policy
→ 规则、轨迹和视觉证据确定性聚合
→ 证据返回 Planner，继续细化或结束
→ 给出可审计、可解释的评估结论
```

当前工程首先使用 RoboTwin + ACT 建立最小可复现通路，再逐步增加论文实验所需的覆盖与
证据。论文中的“大规模、多 policy、多 benchmark”是后续验证目标，不应妨碍日常 1 / 3 / 5
预算的敏捷开发。

### 1.1 “功能复现完整”的顶层判定

功能完整不是“组件文件齐全”，而是论文 Fig. 2 的一条开放问题在同一次 evaluation 中真实
驱动以下闭环：

```text
开放 Query
→ 全局 Router 从可信 catalog 选择一次 task + ACT checkpoint
→ Bound PlanSession 在该 task 内分解/调整 sub-aspect
→ TaskGen 复用或生成任务，并通过场景 gate
→ ToolGen 复用或生成测量工具
→ ACT 执行少量 rollout
→ Rule Tool + 动态 Execution VQA 形成互补证据
→ 当前证据改变下一轮方向或触发停止
→ 最终回答原始问题的强项、弱项、建议和局限
```

缺少任一环，只能称为组件原型或局部通路。尤其不得用预写两轮冒充证据驱动规划，不得用
缓存图片扰动冒充真实 simulator 扰动，也不得把 plan-only、expert solvability 或 pipeline
通过率写成 ACT 表现。

当前已经具备这一门槛的**小规模功能实现**：全局 Query 可从可信 ACT catalog 选择
`task/profile/aspect/checkpoint id`，并把一次 evaluation 固定为一个 task/checkpoint；通用
PlanSession 与证据合同可驱动该 task 内的 `drill_down / switch_aspect / stop`；
`click_bell` 与 BBH 均有真实单任务闭环；ToolGen 可在显式审核后跨 evaluation 复用；真实
TaskGen 场景错误已经走通 visual reject→diagnosis→repair→revalidate；跨任务 portfolio 父层
也能为同一 Query 生成两个受硬预算约束的 child 计划，并对已完成 child 的账本、轮级
provenance、policy outcome 做确定性审核和汇总。

这里的“具备”指 Fig. 2–5 主体数据流已有最小实现，不等于论文实验复现完成。portfolio 的
`reuse` 模式本身不会因果启动 child；当前只评 ACT，任务族和受控轴很少，标注仍含
`development_agent_proxy`，真实 seed/repetition 也远少于论文。增加第四、第五个任务或更多
细碎 metric 仍不是默认优先级；下一阶段优先把通用性、人工 gold、真实扰动覆盖和实验合同
逐项补齐。

## 2. 与 RoboTwin 的关系和可信边界

MEA 是 RoboTwin 的评估层扩展。以下事实必须始终保持清楚：

- RoboTwin 负责物理仿真、任务环境、控制循环、official expert、ACT 推理和
  `check_success()`；MEA 不重写这些权威语义。
- MEA 负责开放问题规划、受限任务变式、工具路由/生成、telemetry、视觉检查、确定性聚合、
  多轮反馈和实验协议。
- official expert 用来验证任务/seed 是否可解；ACT 才是当前主要被评 policy。运行 expert
  不需要 checkpoint，运行某任务的 ACT 需要该任务专属 checkpoint。
- `check_success()`、simulator state 和确定性工具优先于语言模型判断。VQA 只判断代码内登记
  的可见现象；冲突要保留，不能用 VQA 覆盖数值证据。
- generated route、official passthrough、Easy/Hard paired comparison 是不同实验合同，不能
  因为都调用 ACT 就混为一谈。
- 一次 evaluation 只绑定一个 RoboTwin task 和一个 ACT checkpoint。task-agnostic 指同一
  PlanSession/proposal/证据状态机可服务不同 task session，不代表一个 evaluation 可以中途换 task；
  跨任务 Query 必须由多个独立 child evaluation 和父层汇总完成。

## 3. 论文方法与项目模块的对应

| 论文部分 | 论文要点 | 当前项目承载位置 | 复现判断标准 |
| --- | --- | --- | --- |
| Fig. 2、Sec. 3.2 | 开放问题驱动、多轮动态 Proposal | `mea/planner/session.py`、task adapters、`mea/history/`、`mea/portfolio.py` | evaluation 先固定 task/checkpoint；公共 PlanSession 根据前轮证据决定 action/aspect/template，并最终裁决 adapter 候选；跨任务父层绑定独立 child，而不把 pipeline pass 当 policy success |
| Sec. 3.3.1、Fig. 3 | reuse-first TaskGen、任务/资产/文档 RAG、视觉自反思 | `TaskProposal`、shared capability catalog、`VariantSpec` v2、`mea/taskgen/`、scene validation | proposal 不含可执行字段且不能越过 capability；非模板变化以 proposal 为本轮权威、registry contract 为 materializer envelope；场景通过结构、渲染和语义 gate |
| Sec. 3.3.2、Fig. 4 | reuse-first ToolGen、规则工具与 VQA | `ToolProposal` v1/v2、`mea/toolgen/`、`mea/toolkit/`、`mea/execution_vqa/` | sub-aspect 映射到可 resolve metric；v2 可携带严格受限的 `run_local.*` 视觉问题，但 VQA 不能覆盖 simulator numeric authority；新工具须 generate→validate→register |
| Eq. 3–4 | rollout、逐样本测量和确定性聚合 | ACT/expert backend、Recorder、Aggregate、`mea/runtime_ledger.py` | 保存真实分母、seed、成功、样本数、调用开始数和 wall-clock，模型不自行做统计；started 与 completed 分开报告 |
| Fig. 5、App. A.3.5 | observation 回流并驱动继续深入 | Agent orchestration、round decision、Feedback | policy failure 是证据而非流水线失败；下一轮由聚合证据决定 |
| App. A.1、Tables 1–5 | 少样本成本与标准 benchmark 结论一致性 | generated protocol v2、ACT 三任务 N=1 pilot | smoke 用 1 / 3 / 5；N=1 只验计量，正式结果再按论文预算统一计算时间/样本与方差 |
| Tables 6–8 | Planner 人类一致性与 VQA 准确/鲁棒性 | 20-query model-draft + cached-montage image-proxy runner | unreviewed/proxy 与 human gold/simulator perturbation 分层；只有后者能形成论文指标 |
| Table 9 | policy 排名与 benchmark 一致 | 后续多 policy 实验 | 当前 ACT-only 阶段不宣称已复现该结论 |

“有文件或接口”不等于“已复现论文”。只有真实 artifact 贯穿对应数据流，并且实验指标能验证
论文主张时，才把该点标记为完成。

## 4. 可复用的工程资产

新增任务或实验时应先组合下列资产，不要再复制一套 task-specific 主链：

| 资产 | 可复用能力 | 扩展时通常只需补什么 |
| --- | --- | --- |
| TaskSchema + generic Recorder | actor 语义、轨迹、事件、视频索引 | 新任务 schema 与真实 seed 验收 |
| TaskGen capability + `VariantSpec` v2 | 统一受控轴、generation mode、preserve contract 和 variant 身份 | 新 capability card、任务级 `changes` 校验和薄编译器 |
| official passthrough | 不改官方任务即可接 expert/ACT/工具/VQA | TaskSchema、任务 checkpoint、受限 VQA 映射 |
| ACT backend + preflight | checkpoint 检查、原生 evaluator、连续 rollout | 服务器侧按任务下载 checkpoint |
| Trusted Tools + Aggregate | 可信数值和跨 episode 统计 | 新 metric 的显式 signal contract |
| ToolGen sandbox/registry | 生成代码静态检查、oracle、确定性验证、evaluation-local 复用；显式审核后跨 evaluation 精确复用 | 新 target、few-shot 工具、review manifest 和单元测试 |
| Task/asset/doc retrieval | 发现 RoboTwin 任务和代码上下文 | 把单任务知识升级为通用 capability card |
| generated overlay/gates | 薄变式、render/rule/expert solvability 检查 | 新 `VariantSpec` 编译器和受限 repair contract |
| Agent artifact contract | plan、round、execution、summary、feedback 的可审计目录 | 新 planner/task profile 接同一结构 |
| BoundTaskPlanSession | 固定单 task/checkpoint target、统一 task adapter 计划、证据驱动 round transition | 新任务 adapter 只补 materialization；不得复制新的顶层 session 状态机 |
| TaskProposal + ToolProposal | 把“生成/复用什么任务、用什么 Rule/VQA 测量”提升为 task-agnostic 语义边界 | 新 capability card、materializer、metric route 和 allowlisted VQA；不向模型暴露 seed/path/checkpoint/gate |
| illustrated evidence report | 从真实 evaluation 汇总 proposal、代码、render、视频、Tool/VQA/Aggregate/decision/final answer，并发布小型 bundle | 新 artifact 类型增加显式 resolver；缺失时显示 N/A，不制造替代证据 |
| call-start ledger + round provenance | provider transport/ACT batch 在外部调用前落盘；轮计划、summary 与 child/Tool/VQA/ledger 文件 hash 绑定 | 新外部 runtime 必须先接账本；新 round artifact 必须加入 sidecar，而不是另写不可核验计数 |
| cross-task portfolio parent | 同一 Query 的两任务硬预算 command plan；completed child 的 outcome、runtime 与 provenance 审核；强项/弱项/建议/局限汇总 | 扩展为通用 EvaluationGraph，并把原 command plan 与完成 child 做反向因果绑定 |
| paired/protocol/validation runners | exact-seed 比较、generated `(variant_id, seed)` 身份、chunk/resume、cached scorer | 新协议范围和人工数据，不重写统计器 |
| benchmark/query/perturbation pilots | 三任务计量、query review 格式、VQA image-proxy 扰动 | 扩大预算前补真实 repetition、human review 和 simulator-level 变化 |

## 5. 开发优先级怎样决定

每个候选开发点都必须同时回答六个问题：

1. 当前代码具体缺什么，而不是泛称“增强鲁棒性”？
2. 它对应论文哪一节、图或表，支撑的是核心方法还是实验佐证？
3. 不做它会阻断哪条论文主张？
4. 最小可运行通路是什么，能否用 1 个 rollout 或缓存 artifact 验证？
5. 实现难度、GPU/API 时间和失败风险分别多大？
6. 完成后产生什么机器可读证据，怎样判断不是“接口存在但未生效”？

默认排序权重为：核心方法缺口 > 核心实验结论 > 支撑性验证 > 工程便利性。若两项论文价值
相近，优先选择可复用性更高、真实运行成本更低的一项。恢复机制、更多日志、更多任务 seed
只有在它们解锁核心闭环或可靠实验时才应提前。

开发采用三层预算：

- `1`：接口和单个真实 rollout smoke；
- `3`：观察趋势、发现随机性和协议问题；
- `5`：论文默认 constructed-task rollout 预算的近似验收；
- 论文中的 10 次完整重复或更大 benchmark 只在通路稳定、问题明确后执行。

### 5.1 自底向上与自顶向下交错开发

后续批次交错使用两种主视角，避免只沿当前代码的小缺口前进，也避免只画论文蓝图而没有
可执行证据：

- **自底向上批次**：从最近真实失败、重复 task-specific 分支或不可审计 artifact 出发，先抽象
  共享合同并做 `0–1 ACT` 最小验证；重点检查代码是否真正执行、失败是否 fail closed、产物能否
  被独立重算。
- **自顶向下批次**：从论文 Fig. 2–5、Secs. 3.2–3.4 的模块主张和数据流反推，逐环标记
  `真实完成 / 小规模代理 / 只有接口 / 尚缺失`，优先补会阻断完整方法主张的最高层缺口。

默认一批以其中一种为主、下一批切换主视角；但每批结束都做一次另一视角的短审计。候选点仍
必须同时给出论文对应、重要性、最小实现、真实成本和不能宣称的结论，不能把机械轮替本身当作
优先级依据。

## 6. 一般开发流程

1. 先读外部项目上下文和最新 handoff；服务器
   `/root/autodl-tmp/mea` 是 canonical working copy。
2. 只读核对 `git status`、`HEAD`、`origin/main`、当前 artifact 和相关源码，不能从旧 handoff
   推断实时状态。
3. 说明本轮候选点的论文映射、重要性、难度和最小实验；与用户确认或按已授权范围选择。
4. 优先扩展共享 schema/spec/registry，再为具体任务写薄适配层；避免新增散落的
   `if task_name == ...`。
   顶层 evaluation 必须先冻结 task/checkpoint；跨 task 需求由多个 child session 表达。
5. 先跑静态/单元测试，再跑 1 个真实 rollout；通过后才按 3、5 放大。
6. 记录输入、seed、checkpoint、Git HEAD、wall-clock、sample count、失败阶段和 artifact 路径。
   generated 多变体实验必须把 `(variant_id, seed)` 作为样本身份，并同时报告 pooled 与逐变体
   coverage；不能把跨 variant 的同 seed 当成重复样本。
7. 更新受影响的运行/架构文档和 development log；执行 `git diff --check` 与测试。
8. 使用 DCO signed-off commit，推送 GitHub，最后再次核对 server HEAD、origin 与 clean status。

## 7. 凭据、大文件与服务器约定

- SSH 密码、UIUI key、Git 私钥和 Hugging Face token 不写入仓库、handoff 或长期 memory。
  只有真正需要时才向用户明确索取具体凭据；进程内临时注入，用后清除。
- GitHub 写权限若已通过仓库本地 `core.sshCommand` 配置并验证，应直接复用；不要因新对话而
  默认它失效，也不要无缘由要求用户重配网页 key。
- checkpoint、数据集和模型权重只在服务器直接下载。优先 AutoDL 学术加速，其次服务器侧
  Hugging Face mirror；不得让常规大文件经过 Windows、`C:` 或 Codex 工作区。
- 若误把大文件下载到本机，要删除 staging 和相关 cache，核对零残留，并在交接中记录原因、
  补救和以后采用的服务器侧路径。
- 不提交运行 artifact、checkpoint、软链接、私钥或 token。提交前检查 Git status 和大文件。

## 8. 文档分工与维护

| 文档 | 保存内容 | 不应保存 |
| --- | --- | --- |
| 本手册 | 稳定目标、边界、论文映射、可复用资产、开发约定 | 单次实验、当前 SHA、密钥 |
| `architecture_and_dataflow_zh.md` | 当前已实现的调用链、artifact 和可信边界 | 未来设想冒充当前能力 |
| `running_guide_zh.md` | 安装、checkpoint、入口命令和故障检查 | 长篇设计讨论 |
| development log | 已完成变更和真实实验结果 | 未验证的宣传性结论 |
| 外部 project context | 服务器路径、稳定入口、跨任务协作规则 | 重复整份架构说明 |
| 外部最新 handoff | 当前 commit/status、最近结果、未完成项和下一步候选 | 长期固定知识、任何秘密 |

入口、route、artifact contract 或可信边界变化时更新架构文档；命令和依赖变化时更新运行
指引；稳定协作规则变化时更新本手册；每次完成开发后刷新最新 handoff。
根 `README.md` 保持 RoboTwin 上游入口不动，MEA 的运行和设计说明集中维护在上述独立文档。

## 9. 复现成熟度的命名约定

为避免“入口存在”被误读成“论文复现”，所有对外总结使用以下层级：

| 层级 | 可以声称 | 还不能声称 |
| --- | --- | --- |
| plumbing | schema、入口、artifact 和失败检查可运行 | 方法有效或指标提升 |
| cached smoke | 既有真实 artifact 能走完 scorer/tool reuse | 新 rollout 可靠、跨 seed 泛化 |
| live N=1 pilot | 单个真实 seed/rollout 贯穿完整链路 | 均值、方差、稳健性或论文表结论 |
| agile 3/5 | 在小预算下观察趋势并暴露随机性 | 等同论文完整 repetition、policy/benchmark 覆盖 |
| paper-eligible | 预算、人工标注、扰动层级、policy/benchmark 和统计合同与论文一致 | 超出该预注册范围的泛化主张 |

当前 generated protocol v2、`click_bell` ToolGen 和三任务聚合分别属于协议骨架、live/cached
smoke 与 N=1 instrumentation pilot。20-query 同时保留 `model_draft_unreviewed` 原始集和
`development_agent_proxy` 复核集；后者可跑 live Planner scorer，但仍不能填 Table 6。VQA
同时保留缓存 montage image proxy、真实 simulator-native clean/clutter `N=1`，以及 unseen
texture/static lighting 各 2 个独立 seed 的 unvalidated proxy coverage。后两类证明了真实扰动
数据流，却仍因代理标签、覆盖不完整或缺少正负可见性平衡而不能填 Tables 7–8。只有完成独立
多人标注、正负/困难扰动覆盖和足量 repetition 后，才升级对应结论的命名。

`BoundTaskPlanSession`、`TaskProposal`/`ToolProposal` 与 illustrated report 属于主体方法 plumbing；
BBH 两轮 live run 是 N=1 功能证据。它证明 Query→appearance codegen→evidence-driven timing→
Tool/VQA/Aggregate/final answer 的真实数据流，不证明 ACT 的 appearance/timing 泛化能力；两轮
ACT 都失败这一 policy 结果必须与成功的 expert controls 和完整 pipeline 分开报告。

## 10. 新对话最短启动清单

```text
1. 读取外部 project context 与 latest handoff
2. 只读核对服务器 git status / HEAD / origin/main
3. 核对最近 artifact，而不是只看报告摘要
4. 用本手册的论文映射和排序规则评估下一步
5. 先说明“论文对应点 + 重要性 + 最小通路”，再开始实现
```

如果 handoff 与服务器冲突，以服务器 Git、真实 artifact 和测试结果为准，并立即刷新 handoff。

## 11. 临时人工代理与恢复的长期约定

- 开发阶段可以由 Codex 充当 `development_agent_proxy` 做二元视觉标签、query/aspect review 与
  registry review，以最小成本验证数据通路；artifact 必须显式保存该身份、
  `human_reviewer_count=0` 与 `paper_table_eligible=false`。
- development-agent proxy 不能改名为 human gold、人工 majority 或论文指标。进入论文表格前，
  必须由独立人工重新标注或复核，且保留替换关系与 provenance。
- 真实 simulator 扰动和缓存图像扰动必须分开命名。前者需由 simulator state 证明变化实际发生；
  后者只能称 image proxy，即使其输入来自真实 rollout。
- 恢复遵循论文 App. A.3.4 的 stage/action 表：planning disagreement 重做 planning，TaskGen
  visual failure 由 TaskGen 内部 regenerate/repair，ToolGen unit-test failure 由 ToolGen 内部
  regenerate；只有 planned Tool 的未预期执行异常可最多重启一次完整 evaluation round。
  整轮重启必须使用新 child run id、新 execution 目录和新的 ACT rollout，并显式统计额外样本；
  policy/simulator failure 是被评结果，禁止重试。旧的 same-telemetry Tool 子阶段 retry 仅作显式
  兼容选项，默认关闭。
- fault injection、cached counterfactual 与 N=1 都是功能证据，不产生成功率、方差、AUROC 或
  论文消融结论。报告必须同时写出不可用指标及原因。

## 12. 预注册与最小实验的长期规则

- 功能优先于扩大实验。新通路先用静态测试、`0-ACT` prepare/audit 或一个真实 rollout 验证；
  只有身份和失败处理稳定后才按 `1 → 3 → 5` 扩大。论文的 10 次完整 repetition 不作为日常
  开发默认值。
- 当前只评 ACT。不得为了“表格更完整”擅自接入第二种 policy；official expert 只做 seed/
  solvability gate，不能替代 ACT checkpoint 验证。
- 每个 evaluation 在首轮前固定一个 task、一个 ACT checkpoint 与最大轮预算。模型可在受信
  sub-aspect/variant 内提 proposal，但不得切 task/policy/checkpoint，也不得直接提供可执行路径、
  seed 或 gate；跨任务评估必须拆成独立 child evaluation。
- 注册 capability contract 表示可执行 materializer 的权限 envelope，不等于模型只能逐字重复一个
  template。`TaskProposal` 可在受控 roots 内提供本轮新 changes；TaskGen 必须同时保存 proposal 与
  envelope，并明确记录哪一个是 round variation authority。
- `ToolProposal` v2/v3 的 run-local VQA question 必须使用 `run_local.*` ID、受控字段枚举、单行问句和
  固定长度上限；保存后的 query 必须可独立重验。v3 可额外携带严格 `MetricSpec`；当前只允许已审核
  typed operator，不能用自然语言绕过 AST、差分 oracle 或 registry collision gate。run-local VQA
  始终是补充视觉证据，不是数值 oracle。
- checkpoint、数据集、模型权重和 rollout 大包仍只在服务器侧下载、生成与保存；Windows/Codex
  工作区只接收代码、配置、小型报告和必要的压缩源码。
- preregistration 必须绑定真实执行：至少把 manifest、registered route、command plan 与观测到
  的 argv 在 Agent preflight 中共同校验，把同一 registration identity 传到 child artifact，并在
  post-hoc comparator 再校验。仅生成一个带 hash 的 JSON 不算预注册执行。
- canonical self-hash 只能发现内容漂移，不能证明执行发生、代码可信或结果独立。报告必须同时
  指向 parent/child completed artifact、checkpoint/source hash 和比较器检查；计划产物固定写
  `act_rollouts_started=0`。
- module-off 的 prepare 与 audit 都是 artifact-only。内置 development execute 必须写入与 formal
  `artifact_root` 双向不重叠的独立目录，且其 manifest 不可交给 formal audit。completed manifest 中的
  `provider_called / simulator_called / act_rollouts_started` 是历史 runner 的 self-attested 声明，
  audit 没有旁观运行；只有真正按冻结 switch 执行、完整 matched、typed outcome 可核验时才可
  报 functional effect，否则 effect 保持 `null`。
- provenance 不是 outcome。RAG source 存在、视觉 gate 接线或 Tool validation 配置存在，只能
  证明对应模块可追踪，不能直接作为 Table 3 生成成功率或消融效果。

## 13. 论文方法覆盖审计的长期规则

- 每次自顶向下批次结束，用 `scripts/manipeval_method_coverage.py` 重算 16 项主体方法覆盖；不要手工
  修改状态。`partial` 表示源码合同缺失，`evidence_pending` 表示代码已就绪但严格运行证据缺失，
  `implemented` 也只表示该项的最小检查通过。
- coverage audit 是 0-runtime bookkeeping，不是实验。源码 AST、文件 hash、N=1、缓存 replay 和
  development-agent proxy 各自只能支持其声明的证据层级；不能因“16/16”而宣称论文有效性复现。
- `PlanningContext` 必须来自受信 Policy/Simulator/Adapter metadata；`EvidencePacket` 必须保留 scalar、
  VQA、pipeline 和 policy 字段及冲突，禁止让模型自行给这些证据发明概率。
- `bounded_each_round` 只有在上一轮真实 observation 或显式缓存 replay 后才算验证了逐轮 proposal；
  `plan-only` 只能证明首轮 proposal/context。一次 evaluation 仍冻结 task 与 ACT checkpoint。
- TaskGen 交付物必须同时说明 scene method 和 success method 的来源。复用官方 `check_success()` 是诚实的
  bounded implementation，不应被描述为“模型生成了完整成功函数”。
- executable Tool 的语义复用只能忽略表述性字段；task、metric、typed spec、signals、output、源码与
  validation contract 必须保持精确。发现 registry collision 时 fail closed。
