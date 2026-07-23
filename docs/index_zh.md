# MEA 文档入口

本项目的目标是在 RoboTwin 2.0 上，以 ACT 为主、DP3 为一个有界部署 pilot，并用最小实验预算逐步复现 ManipEvalAgent 的
主体数据流。当前是**两个任务族、少量可信 capability 内的受限功能原型**：开发优先验证论文
Fig. 2–5 的开放 Query、动态规划、TaskGen、ToolGen、rollout、Rule/VQA evidence、反馈与最终回答；
尚未完成论文规模的多 policy、足量 seed、人工 gold 或完整统计表。单个 DP3 seed 跑通不代表
policy ranking 已复现。

## 1. 从哪里开始

| 需求 | 入口 |
| --- | --- |
| 第一次安装和运行 | [简明运行指引](running_guide_zh.md) |
| 理解调用链、数据流和可信边界 | [架构与数据流](architecture_and_dataflow_zh.md) |
| 理解项目目标、论文映射和开发约定 | [项目手册](project_playbook_zh.md) |
| 查看当前源码/服务器证据边界 | [`evidence_snapshot_current.json`](evidence_snapshot_current.json) |
| 从论文主体审查 claim、当前对应与真实 gap | [论文主张与当前差距（自顶向下）](paper_claim_gap_zh.md) |
| 查看动态 sub-aspect / MetricSpec 最新批次 | [2026-07-22 动态规划与受限恢复记录](development_log_20260722_dynamic_plan_toolgen_zh.md) |
| 查看 Task resolution / SuccessSpec v2 / registered adaptive 批次 | [2026-07-22 TaskGen 主链补全记录](development_log_20260722_taskgen_reuse_success_v2_zh.md) |
| 查看最近一个已定稿批次的实现与真实验收 | [2026-07-22 最小论文闭环开发记录](development_log_20260722_minimal_paper_loop_zh.md) |
| 查看 reviewed Task、partial route 与 clean-head v4 完成态验收 | [2026-07-23 开发记录](development_log_20260723_reviewed_partial_route_clean_head_zh.md) |
| 查看 Query contract、TaskGen v2、AnswerScope 与有效性协议批次 | [2026-07-23 P1–P5 开发记录](development_log_20260723_query_contract_taskgen_validity_zh.md) |
| 查看最新手工 open-query chain、scene+checker ACT、bounded Tool v3b、efficiency v4 与 DP3 pilot | [2026-07-23 batch19 claim-first 开发记录](development_log_20260723_batch19_claim_evidence_zh.md) |
| 查看当前架构边界与 Query sufficiency 首要 gap | [架构与数据流（见 §17）](architecture_and_dataflow_zh.md) |
| 直接检查当前 clean-head 两轮图文链路 | [2026-07-23 v4 compact evidence bundle](evidence_runs/eval_20260723_batch17_clean_head_click_live_n1_v4/) |
| 检查本批 cached/live-TaskGen/synthetic 分层证据 | [2026-07-23 batch18 compact evidence](evidence_runs/batch18_contracts/) |
| 检查 batch19 小型真实/proxy/inconclusive 分层证据 | [2026-07-23 batch19 compact evidence](evidence_runs/batch19_claim_evidence/) |
| 检查旧 task-specific planner 历史链路 | [2026-07-22 最小论文闭环开发记录](development_log_20260722_minimal_paper_loop_zh.md)（原始大产物仅保留在服务器） |
| 检查本批 0-ACT 覆盖审计与缓存反事实 | [2026-07-23 紧凑验证证据](evidence_runs/batch17_validation/) |
| 深入 TaskGen / ToolGen / telemetry | `taskgen_prototype.md`、`toolgen_prototype_zh.md`、`trajectory_toolkit_zh.md` |

文档按三层维护：

```text
README.md
└── docs/index_zh.md                 # 人工入口，不堆实现细节
    ├── running_guide_zh.md          # 安装、checkpoint、命令、产物
    ├── architecture_and_dataflow_zh.md
    │                                # 当前真实调用链与 artifact 合同
    ├── project_playbook_zh.md       # 长期目标、边界、开发循环
    ├── evidence_snapshot_current.json
    │                                # 紧凑的可机器读取状态快照
    ├── development_log_*.md         # 每批变更和验收，按时间追加
    └── evidence_runs/               # 可提交的小型图文证据包；不放原始大产物
```

`README.md` 保持短小；实时服务器地址、密码、API key 和临时实验状态不应写入长期文档。

工作位置与维护职责集中如下：

| 位置 | 维护内容 | 是否可作正式验收 |
| --- | --- | --- |
| Windows staging | 源码阅读、轻量编辑、changed-file 编译/静态检查、`git diff --check`、小型文档/证据中转 | 否 |
| canonical AutoDL tracked repo | 源码、测试、版本化 docs、小型 compact evidence；在这里运行测试并提交 | 是 |
| AutoDL ignored artifact 目录 | checkpoint、raw rollout、evaluation、generated task、validation、telemetry 与大视频 | 仅作为 server machine audit，不直接提交 |
| GitHub `main` | 已签署提交的源码、文档和小证据包 | 版本化发布面；不替代 server-only 原始产物 |

## 2. 当前方法主链

```text
开放 Query
→ Global Router 选择 checkpoint-ready task 和 supported subset，并显式保留 gaps
→ 单 task / 单 ACT checkpoint 的 PlanSession
→ evidence-conditioned TaskProposal + ToolProposal
→ TaskGen retrieve / reuse / generate + visual gate / repair
→ ACT 少量 rollout
→ Trusted Rule Tool / typed MetricSpec + Dynamic Execution VQA
→ EvidencePacket + Aggregate
→ Planner 继续、切换 aspect 或停止
→ 强项、弱项、建议和局限
```

一个 ACT checkpoint 只执行其训练任务是正常前提。系统的 task-agnostic 性来自共享协议和 task
adapter；跨任务 Query 应创建多个固定任务的 child evaluation，再由父层选择和汇总，不能在单个
rollout 中途更换 checkpoint。

## 3. 可复用入口

| 能力 | 推荐入口 |
| --- | --- |
| 开放 Query 端到端 Agent | `scripts/manipeval_agent.py` |
| TaskGen / official passthrough | `scripts/manipeval_taskgen.py` |
| ACT wrapper | `policy/ACT/eval_mea.sh` |
| DP3 exact-seed pilot adapter | `policy/DP3/deploy_policy.py` |
| claim-first 无预编排单步 Planner | `scripts/manipeval_claim_first_plan.py` |
| Query-induced ToolGen | `scripts/manipeval_query_induced_toolgen.py` |
| 论文 claim 小型协议 | `scripts/manipeval_paper_claim_demo.py` |
| TaskSchema、Recorder、Trusted Tools | `mea/toolkit/` |
| 公共 planning 与 evidence 合同 | `mea/planner/` |
| cached Query sufficiency | `scripts/manipeval_query_sufficiency.py` |
| Proposal + bounded experimental TaskGen acceptance | `scripts/manipeval_proposal.py` |
| capability / VariantSpec | `mea/capability_adapter.py`、`mea/taskgen/` |
| ToolGen、MetricSpec、registry | `mea/toolgen/` |
| Dynamic VQA | `mea/execution_vqa/` |
| paired / protocol / resume | `mea/paired.py`、`mea/protocol.py` |
| matched efficiency preregistration | `scripts/manipeval_matched_efficiency.py` |
| independent annotation/VQA validity | `scripts/manipeval_independent_validity.py` |
| 跨任务父层 | `mea/portfolio.py`、`mea/evaluation_graph.py`、`scripts/manipeval_evaluation_graph.py` |

新增任务时，优先补 TaskSchema、capability card、薄 materializer、metric/VQA route 和任务专属 ACT
checkpoint；不要复制一套新的顶层 Planner 或 runtime。

## 4. 开发与运行约定

- canonical 开发副本是 AutoDL 的 `/root/autodl-tmp/mea`；开始和结束都核对 `git status`、`HEAD`、
  `origin/main`。服务器迁移后只要路径、deploy key 和 checkpoint 链接保持有效，Git 流程不变。
- 使用已配置的 GitHub deploy key，DCO signed-off commit，并在推送后再次确认远端 SHA。凭据只在
  进程环境中使用，不进入源码、文档或 memory。
- checkpoint、数据集和模型权重只在服务器直接下载，优先 AutoDL 学术加速，其次服务器侧
  Hugging Face mirror；不经过 Windows、`C:` 或 Codex 工作区中转。
- ACT 仍是主策略；本批只增加一个官方 DP3 BBH checkpoint 的单 seed 部署 pilot。RDT、π0、
  LIBERO 与大规模多 policy 暂不扩张。实验预算按 `0 → 1 → 3 → 5` 放大：静态/缓存证据先行，
  路径稳定后才支付 N=1，确需观察随机性时才做 N=3/5。
- 每批先从论文主张反推最高层 gap，再从最近真实失败自底向上复核；实现最小公共合同，跑定向测试，
  必要时做少量 live 验收，更新文档和证据，再提交推送。
- 不把 plan-only、expert solvability、pipeline pass、缓存 replay 或代理标注写成 ACT 性能与论文统计。

## 5. 证据层级

| 层级 | 能说明什么 | 不能说明什么 |
| --- | --- | --- |
| source/static | 接口、约束和 fail-closed 检查存在 | 运行时真实贯通 |
| synthetic fixture | 编译器、router、oracle 等控制流正确 | RoboTwin 场景或策略表现 |
| cached real artifact | Tool/VQA/恢复可消费既有真实 rollout | 新 rollout 稳定性 |
| live N=1 | 一条真实链路贯通 | 均值、方差、泛化结论 |
| agile N=3/5 | 小预算趋势和随机性 | 论文完整 repetition |
| paper-eligible | 预算、人工 gold、扰动和统计合同对齐 | 超出预注册范围的泛化 |

证据报告必须分别标明 source、synthetic、cached、live 和 server-only。Git 仓库通常不提交原始
rollout、checkpoint 或大型视频，因此“GitHub 中没有 artifact”不等于服务器没有跑过，但也不能只凭
文档文字把 server-only 结果当成可独立复算的仓库内证据。

## 6. 自顶向下路线

| 轮次 | 最小交付 | 论文对应 | 当前证据 | 默认预算 |
| --- | --- | --- | --- | ---: |
| 0 | 固定跨平台文本/二进制规则、资产前置条件、紧凑证据快照 | 工程前置 | 已完成 source | 0 ACT |
| 1 | 将手工 capability-conditioned chain 接进统一 VQA/Rule→Aggregate→Feedback runtime，并在预算内安排 paired control | Sec. 3.2；Figs. 2/5 | 三轮由开发者串联；post-budget clean seed1000 也失败，property attribution 不成立，Planner 仍请求 seed1010 control | 0 ACT replay 设计，再 1–2 ACT |
| 2 | 把 standalone scene+checker 单例接入主 Agent，并提升为 target + physical distractor | Sec. 3.3.1；Fig. 3 | standalone 路径已有 1 ACT；主 Agent 仍阻止 experimental v2，且只覆盖 bounded appearance | 0–1 ACT，3–5 probes |
| 3 | 将 bounded Query-induced Tool 接入 Agent/VQA 并做 live confirmation | Fig. 4 | v3b active/right-arm DSL 已在 cached real telemetry 上 generate/validate/register/reuse；旧 v2 被撤回 | 先 0 ACT integration，再最多 1 ACT |
| 4 | 预注册并独立运行 fixed/adaptive arms，再放大到 N=3 | Tables 1–2 | v4 是 outcomes 已知后的 post-hoc cached-prefix counterfactual；actual rollout/wall saving 均为 null | 约 9–12 ACT |
| 5 | 扩 ACT/DP3 ranking、人工 gold 与真实 codegen ablation | Tables 3、6–9 | ACT/DP3 同 seed tie；5-query/8-clip 是 development proxy；7-call 只是 proposal prompt ablation | 先冻结 universe；禁止 post-hoc seed |

`EvaluationGraph` 只是一种可选 `cross_checkpoint_portfolio`：每个 child 都固定自己的单任务 ACT
checkpoint。它不再列为论文主体闭环的前置 gap。动态状态以当前 evidence snapshot、最新 development
log 与 [自顶向下论文审查](paper_claim_gap_zh.md) 为准。
