# MEA 文档入口

本项目的目标是在 RoboTwin 2.0 上，以 **ACT 单任务策略和最小实验预算，完成功能层面的
ManipEvalAgent 复现**。开发优先证明论文 Fig. 2–5 的开放 Query、动态规划、TaskGen、ToolGen、
rollout、Rule/VQA evidence、反馈与最终回答确实连成闭环；暂不追求论文规模的多 policy、海量 seed
或完整统计表。

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
| 查看最近一批实现与真实验收 | [2026-07-22 最小论文闭环开发记录](development_log_20260722_minimal_paper_loop_zh.md) |
| 直接检查一条真实图文链路 | [click_bell 两轮 N=1 证据包](evidence_runs/eval_20260722_batch14_click_flagship_n1_v2/) |
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

## 2. 当前方法主链

```text
开放 Query
→ Global Router 选择 checkpoint-ready task
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
| TaskSchema、Recorder、Trusted Tools | `mea/toolkit/` |
| 公共 planning 与 evidence 合同 | `mea/planner/` |
| capability / VariantSpec | `mea/capability_adapter.py`、`mea/taskgen/` |
| ToolGen、MetricSpec、registry | `mea/toolgen/` |
| Dynamic VQA | `mea/execution_vqa/` |
| paired / protocol / resume | `mea/paired.py`、`mea/protocol.py` |
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
- 只评 ACT，直到项目范围明确改变。实验预算按 `0 → 1 → 3 → 5` 放大：静态/缓存证据先行，路径
  稳定后才支付 N=1，确需观察随机性时才做 N=3/5。
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
| 1 | 同一 task/checkpoint 内由 `Query + Y1:t` 动态发现/调整 sub-aspect | Sec. 3.2；Figs. 2/5 | common runtime source + synthetic branch；当前 live 待补 | 0，再 1–2 ACT |
| 2 | TaskProposal 真正 reuse-first；Proposal-derived bounded SuccessSpec v2 | Sec. 3.3.1；Fig. 3 | 当前 resolver 与 SuccessSpec v1 仍 partial | 0 ACT |
| 3 | 当前动态 runtime 与 fixed 用同 Query/seed/两轮预算 matched N=1 | Figs. 2–5；Tables 1–3 | 旧 runtime pilot 有；当前 runtime 待补 | 2–4 ACT |
| 4 | 统一 code/SuccessSpec/render/vision/expert 的 TaskGenerationAttempt recovery | Fig. 3；App. A.3.4 | 分阶段 recovery 已有，统一合同待补 | 0–1 expert |
| 5 | 独立人工 gold、少量真实扰动 clips、稳定后 N=3 | Tables 6–8 | assistant-proxy/既有 pilot | 2–4 ACT 起 |

`EvaluationGraph` 只是一种可选 `cross_checkpoint_portfolio`：每个 child 都固定自己的单任务 ACT
checkpoint。它不再列为论文主体闭环的前置 gap。动态状态以当前 evidence snapshot、最新 development
log 与 [自顶向下论文审查](paper_claim_gap_zh.md) 为准。
