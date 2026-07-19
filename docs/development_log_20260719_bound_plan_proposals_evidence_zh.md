# 2026-07-19：Bound PlanSession、Task/Tool Proposal 与 illustrated evidence

## 1. 论文主张与本批目标

本批从论文 Fig. 2–4 自顶向下补“Proposal 到可读证据”的公共边界，仍只评 ACT，并把真实运行
保持在 N=1：

| 论文位置 | 本批最小实现 | 证据边界 |
| --- | --- | --- |
| Fig. 2、Sec. 3.2 | 一次 evaluation 冻结一个 task/checkpoint 的 `BoundTaskPlanSession`；同一 session 内按证据换 sub-aspect | 核心状态机跨 task session 复用；runtime 后方仍有 BBH/click_bell adapter |
| Fig. 3、Sec. 3.3.1 | 严格 `TaskProposal`；reuse-first TaskGen 或真实 codegen/materialization | proposal 只在 catalog capability 内变化，不是任意任务生成 |
| Fig. 4、Sec. 3.3.2 | 严格 `ToolProposal`；Rule metric、ToolGen route 与 VQA assignment | VQA 仍从 allowlist 选择，不能自由生成问题实现 |
| Sec. 3.4 | illustrated evidence report 展示 proposal、代码/render/video、Tool/VQA/Aggregate/decision/final | 是可读真实证据，不替代 raw artifact、provenance 或论文统计 |

## 2. 实现与数据流

### 2.1 单 task / 单 checkpoint PlanSession

`mea/planner/session.py` 的 `BoundTaskPlanSession` 从 checkpoint-ready ACT catalog 建立
`EvaluationTarget`，固定 task、profile、planner kind、ACT policy/checkpoint、allowed aspects 和
`max_rounds`。它把 task adapter 的计划规范化成同一 schema，并拒绝：

- task 或 policy/checkpoint 漂移；
- catalog 外 aspect/template；
- 超过绑定预算的 round；
- TaskProposal 与 ToolProposal 的 task/aspect 不一致。

`task-agnostic` 的含义是同一个 session/transition API 可分别包装 BBH 与 `click_bell`，不是一个
evaluation 中途切任务。跨任务仍由 portfolio 创建独立 child evaluation。

### 2.2 Proposal 成为执行边界

`mea/proposals.py` 定义 paper-facing `TaskProposal` 与 `ToolProposal`。TaskProposal 只描述
task/aspect/capability/intent/bounded changes，固定 `reuse_first=true` 并保留 success semantics；
ToolProposal 只描述 evaluation goal、可 resolve metric、question 和 allowlisted VQA phenomena。
module path、checkpoint、seed、gate 和执行 argv 都由 runtime 掌握。

Agent 会把现有 task adapter 的轮次提升为 proposal，并在 TaskGen 前后、Tool route 和 VQA query
处做 exact binding。`BoundedProposalAgent` 则在固定 target/capability card 内调用模型生成一个
不同于已注册 template 的新 proposal；当前它通过独立 CLI 演示，尚未替代主 runtime 的
task-specific adapter。

### 2.3 可读 evidence bundle

`mea/feedback/evidence_report.py` 从 completed evaluation 只读选择真实 artifact：固定
task/checkpoint、初始分解、每轮 TaskProposal、task code/overlay、VariantSpec、render、ACT 结果和
小视频、ToolProposal/Tool source/result、VQA montage、Aggregate/decision、最终回答与 raw index。
缺失内容显示 `N/A`；视频超过上限不复制。

Agent 自动写 `evidence_report.md`；`scripts/manipeval_evidence_report.py --publish-dir` 生成适合
GitHub/手机阅读的 `README.md + assets/code/data + evidence_bundle_manifest.json`。

## 3. 真实最小验证

### 3.1 三个 plan-only 与一个 0-ACT materialization

- BBH Query：appearance + timing 被同一 bound session 分解，0 ACT；
- `click_bell` Query：position + instance 被同一 bound session 分解，0 ACT；
- bound `click_bell` friction Query：能力显式 unsupported，没有改选 BBH，0 ACT；
- bounded proposal：模型提出 `click_bell xy=[-0.14,-0.12]`，TaskGen 物化并完成真实 render/probe，
  ACT 为 0。

这些结果证明 task/checkpoint 边界、proposal schema 和 materialization 接线，不是 policy 结果。

### 3.2 两轮 BBH live evaluation

真实 run：`eval_20260719_batch11_bbh_adaptive_n1_v3`。

```text
open Query
→ bound BBH + ACT demo_clean-50
→ appearance TaskProposal
→ true codegen → render/expert gate → ACT N=1
→ evidence decision: continue timing
→ reuse task + generated timing Tool → ACT N=1
→ dynamic VQA + Aggregate → final feedback
```

两轮 ACT `policy_success=0`；expert controls 均成功，说明任务/seed 可解而 ACT 未成功。完整 wall-clock
为 `587.7 s`。因此可以声称 proposal/reuse/codegen/Tool/VQA/decision/final 数据流真实贯通，不能
声称 ACT 对 appearance 或 timing 的泛化表现已经统计成立。

### 3.3 published evidence

v3 已发布
`docs/evidence_runs/eval_20260719_batch11_bbh_adaptive_n1_v3/README.md`，bundle 约 `1.1 MB`。
它包含报告引用的真实 task/Tool 代码、render、ACT 视频、proposal 与 compact JSON 结果；大体积
raw evaluation 和 checkpoint 未复制进文档目录。

## 4. 失败、修复与真实预算

两次完整集成尝试各在启动 1 条 ACT 后暴露控制流问题：

1. round budget 曾在一个接线点被硬编码，导致证据要求继续时 session 上限不一致；修复后统一使用
   effective `max_rounds`。
2. 某些 adapter 合法省略可选 `task_name`，旧 equality check 却把缺省值当成与 bound task 不相等；
   修复后缺省值继承 session target，显式不同值仍 fail closed。

失败 run 不删除、不从成本中扣除。本批 ACT started 为：

```text
failed integration 1: 1
failed integration 2: 1
successful v3:       2
total started:       4
```

最终 v3 报告只消费其自身两条 completed round；前两条只作为开发成本和失败诊断保留。

## 5. 当前完成度与剩余 gap

本批把论文 Proposal 层从 task-specific plan 字段提升为共享 PlanSession + Task/Tool proposal，并把真实
方法证据整理成可直接检查的图文 bundle。主要剩余距离是：

1. proposal 仍受 catalog capability card 约束，未覆盖任意任务/资产生成；
2. VQA question implementation 仍来自 allowlist，ToolProposal 只做选择与绑定；
3. 主 runtime 仍由 BBH/click_bell adapter 在 PlanSession 后物化轮次，尚未完全由通用 Proposal
   Agent 驱动；
4. 本批每轮 N=1、仅 ACT，没有论文规模 repetition、人工 gold、多 policy 或 benchmark 结论。

因此当前可以称为“论文主体方法的小规模功能闭环与可读证据进一步完整”，不能称为完整论文
实验复现。
