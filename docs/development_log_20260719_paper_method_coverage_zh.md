# 2026-07-19：论文主体方法 16 项覆盖审计

## 1. 本批目标

本批从论文 Fig. 2–5、Secs. 3.2–3.4 和 Appendix A.3 自顶向下检查主体方法，不增加 policy，
也不扩大 repetition。目标是让每个论文主张都有一个受限、可执行、可审计的最小实现，并用
`scripts/manipeval_method_coverage.py` 区分“源码合同存在”和“真实运行证据存在”。默认预算仍是
`0 → 1 → 3 → 5 ACT`；只有链路稳定且确有统计需要时才扩大，日常开发不执行论文的 10 次完整重复。

## 2. 16 项方法映射

| # | 论文主张与位置 | 本批后的最小实现 | 当前可信边界 |
|---:|---|---|---|
| 1 | Plan Agent 根据观测决定继续、切换方面或停止；Sec. 3.2、Figs. 2/5 | `BoundTaskPlanSession` 从 typed evidence 产生允许的 transition 集；live Planner 可在合法 aspect/template 候选中选择，Session 再裁决 | 受限状态机，不是无约束自然语言规划 |
| 2 | 每轮 Observation 回到 Proposal；Figs. 2–4 | `--proposal-mode bounded_each_round` 在真实 `continue` 后重新生成受 capability 约束的 Task/Tool Proposal | `plan-only` 只能验首轮；逐轮行为至少需要前一轮真实或可审计 replay 证据 |
| 3 | TaskGen/ToolGen 不应特定于单一任务；Fig. 2、Sec. 3.3 | 公共 `EvaluationTarget`、`PlanningContext`、capability adapter 与 `VariantSpec` 合同 | 当前真实 adapter 主要覆盖 BBH 与 `click_bell`，并非任意 RoboTwin task |
| 4 | TaskGen 交付完整可执行任务；Sec. 3.3.1、Fig. 3 | `TaskArtifactBundle` 显式绑定 scene method 与 official `check_success()`；BBH codegen、bell bounded overlay、official reuse 共用产物合同 | success method 仍复用官方语义，不声称模型生成任意 success code |
| 5 | ToolGen 从 Proposal/任务代码产生新 Tool；Sec. 3.3.2、Fig. 4 | `ToolProposal v3 + MetricSpec v1` 可编译、静态校验、双 episode 差分验证、注册和复用 | DSL 目前只有 `minimum_distance`，不是开放式任意 Python ToolGen |
| 6 | Planner 读取 policy 与 simulator 能力；Sec. 3.2 | `PlanningContext` 保存 `PolicyCard`、`SimulatorCard` 和 adapter view，并在初始全局 Query route 前进入模型 prompt | card 来自受信项目元数据；不由模型猜测 checkpoint 能力 |
| 7 | scalar Tool、VQA 与 pipeline 状态共同形成证据；Sec. 3.2、App. A.3.5 | `EvidencePacket` 保存 pipeline、policy、rule、VQA 与 `sufficient/uncertain/conflicting/pipeline_invalid`；请求了 VQA 但缺失/失败时只能为 uncertain | 使用可审计类别，不伪造概率或置信区间；字符串布尔和 bool 计数会被拒绝 |
| 8 | Task/asset/document RAG，reuse first；Sec. 3.3.1、Fig. 3 | knowledge index 增加 `click_bell` task card、`050_bell` asset card，TaskGen 保存命中与 freshness | 小型项目知识库，不等于论文规模检索效果 |
| 9 | render 后视觉诊断与修复；Sec. 3.3.1、Fig. 3、App. A.3.4 | Proposal/VariantSpec 派生 `SceneCheckSpec`；BBH 可 bounded repair，bell validate-only | 只有 BBH 已有真实语义修复通路；不能声称通用视觉代码修复 |
| 10 | 历史 evaluation 支持一致规划；App. A.3.3 | history retrieval 同时使用 task、canonical aspect overlap 与文本相关性排序 | 仍是项目内历史库；没有论文规模长期记忆实验 |
| 11 | 审核后的 Tool 跨 evaluation 复用；Sec. 3.3.2、App. A.3.3 | reviewed registry 按可执行 ToolSpec 语义匹配；自然语言问句改写不破坏同一 Tool 的复用 | task/metric/schema/code 仍须完全一致，避免把“语义复用”变成宽松误命中 |
| 12 | Planner taxonomy 与 unsupported 边界；Sec. 3.2、App. A.4 | ontology 覆盖 query-gold 中的 appearance/physics/scale/camera/occlusion/performance/language/safety 等轴 | ontology 中存在不代表当前 ACT task 已有 materializer；unsupported 必须显式返回 |
| 13 | `README.Agent` 提供 TaskGen 知识；Sec. 3.3.1、Fig. 3 | task-specific knowledge snapshot、source symbol 与 freshness hash | hash 只证明来源未漂移，不证明文档正确或任务生成有效 |
| 14 | 分阶段失败恢复与长实验 resume；App. A.3.4 | 已有 stage-aware recovery、chunk/resume protocol，本批补 mock resume 验收 | mock 证明控制流；policy/simulator failure 仍是结果，禁止为“成功”重试 |
| 15 | rollout keyframe 上的 Dynamic Execution VQA；App. A.3.5 | `manipeval_execution_vqa_replay.py` 可对已完成真实 ACT 视频增加 run-local 问题，0 新 ACT | coverage 还要求 provider metadata、视频/episode、关键帧选择和完整 PNG 可追溯；仅模型名加伪 montage 不算证据 |
| 16 | fixed 与 adaptive 使用 matched ACT 预算比较；Tables 1–2 的机制 | 已有同 policy/task/checkpoint 候选集的 comparator 与 N=1 artifact validator | N=1 只证明机制与计量；不支持效率均值、显著性或论文表结论 |

## 3. 当前主体数据流

```text
open Query
  → GlobalQueryRouter：选择一个 checkpoint-ready task，显式列 unsupported aspects
  → BoundTaskPlanSession：冻结 ACT policy/checkpoint/round budget
  → PlanningContext：PolicyCard + SimulatorCard + capability adapter
  → bounded TaskProposal + ToolProposal（每个 evidence-driven continue 可重新生成）
  → TaskGen retrieve/reuse/generate
       → TaskArtifactBundle + SceneCheckSpec
       → static gate → render/visual diagnosis → bounded repair → expert gate
  → ACT N=1 execution（开发默认）
  → trusted Rule Tool / typed MetricSpec + rollout-keyframe Dynamic VQA
  → Aggregate → EvidencePacket
  → Planner 继续、切 aspect 或停止
  → final strengths / weaknesses / recommendations / limitations
```

一次 evaluation 固定单任务 ACT policy 是设计前提，而不是缺陷；系统层面的通用性由公共合同和多个
task adapter 承担。跨任务问题拆成多个固定 task child，再由 portfolio 汇总，不能让单任务 ACT
checkpoint 在 evaluation 中间切换任务。

## 4. 0-ACT 审计与最小验收

16 项审计只读源码和既有 JSON artifact，不调用 provider、simulator 或 ACT：

```bash
PYTHON=/root/autodl-tmp/conda/envs/RoboTwin/bin/python

"$PYTHON" scripts/manipeval_method_coverage.py \
  --repo-root "$PWD" \
  --output mea/validation_runs/batch13_method_coverage/report.json \
  --markdown mea/validation_runs/batch13_method_coverage/report.md
```

`implemented` 表示该项的源码检查和声明的运行证据检查均通过；`evidence_pending` 表示代码已就绪，
但找不到通过严格 validator 的运行 artifact；`partial` 表示源码合同仍缺。审计本身不是实验，也不会
把 N=1、缓存 replay 或 development-agent proxy 升格成论文结果。

逐轮 Proposal 的最低成本验收分两步：先用 `--plan-only --proposal-mode bounded_each_round` 检查首轮
Proposal 与 PlanningContext（0 ACT）；只有决定支付一次 ACT 后，首轮真实 observation 才能触发第二轮
Proposal。`MetricSpec` 可先运行单元测试验证严格 DSL、编译和差分复用；真实缓存 telemetry smoke 仍需
使用两个 task/schema 匹配且 oracle 值不同的 episode。run-local VQA replay 复用既有 ACT 视频，支付
一次视觉 provider 调用但启动 0 个新 ACT。

## 5. 服务器最小验收结果

- 完整回归：`454/454` 通过。
- live `plan-only`：`eval_20260719_batch13_bounded_each_round_plan_smoke_v1` 成功；初始
  GlobalQueryRouter 消费 `click_bell` 的 RoboTwin PlanningContext，保存首轮 bounded Proposal，
  `act_rollouts_started=0`。
- typed MetricSpec：复用同一真实 click_bell child 的 expert 与 ACT telemetry；首次
  `typed_metric_spec_compile`，问句改写后 `run_local_reuse`，两次 provider 均未调用；oracle 值分别约为
  `0.000659 m` 与 `0.048398 m`，`act_rollouts_started=0`。
- run-local Dynamic VQA：复用 `eval_20260717_stage1_global_click_flagship_v3` 的真实 ACT 视频，
  live `gpt-5.6-luna` 返回 4 个关键帧和一个 `run_local.*` 观察，`act_rollouts_started=0`。
- 最终 coverage：`15 implemented / 1 partial / 0 evidence_pending`。partial 仅为第 4 项：
  TaskGen 仍复用 official `check_success()`，尚未生成并验证通用 success function。
- matched fixed/adaptive 沿用已有注册 N=1 artifact；本批没有增加 ACT rollout。上述结果都是功能/机制
  证据，不是论文规模的统计结论。

## 6. 已知剩余 gap

1. Proposal Agent 仍只在受信 capability 中选择/生成变化；尚未覆盖论文所暗示的开放式任意任务生成。
2. TaskGen 的 success semantics 主要复用官方 `check_success()`；尚无通用生成、验证和修复成功函数。
3. `MetricSpec v1` 只有一个安全数值算子；要覆盖更多论文 ToolGen 场景，需增量加入 typed operator，
   而不是放开任意代码执行。
4. BBH legacy `verify` 与 `click_bell` adaptive 路径尚未完全统一到同一运行时 adjudication。
5. 真实 evidence-driven 分支差异仍缺 matched 小实验；下一步先 N=1，稳定后最多 3/5。
6. Planner gold、VQA gold 和 Tool/Task generation review 目前部分由 development-agent proxy 代替；
   论文 Table 3、6–8 仍需独立人工与合格正负样本。
7. 当前只用 ACT、任务与 seed 很少；这符合功能优先范围，但不能声称复现论文的大规模有效性结论。

因此，本批目标应表述为“论文主体方法的受限功能覆盖与可执行审计”，不是“整篇论文实验复现完成”。
