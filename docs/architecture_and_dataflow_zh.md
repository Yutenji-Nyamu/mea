# 架构与干净数据流

## 当前生产边界

MEA 的生产方法链对 policy backend 中立。系统先解释开放 Query，再从 official task
source、runtime TaskContext 与已声明的 policy scope 中建立 `PolicyTaskBinding`；ACT、SmolVLA
或其他 policy 只是该 binding 选择的 rollout backend，不决定用户能问什么、Plan Agent
能提出什么 Proposal。

`mea/artifact_retrieval_index.py` 仅检索已审查的 Task/Tool/VQA artifact；
official inventory 仅发现可用 task。两者都不是执行许可表。执行许可来自：

- official task source 与 runtime reset 可用；reviewed TaskSchema 只是可选语义缓存，
  不是任务 allowlist；缺失时从 source-bound public root 下的原生 list/tuple/dict actor
  建立本次运行的 TaskContext；
- policy checkpoint 明确支持该 task；
- simulator/backend 发布了 Proposal 所需的 scene、checker、telemetry 或 VQA hook；
- 生成物通过对应的 simulator、fixture、render/VLM 与 expert gate。

生产入口使用 Plan Agent，不实例化 catalog 或任务专属 legacy planner。首轮计划由
`PlanAgentInitialPlanBuilder` 直接建立；只有 Query 需要对照时才先运行 neutral official
control。scene、checker、Rule Tool 与 VQA Tool 是相互独立的 typed need，系统只检索
或生成本轮实际需要的部分。

```text
开放 Query
  → Query interpretation
  → Runtime Task Binding（task + policy backend + checkpoint + seed）
  → Query contract
  → Plan Agent session
      → Proposal
      → TaskGen（仅 scene/checker need）
          → retrieve or generate
          → fixture/state + render/VLM + expert
          → 至多一次局部 repair
      → policy rollout
      → ToolGen（Rule/VQA retrieve or generate + validate + register）
      → Aggregate
      → Plan Agent：依据 completed evidence 提出下一 sub-aspect 或停止
      → QueryContract：验证停止是否被证据支持
  → Answer：回答原 Query，并列出 N、未覆盖项、冲突和限制
```

Plan Agent 可以提出 catalog 外 concern，但不能越过 backend 的真实执行面。无法测量或
无法干预的 concern 必须改写为可执行 Proposal，或报告 `unsupported/inconclusive`；
不能用相近旧 metric 或 VLM 判断代替 simulator authority。

## 模块边界

| 阶段 | 主要位置 | 最小职责 |
| --- | --- | --- |
| 入口编排 | `scripts/manipeval_agent.py`、`mea/agent_cli.py`、`mea/plan_agent_application.py` | CLI 解析参数和 binding；Application 原生拥有 round → evidence → next/stop → Answer 生命周期 |
| Query / binding | `mea/planner/open_task_resolver.py`、`runtime_task_binding.py` | 先解释 Query，再验证 task、policy scope、checkpoint 与 runtime hooks |
| Plan Agent | `mea/planner/claim_first_initial.py`、`claim_first_runtime.py`、`query_contract.py` | `PlanAgentSession` 建立首轮计划并依据完整 evidence 提出继续/停止；`open_world_session.py` 仅保留内部冻结执行运输与历史 reader 兼容 |
| TaskGen | `mea/taskgen/runtime.py`、`generic_backend.py`、`artifact_index.py` | 按 typed need 精确复用或生成 scene/checker，并完成有界验证与一次 repair |
| Policy backend | `PolicyTaskBinding` 与对应 runner | 在冻结的 task/checkpoint/seed 下输出 video、telemetry 与 official success |
| 单轮执行 | `mea/round_executor.py`、`mea/robotwin/runtime.py`、`native_agent_round.py` | `MethodRuntime.materialize_candidate()` 是唯一生产 TaskGen materialization owner，随后 rollout → Rule/VQA → Aggregate；冻结 artifact binder 仅供兼容/重放 |
| ToolGen / VQA | `mea/toolgen/`、`mea/execution_vqa/` | retrieve-first；miss 时按实际 telemetry 或视觉问题生成、验证、注册并运行 |
| Answer | `mea/toolkit/aggregate.py`、`mea/feedback/` | 汇总证据并生成受限回答 |
| 实验层 | `experiments/paper/` | fixed/adaptive、消融、ranking、人工/VQA 协议；不得反向成为生产入口 |

`CapabilityAdapter` 与旧 planner/dialect 只服务历史 artifact、兼容和论文消融。新增 task
不应要求增加任务名分支；理想新增项只提供 task identity、schema、policy binding 与
simulator/runtime hooks。

## Backend 与 simulator 现状

- RoboTwin 的 ACT 与 SmolVLA 已进入同一生产 `RoundExecutor`；SmolVLA 也可调用通用
  TaskGen，不再只是 `experiments/paper/` 的旁路 adapter。
- 代码路径统一不等于已有 SmolVLA 生成式正证据；最新运行状态只维护在
  [论文 claim 与 gap](paper_claim_gap_zh.md)，不固化在架构文档中。
- LIBERO 已有 SmolVLA benchmark/basic-adaptation chain，但尚未共享完整的 Plan Agent
  session、RoundExecutor、stop contract 与 Answer loop。
- generated checker 是实验评价语义，必须与 simulator official success 分开记录和回答。

TaskGen 的具体 preservation、Pose、geometry authority 和 repair 规则见
[开发者参考](developer_reference_zh.md)；安装、网络与 backend 环境问题见
[RoboTwin / SmolVLA 复现](robotwin_smolvla_reproduction_zh.md)和
[LIBERO / SmolVLA 复现与 MEA 接入](libero_smolvla_reproduction_zh.md)。架构文档不重复
维护这些故障手册。

## 每次运行保留的干净证据

```text
query.txt
plan/
  query_interpretation.json
  open_task_resolution.json
  query_sufficiency_contract.json
  initial_sub_aspect_proposal/proposal.json
  plan_agent_steps/
task/
  round_*/task.py 或 overlay.yml
  round_*/check_success.py（仅 checker need）
  round_*/render.png
rollout/
  round_*/video.mp4
  round_*/episode.json
evaluation/
  round_*/rule.json
  round_*/vqa.json
  aggregate.json
answer/
  answer.json
  report.md
manifest.json
```

`manifest.json` 是唯一公共运行清单，只记录 Query、binding、seed/N、逐轮 Proposal、
artifact 相对路径、结果和限制。完整 raw bundle 留在服务器；Git 只发布一个紧凑的
`docs/evidence/current/`，旧结果压成
[`docs/evidence/history.jsonl`](evidence/history.jsonl)。正式 preregistration 可在
`experiments/paper/` 额外冻结 hash，普通方法运行不增加 receipt/ledger 层。

运行结论和样本边界不固化在架构文档中。当前真值见
[论文 claim 与 gap](paper_claim_gap_zh.md)和
[当前证据](evidence/current/README.md)。
