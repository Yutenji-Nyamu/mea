# 架构与干净数据流

当前项目是在 RoboTwin 上复现 ManipEvalAgent 论文方法的受限实现。生产路径只保留一条：

```text
原始 Query
  → Global route（选择可运行的 RoboTwin task/policy）
  → ClaimFirst Planner + QueryContract（选择本轮要区分的 claim）
  → TaskProposal / ToolProposal
  → TaskGen（检索或生成 scene + check_success）
  → render + 一次局部 visual repair + expert/fixture gate
  → policy rollout（当前生产主链为 ACT；DP3 只用于 ranking pilot）
  → Rule Tool / VQA
  → Aggregate
  → evidence-conditioned next plan 或 evidence-sufficient stop
  → 回答原始 Query，并列出未覆盖候选、N 和限制
```

## 模块边界

| 阶段 | 主要位置 | 最小职责 |
| --- | --- | --- |
| 编排 | `scripts/manipeval_agent.py` | 创建 evaluation，逐轮调用下述阶段，写出紧凑 run bundle |
| 路由与规划 | `mea/planner/global_query.py`、`claim_first.py`、`query_contract.py` | 从 Query 和已有证据选择下一测试；不预写 aspect 顺序 |
| TaskGen | `mea/taskgen/`、`scripts/manipeval_taskgen.py` | retrieve-first；必要时生成 scene 与实验 checker；渲染、fixture 与 expert 验证 |
| Policy | `policy/ACT/eval_mea.sh` 及 paper experiment adapter | ACT 主链在明确 task、checkpoint、seed 下产生 rollout、video 与 telemetry；DP3 不伪装成生产主链 |
| ToolGen/VQA | `mea/toolgen/`、`mea/execution_vqa/` | retrieve-first；生成并验证缺失 metric；对 rollout 产生可追踪 observation |
| Aggregate/Answer | `mea/toolkit/aggregate.py`、`mea/feedback/` | 汇总样本，决定证据是否充分，回答 Query |

catalog 只是运行能力清单，不是另一套 Planner。论文消融、效率比较、人类/VQA
有效性和 policy ranking 属于 `experiments/paper/`，不得被生产入口隐式调用。

## 每次运行应保留的干净证据

每次 live evaluation 只需保留以下逻辑内容：

```text
query.txt
plan/
  query_contract.json
  round_01_proposal.json
  round_02_proposal.json
task/
  round_*/task.py 或 overlay.yml
  round_*/check_success.py
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

`manifest.json` 是唯一公共运行清单，只记录 Query、task、policy/checkpoint、seed、N、
各轮 proposal、artifact 相对路径、结果和限制。普通开发运行不再生成多层
receipt/ledger/provenance hash；正式 preregistration 实验可在实验目录额外冻结 hash。

Git 只发布一个最近运行的紧凑证据包 `docs/evidence/current/`：保留 Query/Proposal、
模型生成代码、两张 render、两个短 rollout、关键 Tool/Aggregate 和最终回答。完整 raw
bundle 留在服务器，历史结果压成 `docs/evidence/history.jsonl`，避免重复提交大体积
telemetry、VQA montage 和开发日志。

## 当前范围

- 生产评估以 ACT 为主，DP3 只用于 BBH 最小双 policy pilot。
- ACT official 入口覆盖 `beat_block_hammer`、`click_bell`、`adjust_bottle`、
  `grab_roller`；新增任务优先复用 official task、TaskSchema 和
  通用 recorder/tool，不复制整套 planner。
- generated checker 是实验评价语义，必须与 RoboTwin official success 分开报告。
- N=1/2 的 smoke 只能证明机制跑通，不能声称论文规模的泛化、效率或 ranking。
