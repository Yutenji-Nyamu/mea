# 架构与干净数据流

当前项目是在 RoboTwin 上复现 ManipEvalAgent 论文方法的受限实现。生产路径只保留一条：

```text
原始 Query
  → FreeConcern（先抽取任务意图与待测变化，不向模型展示 task inventory）
  → official task discovery + retrieve/generate decision
  → policy/checkpoint compatibility gate
      ├─ 语义相近且 checkpoint 可执行：retrieve and adapt
      ├─ open-policy 且确有新任务需求：generate new
      └─ 不可执行或 scope 不明：unsupported
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
| 开放任务解析 | `mea/planner/open_task_resolver.py`、`global_query.py` | Query-first 抽取 concern；从 official inventory 检索候选；在 policy scope 边界内决定复用、生成或 unsupported |
| Claim 规划 | `mea/planner/claim_first.py`、`query_contract.py` | 从已绑定 task、Query 和已有证据选择下一测试；不预写 aspect 顺序 |
| TaskGen | `mea/taskgen/`、`scripts/manipeval_taskgen.py` | retrieve-first；必要时生成 scene 与实验 checker；渲染、fixture 与 expert 验证 |
| Policy | `policy/ACT/eval_mea.sh` 及 paper experiment adapter | ACT 主链在明确 task、checkpoint、seed 下产生 rollout、video 与 telemetry；ranking pilot 让 ACT/DP3 共享同一 expert scene gate |
| ToolGen/VQA | `mea/toolgen/`、`mea/execution_vqa/` | retrieve-first；生成并验证缺失 metric；对 rollout 产生可追踪 observation |
| Aggregate/Answer | `mea/toolkit/aggregate.py`、`mea/feedback/` | 汇总样本，决定证据是否充分，回答 Query |

official inventory/catalog 只是可检索的任务与运行能力清单，不是另一套 Planner。发现某个
task 不等于当前 checkpoint 能执行它：单任务 checkpoint 只能在其声明任务上运行，除非
另有可验证的 open-policy scope。论文消融、效率比较、人类/VQA
有效性和 policy ranking 属于 `experiments/paper/`，不得被生产入口隐式调用。

## 每次运行应保留的干净证据

每次 live evaluation 只需保留以下逻辑内容：

```text
query.txt
plan/
  free_concern.json
  task_resolution.json
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
模型生成代码、current manifest 收录的 render/短 rollout、关键 Tool/Aggregate 和最终
回答。完整 raw bundle 留在服务器，历史结果压成 `docs/evidence/history.jsonl`，避免
重复提交大体积 telemetry、VQA montage 和开发日志。

## 当前范围

- 生产评估以 ACT 为主，DP3 只用于 BBH 最小双 policy pilot。
- RoboTwin official env 与 instruction 的交集当前可发现 50 个 task；这表示检索空间，
  不是 50 个 checkpoint-ready task，也不是 50-task 论文复现。
- ACT official 入口覆盖 `beat_block_hammer`、`click_bell`、`adjust_bottle`、
  `grab_roller`；新增任务优先复用 official task、TaskSchema 和
  通用 recorder/tool，不复制整套 planner。
- ClickBell 的 clean flagship 已由一个生产 CLI 完成：inventory-free FreeConcern
  自动绑定唯一 distractor concern，official control 后由 evidence 触发 provider-written
  scene+checker，经过 6/6 fixtures、render/expert、第二次 ACT、Tool/VQA/Aggregate，
  最终以 `evidence_sufficient` 停止。全程无 aspect CLI hint、缓存 replay 或人工串接；
  仍只覆盖 seed `100405` 和一个有限候选。
- LIBERO/SmolVLA 由 `mea/libero/` 的独立 adapter/chain 负责，不进入 RoboTwin resolver。
  当前 checkpoint 未声明训练 task scope；未显式绑定时 fail closed，显式绑定只授权
  official control，不证明 checkpoint 兼容。batch26 以已知 adapter parity 参数执行
  `libero_object/task0` official control，1 个 rollout 失败后正确短路，未生成或执行
  custom BDDL；因此只证明 fail-closed 协议，不是 LIBERO 方法链正例。
- generated checker 是实验评价语义，必须与 RoboTwin official success 分开报告。
- N=1–5 的 smoke 只能证明机制或受限有限域结论，不能声称论文规模的泛化、效率或
  多策略 ranking。
