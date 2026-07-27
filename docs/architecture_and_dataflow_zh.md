# 架构与干净数据流

当前项目是在 RoboTwin 上复现 ManipEvalAgent 论文方法的受限实现。`--auto-route`
开放 Query 的生产路径只保留一条：

```text
原始 Query
  → FreeConcern（先抽取任务意图与待测变化，不向模型展示 task inventory）
  → official task discovery + retrieve/generate decision
  → policy/control binding（checkpoint、official control 与 task identity）
  → policy/checkpoint compatibility gate
      ├─ 语义相近且 checkpoint 可执行：retrieve and adapt
      ├─ open-policy 且确有新任务需求：generate new
      └─ 不可执行或 scope 不明：unsupported
  → ClaimFirst Planner + QueryContract（选择本轮要区分的 claim）
  → runtime ExperimentCandidate（scene/checker/tool need；不要求 template id）
  → TaskGen（精确检索；miss 时生成 scene + check_success）
  → render + 一次局部 visual repair + expert/fixture gate
  → policy rollout（当前生产主链为 ACT；DP3 只用于 ranking pilot）
  → 从实际 telemetry schema 生成或精确复用 Rule Tool / VQA
  → Aggregate
  → evidence-conditioned next plan 或 evidence-sufficient stop
  → 回答原始 Query，并列出未覆盖候选、N 和限制
```

## 模块边界

| 阶段 | 主要位置 | 最小职责 |
| --- | --- | --- |
| 编排 | `scripts/manipeval_agent.py` | 创建 evaluation，逐轮调用下述阶段，写出紧凑 run bundle |
| 任务执行边界 | `mea/taskgen/generic_backend.py` | `GenericRoboTwinTaskAdapter` 从 official source、TaskSchema、文档和资产发现薄适配；不枚举 aspect、variant、metric 或 planner route |
| 兼容性检索 | `mea/capability_adapter.py` | 旧任务/模板的 retrieval index 与消融兼容层；生产 open-world round 只把它当检索提示，不把成员关系当执行许可 |
| 开放任务解析 | `mea/planner/open_task_resolver.py`、`global_query.py` | Query-first 抽取 concern；从 official inventory 检索候选；在 policy scope 边界内决定复用、生成或 unsupported |
| Claim 规划 | `mea/planner/claim_first.py`、`query_contract.py`、`open_world_session.py` | 从已绑定 task、Query 和已有证据产生/追加 runtime candidate；按 claim truth condition 判断继续或停止 |
| TaskGen | `mea/taskgen/generic_backend.py`、`artifact_index.py`、`scripts/manipeval_taskgen.py` | exact reuse-first；miss 时生成 scene 与实验 checker；每次 reuse 仍重跑当前 seed 的渲染、fixture 与 expert 验证 |
| Policy | `policy/ACT/eval_mea.sh` 及 paper experiment adapter | ACT 主链在明确 task、checkpoint、seed 下产生 rollout、video 与 telemetry；ranking pilot 让 ACT/DP3 共享同一 expert scene gate |
| ToolGen/VQA | `mea/toolgen/`、`mea/execution_vqa/` | retrieve-first；生成并验证缺失 metric；对 rollout 产生可追踪 observation |
| Aggregate/Answer | `mea/toolkit/aggregate.py`、`mea/feedback/` | 汇总样本，决定证据是否充分，回答 Query |

official inventory/catalog 是可检索清单，不是另一套 Planner。发现某个 task 不等于当前
checkpoint 能执行它：单任务 checkpoint 只能在其声明任务上运行，除非另有可验证的
open-policy scope。若原始 Query 提出 catalog 外 concern，Planner 会保留
`ExperimentCandidate` 的 scene/checker/tool need；generic backend 可以在已绑定 base
task 上 exact reuse 或生成并验证。只有生成、当前 seed preflight 和 rollout 全部完成后，
该 concern 才能成为评价证据。
论文消融、效率比较、人类/VQA
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
- ACT official 入口现有 5 个 `TaskAdapter`：`beat_block_hammer`、`click_bell`、
  `adjust_bottle`、`grab_roller`、`place_phone_stand`。其中 BBH/ClickBell 有深入的
  Planner→TaskGen→Tool/VQA→Answer 闭环；后三个仅是 official adapter。准确表述是
  “五任务接入、两任务深入”，不是五任务论文复现。
- `place_phone_stand` 的 expert N=1 成功而 ACT N=1 失败；该运行只证明第五个
  adapter、checkpoint、render/telemetry、Rule/VQA/Answer 接口连通，不支持稳定策略
  弱点或成功率结论。
- catalog-external concern 已不再因缺少 template id 直接终止：运行时会形成稳定
  `ExperimentCandidate`，catalog 仅提供相近工件检索提示，随后进入通用
  scene/checker/Tool materialization。该路径已有 provider fixture 与主 CLI 组合测试，
  但本批尚无一次新的 provider+ACT 同 bundle live 验收，因此不能写成新的实验正例。
- generic TaskGen 会自动发现任意 source/schema-backed RoboTwin base task；新增 concern
  不再要求修改 BBH/ClickBell 方言。它目前只接受能由相同 seed simulator state 或 render
  观察到的 scene 变化；纯隐式物理变化仍需新的 simulator measurement hook。
- generated actor 会扩展本次 rollout 的 telemetry schema；ToolGen 在 ACT 后读取该实际
  schema。evaluation-local Tool 可同会话复用，显式 reviewed 的 typed Tool 可在新 Query
  中 exact reuse，并在当前 episode 上再次做确定性与 oracle 校验。
- ClickBell 的 clean flagship 已由一个生产 CLI 完成：inventory-free FreeConcern
  自动绑定唯一 distractor concern，official control 后由 evidence 触发 provider-written
  scene+checker，经过 6/6 fixtures、render/expert、第二次 ACT、Tool/VQA/Aggregate，
  最终以 `evidence_sufficient` 停止。全程无 aspect CLI hint、缓存 replay 或人工串接；
  仍只覆盖 seed `100405` 和一个有限候选。
- LIBERO/SmolVLA 由 `mea/libero/` 的 benchmark adapter/chain 负责，不进入 RoboTwin
  resolver。batch27 在 `libero_object/task0` 完成两回合结构闭环：official control
  成功，evidence 触发一个 state-compatible custom BDDL；custom rollout 失败，但合法
  进入 Tool/Aggregate/Planner，且相同 Tool need exact reuse 不增加 rollout。总计
  2 rollouts、132.698 s，`method_chain_valid=true`、`query_sufficient=false`。
  这只是 basic-adaptation method-chain smoke，不是鲁棒性、效率或跨模拟器一致性证据。
- generated checker 是实验评价语义，必须与 RoboTwin official success 分开报告。
- N=1–5 的 smoke 只能证明机制或受限有限域结论，不能声称论文规模的泛化、效率或
  多策略 ranking。
