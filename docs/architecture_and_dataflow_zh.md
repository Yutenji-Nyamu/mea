# 架构与干净数据流

当前项目是在 RoboTwin 上复现 ManipEvalAgent 论文方法的受限实现。`--auto-route`
开放 Query 的唯一生产主链是 `FreeConcern → ClaimFirst`；official inventory/catalog
只负责发现与绑定，legacy task-specific planner 也不是并行生产入口：

```text
原始 Query
  → FreeConcern（先抽取任务意图与待测变化，不向模型展示 task inventory）
  → official task discovery + retrieve/generate decision
  → policy binding（checkpoint 与 task identity）
  → policy/checkpoint compatibility gate
      ├─ 语义相近且 checkpoint 可执行：retrieve and adapt
      ├─ open-policy 且确有新任务需求：generate new
      └─ 不可执行或 scope 不明：unsupported
  → ClaimFirst Planner + QueryContract（选择 claim，并决定是否需要 control）
      ├─ Query 需要对照：materialize neutral official control
      └─ Query 不需要对照：直接进入首个 Query-derived candidate
  → runtime ExperimentCandidate（scene/checker/tool need；不要求 template id）
  → TaskGen（精确检索；miss 时生成 scene + check_success）
  → VLM + simulator state/checker fixture + render + expert gate
    （只允许一次局部 visual repair）
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
neutral official control 只是 QueryContract 在 claim 需要对照时调用的 unchanged-scene
materialization，不是每个 Query 的默认首轮。generic TaskGen 若要从同一采样 Pose 派生
另一个 actor 的位置，必须复制位置数组并新建
`sapien.Pose(new_position, old_pose.q)`；禁止复用原 Pose 后修改 `Pose.p/q` 的元素或做
`+=/-=`。VLM 判断不能单独放行，simulator state/checker fixture、render 与 expert gate
必须同时通过。
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
  scene/checker/Tool materialization。batch28 v4 已从 broad Query 在线产生一个
  position-translation candidate，经过 provider scene+checker 与第二次 ACT；两轮后因
  coverage 不足返回 inconclusive。这是单任务/单 seed 的接线验收，不是开放域泛化正结论。
- generic TaskGen 会自动发现任意 source/schema-backed RoboTwin base task；新增 concern
  不再要求修改 BBH/ClickBell 方言。它目前只接受能由相同 seed simulator state 或 render
  观察到的 scene 变化；纯隐式物理变化仍需新的 simulator measurement hook。
- generated actor 会扩展本次 rollout 的 telemetry schema；ToolGen 在 ACT 后读取该实际
  schema。evaluation-local Tool 可在同一 evaluation 的后续 Query 中 exact reuse；
  跨独立 evaluation 的复用必须来自显式 reviewed registry，并在当前 episode 上再次做
  确定性与 oracle 校验。
- ClickBell 现有两个互补旗舰：batch26 的 singleton distractor Query 在有限合同下以
  `evidence_sufficient` 停止；batch28 v4 从 broad Query 在线发现 position translation，
  完成两次 ACT 后以 `budget_exhausted`/inconclusive 停止。二者均无 aspect CLI hint 或
  缓存 replay，但都只覆盖 seed `100405`，不能合并成广泛泛化证据。
- LIBERO/SmolVLA 由 `mea/libero/` 的 benchmark adapter/chain 负责，不进入 RoboTwin
  resolver。batch27 在 `libero_object/task0` 完成两回合结构闭环：official control
  成功，evidence 触发一个 state-compatible custom BDDL；custom rollout 失败，但合法
  进入 Tool/Aggregate/Planner，且相同 Tool need exact reuse 不增加 rollout。总计
  2 rollouts、132.698 s，`method_chain_valid=true`、`query_sufficient=false`。
  这只是 basic-adaptation method-chain smoke，不是鲁棒性、效率或跨模拟器一致性证据。
- generated checker 是实验评价语义，必须与 RoboTwin official success 分开报告。
- N=1–5 的 smoke 只能证明机制或受限有限域结论，不能声称论文规模的泛化、效率或
  多策略 ranking。
