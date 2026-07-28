# 架构与干净数据流

当前项目是在 RoboTwin 上复现 ManipEvalAgent 论文方法的受限实现。`--auto-route`
开放 Query 的唯一生产主链是
`FreeConcern → runtime binding → direct ClaimFirst`。首轮计划由
`ClaimFirstInitialPlanBuilder` 直接创建，不再以 `CatalogPlanAgent` 或任务专属 Planner
生成壳计划；official inventory/catalog 只负责检索与绑定，不决定 Query 能问什么或
runtime candidate 能否执行。legacy task-specific planner 仅由论文消融/兼容入口延迟加载：

```text
原始 Query
  → FreeConcern（先抽取任务意图与待测变化，不向模型展示 task inventory）
  → official task discovery + retrieve/generate decision
  → policy binding（checkpoint 与 task identity）
  → policy/checkpoint compatibility gate
      ├─ 语义相近且 checkpoint 可执行：retrieve and adapt
      ├─ open-policy 且确有新任务需求：generate new
      └─ 不可执行或 scope 不明：unsupported
  → QueryContract（按原 Query 真值条件决定是否需要 control）
  → ClaimFirstInitialPlanBuilder（直接建立初始计划）
      ├─ Query 需要对照：materialize neutral official control
      └─ Query 不需要对照：等待/接入首个 Query-derived candidate，不制造 control 壳
  → OpenWorldPlanSession.from_target（只规范化已冻结的 task/checkpoint binding）
  → runtime ExperimentCandidate
    （scene/checker/tool 是相互独立的 typed need；不要求 template id）
  → 若需要 scene/checker：TaskGen 精确检索；miss 时生成 scene + check_success
  → 若生成 Task：VLM + simulator state/checker fixture + render + expert gate
    （只允许一次 TaskGen 局部 visual repair）
  → policy rollout（当前生产主链为 ACT；DP3 只用于 ranking pilot）
  → 按实际 telemetry schema 生成或精确复用 Rule Tool
  → task-owned VQA；无已审查问题时使用通用 tracked-object 问题
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
| Claim 规划 | `mea/planner/claim_first_initial.py`、`claim_first.py`、`query_contract.py`、`open_world_session.py` | 直接创建首轮计划；从已绑定 task、Query 和已有 evidence 产生/追加 runtime candidate；按 claim truth condition 判断继续或停止 |
| TaskGen | `mea/taskgen/generic_backend.py`、`artifact_index.py`、`scripts/manipeval_taskgen.py` | 仅在 typed need 要求 scene/checker 时运行；exact reuse-first，miss 时生成 scene 与实验 checker；每次 reuse 仍重跑当前 seed 的渲染、fixture 与 expert 验证 |
| Policy | `policy/ACT/eval_mea.sh` 及 paper experiment adapter | ACT 主链在明确 task、checkpoint、seed 下产生 rollout、video 与 telemetry；ranking pilot 让 ACT/DP3 共享同一 expert scene gate |
| ToolGen/VQA | `mea/toolgen/`、`mea/execution_vqa/` | retrieve-first；按已声明 telemetry 字段生成并验证缺失 metric；VQA 优先使用 task-owned 问题，再退回通用 tracked-object 问题 |
| Aggregate/Answer | `mea/toolkit/aggregate.py`、`mea/feedback/` | 汇总样本，决定证据是否充分，回答 Query |

official inventory/catalog 是可检索清单，不是另一套 Planner。生产入口不会实例化
`CatalogPlanAgent`、BBH/ClickBell Planner 或 Official Planner；显式选择历史协议时，
`experiments/paper/legacy_planner_factory.py` 才延迟导入它们。发现某个 task 不等于当前
checkpoint 能执行它：单任务 checkpoint 只能在其声明任务上运行，除非另有可验证的
open-policy scope。若原始 Query 提出 catalog 外 concern，Planner 会保留
`ExperimentCandidate` 中实际需要的 scene/checker/tool 子集；Tool-only Query 不再被迫
生成新场景或 checker。generic backend 可以在已绑定 base task 上 exact reuse 或生成并
验证。只有本轮要求的生成、当前 seed preflight、rollout 和 Tool/VQA 全部完成后，该
concern 才能成为评价证据。
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
  `adjust_bottle`、`grab_roller`、`place_phone_stand`。BBH/ClickBell 有较成熟的
  Planner→TaskGen→Tool/VQA→Answer 闭环；AdjustBottle 新增了一次在线生成式方法链，
  但仍需 0-ACT 追加修复才能得到正确的 terminal Tool 与 task-owned VQA。
  GrabRoller/PlacePhoneStand 仍是 official adapter。准确表述是
  “五任务接入、两任务较深入、一个任务有生成式验收”，不是五任务论文复现，也不是
  三个无瑕的在线旗舰。
- `place_phone_stand` 的 expert N=1 成功而 ACT N=1 失败；该运行只证明第五个
  adapter、checkpoint、render/telemetry、Rule/VQA/Answer 接口连通，不支持稳定策略
  弱点或成功率结论。
- catalog-external concern 已不再因缺少 template id 直接终止：运行时会形成稳定
  `ExperimentCandidate`，catalog 仅提供相近工件检索提示，随后进入通用
  scene/checker/Tool materialization。batch28 v4 已从 broad Query 在线产生一个
  position-translation candidate，经过 provider scene+checker 与第二次 ACT；两轮后因
  coverage 不足返回 inconclusive。这是单任务/单 seed 的接线验收，不是开放域泛化正结论。
- `eval_20260728_adjust_bottle_open_live_v2` 从未提供 aspect/template：FreeConcern
  提出未见瓶体几何，official control evidence 随后促使 Planner 在线提出
  `task_execution.success_margin_components`。第二轮由 provider 编写 scene/checker，
  通过静态、2/2 checker fixture、VLM（0.88）和 expert gate 后执行 ACT。原始在线
  Tool 选择了 episode 中不存在终止事件的时间度量而返回 null，VQA 又错误继承了
  BBH 问题；因此源 bundle 以 `budget_exhausted`/inconclusive 停止，必须原样保留。
- 源 bundle 下的追加式
  `repairs/terminal_signal_component_repair_v3` 在 0 ACT 下用新的通用
  `terminal_signal_component` 读取
  `bottle_functional_position.z=0.771909236907959 m`：首次为
  `typed_metric_spec_compile`、第二个相同 Query 为 `run_local_reuse`，两次均
  `provider_called=false`；Aggregate 通过，Planner replay 得到
  `evidence_sufficient`/`diagnosed`。但 v3 Planner replay 仍消费源 bundle 的错误 BBH
  VQA，不能单独作为“修复后 Tool+VQA”的组合结论。独立
  `repairs/vqa_task_owned_replay_v1` 只询问
  `bottle_visibly_repositioned`，VLM 观察为 true（0.98），同时因 generated/official
  核心 predicate 为 false 保留 `evidence_conflict=true`。canonical composed replay
  `repairs/terminal_tool_plus_task_vqa_repair_v7` 在一次追加式 0-ACT 审计中同时消费
  terminal Tool 和 task-owned VQA：exact reuse 与 Aggregate 通过，EvidencePacket 为
  `conflicting`，Planner `should_stop=true`、`stop_reason=evidence_conflict`、
  `verdict=inconclusive`、`evidence_sufficient=false`。v3/v1 是分阶段诊断，v6 才是组合
  真值；三者都不能回写为源在线 bundle 的正例，也不能把 generated checker 等同于
  official success。
- generic TaskGen 会自动发现任意 source/schema-backed RoboTwin base task；新增 concern
  不再要求修改 BBH/ClickBell 方言。它目前只接受能由相同 seed simulator state 或 render
  观察到的 scene 变化；纯隐式物理变化仍需新的 simulator measurement hook。
- generated actor 会扩展本次 rollout 的 telemetry schema；ToolGen 在 ACT 后读取该实际
  schema。显式 final/terminal 位置分量由 `terminal_signal_component` 消费，语义
  alignment gate 会拒绝用 event-time/distance Tool 逃避该 measurement need。
  evaluation-local Tool 可在同一 evaluation 的后续 Query 中 exact reuse；
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
