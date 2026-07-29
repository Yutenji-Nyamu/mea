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
  → EvaluationIntent（合并原 Query 与 FreeConcern 中的显式 preservation 条件）
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
  → 若需要 scene/checker：TaskGen 精确检索；miss 时只生成声明所需的 scene/checker 子集
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
| TaskGen | `mea/taskgen/generic_backend.py`、`artifact_index.py`、`scripts/manipeval_taskgen.py` | 仅在 typed need 要求 scene/checker 时运行；exact reuse-first，miss 时生成声明所需的 scene 和/或实验 checker；每次 reuse 仍重跑当前 seed 的渲染、fixture 与 expert 验证 |
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
`EvaluationIntent` 必须保留原 Query 中的显式 keep/preserve/remain-unchanged 条件；
provider 的 FreeConcern 若漏写这些条件，不能把它们从执行合同中删除。单条条件同时
包含 contact、position、orientation 时，simulator-state gate 必须逐分量合取，而不是
命中第一个关键词就通过。
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
  round_*/check_success.py（仅 checker need 要求时）
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
  Planner→TaskGen→Tool/VQA→Answer 闭环；AdjustBottle 有一次在线生成式链，
  GrabRoller/PlacePhoneStand 仍为 official adapter。任务数量表示已验证执行边界，
  不限制开放 Query 能提出的 concern。
- `place_phone_stand` 的 expert N=1 成功而 ACT N=1 失败；该运行只证明第五个
  adapter、checkpoint、render/telemetry、Rule/VQA/Answer 接口连通，不支持稳定策略
  弱点或成功率结论。
- 当前 v19 flagship 从未提供 aspect 或 template：runtime 先选择 bell color，看到该轮
  成功后再选择 80% size，而不是预先执行固定菜单。三轮均使用 seed `100000`，
  official/color/size 的 ACT 结果分别为 `1/1`、`1/1`、`0/1`，并真实执行 VLM visual
  diagnosis 与 expert preflight。这证明在线机械编排和 evidence-conditioned 转向实际
  运行过，但不自动证明生成场景满足其语义合同。
- 最终 preservation audit 推翻了 v19 的正验收：round2 的 bell center 通过检查，但
  shape/size 没有 simulator 或 AST authority，不能从“未发现变化”推断为 preserved，
  因此 trace 为 `direct+partial`；round3 center 不变，但 contact-point z 从
  `0.7667903972` 变为 `0.7616323171`，差值 `-0.0051580801 m`，明确违反
  preservation contract，trace 为 `direct+partial/repair_required`。
- v19 第二轮按 Query need 生成新的 XY Tool，live 值为 `0.0077674431 m`；第三轮对相同
  semantic key 做 evaluation-local exact reuse，live 值为 `0.0452577472 m`。这些值是
  实际执行场景中的有效轨迹测量，但因为属性变化与 preservation failure 混杂，不能回答
  “只改变 color/size”的因果 Query。最终 audit 为 `accepted=false`，原 Query 必须保持
  inconclusive。
- 修正后的 preservation gate 对 exact spatial/contact 约束比较 same-seed simulator
  state；对没有 simulator/AST authority 的 geometry 返回 `unknown`，而不是乐观通过；
  `false` 会触发唯一一次局部 repair。generic backend 还会在 lookup/provider 之前拒绝
  同时要求 uniform scale、center/origin 不变与 contact-point world position 不变、且
  没有 custom pivot capability 的不可实现 proposal；该 gate 已通过服务器定向反例测试，
  但仍需一个可实现 candidate 的新在线运行才能取得正旗舰。
- generic TaskGen 会自动发现任意 source/schema-backed RoboTwin base task；新增 concern
  不再要求修改 BBH/ClickBell 方言。它目前只接受能由相同 seed simulator state 或 render
  观察到的 scene 变化；纯隐式物理变化仍需新的 simulator measurement hook。
- generated actor 会扩展本次 rollout 的 telemetry schema；ToolGen 在 ACT 后读取该实际
  schema。显式 final/terminal 位置分量由 `terminal_signal_component` 消费，语义
  alignment gate 会拒绝用 event-time/distance Tool 逃避该 measurement need。
  evaluation-local Tool 可在同一 evaluation 的后续 candidate/Query 中 exact reuse；
  跨独立 evaluation 的复用必须来自显式 reviewed registry，并在当前 episode 上再次做
  确定性与 oracle 校验。
- LIBERO/SmolVLA 由 `mea/libero/` 的 benchmark adapter/chain 负责，不进入 RoboTwin
  resolver。batch27 在 `libero_object/task0` 完成两回合结构闭环：official control
  成功，evidence 触发一个 state-compatible custom BDDL；custom rollout 失败，但合法
  进入 Tool/Aggregate/Planner，且相同 Tool need exact reuse 不增加 rollout。总计
  2 rollouts、132.698 s，`method_chain_valid=true`、`query_sufficient=false`。
  这只是 basic-adaptation method-chain smoke，不是鲁棒性、效率或跨模拟器一致性证据。
- generated checker 是实验评价语义，必须与 RoboTwin official success 分开报告。
- v19 只有一个 task、一个 seed、每个 candidate 一个 rollout（总 `N=3`）；它证明方法
  的在线机械链能够执行，并暴露了语义 preservation 验收缺陷。它不是有效属性弱点证据，
  更不证明广泛属性泛化、采样效率或多策略 ranking。
