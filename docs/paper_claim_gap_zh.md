# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–5、Tables 1–10 与 Appendix。本文只维护当前真值；
历史运行摘要见 [`docs/evidence/history.jsonl`](evidence/history.jsonl)。

## 当前方法真值

当前证据入口是
[`eval_20260731_batch32_clean_flagship_live_v18`](evidence/current/README.md)。
这是首个把论文方法主干合入同一真实 RoboTwin bundle 的小范围正例：

1. broad Query 未给 aspect/template；official control 成功后，Plan Agent 才选择
   `grab_roller` 的世界 x 轴 `+0.05 m` 变化。
2. 通用 TaskGen 由 provider 编写 scene 与实验 checker；AST、`2/2` fixture、
   render/VLM、expert 与 preservation gate 均通过。
3. SmolVLA 在新场景仍满足 official core，但未满足“official core 且双侧 terminal
   TCP-to-contact 距离均不超过 `0.025 m`”的实验谓词。
4. ToolGen 生成 `terminal_max_tcp_contact_distance`，从真实 telemetry 得到
   `0.24384725093841553 m`，进入 Aggregate；VQA 无证据冲突。
5. Plan Agent 主动提出 stop，QueryContract 验证
   `evidence_sufficient=true / counterexample_found`，随后回答原 Query。
6. 一个独立 follow-up Query 在零 rollout、零 provider 下命中同一 registration，
   `run_local_reuse` 得到相同数值；registry 与原始 summary 字节未改变。

本次只有两个 policy episode、一个 seed `100401`、一个任务与一个生成候选。
generated checker 是有界实验语义，不是官方 benchmark checker；因此这是方法链
验收，不是统计泛化、benchmark 排名或论文全部实验结论。原始 v18 summary 在运行时
仍使用旧 acceptance 投影而显示 false；当前代码的 append-only 重投影为 true，两者均
保留在证据包。原始 final-summary 还多写了一条“缺少轨迹峰值”的保守限制，而 Query
只要求从 rollout telemetry 得到一个诊断标量；提示词已由该失败例修正，原始产物未被
回写。

### 近期补充事实（不替换当前旗舰）

Batch33–36 扩展了跨任务、第二个多任务 policy 与 schema-free TaskGen 证据，但没有产生
比 v18 更完整的可接受旗舰：

- `press_stapler` v3 从未指定 aspect/template 的 Query 出发，在同一 bundle 中完成
  evidence-conditioned scene refinement，并生成有 live finite 值的 Python Tool；第三轮精确
  `run_local_reuse` 同一 registration。三次 policy episode 均成功，最终因 `N=3` 预算停止，
  `evidence_sufficient=false`；checker 是 official reuse，因此不能替代 v18 的完整
  scene+checker 正例。
- v4 在前三轮成功后由 Plan Agent 提出更强的第四候选；TaskGen 的 expert gate 两次均以
  `target_pose cannot be None` 拒绝该候选，第四次 policy rollout **没有启动**。这属于
  candidate materialization failure，不是 policy failure。
- v5 在同一在线 bundle 中闭合了这条恢复链：三轮成功证据促使 Agent 提出 vertical
  `+0.16 m` Proposal；其 TaskGen 拒绝成为 typed `N=0` planning evidence；Agent 随即切换为
  lateral `+0.20 m`，TaskGen、SmolVLA rollout 与复用 Tool 再次成功。四个 policy episode
  均成功，最终因五个 method-step 外部上限停止，仍为 inconclusive，而非泛化正结论。
- v5 同时暴露 source answer 把 planning rejection 误算成全局 `pipeline_invalid`。当前代码已把
  policy pipeline health 与 planning gap 分开，并有缓存 fixture 回归；没有篡改 source 产物。
- SmolVLA 新增五任务 official N=1 为 `2/5`；连同既有结果，目前共有 13 个任务得到明确
  official outcome，为 `8/13` 成功。它只说明 policy/backend 广度，不代表 13 个任务均完成
  TaskGen、ToolGen 或多轮 MEA。
- Hy-VLA official RoboTwin checkpoint 已在服务器完成离线 official-wrapper 验证，并在
  `press_stapler / demo_clean / seed=10000` 的 official N=1 中成功。生产 v9 随后复用
  RoboTwin 共享方法外层，完成 official rollout、Rule Tool 精确复用、Aggregate、Agent
  stop、QueryContract evidence sufficiency 与 cached Answer finalization。它证明第二个
  多任务 policy backend 可进入一轮受限 official-control MEA；没有证明生成式
  scene/checker/Tool、多轮 refinement 或多任务排名。
- Batch34 将 reviewed TaskSchema 从生产准入条件降为缓存；仓库当前五份 reviewed schema
  只是可选 fast path：同一 runtime probe 已在
  `put_bottles_dustbin`、`place_bread_basket`、`press_stapler` 三个无 reviewed schema 任务
  建立 TaskContext，并在 `grab_roller` 保持 reviewed schema 兼容；两个容器任务分别发现
  3 与 2 个嵌套 actor。`dump_bin_bigbin` 在两个 seed 均被 RoboTwin 的 `UnStableError`
  提前拒绝，未进入 TaskContext。该结果只是 reset/schema 证据，不是 TaskGen 或 policy 结果；
  紧凑记录见
  [`batch34_task_independent_context/probe_summary.json`](../experiments/paper/results/batch34_task_independent_context/probe_summary.json)。
- Batch35 用同一 production Query 和 SmolVLA 跑了五任务、7-rollout 的 N=1 方法矩阵。
  `press_stapler` 三轮均完成，evidence 连续细化位移且第三轮 exact Tool reuse；三个任务的
  official control 为真实 policy negative；`grab_roller` 在 official success 后因 TaskGen
  expert hook 失败而停止，generated rollout 未启动。该矩阵把方法状态、policy outcome 与
  Answer sufficiency 分开，但不是 benchmark 成功率实验。
- 独立 v2 补充在无 reviewed schema 的 `press_stapler` 上首次让 generic provider 在一个
  attempt 内生成 scene 与 official-core-conjunct checker，并实际完成第二次 SmolVLA
  rollout；新 Python Tool 从 telemetry 得到 `0.09898103773593903 m` 并进入 Aggregate。
  运行因两轮预算耗尽而停止，`evidence_sufficient=false`，Tool 无独立数值 oracle且本次未
  exact reuse，因此只关闭 schema-free live TaskGen 执行 gap，不替换 v18 旗舰。
- Batch36 v5 首次在同一 broad Query 中完成
  `TaskGen typed N=0 → evidence-conditioned orthogonal concern → model-written scene/checker
  → successful SmolVLA rollout → independently validated live Tool → new concern`。Round 3
  Tool 值为 `0.08938229084014893 m`；第二个独立 Query 以零 provider、零 rollout 精确命中
  同一 registration。Round 4 的 checker 被 expert-solvability gate 拒绝，随后 provider 空响应，
  因而没有 Agent 主动 stop 或 final Answer。紧凑原始产物见
  [`batch36_v5_refinement`](evidence/supplements/2026-08-04/batch36_v5_refinement/README.md)。

冷结果索引与服务器原始路径见
[`batch33_open_cross_task/README.md`](../experiments/paper/results/batch33_open_cross_task/README.md)
和
[`batch35_generic_method_matrix/README.md`](../experiments/paper/results/batch35_generic_method_matrix/README.md)。

## 方法 claim

| 论文 claim | 当前项目 | 判断 |
| --- | --- | --- |
| Fig. 2/5：开放 Query 驱动 Plan Agent 自主提出 sub-aspect | v18 在 completed control evidence 后才选择轴与精确位移；Batch36 v5 又让 TaskGen 负 evidence 从 x 关系切换到正交 y concern，随后由 live Tool 转向 vertical concern | **小范围行为完成**；仍缺同一 broad Query 最终主动 stop 的合一正例 |
| evidence 决定下一轮，并在充分时停止 | v18 为 `continue → 新 Proposal → Agent stop → QueryContract 验证`；Batch36 v5 为 `TaskGen rejection → N=0 evidence → switch concern → executable rollout → new concern` | **两类行为均有 live 证据**；它们尚未在同一 broad Query 中合一为 `evidence_sufficient` 主动停止，有限 existential 合同也不代表广泛充分性 |
| Fig. 3：Proposal → retrieve/generate scene + `check_success()` → rollout | v18 同链生成 scene/checker 并裁决真实 policy episode；Batch35 v2 又在无 reviewed schema 的 runtime TaskContext 上完成 model-written scene/checker 与真实 rollout | **小范围双例完成**；覆盖两个任务、每例一个 seed，simulator hook 与语义 preservation 仍是真实边界 |
| 首帧视觉诊断与局部重新生成 | render/VLM 与一次局部 repair 路径已在历史真实案例触发；v18 无需 repair | **组件完成**；视觉只审外观，数值关系仍由 simulator/fixture 审计 |
| Fig. 4：ToolGen retrieve/generate/validate/register/reuse | v18 新 Python Tool 有 live finite 值；Batch36 v5 在同一 evaluation 完成独立 oracle、live finite 值、影响下一 Proposal 与第二 Query exact reuse | **同一 evaluation 内完成**；尚未证明 reviewed registry 的跨 evaluation 长期复用 |
| rollout → Rule/VQA → Aggregate → Plan Agent → Answer | RoboTwin 的 SmolVLA/ACT 共用 `RoundExecutor` 与方法外层；Hy-VLA v9 也完成 official-control round、validated stop 与 cached Answer | **RoboTwin 小范围跨 policy 完成**；LIBERO 仍有独立外层 |
| 回答原 Query 并约束确定性 | `AnswerScope` 报告 N、seed、候选域、冲突、停止原因与语义边界 | **完成度较高**；还需避免模型凭空收紧 Query 子要求 |

## 实验 claim

| 论文 claim | 当前证据 | 判断 |
| --- | --- | --- |
| Tables 1–2：更少 samples/time 保持 dense 结论 | 三 seed toy 从 12 降到 6 ACT，但完整 failure set 仅保持 2/3 | **真实节省、结论不完全一致** |
| Table 3：RAG、visual self-check、README.Agent 提升 codegen | 五个 frozen Proposal 的小型消融，只有 RAG 有方向信号 | **未复现结论** |
| Table 6：Plan 与机器人研究者一致 | 30 条 development-agent proxy；无人类 gold | **未复现** |
| Tables 7–8：VQA 四条件 accuracy/AUROC | 8 个缓存样本、单 VLM、proxy gold | **仅协议 smoke** |
| Table 9：少样本保持多 policy 排名 | ACT/DP3 三 seed 为 2/3 对 2/3，Spearman 不可算 | **未复现** |
| Fig. 6：系统错误率与模块分布 | 固定 operation 分母很小 | **不可比较** |
| RoboTwin/LIBERO 跨任务适配 | SmolVLA 已有 13 任务 official outcome（8/13）；Hy-VLA 有 official N=1 及共享外层 official-control 正例；LIBERO 有 basic adaptation | **RoboTwin 已跨两个 generalist backend；LIBERO 方法外层仍未统一** |

## 下一步主干

1. **完成一个 broad Query 的主动停止 clean flagship。** 在 Batch36 v5 已把 rejection、
   concern switch、scene/checker、rollout、Tool 与 reuse 合入同一 evaluation 的基础上，只需
   让同一 bundle 最终由 Plan Agent 提出 stop、QueryContract 验证
   `evidence_sufficient`；不得靠 hard cap，也不得把 planning gap 计作 policy failure。
2. **保持 checker 的 expert-solvable 语义。** Batch36 的 vertical 失败来自 checker 把场景
   改动本身编码为必败条件；当前 prompt 与 bounded validation 已修正。下一正验收不再新增
   任务专属 schema、方言、中央 recovery 或额外重试层。
3. **发布 Hy-VLA v9 的紧凑证据，并统一 LIBERO 方法外层。** v9 的 UIUI 503
   已通过零 rollout cached finalization 恢复；只需保留原始失败与完成边界，不再为
   Hy-VLA 新建方法链。LIBERO 仍应复用 QueryContract、Plan Agent session、
   RoundExecutor、Aggregate、stop 与 Answer。
4. **继续收束生产结构。** `PlanAgentApplication` 拥有 route/round/stop/answer；
   `MethodRuntime` 是唯一 TaskGen materialization owner。迁移 caller 后再删除 legacy
   planner、任务方言和重复 registry，不再增加 façade。
5. **方法稳定后补实验。** 先做小型三 seed dense/adaptive 保真；独立人工 Plan/VQA、
   多 policy ranking 与大规模任务后置。

## 软件工程边界

- 生产只保留 Plan Agent 主链；fixed/catalog/task-specific planner 属于
  `experiments/paper/` 或 compat。
- Task binding 只保存 task/checkpoint/schema/official-success/runtime hooks，不承载
  aspect、metric 或 Planner 菜单。
- TaskGen、ToolGen 各只保留一次局部 repair；失败不触发中央 whole-round restart。
- 动态运行真值只维护在本文与 `docs/evidence/current/`；安装/网络故障放 cold
  runbook，历史结果放 `docs/evidence/history.jsonl`。
