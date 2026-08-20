# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–5、Tables 1–10 与 Appendix。本文只维护当前真值；
历史运行摘要见 [`docs/evidence/history.jsonl`](evidence/history.jsonl)，当前紧凑证据入口见
[`docs/evidence/current/`](evidence/current/README.md)。

## 当前方法真值

当前证据有两条互补的 RoboTwin 主线。较宽的多轮探索旗舰仍是
`eval_20260805_batch37_clean_flagship_press_stapler_s1000_v7`：

1. 输入是未指定 aspect、template、对象、轴、阈值、checker 或 metric 的 broad Query；
   Plan Agent 根据逐轮 evidence 选择并细化 position、orientation 等 Proposal。
2. 共执行 **10 个实际 SmolVLA rollout**：1 个 official control 与 9 个实验候选，均使用
   seed `1000`。另有 1 个 contact candidate 在 expert positive fixture 被拒绝，记录为
   typed `N=0` planning evidence；它没有启动 policy rollout，也不是 policy failure。
3. 可执行候选经过通用 TaskGen 的 scene/checker materialization、真实 rollout、Python Rule
   Tool、Aggregate 与下一轮 Plan Agent；Tool 得到非空 live 值并在后续轮复用。
4. 第 11 个已执行 round 后，Plan Agent 声明没有进一步有根据的 Proposal，并以当时的历史
   writer 名称 `agent_saturation_inconclusive` 主动停止。当时的后置 stop validator 只验证该判断没有被
   升级为充分证据或确定结论，并未独立证明开放能力空间已经饱和；新 writer 使用更诚实的
   `agent_inconclusive_stop`，同时强制保留 `evidence_sufficient=false`、
   `claim_verdict=inconclusive` 与 `answered_query=false`。
5. 最终 Answer 明示：10 个 episode 来自同一 seed；原 contact candidate 未执行；generated
   checker 不是 official benchmark checker；候选域仍开放。因此本次证明的是开放反馈链能
   主动收敛到诚实的 inconclusive Answer，不是属性泛化、benchmark 排名或统计充分性正结论。

这个旗舰关闭了“只能靠 hard cap 停止”和“typed `N=0` 被误算成 policy failure”的方法 gap；
它没有证明 broad open-world Query 已取得 `evidence_sufficient=true`。

当前精简代码上最贴合论文主循环的单次闭环是 Batch46 v5：它把 broad Query、真实对照、
evidence-conditioned refinement、后续真实 rollout、充分 Answer 与 Agent 主动停止合在同一运行。
该运行只有一个 task、一个 seed 和三个 episode；它补齐方法闭环，不替换 Batch37 的较宽探索证据。

## Batch46 当前精简主干的 anchor-free 充分闭环

`eval_20260820_batch46_move_playingcard_anchorfree_adaptive_s1000_v5` 基于
`e9e3880b3fbf4cecd5877a4732d2aca0701d6295`。Query 没有给 actor、axis、direction、magnitude、
checker、diagnostic metric、历史候选或停止脚本；任务卡只含 official source facts，history
retrieval disabled，execution VQA 未请求。

R1 unchanged official control 成功后，Plan 首次提出 `playingcards y +0.05 m`。TaskGen fresh
生成并精确 materialize 该 same-seed delta，preservation、2/2 fixtures、vision 与 expert 均通过；
真实 rollout 的 unchanged official checker 失败，live Rule 得到 terminal
`abs(playingcards.x)=0.062070973217487335 m`。R2 evidence 随后直接改变下一 Proposal：Plan 引用
该失败与数值，将同一因素减半为 `y +0.025 m`；第二个 fresh Task 再次精确 materialize 并通过
全部真实边界，真实 rollout 仍 official failure，Rule 值为 `0.05292544141411781 m`。

R3 后 Plan 在五轮上限尚未用尽时原始输出 `stop`、`supported`、
`evidence_sufficient=true`。最终结构化范围为 `N=3`、seeds `[1000]`、两个已测试 candidate、
无 evidence conflict、`termination=agent_stop`；Plan query answer、最终 answer 与 manifest 文本一致。
这关闭的是单 task、单 checkpoint、单 seed 下的 existential 方法正例，不证明跨 seed/任务泛化、
policy 总体成功率、VQA accuracy 或样本效率。完整记录见
[`experiments/paper/results/batch46_move_playingcard_anchorfree_adaptive/`](../experiments/paper/results/batch46_move_playingcard_anchorfree_adaptive/README.md)。

## Batch38 开发回归（未晋升旗舰）

Batch38 用 `grab_roller` 的历史失败做 prompt/context 纵向修复；它不替换上述 Batch37
旗舰，也不更新 `docs/evidence/current/`。v5 完成 4 个 round、2 个 SmolVLA episode：
official control 与 lateral scene 均成功，新 Rule Tool 得到 `0.0606393 m` live 值；随后
evidence 使 Plan 从 lateral 转向 orientation，再转向 longitudinal，后二者被 expert gate
判定为不可执行且未消耗 rollout。运行最终仍因四轮上限停止并返回 inconclusive，因此新增的
只是“失败上下文改变下一 Proposal”的开发证据，不是主动充分停止正例。

本批还让 Plan、TaskGen 与 ToolGen 共用短任务卡和“上一输出 + 具体错误”的一次局部 repair，
并让 VQA 的证据不足/冲突进入 `abstained`。v5 的最终 Answer prompt 仍有 56,150 tokens；其后
新增的紧凑 evidence projection 未被 v5 live Answer 使用；服务器已在冻结 v5 evidence 上完成
离线回归并保留关键字段，因此只能评价上下文压缩本身，不能回写 v5 的 token 记录。完整冷流水见
[`experiments/paper/results/batch38_prompt_context/`](../experiments/paper/results/batch38_prompt_context/README.md)。

## Batch39 方法回归（未晋升旗舰）

Batch39 继续使用 `grab_roller` 的真实失败纵向迭代，而没有新增任务专属分支。v4 在一个
broad Query 中完成 4 round、3 个 SmolVLA episode：official control 与同一正向 x 场景语义的
两次 generated-scene episode 均成功（第二次改测右侧），两个新 Rule Tool 分别得到
`0.04107694 m` 与 `0.04231440 m`；中间的 model-2
candidate 因 generated 与 unchanged same-seed official expert 都得到 `target_pose=None`，作为
expert-oracle limitation 和 typed `N=0` evidence 返回 Plan。evidence 改变后续 Proposal，Agent
最终主动停止并诚实返回 inconclusive，而不是再次撞上 hard cap。

这比 Batch38 前进了一步：prompt/task guide 和数值 preservation 边界的修正已经改变真实主链
结果。但它仍不替换 Batch37 旗舰：`evidence_sufficient=false`、只有单 seed `N=3`，而且左右
距离来自两个不同 Tool，未证明同一 Tool exact reuse。完整冷记录见
[`experiments/paper/results/batch39_grab_roller_prompt_mainline/`](../experiments/paper/results/batch39_grab_roller_prompt_mainline/README.md)。

## Batch40 充分停止正例（冷补充，未替换 current evidence）

Batch40 先修正三处会截断论文语义的运输层问题：原始 pipeline completion 与
`valid_for_planning` 分离；Plan Agent 拥有 continue/stop 决策而后置 stop validator 只核验；
preservation 与 Proposal-to-artifact alignment 改用结构化事实而不再依赖词项重叠。

最终 v4 从未指定 aspect、对象、轴、scene edit、checker、metric 或停止脚本的 Query 出发，
完成两个 SmolVLA episode：unchanged official control 与一个 generated scene 均成功。通用
TaskGen 一次生成并通过 2/2 fixtures、VLM 和 expert；provider Python Rule Tool
`query_terminal_roller_z_position` 经独立 oracle 验证后，从 generated episode 返回
`0.8000384569168091 m`。schema-v2 Aggregate 同时满足 `pipeline.passed=true`、
`valid_for_planning=true`、intent/preservation complete 与 `sufficient=true`。Plan 原始响应
给出 `stop/evidence_sufficient`、`claim_verdict=supported` 和 `answered_query=true`，
后置 stop validator 随后验证通过。不过该次 live prompt 暴露了预计算的 sufficiency verdict，不能
单独证明 Agent 独立判断停止。

Plan 的候选选择使用了检索任务卡中已有的同 seed 正向锚点，然后 fresh 生成并重新通过全部
验证；因此这是 retrieval-guided 选择与 live revalidation，不是只凭本次 control 冷发现全新
concern。冻结 artifact 的 0-rollout 真值重算还将 3 条 metric result 正确归并为 2 个物理
episode，并补齐 compact decision 的充分停止字段；它没有改写冻结 live 产物。

当前代码随后同时收紧 Plan 调用合同与 prompt，只传预算和具体运行限制。对冻结的两轮 evidence
进行一次 provider-only、0 simulator、0 policy rollout 回放时，prompt 不再含预计算的
`should_stop/evidence_sufficient/stop_reason/claim_verdict` 字段；Agent 一次调用独立输出 stop，
后置 validator 验证为 `validated_stop/supported/answered=true`。因此停止所有权
由“live 执行 + 当前 prompt 的冻结回放”共同证明，而不是倒写原 live prompt 已经干净。

v4 的实际 `x=0.15001997 m` 由 simulator state 直接确认；通用 matcher 仍只证明 scene owner
和可观测变化，尚不能证明任意自然语言 requested delta 的 actor/axis/value 与 artifact 精确蕴含。

这关闭了小范围“Agent 主动充分停止尚无正例”的 gap，但不替换 Batch37 较宽的多轮证据包：
Batch40 只有 `grab_roller`、单 seed、`N=2`，没有证明同一运行中的多步 concern refinement、
跨 evaluation Tool reuse、跨任务稳定性或任何统计实验结论。完整冷记录见
[`experiments/paper/results/batch40_paper_mainline_cleanup/`](../experiments/paper/results/batch40_paper_mainline_cleanup/README.md)。

## Batch41 生产主链裁剪（结构回归，不是方法正证据）

Batch41 把普通运行中没有论文方法对应、也没有真实故障理由的运输层移出主链：formal
QueryContract、receipt/evidence seal、reviewed Task/Tool/VQA 审批 registry、registered strategy、
cached finalization/resume 与 append-only TaskGen attempt ledger 已删除或迁冷。TaskGen 现在只有
generation、最多一次针对性 repair 与单一 result；Rule/VQA 复用缩成 semantic key 到 executable
artifact，并在当前 telemetry/frames 上重验。Query interpretation 也不再用 token、alias、ontology
或 catalog ranking 裁决 concern；typed needs 直接交给 Plan/Task/Tool。

最终代码在 AutoDL server 通过 225 项默认 mainline（另 10 个 subtests）与 623 项 cold/compat
回归（另 162 个 subtests）；3 条 warning 均来自 robosuite 上游弃用接口。本批没有 provider、
simulator 或 policy rollout，因此只证明裁剪后的软件结构仍可运行，不增加论文 claim 证据。
仍待迁移的明确债务是旧 `artifact_retrieval_index` 菜单提示、`ExperimentCandidate` 内部命名/
identity、HistoryDB 的技术性摘要，以及 `RoundExecutor`/TaskGen 大文件的 owner 拆分。完整冷记录见
[`experiments/paper/results/batch41_mainline_cleanup/`](../experiments/paper/results/batch41_mainline_cleanup/README.md)。

## Batch42 cold transfer 与 Tool 复用（focused method regression）

`move_playingcard_away` 在没有任务卡的情况下完成 fresh reset / TaskContext preflight，耗时
`33.167731 s`，没有 provider 或 policy rollout；这只证明通用 source/runtime binding 可进入，
Batch42 当时尚未执行真正的 concern 冷发现与 live evidence loop；该缺口由下述 Batch43
继续验证。

Rule Tool 的轻量 semantic library 在真实冻结 telemetry 上完成一次跨 evaluation 精确复用：
Batch40 的 `query_terminal_roller_z_position` artifact 在 Batch39 的另一真实 episode 上以
`provider_called=false` 命中，并重新完成当前 telemetry、双执行、typed oracle 与 finite 验证；
值从 `0.8000384569168091 m` 变为 `0.8001335859298706 m`。这关闭 Rule Tool executable
artifact 的小范围跨 evaluation reuse gap，不是新 rollout 或 policy 稳定性证据。

VQA observer 的二值示例已改成 `observed=null`，并明确稀疏 temporal frames 中“没看到”
不能判 false。Batch42 当时未执行冻结 montage 五次 provider repeat；当前结果见 Batch43，
Batch37 的原模型稳定性负证据不能被跨模型结果倒写。完整边界见
[`experiments/paper/results/batch42_cold_transfer_and_tool_reuse/`](../experiments/paper/results/batch42_cold_transfer_and_tool_reuse/README.md)。

## Batch43 无任务卡 cold live 与 scene-fact 修复

`move_playingcard_away` v4 在没有任务卡、aspect、actor、axis、scene edit、checker、metric 或
停止脚本时完成三个 SmolVLA episode：unchanged official control 成功；Plan cold 提出未定量的
playing-card lateral relocation，TaskGen 将其 materialize 为 `y +0.03 m`，一次生成并通过
2/2 fixtures、vision 与 expert，
该 episode official failure，Rule Tool 得到相对 generated reset 的终态位移
`0.007887560874223709 m`；随后又完成一个 contact-margin follow-up，official success，第二个
Rule Tool 得到最近 TCP-card 距离 `0.06134439632296562 m`。Plan 在还剩一个 candidate allowance
时主动停止并给出 inconclusive Answer。

冻结 artifact 审查同时否定了 raw Plan/Answer 的“same-scene refinement”措辞：第三轮 card
position 回到 official location，orientation 也发生改变，不是第二轮场景的重复。因而这次运行
证明第二个 task 的 cold Proposal、TaskGen、两次 generated rollout、Rule evidence、refinement 与
主动停止已经连通，但不证明可复现的位置弱点、精确 scene reuse 或充分正 Answer。

最小修复把 same-seed simulator setup 的 actor position change 以 axis、signed delta、unit、seed、
authority 和 `1e-6 m` 比较容差送入 Plan；Planner 对 exact/prior/refinement 必须重述明确数值，
否则改做独立 official-base concern 或停止。在 v4 R2 冻结 evidence 上的 Terra provider-only replay
经过一次 schema repair 后，选择独立 official-base `x +0.030 m`，明确不保留 prior y delta；新增
simulator 与 policy rollout 均为 0。该回放当时只证明 Plan handoff；Batch43 收束时尚未执行
TaskGen preflight 或新 policy episode。TaskGen preflight 的后续结果见 Batch44，policy 仍未重跑。

同一 current prompt、冻结 montage 与 `gpt-5.6-sol` 的五次 temperature-0 VQA 调用均返回
`observed=null`，没有多数投票。这是稳定弃答正例，不是 accuracy 证据，也不是原 Luna 矛盾输出的
同模型复现。完整冷记录见
[`experiments/paper/results/batch43_move_playingcard_cold_mainline/`](../experiments/paper/results/batch43_move_playingcard_cold_mainline/README.md)。

## Batch44 typed scene materialization 与冻结 Tool 复用

前置诊断先用手工 typed candidate 要求 same-seed official reset 的
`playingcards y +0.03 m`。TaskGen 一次生成、0 repair，经过 2 次 simulator probe、1 次 vision
和 1 次 expert；checker fixtures `2/2`、vision、expert 与 preservation 全部通过，simulator
observed delta 与 expected delta 均为 `+0.03 m`，容差 `1e-5 m`。这只证明明确 scene request
可以 materialize，尚未证明 Plan 能生成该结构。

最终正验收从 Batch43 frozen R2 evidence 开始。Terra 单次、0-retry Plan replay 同时读取 control
success、simulator-authoritative `y +0.03000000212341547 m` failure scene 与终态 displacement
`0.007887560874223709 m`，自主选择 typed official-base `y +0.015 m` midpoint；bound scene need
完整保留 actor、axis、signed delta、unit 与 reference。v1 示例曾重复写入外层已固定的
`task_identity/policy_checkpoint`，导致 TaskGen 虽精确生成 `+0.015 m` 且通过 checker/vision/expert，
preservation 仍为 partial；prompt owner 删除这两个重复 scene-preservation claim 后，v2 实际
Plan→TaskGen preflight 一次生成、0 repair，expected=`0.015 m`、observed=
`0.015000000000000001 m`，preservation verified，checker `2/2`、vision 与 expert 全部通过。
整个闭环新增 policy rollout 为 0。

另一项前置诊断将 Batch43 R3 的 nearest-TCP Rule Tool 通过 semantic library 复用到 R2 真实失败的
`y +0.03 m` 冻结 episode；0 provider、0 simulator、0 policy，当前 telemetry 重验与 independent
numeric oracle 均通过，得到 `0.17870217561721802 m`。R3 异场景成功 episode 的
`0.06134439632296562 m` 只作对照，不能据此推断同 scene contact 因果。完整冷记录见
[`experiments/paper/results/batch44_scene_delta_materialization/`](../experiments/paper/results/batch44_scene_delta_materialization/README.md)。

## Batch37 补充证据

- **Rule Tool 跨 evaluation 复用。** 一个新 evaluation 以 **0 rollout、0 provider call** 从
  当时的 reviewed persistent registry 精确复用 `terminal_minimum_tcp_to_stapler_distance`；代码 hash、
  Tool contract 与 telemetry schema 均匹配，并在新的缓存 episode 上重新执行得到 finite 值。
  这关闭了“只在 run-local registry 内复用”的代码/证据 gap，但仍是单 Tool、单任务案例。
- **开放 VQA ToolGen。** Query 自然诱发了“首次成功按压前，夹爪是否越过订书机再反向重对齐”
  的 temporal question；问题由 provider 生成、注册，并在第二个 evaluation 以 exact semantic
  key 复用，文本生成调用为零。可是同一冻结 episode、同一问题与同一组关键帧上，两次 VQA
  分别得到 `false (0.82)` 与 `true (0.86)`。因此这里只证明 VQA Tool artifact 的生成与复用，
  同时暴露输出不稳定；不能据此声称 VQA robustness，当前跨 evaluation 冲突还需要显式聚合。
- **LIBERO basic adaptation。** `eval_20260805_batch37_libero_shared_method_v4` 完成两个 SmolVLA
  episode：official control 为 positive，provider-written custom BDDL/object-identity candidate
  为 negative；同时产出实验 checker、live Tool、Aggregate、复用结果与受限 Answer。该运行
  明示 `method_chain_valid=false`、`scientific_evidence_eligible=false`，因为 Plan Agent 未主动
  stop，custom checker 也不是 official-equivalent。这是论文所称 basic adaptation 的结构
  smoke，不是 RoboTwin 同等完整度或 LIBERO 性能结论。

## 方法 claim

| 论文 claim | 当前项目 | 判断 |
| --- | --- | --- |
| Fig. 2/5：开放 Query 驱动 Plan Agent 自主提出 sub-aspect | Batch46 的 anchor-free Query 先得到 control evidence，再由 Plan 提出 `y +0.05 m`，并按失败 evidence 自主细化为 `y +0.025 m` | **同一 live 运行小范围完成**；仍是单 task、单 seed，执行能力域有限 |
| evidence 决定下一轮，并在充分时停止 | Batch46 R2 的 official failure、simulator delta 与 Rule 数值直接进入下一 Plan；R3 完成后 Agent 在 allowance 尚余时主动输出 supported stop | **多步 refinement 与充分正结论已在同一 live 运行小范围合一**；未证明跨 seed 稳定性 |
| Fig. 3：Proposal → retrieve/generate scene + `check_success()` → rollout | Batch46 两个 typed scene 均由通用 TaskGen fresh 生成，observed delta 精确等于 `0.05/0.025 m`，preservation、fixtures、vision、expert 全过，随后各有一个真实 official outcome | **Plan→TaskGen→policy 数值闭环小范围完成**；量词/关系语义和更宽 task coverage 仍是边界 |
| 首帧视觉诊断与局部重新生成 | render/VLM 与一次有界局部 repair 已接入；数值 preservation 由 simulator state、AST 与 fixture 审计 | **组件完成**；视觉不能替代数值语义验证 |
| Fig. 4：ToolGen retrieve/generate/validate/register/reuse | 新 Python Rule Tool 有独立验证、live finite 值与 Planner 消费；Batch42 完成跨 evaluation exact reuse；Batch44 又把 R3 Tool semantic-library 复用到真实 R2 失败 episode 并重过 telemetry/oracle；Batch43 frozen VQA 为 `5×null` | **Rule Tool 小范围闭合、VQA 有稳定弃答正例**；VQA accuracy 与原模型稳定性仍未证明 |
| rollout → Rule/VQA → Aggregate → Plan Agent → Answer | Batch46 在同次生产运行中完成 control、两个 generated rollout、两次 finite Rule、evidence-conditioned Proposal 与 supported Answer；Batch37 保留较宽 inconclusive 探索证据 | **RoboTwin 小范围完成**；LIBERO 仍仅 basic adaptation |
| 回答原 Query 并约束确定性 | Batch46 原样保留 Plan Answer，并只附加 N、seed、tested candidates、conflict、termination 与 verdict 等结构化范围 | **完成度较高**；当前确定正例仍仅是单 seed existential claim |

## 实验 claim

| 论文 claim | 当前证据 | 判断 |
| --- | --- | --- |
| Tables 1–2：更少 samples/time 保持 dense 结论 | 三 seed toy 从 12 降到 6 ACT，但完整 failure set 仅保持 2/3 | **真实节省、结论不完全一致** |
| Table 3：RAG、visual self-check、README.Agent 提升 codegen | 五个 frozen Proposal 的小型消融，只有 RAG 有方向信号 | **未复现结论** |
| Table 6：Plan 与机器人研究者一致 | 30 条 development-agent proxy；无人类 gold | **未复现** |
| Tables 7–8：VQA 四条件 accuracy/AUROC | 小型 proxy；Batch37 同 episode exact reuse 出现相反 VQA 结论 | **未复现，且已有稳定性负证据** |
| Table 9：少样本保持多 policy 排名 | ACT/DP3 三 seed 为 2/3 对 2/3，Spearman 不可算 | **未复现** |
| Fig. 6：系统错误率与模块分布 | 固定 operation 分母很小 | **不可比较** |
| RoboTwin/LIBERO 适配 | RoboTwin 已有多个 generalist backend 与小范围方法闭环；LIBERO v4 为 official-positive/custom-negative 两回合 basic adaptation | **RoboTwin 小范围完成；LIBERO 仅结构 smoke** |

## 当前主干 gap

1. **从单 seed existential 正例走向科学验证。** Batch46 已把多步 refinement、后续真实 rollout、
   充分 Answer 与主动停止合在同一 broad Query；下一步应先做少量预注册 seed 复验，报告 witness
   保持率和失败边界，而不是继续增加 Planner 协议或 task-specific 方言。
2. **第三个无正锚点 task。** 复用当前 Query→Plan→Task/Tool→evidence 主链，选择 official control
   可用且 source facts 足够的第三个 task；先验证 transfer，再考虑扩大任务数。
3. **VQA 准确性与同模型稳定性。** current prompt+Sol 对一个冻结模糊输入已得到 `5×null`，但
   没有 independent gold，也不是原 Luna 输出的同模型复现。Aggregate 仍须显式保留跨 evaluation
   的相反观测，不能把 question artifact 复用或稳定弃答等同于 VQA accuracy。
4. **LIBERO 外层对齐。** 保留独立 simulator/policy backend，但复用同一 Plan Agent session、
   RoundExecutor、stop validation 与 Answer；先让两回合案例由 Agent 主动 stop，再谈覆盖扩展。
5. **方法稳定后补科学实验。** 先做小型三 seed dense/adaptive 保真；独立人工 Plan/VQA、
   多 policy ranking 与大规模统计后置。

## 软件工程边界

- 生产只保留 Plan Agent 主链；fixed/catalog/task-specific planner 可执行框架已删除，
  旧实验只保留冻结 paper inputs/results，并由 Git 历史恢复实现。
- Task binding 只保存 task/checkpoint/schema/official-success/runtime hooks，不承载
  aspect、metric 或 Planner 菜单。
- TaskGen、ToolGen 各只保留一次局部 repair；失败不触发中央 whole-round restart。
- exact/prior/refinement scene 必须由 same-seed simulator fact 提供 actor、axis、数值和单位；
  没有该事实时 Plan 只能改做独立 official-base concern 或停止。
- 动态运行真值只维护在本文与 `docs/evidence/current/`；安装/网络故障放 cold runbook，
  历史结果放 `docs/evidence/history.jsonl`。
