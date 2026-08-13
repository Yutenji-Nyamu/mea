# 论文 claim、当前证据与主要 gap

判断依据为论文 Abstract、Figs. 2–5、Tables 1–10 与 Appendix。本文只维护当前真值；
历史运行摘要见 [`docs/evidence/history.jsonl`](evidence/history.jsonl)，当前紧凑证据入口见
[`docs/evidence/current/`](evidence/current/README.md)。

## 当前方法真值

当前 RoboTwin 旗舰是
`eval_20260805_batch37_clean_flagship_press_stapler_s1000_v7`：

1. 输入是未指定 aspect、template、对象、轴、阈值、checker 或 metric 的 broad Query；
   Plan Agent 根据逐轮 evidence 选择并细化 position、orientation 等 Proposal。
2. 共执行 **10 个实际 SmolVLA rollout**：1 个 official control 与 9 个实验候选，均使用
   seed `1000`。另有 1 个 contact candidate 在 expert positive fixture 被拒绝，记录为
   typed `N=0` planning evidence；它没有启动 policy rollout，也不是 policy failure。
3. 可执行候选经过通用 TaskGen 的 scene/checker materialization、真实 rollout、Python Rule
   Tool、Aggregate 与下一轮 Plan Agent；Tool 得到非空 live 值并在后续轮复用。
4. 第 11 个已执行 round 后，Plan Agent 声明没有进一步有根据的 Proposal，并以当时的历史
   writer 名称 `agent_saturation_inconclusive` 主动停止。QueryContract 只验证该判断没有被
   升级为充分证据或确定结论，并未独立证明开放能力空间已经饱和；新 writer 使用更诚实的
   `agent_inconclusive_stop`，同时强制保留 `evidence_sufficient=false`、
   `claim_verdict=inconclusive` 与 `answered_query=false`。
5. 最终 Answer 明示：10 个 episode 来自同一 seed；原 contact candidate 未执行；generated
   checker 不是 official benchmark checker；候选域仍开放。因此本次证明的是开放反馈链能
   主动收敛到诚实的 inconclusive Answer，不是属性泛化、benchmark 排名或统计充分性正结论。

这个旗舰关闭了“只能靠 hard cap 停止”和“typed `N=0` 被误算成 policy failure”的方法 gap；
它没有证明 broad open-world Query 已取得 `evidence_sufficient=true`。

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

## Batch37 补充证据

- **Rule Tool 跨 evaluation 复用。** 一个新 evaluation 以 **0 rollout、0 provider call** 从
  reviewed persistent registry 精确复用 `terminal_minimum_tcp_to_stapler_distance`；代码 hash、
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
| Fig. 2/5：开放 Query 驱动 Plan Agent 自主提出 sub-aspect | Batch37 broad Query 未给实验菜单；上一轮 evidence 触发 position 数值细化及 position→orientation 转向 | **小范围 live 完成**；仍是单任务、单 seed，执行能力域有限 |
| evidence 决定下一轮，并在充分时停止 | Batch37 的 evidence 改变下一 Proposal；Agent 最终声明没有进一步有根据的 Proposal，QueryContract 仅验证该终止保持 inconclusive | **主动终止完成，充分性正例未合一**；当前旗舰不是 `evidence_sufficient=true` |
| Fig. 3：Proposal → retrieve/generate scene + `check_success()` → rollout | 通用 TaskGen 已在 reviewed-schema-free 的 `press_stapler` 中 materialize scene/checker，并实际裁决 SmolVLA episode；一个不忠实 checker 被 fixture fail-closed 拒绝 | **小范围完成**；跨任务稳定性和量词/关系语义忠实性仍是边界 |
| 首帧视觉诊断与局部重新生成 | render/VLM 与一次有界局部 repair 已接入；数值 preservation 由 simulator state、AST 与 fixture 审计 | **组件完成**；视觉不能替代数值语义验证 |
| Fig. 4：ToolGen retrieve/generate/validate/register/reuse | 新 Python Rule Tool 有独立验证、live finite 值、Planner 消费、run-local reuse，并新增 0-rollout reviewed cross-evaluation exact reuse | **Rule Tool 小范围闭合**；开放 VQA artifact 可复用但观测不稳定 |
| rollout → Rule/VQA → Aggregate → Plan Agent → Answer | RoboTwin 的 ACT、SmolVLA、Hy-VLA 已复用共享方法组件；Batch37 clean flagship 完成多轮真实反馈 | **RoboTwin 小范围完成**；LIBERO 仍仅 basic adaptation |
| 回答原 Query 并约束确定性 | `AnswerScope` 报告 N、seed、候选域、typed `N=0`、冲突、停止原因与语义边界 | **完成度较高**；当前旗舰正确回答为 inconclusive |

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

1. **在 broad open-world Query 中取得有证据约束的充分 Answer。** Batch37 已证明 Agent 可以
   主动声明无进一步有根据的 Proposal，并诚实返回 inconclusive；下一步不是删除限制，而是让新的可执行 Proposal
   扩展证据域，或选择具备有限可验证真值合同的 Query，形成 `evidence_sufficient=true` 正例。
2. **TaskGen 跨任务语义稳定性。** 不新增任务专属方言；用 runtime TaskContext、official source、
   simulator actors 与通用 fixture，在另外 2–3 个任务上验证 scene/checker 的量词、对象关系、
   同时性和 official-core preservation。
3. **VQA 可重复性与冲突处理。** 对同一冻结证据做少量重复判定或引入独立 gold；Aggregate 必须
   显式标记跨 evaluation 的相反观测，不能把 question artifact 复用等同于 VQA 结论复现。
4. **LIBERO 外层对齐。** 保留独立 simulator/policy backend，但复用同一 Plan Agent session、
   RoundExecutor、stop validation 与 Answer；先让两回合案例由 Agent 主动 stop，再谈覆盖扩展。
5. **方法稳定后补科学实验。** 先做小型三 seed dense/adaptive 保真；独立人工 Plan/VQA、
   多 policy ranking 与大规模统计后置。

## 软件工程边界

- 生产只保留 Plan Agent 主链；fixed/catalog/task-specific planner 属于
  `experiments/paper/` 或 compat。
- Task binding 只保存 task/checkpoint/schema/official-success/runtime hooks，不承载
  aspect、metric 或 Planner 菜单。
- TaskGen、ToolGen 各只保留一次局部 repair；失败不触发中央 whole-round restart。
- 动态运行真值只维护在本文与 `docs/evidence/current/`；安装/网络故障放 cold runbook，
  历史结果放 `docs/evidence/history.jsonl`。
