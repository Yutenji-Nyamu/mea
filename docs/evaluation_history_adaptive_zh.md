# Historical Evaluation Retrieval 与自适应轮次

## 本轮实现的最小闭环

本轮补齐论文中两个相互独立的环节：

1. **跨 evaluation 的 planning history**：新 query 到来时检索相似的已完成评估，
   把过去的 query、sub-aspect/template 顺序和 policy 标签提供给 Plan Agent；
2. **observation-driven round control**：每轮 Aggregate 与 Execution VQA 返回后，
   先由可信代码判断证据是否充分、冲突或数据质量不完整，再约束 Plan Agent
   `continue`、`verify` 或 `stop`。

这对应论文 Appendix A.3.3 的 historical planning database，以及 §3.1–3.2、
Figure 2/5 中“根据 intermediate observations 继续提出 sub-aspect，直到证据充分再总结”
的过程。论文没有规定具体阈值；本项目的阈值是可审计的工程化原型。

## 当前调用链

```text
自然语言 query
→ EvaluationHistoryDB.retrieve_similar(top-k=3)
→ plan/history_retrieval.json
→ Plan Agent 选择本次明确要求的 trusted template
→ TaskGen / validation / expert gate / ACT
→ Aggregate Toolkit + Execution VQA
→ deterministic EvidenceAssessment
   ├─ sufficient + 还有明确请求的 aspect → continue
   ├─ sufficient + 请求已覆盖 → stop
   ├─ evidence_conflict / aggregate_uncertain → verify 同一 template
   ├─ pipeline_failure → stop
   └─ verification 后仍未解决或预算耗尽 → stop + unresolved
→ Feedback Agent / evaluation_report.md
→ completed evaluation 写 history_record.json 并 upsert SQLite
```

Plan Agent 仍负责解释 observations，以及在多个 remaining template 中决定顺序；
它不能覆盖 `EvidenceAssessment.required_action`。

## Historical Evaluation Database

实现位于 `mea/history/database.py`，只使用 Python 标准库 `sqlite3`。

- 默认缓存：`mea/evaluation_runs/history.sqlite3`，已加入 `.gitignore`；
- canonical rebuild source：每个完成 run 的 `summary/history_record.json`；
- 新 run 只在 `lifecycle_status=completed` 后入库；
- 旧 schema 只在 `status=completed`、存在 `execution_finished_at`、plan 与
  evidence artifacts 完整时兼容导入；
- 同 `evaluation_id` 使用 content hash 幂等 upsert；
- task 做 exact filter；policy 不做硬过滤，以支持论文要求的 cross-policy
  planning consistency，但每条结果保留 `same_policy` / `same_checkpoint` 标签；
- query similarity 为确定性的 Unicode normalization、SequenceMatcher 与
  character bigram Jaccard 组合；不依赖向量数据库；
- 历史只含 compact planning/outcome/provenance，不含 trajectory 或数值 ToolResult。

重建命令：

```bash
python scripts/rebuild_evaluation_history.py --reset
```

正常调用默认启用 history：

```bash
python scripts/manipeval_agent.py \
  --request '评估 ACT 的蓝色方块表现' \
  --model-profile balanced \
  --plan-only
```

可使用 `--history-database PATH`、`--history-limit N`，或通过
`--no-history` 完全关闭。

## EvidenceAssessment

实现位于 `mea/planner/evidence_policy.py`。第一版只判断“数据质量是否足够”，
不把少量 episode 伪装成统计置信区间。

Aggregate quality 检查包括：

- 当前 template 对应 metric 是否存在 policy cohort；
- `valid / missing / invalid` 数量；
- 是否覆盖预期 policy episode 数；
- 是否有 Aggregate input issues；
- `pickup_not_observed`、`contact_not_observed_after_pickup` 等有明确语义的
  `null` 是否只是“事件没有发生”，而不是 telemetry 丢失。

因此，3 个合法 `false` 是明确负证据，不会被误判为 uncertainty；
`contact_before_pickup` 等契约无效结果仍会触发 verification。

Verification 采用：

- 只复核最新发生冲突或数据不完整的同一 template；
- route 固定为 `reuse`；
- 只跑 1 episode；
- seed 为本 evaluation 已用最大 seed 加 1；
- 每个 template 最多 verification 一次；
- `max_rounds=3` 仍是硬上限。

每轮保存：

```text
plan/evidence_after_round_<N>.json
plan/decision_after_round_<N>.json
```

最终 report 会展示 history retrieval、EvidenceAssessment 与 decision，历史结果不会
进入本次 Aggregate。

## 本轮验证

- 服务器完整单元测试：112/112 通过；其中包括 `verify` round 持久化执行、
  canonical-only rebuild 与跨 artifact 一致性回归；
- history rebuild：从旧 artifacts 中安全导入 6 条 completed evaluation，
  其余 incomplete/corrupt/缺 artifact 的目录只记录 issue；
- live UIUI `plan-only`：改写后的蓝色方块 query 检索到 3 条相似 planning，
  Plan Agent 仍只选择本次明确要求的 `object_appearance.color_blue`；
- 真实 artifact smoke：复用已有 3-episode Aggregate 与 corrected Execution VQA，
  得到 `sufficient → stop`；将 conflict 位设为 true 后得到
  `evidence_conflict → verify`，新 round seed 为 `100001`；
- artifact smoke 不调用 ACT 或 GPT。

## 与论文仍存在的主要 gap

按“方法重要性 / 当前实现难度”综合排序：

1. **动态 Execution VQA query（高 / 中）**：目前视觉问题仍是固定的颜色、抬起、
   位移三项；下一步可由 sub-aspect/ToolSpec 生成受限问题 schema。
2. **更开放的 sub-aspect taxonomy（高 / 中）**：当前 Plan Agent 只有 3 个
   BeatBlockHammer trusted templates；可先扩 generalization、performance、safety、
   robustness 的 bounded taxonomy。
3. **第二个 RoboTwin task + generic TaskSchema（高 / 中）**：证明 Recorder、Tool、
   history 与 planning 不是只适用于 BeatBlockHammer。
4. **`balanced_v1` Recorder runtime（中高 / 中）**：按 250 Hz semantic/contact、
   50 Hz dynamics、policy-boundary action 与事件关键帧实现通用 telemetry。
5. **跨 policy 对照（高 / 中高）**：同 query 复用相同 decomposition、task、tool，
   比较 ACT 与另一个 policy，验证论文强调的 consistency。
6. **统一 failure recovery（中 / 中）**：把 round restart、Tool exception、scene
   regeneration 和失败历史统一为可审计状态机。
7. **论文级实验规模（高 / 实验重）**：默认每 constructed task 多次 trials、
   human-plan agreement、VQA perturbation accuracy 与 benchmark ranking consistency。

下一批最推荐做第 1 项，再配合一个新的视觉 sub-aspect；它直接补论文 Observation
分支，且可以继续复用当前视频、关键帧、Provider 与 conflict guard。
