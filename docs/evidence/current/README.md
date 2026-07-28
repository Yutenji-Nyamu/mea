# 当前紧凑证据：AdjustBottle 开放主链与冲突保真

本包对应在线运行 `eval_20260728_adjust_bottle_open_live_v2` 及其追加式
0-ACT 审计 `terminal_tool_plus_task_vqa_repair_v7`。

输入只有开放 Query，没有提供 aspect、template 或候选顺序：

> 这个 ACT 策略执行调整瓶子任务时，对未见对象属性的泛化能力如何，最先在哪里暴露弱点？

## 在线数据流

```text
开放 Query
→ inventory-free FreeConcern
→ 运行时绑定 adjust_bottle / ACT checkpoint
→ direct ClaimFirst initial plan
→ round 1 official control：ACT success
→ evidence-conditioned Planner 提出 catalog 外 runtime candidate
→ Generic TaskGen 编写 scene/check_success()
→ AST、2/2 fixtures、render、VLM visual diagnosis、expert gate
→ round 2 ACT
→ Rule / VQA / Aggregate / Planner / Answer
```

在线链共启动 2 次 ACT，seed 均为 `100201`。它证明第三个 RoboTwin
任务可以走通通用生成式主链，且未新增 `adjust_bottle` 专属 Planner 或 TaskGen
方言。

源在线结果仍是 `budget_exhausted` / `inconclusive`。原因不是隐去的策略成功：

- provider 生成的实验 checker 判定 round 2 为失败；它尚未经过
  official-equivalent 认证，不能当作 official benchmark 成功率；
- 当时 Tool 错选为 contact→success 时间，因没有 success event 得到 `null`；
- 当时动态 VQA 错误继承 BBH 的 block/hammer 问题。
- 原始 answer 把未认证 checker 写成 “expected semantic extension”；这是框架的
  保守标签，不是代码等价性分析结论，canonical v7 已改为“尚未认证”。

## 追加式方法修复

修复后的代码不重跑 ACT，也不覆盖源 bundle，而是只复用已完成 telemetry：

1. 通用 `terminal_signal_component` 首次编译并测得
   `bottle_functional_position.z = 0.771909236907959 m`。
2. 第二个完全相同 Tool Query 命中 `run_local_reuse`，provider 未再次调用。
3. task-owned VQA 只问 `bottle_visibly_repositioned`，观察为 `true`
   （confidence `0.98`）。
4. 该视觉观察与 generated/official-core predicate 的 `false` 发生冲突。
5. composed v7 重算 Aggregate、EvidencePacket 和 Planner：
   `EvidencePacket=conflicting`，`stop_reason=evidence_conflict`，
   `claim_verdict=inconclusive`，`evidence_sufficient=false`。

因此，v7 证明的是“新 Tool 的生成、验证、注册、复用以及冲突证据能够进入
Planner 并触发 fail-closed 停止”，不是对原 Query 的正向泛化结论。

## 阅读顺序

| 阶段 | 主要产物 |
| --- | --- |
| Query 与任务绑定 | [request](artifacts/query/request.json)、[global route](artifacts/planner/global_query_route.json)、[FreeConcern prompt](artifacts/planner/free_concern_prompt.md)、[response](artifacts/planner/free_concern_response.txt)、[runtime binding](artifacts/planner/runtime_task_binding.json) |
| evidence-conditioned planning | [round-1 prompt](artifacts/planner/after_round_1_prompt.md)、[response](artifacts/planner/after_round_1_response.txt)、[runtime candidate](artifacts/planner/experiment_candidate.json)、[decision](artifacts/planner/decision_after_round_1.json) |
| TaskGen | [code prompt](artifacts/task/code_prompt.md)、[provider response](artifacts/task/provider_response.txt)、[task.py](artifacts/task/task.py)、[fixtures](artifacts/task/checker_fixtures.json)、[scene comparison](artifacts/task/official_vs_generated.png) |
| visual diagnosis / expert | [vision prompt](artifacts/task/vision_prompt.md)、[response](artifacts/task/vision_response.txt)、[result](artifacts/task/vision.json)、[expert preflight](artifacts/task/expert_preflight.json) |
| 两轮 rollout | [official video](artifacts/rollout/round_1_official_seed_100201.mp4)、[generated-task video](artifacts/rollout/round_2_generated_seed_100201.mp4)、[montage](artifacts/rollout/round_2_montage.png) |
| 在线缺陷原貌 | [null Tool execution](artifacts/original/round_2_null_tool_execution.json)、[错误 VQA Query](artifacts/original/round_2_vqa_wrong_query.json)、[原始 answer](artifacts/planner/answer_original.json) |
| 修复 Tool 与 exact reuse | [MetricSpec](artifacts/repair/terminal_metric_spec.json)、[generated Tool](artifacts/repair/terminal_tool.py)、[execution](artifacts/repair/terminal_tool_execution.json)、[exact reuse](artifacts/repair/terminal_tool_exact_reuse.json) |
| task-owned VQA | [query](artifacts/repair/vqa_query.json)、[prompt](artifacts/repair/vqa_prompt.md)、[response](artifacts/repair/vqa_response.txt)、[result](artifacts/repair/vqa_result.json) |
| composed 结论 | [Aggregate](artifacts/repair/aggregate.json)、[Evidence record](artifacts/repair/evidence_record.json)、[Planner replay](artifacts/repair/planner_replay.json)、[result](artifacts/repair/result.json) |

完整 telemetry、raw logs、checkpoint 和中间 repair v1–v5 只保留在服务器 canonical
目录；Git 只跟踪本包中的无重复紧凑数据流。上一份 ClickBell current bundle 可由
Git 历史恢复。
