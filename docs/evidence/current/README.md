# 当前紧凑证据：开放 Query 的 ClickBell 两轮闭环

本包对应 `eval_20260727_batch28_neutral_click_live_v4`。输入只有 broad Query：

> Where does this ACT policy first expose a weakness when generalizing over manipulated-object properties?

CLI 没有提供 aspect、template 或候选顺序。一个生产命令完成了：

```text
Query
→ inventory-free FreeConcern
→ QueryContract 要求 neutral official control
→ round 1 ACT official success
→ evidence-conditioned ClaimFirst 产生 catalog 外 runtime candidate
→ provider 编写 bell translation scene + check_success()
→ AST / fixtures / state / render / VLM / expert gate
→ round 2 ACT generated-checker success
→ Rule / VQA / Aggregate
→ budget_exhausted，evidence_sufficient=false
→ 对原 Query 返回 inconclusive
```

两轮均为真实 ACT，seed 都是 `100405`。第二轮 checker 的
`official_core_predicate_satisfied=true`，但它仍是实验语义扩展，不是
official-equivalent benchmark 结果。系统没有把两次成功冒充“已找到首个弱点”或泛化保证。

## Tool 语义修复

原运行中的 provider Tool 把 active-gripper 距离错误收缩为固定
`left_tcp_position`；本场景实际 active arm 是 right。该 `0.513703 m` 数值现已明确排除，
并保留为[被拒绝的原请求](artifacts/tool/rejected_left_tcp_request.json)，不能进入当前结论。

当前代码会 fail-closed。复核只复用了上面两条缓存真实 telemetry，新增 `0` 次 ACT：

- 生成并验证 `bell_active_tcp_min_xy_error`；
- control 为 `0.0092059225 m`，translated 为 `0.0057088756 m`；
- 两者均正确选择 right arm；
- 第二个改写 Query 命中 `run_local_reuse`，没有再次调用 Tool codegen provider。

这修复了测量语义，不增加 policy 性能样本；原 Query 结论仍是 `inconclusive`。

## 阅读顺序

| 阶段 | 产物 |
| --- | --- |
| Query 与开放 concern | [request](artifacts/query/request.json)、[FreeConcern prompt](artifacts/planner/free_concern_prompt.md)、[response](artifacts/planner/free_concern_response.txt)、[concern](artifacts/planner/free_concern.json) |
| evidence-conditioned proposal | [round-2 binding](artifacts/planner/round2_bound_proposal.json)，其中包含 runtime candidate 与更新后的开放 QueryContract |
| provider TaskGen | [code prompt](artifacts/task/code_prompt.md)、[response](artifacts/task/provider_response.json)、[task.py](artifacts/task/task.py)、[6/6 fixtures](artifacts/task/checker_fixtures.json) |
| 视觉验收 | [official/generated 对比图](artifacts/task/official_vs_generated.png)、[VLM prompt](artifacts/task/vision_prompt.md)、[response](artifacts/task/vision_response.txt)、[result](artifacts/task/vision.json) |
| 两轮 ACT | [official video](artifacts/act/official_seed_100405.mp4)、[translated video](artifacts/act/translated_seed_100405.mp4) |
| Tool repair | [request prompt](artifacts/tool/repair_request_prompt.md)、[rejected response](artifacts/tool/repair_request_response_rejected.txt)、[accepted response](artifacts/tool/repair_request_response_accepted.txt)、[generated code](artifacts/tool/generated_tool.py)、[validated execution](artifacts/tool/first_execution.json) |
| Tool exact reuse | [second Query](artifacts/tool/reuse_query.json)、[run-local reuse execution](artifacts/tool/reuse_execution.json) |
| Task exact reuse | [independent rephrased Query receipt](artifacts/reuse/exact_task_reuse.json)，provider call 为 0、ACT 为 0 |
| 结论 | [corrected projection](corrected_aggregate.json)、[structured answer](artifacts/answer.json)、[Chinese feedback](artifacts/feedback.json)、[run summary](run_summary.json) |

[manifest.json](manifest.json) 是本包的文件、来源、大小与 SHA-256 索引。完整 telemetry、
raw logs 和 checkpoint 只保留在 manifest 指向的 canonical AutoDL 目录；Git 只跟踪这份
紧凑数据流。上一份 current bundle 可由 Git 历史恢复。
