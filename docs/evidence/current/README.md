# 当前紧凑证据：无缓存的 ClickBell 两轮旗舰

本包对应 `eval_20260727_batch26_clean_online_click_live_v6`。唯一用户 Query 是：

> Can this ACT policy click the intended bell without touching a nearby visually similar distractor bell?

没有从 CLI 指定 aspect、template 或后续分支，也没有历史缓存 replay。一个
`scripts/manipeval_agent.py` 进程完成：

```text
Query
→ inventory-free FreeConcern
→ policy-compatible task resolution
→ QueryContract 绑定唯一 distractor 候选
→ round 1 official control（ACT success）
→ evidence-conditioned ClaimFirst proposal
→ provider 编写 scene + check_success()
→ AST + 6/6 fixtures + render/VQA + expert gate
→ round 2 ACT
→ generated checker Tool + Execution VQA + Aggregate
→ evidence_sufficient stop
→ bounded answer
```

## 结论边界

- 旗舰验收 `accepted=true`，两轮均为真实 ACT，seed 均为 `100405`，总 N=2。
- official control 使用 RoboTwin official success；第二轮使用模型编写并验证的
  `click_target_without_distractor_success`。
- 第二轮 generated checker 为 true，且其 official core projection 为 true；RoboTwin
  terminal official success 为 false。系统将二者标成
  `official_only` 与 `expected_semantic_extension`，没有把实验 checker 冒充 official
  benchmark success。
- 对唯一有限候选，答案为 `no_failure_observed`，并因
  `evidence_sufficient` 停止。这不是广泛泛化、统计效率或 benchmark 成功率结论。

## 阅读顺序

| 阶段 | 产物 |
| --- | --- |
| Query 与开放 concern | [request](artifacts/request.json)、[prompt](artifacts/free_concern_prompt.md)、[response](artifacts/free_concern_response.txt)、[concern](artifacts/free_concern.json) |
| 检索与停止合同 | [task resolution](artifacts/open_task_resolution.json)、[candidate resolution](artifacts/concern_candidate_resolution.json)、[QueryContract](artifacts/query_contract.json) |
| official control | [Proposal](artifacts/r1_task_proposal.json)、[render](artifacts/r1_scene.png)、[rollout](artifacts/r1_video.mp4)、[episode](artifacts/r1_episode.json)、[Tool](artifacts/r1_tool_execution.json)、[VQA](artifacts/r1_vqa.json)、[Aggregate](artifacts/r1_aggregate.json) |
| evidence-conditioned proposal | [decision](artifacts/decision_r1.json)、[semantic proposal](artifacts/r2_semantic_proposal.json)、[bounded binding](artifacts/r2_bounded_proposal.json) |
| provider TaskGen | [Proposal](artifacts/r2_task_proposal.json)、[code prompt](artifacts/r2_code_prompt.md)、[provider response](artifacts/r2_provider_response.json)、[task.py](artifacts/r2_task.py)、[fixtures](artifacts/r2_checker_fixtures.json) |
| custom evaluation | [render](artifacts/r2_scene.png)、[rollout](artifacts/r2_video.mp4)、[episode](artifacts/r2_episode.json)、[tool.py](artifacts/r2_tool.py)、[Tool result](artifacts/r2_tool_execution.json)、[VQA](artifacts/r2_vqa.json)、[Aggregate](artifacts/r2_aggregate.json) |
| 停止与回答 | [round-2 decision](artifacts/decision_r2.json)、[query answer](artifacts/query_answer.json)、[feedback](artifacts/feedback.json)、[run summary](run_summary.json) |

[manifest.json](manifest.json) 是本包唯一文件索引，包含每个公开文件的来源、大小和
SHA-256。完整 telemetry、raw logs、checkpoint 与生成任务目录仍保留在 manifest
记录的 canonical AutoDL 路径中。

后续的统一 `TaskAdapter`、catalog-external 0-ACT 边界、第五个 RoboTwin official
adapter 与 LIBERO 两回合方法链见
[batch27 紧凑结果](../../../experiments/paper/results/batch27_unified_adapter_libero/)；
该批不替换本页的 ClickBell 干净旗舰。
