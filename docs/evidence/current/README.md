# 当前紧凑证据：distractor Query 主链

当前发布对应 `eval_20260724_batch22_distractor_live_n1_v7`。它保留一条可直接检查的论文式数据流：

```text
Query
→ ClaimFirst 先运行 official control
→ ACT official rollout 成功
→ 证据尚不足，Planner 提出 lookalike physical distractor
→ 同一 Proposal 触发模型编写 scene + check_success()
→ AST 边界、6 个语义 fixture、render/visual/expert gate
→ 第二次 ACT rollout
→ 生成 checker 作为绑定 Tool 裁决该 episode
→ VQA / Aggregate
→ QueryContract: evidence_sufficient
→ 有边界的原 Query 回答
```

## 结果与边界

- 两次 ACT 均使用 `beat_block_hammer`、checkpoint `act-beat_block_hammer/demo_clean-50` 和 seed `100600`。
- official control 的 RoboTwin success 为 `true`。
- distractor episode 的模型生成 checker 为 `true`，但 official success 为 `false`；因此它只能支持实验语义
  `target hit and distractor not hit`，不能当作官方 benchmark success。
- 最终 verdict 是 `no_failure_observed`，只覆盖一个已登记候选、一个 task、一个 seed、总 N=2。
- `evidence_sufficient` 表示有限 QueryContract 已覆盖，不表示统计泛化充分。
- live 后发现 summary 的 AST gate 只读取旧字段。修复后对原两条 episode 做了 0-ACT summary replay；
  原结论已备份，未增加或改写 rollout。

## 直接查看产物

| 阶段 | 紧凑产物 |
| --- | --- |
| Query/停止合同 | [query_contract.json](artifacts/query_contract.json) |
| 第二轮 Proposal | [round_2_proposal.json](artifacts/round_2_proposal.json) |
| TaskGen 输入/输出 | [taskgen_prompt.md](artifacts/taskgen_prompt.md)、[provider_response.json](artifacts/provider_response.json) |
| 模型生成任务代码 | [task.py](artifacts/task.py)、[checker_fixtures.json](artifacts/checker_fixtures.json) |
| official control | [scene](artifacts/round_1_scene.png)、[rollout](artifacts/round_1_act.mp4)、[episode](artifacts/round_1_episode.json) |
| distractor candidate | [scene](artifacts/round_2_scene.png)、[rollout](artifacts/round_2_act.mp4)、[episode](artifacts/round_2_episode.json) |
| Tool/Aggregate | [tool_execution.json](artifacts/tool_execution.json)、[aggregate_result.json](artifacts/aggregate_result.json) |
| 最终回答 | [query_answer.json](artifacts/query_answer.json)、[feedback.md](artifacts/feedback.md) |

完整 raw bundle 仍保留在 canonical AutoDL：

```text
/root/autodl-tmp/mea/mea/evaluation_runs/
  eval_20260724_batch22_distractor_live_n1_v7/
/root/autodl-tmp/mea/mea/generated_tasks/
  run_20260724_batch22_distractor_live_n1_v7_round_2/
```

[manifest.json](manifest.json) 是这次公开证据的唯一索引。它只收录论文判断需要的路径和限制，不复制开发日志。
