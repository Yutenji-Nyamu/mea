# 当前紧凑证据：宽泛 Query 三轮主链

当前发布对应 `eval_20260726_batch23_open_query_live_n1_v5`。用户只问：

> 这个 ACT 策略最先在哪种被操作物体或场景变化上暴露弱点？

Query 没有给 aspect 顺序。系统保留的论文式数据流是：

```text
Query
→ runtime-owned official control（ACT success）
→ ClaimFirst 自主选择 1.2× object scale
→ TaskGen + render/expert → ACT success
→ 现场生成 XY-distance Tool，live value=0.024507 m
→ evidence 驱动 Planner 选择 lookalike physical distractor
→ provider 一次编写 scene + check_success()
→ AST + 6/6 fixtures + render/visual/expert
→ 第三次 ACT；生成 checker=true，official success=false
→ Rule/VQA/Aggregate/Planner
→ budget_exhausted / inconclusive / answered=false
```

## 结果与边界

- 三轮都使用 `beat_block_hammer`、ACT checkpoint
  `act-beat_block_hammer/demo_clean-50`、seed `100600`，总 N=3。
- Planner 在看到前一轮 evidence 后依次选择 scale 和 distractor；顺序不是 Query 预写的。
- scale 使用 official-equivalent success；distractor 使用模型生成、AST/fixture 验证的实验
  checker。后者判定成功，但 RoboTwin official success 为 false，不能当作 benchmark success。
- scale 在 official-equivalent authority 下通过。distractor 只在生成的实验 checker 下
  通过，而 RoboTwin official success=false；这是语义不一致，不能合并解释为“两个候选
  都没有弱点”。color、official-random position、timing 尚未测试，因此系统没有回答
  “最先在哪里失败”，而是以预算停止。
- 本包证明受限主链能自动走通，不证明广泛泛化、证据充分、采样效率或 policy ranking。

## 直接查看产物

| 阶段 | 紧凑产物 |
| --- | --- |
| Query/停止合同 | [query_contract.json](artifacts/query_contract.json) |
| evidence-conditioned decisions | [round 1](artifacts/decision_r1.json)、[round 2](artifacts/decision_r2.json)、[round 3](artifacts/decision_r3.json) |
| official control | [Proposal](artifacts/r1_task_proposal.json)、[scene](artifacts/r1_scene.png)、[rollout](artifacts/r1_video.mp4)、[episode](artifacts/r1_episode.json) |
| scale Proposal/TaskGen | [Proposal](artifacts/r2_task_proposal.json)、[proposal prompt](artifacts/r2_proposal_prompt.md)、[code prompt](artifacts/r2_code_prompt.md)、[task.py](artifacts/r2_task.py) |
| scale rollout/Tool | [scene](artifacts/r2_scene.png)、[rollout](artifacts/r2_video.mp4)、[episode](artifacts/r2_episode.json)、[tool.py](artifacts/r2_tool.py)、[Tool result](artifacts/r2_tool_execution.json) |
| distractor TaskGen | [Proposal](artifacts/r3_task_proposal.json)、[bounded proposal](artifacts/r3_bounded_proposal.json)、[code prompt](artifacts/r3_code_prompt.md)、[provider response](artifacts/r3_provider_response.json)、[task.py](artifacts/r3_task.py) |
| distractor validation/rollout | [fixtures](artifacts/r3_checker_fixtures.json)、[scene](artifacts/r3_scene.png)、[rollout](artifacts/r3_video.mp4)、[episode](artifacts/r3_episode.json)、[checker result](artifacts/r3_checker_execution.json)、[checker aggregate](artifacts/r3_checker_aggregate.json) |
| 最终回答 | [query_answer.json](artifacts/query_answer.json)、[feedback.json](artifacts/feedback.json) |

完整 raw bundle 保留在 canonical AutoDL：

```text
/root/autodl-tmp/mea/mea/evaluation_runs/
  eval_20260726_batch23_open_query_live_n1_v5/
/root/autodl-tmp/mea/mea/generated_tasks/
  run_20260726_batch23_open_query_live_n1_v5_round_{1,2,3}/
```

[manifest.json](manifest.json) 是公开证据的唯一索引。Git 未复制完整 telemetry、近 1MB
Aggregate、VQA montage 或开发日志。
