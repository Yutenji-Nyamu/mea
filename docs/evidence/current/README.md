# MEA method evidence: `eval_20260805_batch37_clean_flagship_press_stapler_s1000_v7`

> 一次真实方法运行的紧凑入口。逐轮 prompt、生成代码、验证记录、render、视频、Tool
> 与 Aggregate 均保留在下列 artifact；完整 raw telemetry 仍在服务器 evaluation 目录。

## Query 与配置

> Does there exist a bounded, executable scene concern beyond the unchanged official
> press_stapler task under which this policy exposes a measured weakness? Observe the
> control, then let the Plan Agent invent and refine the most informative concerns from
> evidence. I provide no aspect, object, axis, magnitude, relation, threshold, template,
> checker code, or metric. Generate only the scene, checker, Rule Tool, or VQA Tool
> actually required by each Proposal. A generated checker must preserve official success
> as a required conjunct and add only directly observable current-state semantics. A
> diagnostic Tool must remain separate from success. After a valid success, the evidence
> must choose a genuinely different semantic concern or an evidence-grounded boundary
> refinement rather than repeat the same test. The Plan Agent must propose stop as soon as
> a definitive failure witness has an evidence-backed diagnosis. If executable supported
> concerns become informationally saturated without such a witness, it must actively stop
> and answer only the tested scope.

- Task / policy / checkpoint: `press_stapler` / `SmolVLA` /
  `lerobot/smolvla_robotwin`
- Seed / round budget / completed policy episodes: `1000` / `20` / `10`
- Final planning state: `stopped_after_round_11`
- Query interpretation: [prompt](artifacts/plan/query_interpretation_prompt.md) ·
  [response 1](artifacts/plan/query_interpretation_response_1.txt) ·
  [response 2](artifacts/plan/query_interpretation_response_2.txt)

## 方法数据流

```text
Open Query -> Plan Agent -> TaskGen -> render / visual check -> policy rollout
           -> Rule/VQA Tool -> Aggregate -> next Proposal or stop -> Answer
```

## 每轮一行

| 轮次 | Proposal / 执行结果 | Tool / 下一决策 | 关键 artifact |
| --- | --- | --- | --- |
| 1 | official control；policy success `1.0` | `official_check_success`；切换到 scene concern | [video](assets/round_1_act.mp4) · [Tool](artifacts/tool/round_1/tool_execution.json) · [Aggregate](artifacts/aggregate/round_1.json) · [decision](artifacts/plan/decisions/after_round_1.json) |
| 2 | world-x `+0.03 m` + terminal contact；checker fixture 未通过，`0` rollout | 无测量；根据失败改写 checker | [Proposal](artifacts/taskgen/round_2/generation/proposal.json) · [Aggregate](artifacts/aggregate/round_2.json) · [decision](artifacts/plan/decisions/after_round_2.json) |
| 3 | world-x `+0.03 m` + distance `<=0.080 m`；success | 新 Tool `0.0799467 m`；边界细化 | [prompt](artifacts/taskgen/round_3/generation/code_prompt.md) · [code](code/round_3_task.py) · [render](assets/round_3_scene.png) · [video](assets/round_3_act.mp4) · [Tool](code/round_3_tool.py) · [Aggregate](artifacts/aggregate/round_3.json) · [decision](artifacts/plan/decisions/after_round_3.json) |
| 4 | world-x `+0.031 m` + distance `<=0.080 m`；success | exact run-local Tool reuse，`0.0799027 m`；继续细化 | [prompt](artifacts/taskgen/round_4/generation/code_prompt.md) · [code](code/round_4_task.py) · [render](assets/round_4_scene.png) · [video](assets/round_4_act.mp4) · [Tool](code/round_4_tool.py) · [decision](artifacts/plan/decisions/after_round_4.json) |
| 5 | world-x `+0.03 m` + distance `<=0.07992 m`；success | reuse，`0.0798250 m`；切换 world-y | [prompt](artifacts/taskgen/round_5/generation/code_prompt.md) · [code](code/round_5_task.py) · [render](assets/round_5_scene.png) · [video](assets/round_5_act.mp4) · [Tool](code/round_5_tool.py) · [decision](artifacts/plan/decisions/after_round_5.json) |
| 6 | world-y `+0.03 m`；success | reuse，`0.0799863 m`；切换 world-z | [prompt](artifacts/taskgen/round_6/generation/code_prompt.md) · [code](code/round_6_task.py) · [render](assets/round_6_scene.png) · [video](assets/round_6_act.mp4) · [Tool](code/round_6_tool.py) · [decision](artifacts/plan/decisions/after_round_6.json) |
| 7 | world-z `+0.03 m`；success | reuse，`0.0799230 m`；转向证据夹定边界 | [prompt](artifacts/taskgen/round_7/generation/code_prompt.md) · [code](code/round_7_task.py) · [render](assets/round_7_scene.png) · [video](assets/round_7_act.mp4) · [Tool](code/round_7_tool.py) · [decision](artifacts/plan/decisions/after_round_7.json) |
| 8 | world-x `+0.03 m` + distance `<=0.07980 m`；success | reuse，`0.0796478 m`；切换 yaw | [prompt](artifacts/taskgen/round_8/generation/code_prompt.md) · [code](code/round_8_task.py) · [render](assets/round_8_scene.png) · [video](assets/round_8_act.mp4) · [Tool](code/round_8_tool.py) · [decision](artifacts/plan/decisions/after_round_8.json) |
| 9 | yaw `+15 deg`；success | reuse，`0.0797498 m`；切换 pitch | [prompt](artifacts/taskgen/round_9/generation/code_prompt.md) · [code](code/round_9_task.py) · [render](assets/round_9_scene.png) · [video](assets/round_9_act.mp4) · [Tool](code/round_9_tool.py) · [decision](artifacts/plan/decisions/after_round_9.json) |
| 10 | pitch `+15 deg`；success | reuse，`0.0797661 m`；切换 roll | [prompt](artifacts/taskgen/round_10/generation/code_prompt.md) · [code](code/round_10_task.py) · [render](assets/round_10_scene.png) · [video](assets/round_10_act.mp4) · [Tool](code/round_10_tool.py) · [decision](artifacts/plan/decisions/after_round_10.json) |
| 11 | roll `+15 deg`；success | reuse，`0.0796181 m`；Agent 因信息饱和主动停止 | [prompt](artifacts/taskgen/round_11/generation/code_prompt.md) · [code](code/round_11_task.py) · [render](assets/round_11_scene.png) · [video](assets/round_11_act.mp4) · [Tool](code/round_11_tool.py) · [Aggregate](artifacts/aggregate/round_11.json) · [decision](artifacts/plan/decisions/after_round_11.json) |

## 对原 Query 的回答

> 结论不确定：在本次已验证的有界实验范围内，未发现该策略的明确弱点见证；但原始
> 接触型 `+0.03 m` world-x 平移候选未能执行，因此不能否定仍存在其他未测试的弱点。

- official control 与 9 个已执行的实验候选均成功；实验 checker 都保留 official core
  predicate 为合取项。
- 9 次实验的独立 Rule Tool 测量范围为 `0.0796181–0.0799863 m`，均值
  `0.0798184 m`；最紧已通过条件是 world-x `+0.03 m` 且距离 `<=0.07980 m`。
- Agent 因当前可执行能力已无法提出有根据的新 concern 而停止；这是
  `agent_saturation_inconclusive`，不是“已证明没有弱点”。
- 下一步应修复唯一未执行的 terminal-contact checker，并用不同 seed 复验。

完整 Answer（含全部 finding 与 limitation）：[answer.json](artifacts/answer/answer.json)。

## 限制与证据入口

- 只有 `N=10` 个 policy episode，且全部使用 seed `1000`；不能推出跨 seed 或总体泛化。
- round 2 的接触型候选没有 policy rollout，不能算成功或失败。
- 9 个实验结果采用 `generated_check_success`；它们是有约束的实验语义扩展，不等于
  official benchmark success。
- 候选空间仍开放；本次没有 VQA 证据，不能从“无冲突”推断 VQA 鲁棒性。
- [机器可读紧凑摘要](run_summary.json) ·
  [紧凑 artifact 索引](artifact_index.json) ·
  [final Aggregate](artifacts/aggregate/final.json) ·
  [final Query answer](artifacts/answer/query_answer.json)
- 完整 raw source：服务器
  `mea/evaluation_runs/eval_20260805_batch37_clean_flagship_press_stapler_s1000_v7`。
