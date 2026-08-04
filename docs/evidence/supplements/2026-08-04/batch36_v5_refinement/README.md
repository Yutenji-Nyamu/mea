# Batch36 v5：evidence-conditioned concern switch 与 Tool exact reuse

本目录是一个**负的整体运行、正的方法子链**。它不替换
[`docs/evidence/current`](../../../current/README.md) 的完整旗舰，也没有最终回答原
Query；它保存目前最干净的“TaskGen 负证据改变下一 Proposal”与同链 Tool 生命周期证据。

## 运行范围

- evaluation：`eval_20260804_batch36_clean_flagship_press_stapler_s1000_v5`
- task / policy / seed：`press_stapler` / SmolVLA / `1000`
- Query：未给 aspect、axis、magnitude、template、checker code 或 metric；要求 Plan
  Agent 根据逐轮 evidence 自选有界 simulator-state concern。
- policy episodes：2 个（official control 与 Round 3 generated task）。Round 2、4 均在
  TaskGen gate 被拒绝，未启动 policy。

[原始 Query](artifacts/query/request.json) ·
[Query interpretation prompt](artifacts/plan/query_interpretation_prompt.md) ·
[response](artifacts/plan/query_interpretation_response_1.txt)

## 关键数据流

1. **Round 1 official control**：SmolVLA 官方任务成功，Plan Agent 选择 `+0.02 m x`
   位移和左右 TCP 距离关系，而不是从预定义 aspect/template 执行脚本。
2. **Round 2 typed N=0 evidence**：TaskGen 在 expert positive fixture 中测得左 TCP 距离约
   `0.077 m`、右 TCP 距离约 `0.502 m`，因此拒绝不成立的左右关系；policy 未启动。
3. **evidence 改变下一 Proposal**：Plan Agent 明确引用上述坐标，改为正交的
   `+0.02 m y` 位移，并用已观测 expert state 给出 `0.10 m` terminal-distance 边界。
4. **Round 3 完整执行**：provider 编写 scene 与 official-core-conjunct checker；AST、
   semantic review、fixture、render/VLM、expert 和 preservation gate 均通过。SmolVLA
   rollout 同时满足 official success 与 generated checker。
5. **新 Tool 进入反馈**：ToolGen 生成
   `query_terminal_left_tcp_to_stapler_distance`，经独立 oracle 验证，从真实 telemetry
   得到 `0.08938229084014893 m`；Aggregate 消费该值。Plan Agent 随后转向新的 vertical
   concern，而非继续放大 y 位移。
6. **第二 Query exact reuse**：零 provider、零 rollout 命中同一 registration，路线为
   `run_local_reuse`，再次得到 `0.08938229084014893 m`。

关键产物：

- Round 2：[Proposal](artifacts/taskgen/round_2/proposal.json) ·
  [拒绝摘要](artifacts/taskgen/round_2/failure_summary.json) ·
  [据此生成的下一 Plan step](artifacts/plan/plan_agent_steps/after_round_02/response_1.txt)
- Round 3：[code prompt](artifacts/taskgen/round_3/code_prompt.md) ·
  [provider response](artifacts/taskgen/round_3/provider_response.txt) ·
  [task.py](artifacts/taskgen/round_3/task.py) ·
  [semantic review](artifacts/taskgen/round_3/checker_semantic_review.json) ·
  [expert gate](artifacts/taskgen/round_3/expert_preflight.json)
- Render 与 rollout：
  [scene comparison](artifacts/taskgen/round_3/scene_comparison.png) ·
  [episode video](artifacts/taskgen/round_3/episode0.mp4) ·
  [result](artifacts/taskgen/round_3/result.json)
- Tool：[generated Python](code/round_3_tool.py) ·
  [live execution](artifacts/tool/round_3/tool_execution.json) ·
  [exact-reuse route](artifacts/tool/exact_reuse/route_decision.json) ·
  [exact-reuse execution](artifacts/tool/exact_reuse/tool_execution.json)

## 为什么整体仍是负结果

Round 4 的 vertical checker 会被 scene 变化本身确定性地破坏，expert gate 正确拒绝；
随后 Plan Agent provider 两次返回空内容，运行未获得新的 Proposal 或主动 stop。因此：

- 本运行证明 `TaskGen rejection → evidence → switch concern → successful generated
  rollout → live Tool → new concern`；
- 本运行也在同一 evaluation 中证明新 Tool 的 exact reuse；
- 它**没有**证明同一 broad Query 最终由 Agent 主动 stop 并通过 QueryContract
  `evidence_sufficient`，也不支持统计泛化或 benchmark 结论。

这些失败分别促成两个精简修正：相同 checker repair 在再次启动 simulator 前即拒绝；
Plan Agent prompt 明确 generated checker 必须是 expert-solvable 成功判据，不能编码预期失败。
没有增加任务名分支、中央 recovery 或额外重试层。

## 索引与服务器真值

- [紧凑运行摘要](run_summary.json)
- [本补充包完整 SHA-256 索引](artifact_index.json)
- 原始服务器目录：
  `mea/evaluation_runs/eval_20260804_batch36_clean_flagship_press_stapler_s1000_v5`

原始 provider failure、未执行轮次和负 evidence 均被保留；N=0 planning evidence 不计为
policy failure，expert evidence 也不计为 policy performance。
