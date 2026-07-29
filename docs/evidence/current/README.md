# 当前证据：开放 Query 三轮运行的语义审计

这份精简包对应服务器运行
`eval_20260728_batch29_open_flagship_v19`。它保留论文方法所需的干净数据流，
不包含 raw telemetry、重复 montage、调试日志或事后 repair 流程。

## 一句话结果

输入没有 aspect/template 的开放 Query 后，Planner 先测试铃铛颜色；该候选的
official rollout 成功，于是 runtime 根据 evidence 改测 80% 尺寸候选，后者的
official rollout 失败。三轮在线机械链与 Tool
生成/复用都真实完成，但事后 authority 审计发现两个动态候选都没有完整满足
preservation contract；最终方法结论必须改为 `inconclusive / accepted=false`。

## 三轮证据

| 轮次 | 测试 | ACT 官方成功 | Rule Tool |
|---|---|---:|---|
| 1 | official control | 1/1 | official metrics |
| 2 | 意图仅改变铃铛颜色；geometry preservation 未获完整 authority | 1/1 | 现场生成；最小 XY 误差 0.0077674431 m |
| 3 | 80% 尺寸候选；同时存在 contact-point z 漂移 | 0/1 | 同一 Tool 精确复用；最小 XY 误差 0.0452577472 m |

三轮均使用 seed `100000`。这不是统计泛化或 benchmark 排名证据。
其中 80% 尺寸候选保持了 bell center，却把成功判定使用的 contact point 的
z 坐标从 `0.7667903972 m` 改为 `0.7616323171 m`（偏移
`-0.0051580801 m`）。因此该失败同时包含尺寸与接触高度变化，不能归因为纯尺寸变化。

## 按论文数据流阅读

1. Query 与开放路由：
   [request.json](artifacts/query/request.json) →
   [FreeConcern prompt](artifacts/plan/free_concern_prompt.md) /
   [response](artifacts/plan/free_concern_response.txt) →
   [free_concern.json](artifacts/plan/free_concern.json)。
2. 首轮动态候选：
   [颜色 candidate](artifacts/plan/candidate_round_2.json)；
   成功 evidence 后见
   [transition_after_round_2.json](artifacts/plan/transition_after_round_2.json)。
3. TaskGen：
   [颜色生成代码](artifacts/taskgen/round_2/task.py) 与
   [尺寸生成代码](artifacts/taskgen/round_3/task.py)；
   每轮目录同时保留 provider prompt/response、静态验证、render/VLM
   visual diagnosis 和 expert preflight。
4. ToolGen：
   [生成 Tool](artifacts/tool/generated_tool.py) →
   [round 2 live execution](artifacts/tool/round_2_live_execution.json) →
   [round 3 exact-reuse route](artifacts/tool/round_3_exact_reuse_route.json) /
   [execution](artifacts/tool/round_3_live_execution.json)。
5. Rollout 与 Aggregate：
   [official](artifacts/rollout/round_1_official_seed_100000.mp4)、
   [颜色](artifacts/rollout/round_2_color_seed_100000.mp4)、
   [80% 尺寸](artifacts/rollout/round_3_size_80_seed_100000.mp4)；
   对应 compact Aggregate 位于 `artifacts/aggregate/`。
6. Planner 停止与回答：
   [final runtime state](artifacts/plan/final_runtime_state.json) →
   [query_answer.json](artifacts/answer/query_answer.json) →
   [final_answer.md](artifacts/answer/final_answer.md)。这三项保留 source runtime 的
   `evidence_sufficient/diagnosed` 输出用于审计，已被下述 semantic audit 覆盖，不能
   当作当前项目结论。

## 最终语义审计

live 运行时使用的旧 reporting projection 只接受恰好两轮，因此原 manifest 中
`accepted=false`。随后一版只放宽到 2–3 轮的 post-run projection 曾得到
`accepted=true`，但它沿用了错误的 VLM preservation authority，现已明确废弃。

最终 [append-only semantic audit](artifacts/plan/semantic_preservation_audit.json)
用 same-seed simulator state 重算：

- round 2 的 center position 有数值 authority，但 shape/size 无 simulator/AST
  authority，`ImplementationTrace=direct+partial`；
- round 3 的 contact-point world position 明确不相等，
  `ImplementationTrace=direct+partial` 且 `repair_required=true`；
- corrected flagship acceptance 为 `false`，原 Query 为 `inconclusive`。

三次 ACT、视频和 Tool 数值仍是对实际执行场景的有效描述；它们不能直接回答原本要求
“其他条件保持不变”的纯属性变化 Query。源运行工件没有被回写，也没有增加 ACT。
