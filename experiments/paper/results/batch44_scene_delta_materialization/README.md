# Batch44：typed scene delta materialization 与冻结 Tool 复用

本目录是 Batch43 暴露 scene handoff 缺口后的冷定向回归，不替换
`docs/evidence/current/`，也不新增 policy 结果。执行代码为
`af93e080aeac6ec18538967b4e74bf23206ebced` 加 scene-delta/prompt-owner working-tree diff；
所有 provider、simulator probe 与服务器测试均在 AutoDL 执行，Windows PC 只编辑文档。

## 1. 前置诊断：手工 typed `+0.03 m` TaskGen preflight

对 `move_playingcard_away`、seed `1000` 明确请求：从 same-seed official reset 将
`playingcards` 沿 y 轴移动 `+0.03 m`，同时保持 x、z、orientation、model identity 与 official
checker。生产 TaskGen 完成：

- TaskGen provider 1 次、local repair 0 次；vision provider 1 次；
- simulator probe 2 次、expert probe 1 次、policy rollout 0 次；
- official checker 复用，正负 fixture `2/2` 通过；vision 与 expert 均通过；
- preservation 为 `verified`；typed scene check 的 expected 与 observed signed delta 均为
  `+0.03 m`，容差 `1e-5 m`。

这项手工构造的 candidate 证明 typed
`{actor, property, axis, signed_delta, unit, reference}` scene request 能被
TaskGen materialize 为 simulator 数值一致的 artifact，并在 policy 前完成现有真实性检查。
它不证明 Plan 能自主写出同一结构，也不证明该 scene 上的 SmolVLA outcome。

## 2. 前置诊断：Batch43 冻结 episode 上的 Rule Tool semantic reuse

Batch43 v4 R3 生成的 `query_terminal_nearest_tcp_to_playingcard_distance` executable Tool 随后
通过 semantic library 命中，并在 **R2 真实失败的 `y +0.03 m` episode** 上重新执行：

- `route=semantic_library_reuse`；provider、simulator 与 policy 调用均为 0；
- current telemetry revalidation、independent numeric oracle 与 oracle agreement 均通过；
- R2 terminal nearest-TCP-to-card distance 为 `0.17870217561721802 m`。

R3 的历史值 `0.06134439632296562 m` 只作异场景对照。R3 position 回到 official location 且
orientation 改变，因此两个数值不能支持“同 scene contact margin 改善”或任何接触因果解释。
本 replay 关闭的是 executable Tool 对正确冻结 episode 的语义复用与当前 telemetry 重验，不是
新的 rollout、policy repeat 或性能比较。

## 3. 最终正验收：冻结 evidence → Plan → TaskGen

v2 Plan replay 读取 unchanged control success 与 R2 的完整冻结 evidence：same-seed simulator
scene fact 为 `playingcards y +0.03000000212341547 m`，official success=false，终态 card
displacement 为 `0.007887560874223709 m`。Terra 1 次调用、0 retry，自主提出 bounded midpoint：

```text
actor=playingcards, property=position, axis=y,
signed_delta=+0.015 m, reference=same_seed_official_reset
```

bound Proposal 的 `scene_need.controlled_changes` 完整保留该 typed delta。此次 replay 为
0 simulator、0 TaskGen、0 policy；它证明数值 scene fact 和失败测量能改变 Plan，而不把一个
手工 candidate 冒充 Agent 输出。

第一次 v1 Plan 示例把外层 runtime 已固定的 `task_identity` 与 `policy_checkpoint` 重复写进
preservation。其 TaskGen 实际已经精确 materialize `+0.015 m`，checker `2/2`、vision 与 expert
均通过，但 preservation 因这两个无 simulator scene authority 的重复事实只能为 `partial`。
修复只落在 prompt owner：outer binding 事实不再重复成为 scene-preservation claim；没有增加
gate、registry 或任务分支。服务器相关回归为 75 passed（另 5 subtests），默认 mainline 为
229 passed（另 10 subtests）。

修复后的 v2 将实际 Plan bound Proposal 直接交给生产 TaskGen：TaskGen provider 1 次、0 repair，
vision 1 次、simulator probe 2 次、expert probe 1 次、policy rollout 0 次；expected delta 为
`+0.015 m`，observed delta 为 `+0.015000000000000001 m`，preservation=`verified`，checker
`2/2`、vision、expert 全部通过。这条 **Plan → TaskGen v2** 才是 Batch44 的最终正验收。

## 4. 结论边界

Batch44 关闭三个相邻缺口：冻结 evidence 能让 Plan 自主提出 typed numeric midpoint；bound
scene need 能无损进入 TaskGen 并精确 materialize；已有 Rule Tool 能在 Batch43 真正的失败 scene
上 provider-free 复用并重新通过 oracle。手工 `+0.03 m` preflight 与 R2 Tool replay 是前置诊断，
最终主链正验收是 v2 Plan→TaskGen。
它没有重写 Batch43 raw artifact，也没有证明 R2/R3 同 scene、contact failure 因果、policy
稳定性、跨 seed 泛化或 evidence-sufficient Answer。Plan 选择的 `+0.015 m` scene 尚未运行新 policy，
因此 `docs/evidence/current/` 保持不变。

## 5. 服务器验证与 artifact

- scene-delta focused：75 passed，另 5 subtests；
- default mainline：229 passed，另 10 subtests；
- Windows PC 未运行测试、import、provider、simulator 或 policy。

AutoDL artifact：

```text
/root/autodl-tmp/mea-run-logs/batch44_move_playingcard_y003_taskgen_preflight_v1/summary.json
/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime/mea/generated_tasks/
  run_batch44_move_playingcard_y003_taskgen_preflight_v1/manifest.json
/root/autodl-tmp/mea-run-logs/batch44_move_playingcard_r2_frozen_tool_reuse_v1/summary.json
/root/autodl-tmp/mea-run-logs/batch44_move_playingcard_round2_plan_replay_v2/summary.json
/root/autodl-tmp/mea-run-logs/batch44_move_playingcard_y0015_plan_taskgen_preflight_v2/summary.json
```

机器可读摘要见 [`summary.json`](summary.json)。
