# Batch43：无任务卡 cold live、场景事实回流与 VQA 稳定弃答

本目录记录 Batch42 之后的 focused method regression，不替换
`docs/evidence/current/`，也不是 policy benchmark。目标是在没有
`mea/knowledge/tasks/move_playingcard_away.md`、没有用户指定 aspect、actor、axis、scene
edit、checker、metric 或停止脚本时，观察第二个 RoboTwin task 是否能进入论文主链。

## 配置与凭据边界

- task：`move_playingcard_away`；policy：SmolVLA shared official checkpoint；scene seed：
  `1000`；每轮 `N=1`。
- 规划 allowance：一个 required official control 加最多三个 candidate round；allowance 是防失控
  边界，不要求跑满。
- v1 使用默认 balanced profile；v2/v3 使用 quality profile；v4 的 Plan、TaskGen、ToolGen 和
  vision 使用 `gpt-5.6-terra`，最终 Answer 使用 `gpt-5.6-sol`。
- v1/v2 基于 `0ea49a73bf433c89b713d089f27a31e9a25a4037`；v3/v4 基于 control-budget
  修复 `b22ff9f6b2c30c71f69255548ff9baf065b2e572`。
- provider credential 只注入对应服务器进程，未写入仓库、命令、summary 或日志索引。v1/v3
  已经到达 provider gateway，失败是 model channel/空响应，不是缺 credential 或认证失败。

## v1–v4 live 真值

| run | policy episode | 结果与边界 |
| --- | ---: | --- |
| v1 | 0 | Query routing 的 Luna 调用先返回空 content、再返回 HTTP 503 no available channel；没有建立 evaluation artifact。属于 provider availability failure，不是 Plan、TaskGen 或 policy 失败。 |
| v2 | 1 | unchanged official control 成功；Sol 随后提出 playing-card lateral relocation cold concern，但 session 把 required control evidence 错算成 candidate evidence，报 `completed_rounds cannot be smaller than evidence count`，generated round 未执行。 |
| v3 | 1 | `b22ff9f` 后 unchanged control 再次成功；Sol 在 control 后连续两次返回空 content，live 终止。一次独立 Terra provider-only replay 能提出 lateral-offset Proposal，但没有被该 live 执行。 |
| v4 | 3 | Terra 完成 control、两个 generated round、两次有限 Rule Tool、evidence-conditioned refinement、Plan 主动 inconclusive stop 与受限 Answer；在还剩一个 candidate allowance 时停止。 |

`b22ff9f` 只修正 required control 与 candidate budget 的计数口径。对 v2 冻结 Plan 输出做
0-provider、0-simulator、0-policy replay 后，candidate `completed_rounds=0`、
`budget_remaining=3`，下一 round 可以正常 materialize；没有引入 resume 或新状态机。

## v4 三轮及 raw artifact 纠正

| round | Task/Tool | live 结果 |
| --- | --- | --- |
| R1 | official passthrough + `official_check_success` exact reuse | pipeline passed，official success=true。 |
| R2 | cold `scene_robustness.object_relocation`；TaskGen 一次生成并通过 2/2 fixtures、vision、expert；新 Python Rule Tool | simulator setup 显示 `playingcards` 相对 same-seed official scene 沿 **y 轴 +0.030000002 m**；official success=false；Tool 测得相对该 generated reset 的终态 card displacement `0.007887560874223709 m`。 |
| R3 | Plan 请求所谓 same-scene contact-margin refinement；TaskGen 首版失败后一次 repair，随后通过 2/2 fixtures、vision、expert；第二个新 Python Rule Tool | official success=true；Tool 得到终态最近 TCP-card 距离 `0.06134439632296562 m`。 |

R3 **不是 R2 的同场景复用**。冻结 TaskGen artifact 的 simulator setup 表明，R3 card position
回到了 R1 official location，orientation 又变成约 `[0.5, 0.5, 0.5, 0.5]`；repair 接受的是
official-base scene，而不是 R2 的 `y +0.03 m` scene。因此 R2 failure 与 R3 success 不能解释为
同一 scene 的重复冲突。raw Plan/Answer 把 R3 称作 same-scene，但同一 Answer 也正确标出
`preserved_conditions` 未覆盖；本记录保留 frozen artifact，并纠正该措辞而不倒写原运行。

v4 证明的是：无任务卡的 control evidence 能让 Plan cold 提出可执行 scene concern，通用
TaskGen 能在第二个 task 上生成并验证场景，两次真实 generated rollout 和有限 Rule evidence
能改变下一 Proposal，Plan 最终能在 allowance 用尽前主动给出诚实的 inconclusive Answer。
它不证明一个可复现的 policy weakness、same-scene reuse、充分正结论、Tool exact reuse、跨 seed
稳定性或 sample efficiency。

## 失败驱动的最小 scene-context 修复

R3 暴露的 owner 是场景事实没有精确进入下一轮 Plan，而不是缺 registry 或恢复机制。当前补丁：

- 从 same-seed official/generated simulator setup 提取显著 actor position change；比较容差为
  `1e-6 m`，事实包含 actor、property、axis、signed delta、generated value、unit、comparison
  seed 与 simulator authority。
- round summary 将该事实直接交给 Plan evidence；不让模型从 prose 猜轴或数值。
- Planner prompt 要求 exact/prior/refinement 显式重述 actor、axis、数值与单位；没有这些事实时
  不得称 same-scene，应改成独立 official-base experiment 或停止。

在冻结 v4 R2 evidence 上进行一次 Terra Plan-only replay，0 simulator、0 policy rollout。
首个响应只因 preservation relation schema 错误被局部修一次；第二个响应选择独立
official-base `playingcards x +0.030 m`，明确不保留原 `y +0.030 m` delta，并请求同类有限
Rule measurement。该 replay 有 2 个 provider response、1 次 schema repair，只证明数值场景事实
已经到达 Plan 并改变 Proposal。补丁后尚未运行 TaskGen preflight，也没有新 policy episode，
所以不能声称新 x scene 已 materialize 或 same-scene 问题已端到端关闭。

## 冻结 VQA repeat

Batch37 的同一 montage/question 曾得到相反 boolean。当前 observer prompt 与
`gpt-5.6-sol`、temperature `0`、provider retry `0` 下做五次独立调用，得到：

```text
null, null, null, null, null
```

五次均为合法响应，没有多数投票，且新增 simulator/policy rollout 都为 0。这证明当前
prompt+Sol 对该模糊冻结输入稳定弃答；没有 independent gold，因此不证明准确性，也不是原 Luna
结果的同模型复现。

## 服务器验证与 artifact

- scene-context 相关回归：62 passed，另 5 subtests；
- 默认 mainline：228 passed，另 10 subtests；
- Windows PC 未运行测试、import、provider、simulator 或 policy；
- `docs/evidence/current/` 未修改。

主要 AutoDL artifact：

```text
/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime/mea/evaluation_runs/
  eval_20260819_batch43_move_playingcard_cold_mainline_s1000_v4
/root/autodl-tmp/mea-run-logs/batch43_move_playingcard_cold_mainline_v1
/root/autodl-tmp/mea-run-logs/batch43_move_playingcard_cold_mainline_v2
/root/autodl-tmp/mea-run-logs/batch43_move_playingcard_cold_mainline_v3
/root/autodl-tmp/mea-run-logs/batch43_move_playingcard_cold_mainline_v4
/root/autodl-tmp/mea-run-logs/batch43_v4_scene_evidence_plan_terra_replay
/root/autodl-tmp/mea-run-logs/batch43_vqa_frozen_repeat_gpt56sol
```

机器可读摘要见 [`summary.json`](summary.json)。
