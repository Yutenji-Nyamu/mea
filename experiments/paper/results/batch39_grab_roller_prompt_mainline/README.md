# Batch39：`grab_roller` prompt-first 方法主链回归

本目录是冷开发证据，不替换 Batch37 当前旗舰，也不更新
`docs/evidence/current/`。Batch39 从 Batch38 的真实失败出发，只修改短任务卡、Agent prompt
和 simulator-authoritative preservation 边界，再用同一条生产主链回归。

## 方法变化

- Plan repair 收到上一版完整 Proposal；跨轮上下文突出最近失败，并明确区分独立实验与累积
  refinement。
- `grab_roller` 短任务卡记录 official actor/contact/telemetry 事实和已观察的 expert-oracle
  限制；Plan、TaskGen 与 Tool Request 读取同一份任务事实。
- preservation 将“指定坐标轴”“contact reference/local offset”“contact world position”和
  model identity 分开验证，避免把合法的 x 平移误判为 y/contact preservation 破坏。
- pre-policy TaskGen validation failure 以 typed `N=0` evidence 返回 Plan；provider/network 等
  system failure 仍保持 terminal。
- 一条 Rule Tool need 只要求一个可执行 metric contract，避免把多个观测塞进一个 Tool。

没有新增任务专属 Planner、重试状态机、registry 或额外 rollout gate。

## AutoDL server 回归

- 服务器工作树：`/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime`
- 实现基线：`061a1652ef958a0d73a2f745e5c6789deb364c62`
- Windows PC 只编辑与检查 diff；测试、provider、simulator 和 SmolVLA 均在 AutoDL server。
- 最终定向回归：`132 passed, 7 subtests passed`。
- 最终默认 mainline：`279 passed, 12 subtests passed`。
- 0-rollout 冻结 artifact 回放：round 2 与 round 4 的真实 simulator state 均由当前
  preservation 实现逐项判为 `verified=true`；冻结 raw artifact 不被回写。

## live v1–v4

| run | policy episode | 结果 |
| --- | ---: | --- |
| v1 | 1 | control 后，旧 preservation 把 actor 平移后的 contact world position 变化误判成 contact reference 被破坏；该 generated round 未启动 policy，最终以 TaskGen system failure 结束。 |
| v2 | 0 | provider 网关连续返回空 assistant 内容；未进入 simulator/policy。 |
| v3 | 2 | control 和新 scene 均成功；Tool Request 因一个 need 同时要求高度与左右距离而无法形成单一 metric contract。 |
| v4 | 3 | 完整退出；同一正向 x 场景语义完成两次 generated-scene policy episode（第二次改测右侧），两个 live Rule Tool 得到有限值，evidence 改变下一 Proposal，Plan Agent 主动停止。 |

v4 的 evaluation id 是
`eval_20260813_batch39_grab_roller_prompt_mainline_s1000_v4`，每轮 `N=1`、scene seed
`1000`，使用 SmolVLA：

1. round 1 official control 成功。
2. round 2 由 Plan Agent 选择 model 0、`x=+0.15 m` 的 scene concern；通用 TaskGen 一次
   通过 fixture、render/VLM 与 expert，并被当时的 production gate 接受，policy 成功；冻结
   raw preservation 是 `partially_unverified`，本批修复后对同一 simulator state 的各项条件
   均可逐项验证；新 Rule Tool 测得左 TCP
   到 declared left contact 的 terminal distance 为 `0.04107694 m`。
3. round 3 转向 model instance 2。generated expert 与 unchanged same-seed official expert
   都在 grasp/IK 路径得到 `target_pose=None`，所以这是 expert oracle limitation，policy 未执行、
   `N=0`，不能归因成 policy failure。v4 raw writer 最初使用了较弱的
   `candidate_unexecutable` 名称；Batch39 的 typed evidence 修正不会改写冻结 raw artifact。
4. round 4 根据该失败返回已验证的 model 0、`x=+0.15 m` scene，改测右侧；policy 成功；
   另一新 Rule Tool 得到右侧 terminal distance `0.04231440 m`。
5. Plan Agent 主动停止：左右距离没有预先验证的失败阈值，instance concern 又缺可用 expert
   oracle；继续猜测 pose/instance 信息增益不足。最终 Answer 为 `inconclusive`，明确报告
   `N=3`、单 seed、未覆盖候选和开放候选空间。

## 结论边界

本批证明 prompt/task-guide-first 迭代能修复一个真实失败，并在同一 broad Query 中形成
`TaskGen -> rollout -> live Tool -> evidence-conditioned next Proposal -> active stop -> Answer`。
它没有证明 `grab_roller` 泛化或 `evidence_sufficient=true`，也没有证明 Tool exact reuse：
round 2 与 round 4 是两个不同的 metric artifact。当前旗舰仍是 Batch37。

机器可读摘要见 [`summary.json`](summary.json)。raw evaluation 位于服务器
`/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime/mea/evaluation_runs/`
`eval_20260813_batch39_grab_roller_prompt_mainline_s1000_v4`；provider credential 未写入仓库。
