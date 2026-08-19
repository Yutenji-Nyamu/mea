# Batch42：cold transfer preflight、Tool 跨 evaluation 复用与 VQA prompt

本目录记录 Batch41 清理后的第一批方法推进。它不替换
`docs/evidence/current/`，也不把 provider-free replay 写成新的 policy 结果。

## 1. 无任务卡 cold preflight

在 AutoDL 的 `main@a290276` 上，对没有
`mea/knowledge/tasks/move_playingcard_away.md` 的 `move_playingcard_away` 运行：

```bash
PYTHONPATH="$MEA_REPO:/root/autodl-tmp/RoboTwin" \
  /root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python \
  experiments/paper/robotwin_breadth.py \
  --repo-root "$MEA_REPO" \
  --output-dir /root/autodl-tmp/mea-run-logs/batch42_move_playingcard_cold_preflight \
  --phase preflight \
  --checkpoint /root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin \
  --task move_playingcard_away --seed 1000 --no-resume
```

fresh reset / TaskContext 在 `33.167731 s` 后通过；provider 与 policy rollout 都为 0。
这证明该任务可由 official source、runtime binding 与 fresh probe 进入通用路径，不证明
Plan、TaskGen、Tool 或 policy 已闭环。第一次 live 仍不得预写任务卡；真实失败发生后才决定
是否补一张只含 source 事实与失败修正的短卡。

## 2. 真实冻结 telemetry 上的跨 evaluation Rule Tool 复用

简化后的 Tool library 先暴露一个真实兼容错误：recorder 的
`semantic_fields` 是 `[{"name": ...}]`，旧读取把 dict 当 set 元素而抛
`TypeError`。修复只接受现有 mapping、string list 与 `{name}` list 三种 schema 形状，
没有增加审批、hash 或 fallback。

随后以 Batch40 的已验证 `query_terminal_roller_z_position` Python Tool 和 episode 为
source，以 Batch39 的另一个真实 `grab_roller` episode 为 target，执行一次 provider-free
replay：

- semantic key 精确命中 `generated_tool_library`；
- `route=semantic_library_reuse`、`provider_called=false`；
- target episode 重新完成 source static check、两次执行、typed interpreter oracle 与
  finite check；
- source value 为 `0.8000384569168091 m`，target value 为
  `0.8001335859298706 m`；
- 0 provider、0 simulator、0 policy rollout。

机器结果在服务器：
`/root/autodl-tmp/mea-run-logs/batch42_tool_cross_eval_replay_v3/summary.json`。
这关闭的是 Rule Tool executable artifact 的跨 evaluation 精确复用，不是整个 Agent run 的
跨 evaluation复现，也不证明 policy 稳定性。

## 3. VQA prompt

Batch37 的同一冻结 montage 曾得到 `false(0.82)` 与 `true(0.86)`。生产下游已经支持
`observed=null → abstained → uncertain evidence`，但 observer prompt 的唯一 schema 示例仍是
`observed=true`。本批只修改这一 prompt owner：示例改为 `null`，并明确 temporal `true`
必须直接展示必要转变，`false` 必须有足够密的帧覆盖并排除事件；稀疏帧中“没看到”必须
返回 `null`。

冻结 montage 的五次 provider repeat 尚未执行：本地与服务器当前均没有
`UIUI_API_KEY`。正式协议为同一 prompt/image/model、temperature 0、五次独立调用；不多数投票，
任何 bool/bool 或 bool/null 混合都记为不稳定。`5×null` 只证明该模糊输入稳定弃答，
没有独立 gold 时 `5×同一 boolean` 也不证明准确。

## 4. 服务器验证与下一步

- Tool 精确跨 evaluation focused：`1 passed`；
- Tool 相关主线：`29 passed`；
- Execution VQA：`11 passed`；
- 默认 mainline：`226 passed`，另 `10 subtests`；
- cold/compat：`623 passed`，另 `162 subtests`；
- Windows 未运行测试/import/provider/simulator/policy。

任务 preflight 运行在 clean `main@a290276`；Tool/VQA 修改及上述测试运行在
`a290276 + Batch42 working-tree diff`，最终提交号见 Git 历史。

下一次 provider 可用时，先跑 `move_playingcard_away` 的 broad Plan/live cold transfer；
`press_stapler` 作为第二个受控回归，检查 Agent 是否停止重复同轴放大。rollout 以信息增益决定，
预期 1 个 control 加 1–2 个不同 concern，软上限分别为 4 与 5，不要求跑满。
