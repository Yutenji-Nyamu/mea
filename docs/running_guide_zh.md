# 运行指南

所有依赖 simulator、checkpoint、GPU 或 provider 的验证在 canonical AutoDL
`/root/autodl-tmp/mea` 执行。根 README 不包含付费运行命令。

## 1. 运行前检查

```bash
cd /root/autodl-tmp/mea
git status --short
git rev-parse HEAD
nvidia-smi
```

确认要评估的 RoboTwin task、policy checkpoint、seed、最大轮数和 ACT 预算。不要在本地
Windows 下载 checkpoint 或启动 simulator。

## 2. Plan-only

```bash
python scripts/manipeval_agent.py \
  --request "这个策略对被操作物体属性的泛化如何，最先在哪里暴露弱点？" \
  --auto-route \
  --plan-only
```

它只验证 Query route、QueryContract 和第一轮 Proposal；不运行 simulator/ACT，也不是
policy 性能证据。检查输出中的 task、policy、候选 universe、停止条件和 unsupported axes。

## 3. Live evaluation

以 `python scripts/manipeval_agent.py --help` 为当前参数真值。live 命令应显式传入：

- 原始 Query；
- policy 与 checkpoint；
- seed/N 和最大轮数；
- evaluation id；
- live/ACT 授权开关。

默认 Planner 应为 ClaimFirst；不要重新启用 legacy task-specific planner、whole-round
recovery 或 fault injection。TaskGen/ToolGen 各允许一次局部修复。执行后至少核对：

1. QueryContract 与首轮 proposal；
2. scene/checker、render 和 gate；
3. 实际 rollout seed、video、telemetry；
4. Rule/VQA 与 Aggregate 是否消费同一 episode；
5. 下一轮是否由上轮 evidence 产生；
6. stop 是 evidence sufficient、unsupported 还是 budget exhausted；
7. Answer 是否列出 N、未覆盖候选和限制。

## 4. 查看证据

最近一次公开索引见 `docs/evidence/current/manifest.json`。原始 bundle 在 manifest 的
`server_run_root` 下。阅读顺序：

```text
query
→ rounds[].proposal
→ rounds[].generated_artifacts
→ rounds[].render
→ rounds[].rollout
→ rounds[].evaluation
→ aggregate
→ answer
```

Git 中不再复制 raw 视频、telemetry 或 provider 响应。需要发布新结果时，用新运行替换
`docs/evidence/current/`，并在 `docs/evidence/history.jsonl` 追加一行旧结果摘要。

## 5. 测试原则

- 纯 schema、Planner、fixture 和 registry 单测可在服务器快速执行。
- 修改主链后运行相关测试，再运行一次 plan-only。
- 触及 TaskGen/ToolGen、simulator adapter 或 rollout 绑定时，追加一个最小 live smoke。
- 不以固定测试数量为目标；被删除的旧链路测试随实现一起删除。
- 大规模 N、更多 policy 或真实消融须另行预注册，不混入日常 smoke。
