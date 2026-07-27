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

单任务 checkpoint 的 query-first 入口必须显式给出执行边界。例如：

```bash
python scripts/manipeval_agent.py \
  --request "这个策略在目标旁存在相似物体时，还能可靠地点击正确目标吗？" \
  --auto-route \
  --bound-task-name click_bell \
  --open-query-planner claim_first_v1 \
  --generated-rounds 2 \
  --plan-only \
  --no-history \
  --evaluation-id <unique-plan-id>
```

这会先调用一次 inventory-free FreeConcern，再做 official task retrieval、policy
compatibility gate、QueryContract 和首个有界 Proposal；不运行 simulator/ACT，也不是
policy 性能证据。重点检查 `free_concern.json`、`task_resolution.json` 中的候选、决策、
provider/repair/retry 计数，以及最终是 `retrieve_and_adapt`、`generate_new` 还是
`unsupported`。

若 online resolver 已输出 `unsupported`，之后基于冻结 concern 的 0-provider
replay 只能证明确定性 resolver/control handoff，不能倒推成一次在线成功。当前
ClickBell flagship 的通过标准更严格：无 aspect CLI hint、无 history replay、一个
进程完成 official→generated scene/checker→Tool/VQA/Aggregate→Answer，并在同一
bundle 中记录 `flagship_acceptance.accepted=true`。

不传 `--bound-task-name` 的 auto-route 只能在已声明的 checkpoint portfolio 中选择，
不是让一个单任务 policy 执行任意发现的 task。当前可发现 50 个 RoboTwin official
task，但 discovery 不等于 checkpoint-ready。语义相近却与单任务 checkpoint 不兼容时
必须 fail closed；不得靠 task 名覆盖绕过 scope gate。

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

Git 只复制 current manifest 收录的短 rollout、render、生成代码、关键 provider 输出和
结论；完整 telemetry/VQA bundle 留在服务器。需要发布新结果时，用新运行替换
`docs/evidence/current/`，并在 `docs/evidence/history.jsonl` 追加一行旧结果摘要。

## 5. 论文协议

`experiments/paper/manipeval_run_live_paper_protocols.py` 只运行已冻结输入：

- efficiency：fixed 与 adaptive 独立执行，比较真实 ACT 数、wall time 和 dense 结论；
- ranking：每个 seed 先冻结一次 expert instruction/scene eligibility，再让各 policy 共享；
- table3：真实 provider scene+checker 经过 compile/render/expert/fixtures，最后读取显式
  `development_agent_proxy` review；proxy 不能写成 human gold。

协议结果保留 preregistration、逐 cell/seed 结果和一个最终 JSON；不要把这些 dispatcher
接回生产 Agent。LIBERO smoke 也属于独立 environment/policy feasibility smoke，不混入 RoboTwin 或 Table 9。
LIBERO 的固定环境、official control 与 MEA 迁移协议见
[LIBERO / SmolVLA 复现与 MEA 接入](libero_smolvla_reproduction_zh.md)。

当前 SmolVLA checkpoint 没有可审计的训练 task manifest，声明 scope 为 unknown。
因此 unbound LIBERO 请求必须在 rollout 前拒绝；`--bound-task-name libero_object/task0`
只授权该次 official control，并不证明 checkpoint 支持该任务。plan-only 的通过标准是
授权候选为 1、change contract 仍为 pending、`rollouts_executed=0`。batch26 live
实际执行 1 个 280-step 上限 official control，结果失败，并以
`control_failed / official_control_failed` 停止；`custom_rollout_authorized=false`，
没有消耗第二 rollout。该负结果只验证协议短路。

## 6. 测试原则

- 纯 schema、Planner、fixture 和 registry 单测可在服务器快速执行。
- 修改主链后运行相关测试，再运行一次 plan-only。
- 触及 TaskGen/ToolGen、simulator adapter 或 rollout 绑定时，追加一个最小 live smoke。
- 不以固定测试数量为目标；被删除的旧链路测试随实现一起删除。
- 大规模 N、更多 policy 或真实消融须另行预注册，不混入日常 smoke。

包含 LIBERO 测试时必须先加载专用环境路径；否则 upstream `libero` 会交互询问
dataset 路径，不能被解释为代码回归：

```bash
cd /root/autodl-tmp/mea
. /root/autodl-tmp/envs/mea-libero/etc/conda/activate.d/10_mea_libero_paths.sh
/root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q tests/manipeval
```

本批唯一紧凑运行入口见[当前证据](evidence/current/README.md)；效率、LIBERO、
ranking 与 proxy 结果统一由
[`batch26_claim_closure/summary.json`](../experiments/paper/results/batch26_claim_closure/summary.json)
索引。
