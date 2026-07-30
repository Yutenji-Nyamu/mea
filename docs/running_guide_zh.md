# 运行指南

所有测试、provider、simulator、checkpoint 与 policy inference 都在 canonical AutoDL
clean worktree 执行；Windows 只编辑代码、文档、diff 和紧凑 artifact。不要使用服务器
上的 dirty/stale clone，也不要把某个 backend 的环境当成 MEA 方法本身。

## 1. 运行前检查

```bash
: "${MEA_REPO:?export MEA_REPO=/absolute/path/to/clean/mea-worktree}"
: "${MEA_PYTHON:?export MEA_PYTHON=/absolute/path/to/backend/python}"
cd "$MEA_REPO"
git status --short
git rev-parse HEAD
"$MEA_PYTHON" -c "import sys; print(sys.executable)"
nvidia-smi
```

运行前冻结原始 Query、task/policy binding、checkpoint、seed/N、最大轮数与
evaluation id。具体环境、checkpoint、policy server 和网络问题分别见
[RoboTwin / SmolVLA 复现](robotwin_smolvla_reproduction_zh.md)、
[LIBERO / SmolVLA 复现与 MEA 接入](libero_smolvla_reproduction_zh.md)及各 policy
官方说明；Hot 指南不复制长安装流水。

### Backend capability

| Backend | 当前方法入口 | 能力边界 | 推荐用途 |
| --- | --- | --- | --- |
| RoboTwin SmolVLA | 生产 Plan Agent → 通用 TaskGen → `RoundExecutor` | 多任务 official rollout；生成式 live 尚无正验收 | 默认的轻量方法 smoke |
| RoboTwin ACT | 同一生产主链 | 单任务 checkpoint，已有最完整正 evidence | checkpoint 特定复核 |
| RoboTwin DP3 | `experiments/paper/` adapter | 仅已有 pilot，不是默认生产 binding | policy 对照实验 |
| LIBERO SmolVLA | 独立 LIBERO chain | basic adaptation；外层方法 loop 尚未统一 | 环境迁移 smoke |

当 policy 比较不是研究问题时，选择已经验证且成本最低的 backend，当前优先
RoboTwin SmolVLA；只有论文协议或 checkpoint 特定问题才固定 ACT/DP3。backend 就绪不
等于方法正验收，TaskGen 在 rollout 前失败时 policy sample 计为 0。
生产 CLI 默认使用 `balanced` model profile；历史 `legacy` profile 只供显式兼容或
消融，不应成为新的 Plan Agent / TaskGen live 默认值。

## 2. Plan-only

使用脚本 `--help` 作为参数真值。代表性 SmolVLA 方法预检为：

```bash
"$MEA_PYTHON" scripts/manipeval_agent.py \
  --request "这个策略最先会在哪种可执行场景变化上暴露轨迹弱点？" \
  --auto-route \
  --policy-backend smolvla \
  --bound-task-name <task> \
  --max-agent-rounds 2 \
  --plan-only \
  --no-history \
  --evaluation-id <unique-plan-id>
```

plan-only 只检查 Query interpretation、task/policy binding、Query contract 与首个
Proposal，不运行 simulator 或 policy，也不是性能证据。重点核对：

1. Query 没有被 task/aspect/template 菜单改写；
2. task 与 checkpoint scope 可执行；
3. `control_requirement` 是否来自 Query，而非固定 official-first；
4. scene/checker/Rule/VQA need 是否彼此独立且只声明必需子集；
5. catalog/task-specific planner 未进入生产路径。

若 `control_requirement=not_required`，初始 rounds 可以等待首个 Proposal
materialization；这不是缺少计划。plan-only 不能预测后续一定生成、复用或成功，更不能
写成 policy evidence。

## 3. Live evaluation

先按对应 backend 的 cold runbook 启动 policy runner，再以

```bash
"$MEA_PYTHON" scripts/manipeval_agent.py --help
```

确认当前 live 参数；命令必须显式给出原始 Query、policy/backend、task/checkpoint、
seed/N、最大轮数、evaluation id、rollout 预算和停止条件。生产链只使用 Plan Agent，不启用
legacy task planner、whole-round restart 或 fault injection。

每轮按以下顺序验收：

1. Query contract 与 Proposal；
2. 若需要，scene/checker retrieve 或 generate；
3. fixture/state、render/VLM 与 expert gate；
4. 同一 binding/seed 的 policy rollout、video 与 telemetry；
5. Rule/VQA 是否消费该 episode，生成 checker 是否与 official success 分开；
6. Aggregate 是否完整进入下一次 Plan Agent decision；
7. Agent 提出的 stop 是否由 QueryContract 验证；
8. Answer 是否列出 N、未覆盖项、冲突、停止原因和限制。

scene、checker、Rule Tool 与 VQA Tool 是独立 need：Tool-only Query 不启动 TaskGen。
TaskGen 与 ToolGen 各至多一次局部 repair；失败即终止当前 live，不循环消耗 policy
sample。Tool exact reuse 后仍须在当前 episode 上重跑确定性/oracle 校验；跨 evaluation
只接受显式 reviewed registry artifact。

生成 checker 是实验语义。它与 official success 冲突或不可比较时必须并列报告，不能
把实验通过写成 benchmark 成功。更细的 simulator authority、preservation 与 codegen
约束见[开发者参考](developer_reference_zh.md)。

## 4. 查看证据

当前公开入口是
`docs/evidence/current/evidence_bundle_manifest.json`：

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

`README.md` 是阅读索引，`run_summary.json` 是机器可读投影；完整 telemetry、VQA 和
raw Aggregate 留在服务器。发布新正证据时替换 `docs/evidence/current/`，并向
[`docs/evidence/history.jsonl`](evidence/history.jsonl)追加一行旧结果摘要。0-rollout
replay 或事后审计必须作为新 artifact 保存，不得覆盖原 Answer 或倒改在线结论。

## 5. 论文协议

效率、fixed/adaptive、Table 3、人工/VQA、ranking 和 replay 只从
[`experiments/paper/`](../experiments/paper/) 的冻结协议入口运行；参数和预算以该目录
索引及脚本 `--help` 为准。它们不得回接生产 Agent，也不属于日常方法 smoke。

## 6. 测试原则

- 所有测试、import 与 compile 都在 AutoDL 执行，不回退到 Windows。
- 先运行触及模块的高信息回归，再运行默认 `tests/mainline/`。
- 修改 Query/Plan Agent 后追加 plan-only；触及 TaskGen/ToolGen/backend binding 时，
  只在静态 gate 通过后追加一个最小 live smoke。
- 不以测试数量为目标；compat 与 paper suite 只在对应改动时显式运行。
- 大 N、多 policy、真实消融与效率比较必须走冻结论文协议，不混入日常验证。

最新样本数、验收状态和限制见[当前证据](evidence/current/README.md)；累计覆盖和下一项
方法缺口见[论文 claim 与 gap](paper_claim_gap_zh.md)。
