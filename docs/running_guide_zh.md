# 运行指南

所有依赖 simulator、checkpoint、GPU 或 provider 的验证在 canonical AutoDL 的一份
clean MEA clone/worktree 执行；不要使用服务器上已有 dirty/stale clone。服务器 live 必须使用
`/root/autodl-tmp/conda/envs/RoboTwin/bin/python`；base `python` 只适合不 import
SAPIEN/RoboTwin、也不执行 setup/render/expert/rollout 的纯单测。根 README 不包含
付费运行命令。

## 1. 运行前检查

```bash
MEA_REPO=/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime
cd "$MEA_REPO"
git status --short
git rev-parse HEAD
/root/autodl-tmp/conda/envs/RoboTwin/bin/python -c \
  "import sys, sapien; print(sys.executable)"
nvidia-smi
```

确认要评估的 RoboTwin task、policy checkpoint、seed、最大轮数和 ACT 预算。不要在本地
Windows 下载 checkpoint 或启动 simulator。

预检按层归因：base Python 的 import/setup 失败是解释器错误；在正确解释器下，若生成
代码用 `pose.p[i] += ...` 或复用同一 Pose 后场景仍与 control 相同，则是 TaskGen
Pose no-op，必须复制位置并新建 `sapien.Pose`。两者都不要归因给 policy、provider 或
VLM；修正后仍须通过 state/checker、render、VLM 与 expert gate。

## 2. Plan-only

单任务 checkpoint 的 query-first 入口必须显式给出执行边界。例如：

```bash
/root/autodl-tmp/conda/envs/RoboTwin/bin/python scripts/manipeval_agent.py \
  --request "这个策略在目标旁存在相似物体时，还能可靠地点击正确目标吗？" \
  --auto-route \
  --bound-task-name click_bell \
  --generated-rounds 2 \
  --plan-only \
  --no-history \
  --evaluation-id <unique-plan-id>
```

`--auto-route` 的生产默认值就是 Plan Agent，无需重复选择规划器。这会先完成一次
inventory-free Query interpretation，再做 official task retrieval、policy compatibility
gate 和 Query contract；`PlanAgentInitialPlanBuilder` 随后直接写初始计划，不调用
`CatalogPlanAgent` 或任务专属 legacy planner。它不运行 simulator/ACT，也不是 policy 性能
证据。重点检查 `plan/query_interpretation.json`、`plan/open_task_resolution.json` 中的候选、
决策、provider/repair/retry 计数，以及 `manifest.planner.kind` 是否为
`plan_agent_direct_initial_v1`、`task_specific_planner_used=false`。

若原始 Query 提出 artifact index 外 concern，plan-only 应保存 Query interpretation、
Proposal-domain resolution 与 Query contract，并明确它没有执行 scene/checker/tool
materialization。只有 `control_requirement=required` 才绑定 neutral official control；
`not_required` 的 live 可直接从 Query interpretation 形成首个 Query-derived Proposal。因此
no-control plan 的初始 `rounds` 可以为空，并明确等待首个 Proposal materialization；
这不是缺少计划，而是避免制造假的 control 壳。plan-only 不能预测后续一定 exact reuse
或 generation，更不得写成 policy evidence。

若 online resolver 已输出 `unsupported`，之后基于冻结 concern 的 0-provider replay
只能证明确定性 resolver/control handoff，不能倒推成一次在线成功。通用在线语义验收
要求：无 aspect/template CLI hint、无 history rollout replay、一个进程完成有界的
control/Proposal→TaskGen→Tool/VQA→Aggregate→Answer；每个动态 Proposal 必须有
`direct+complete` ImplementationTrace，且 post-run acceptance projection 必须与最终
回答一致。`accepted=true` 是正验收条件，不是尚未运行时的预设结果。

不传 `--bound-task-name` 的 auto-route 只能在已声明的 checkpoint portfolio 中选择，
不是让一个单任务 policy 执行任意发现的 task。official discovery 不等于
checkpoint-ready。语义相近却与单任务 checkpoint 不兼容时必须 fail closed；不得靠
task 名覆盖绕过 scope gate。

## 3. Live evaluation

以
`/root/autodl-tmp/conda/envs/RoboTwin/bin/python scripts/manipeval_agent.py --help`
为当前参数真值。live 命令应显式传入：

- 原始 Query；
- policy 与 checkpoint；
- seed/N 和最大轮数；
- evaluation id；
- live/ACT 授权开关。

`--auto-route` live 的默认规划器应为 Plan Agent；不要在这条生产链重新启用 legacy
task-specific planner、whole-round recovery 或 fault injection。generic TaskGen 的
所有失败阶段共用至多一次局部 repair；checker fixture 失败可保持已验证 scene、只修 checker；
ToolGen 最多一次局部修复。执行后至少核对：

1. Query contract 的 `control_requirement` 与首轮 Proposal；
2. scene/checker、render 和 gate；
3. 实际 rollout seed、video、telemetry；
4. Rule/VQA 与 Aggregate 是否消费同一 episode；
5. 下一轮是否由上轮 evidence 产生；
6. stop 是 evidence sufficient、unsupported 还是 budget exhausted；
7. Answer 是否列出 N、未覆盖候选和限制。

open-world round 先由 Query contract 决定是否需要 neutral official control；仅 Query
需要对照时才执行。随后用 Proposal 做 Task exact lookup，miss 才调用
provider。scene/checker/tool need 是独立的：Tool-only Query 不启动 TaskGen；
scene/checker need 才走 exact reuse 或 generate。无论 Task 生成还是复用，都必须在当前
seed 重跑 state/checker、render、VLM 与 expert gate；从同一采样 Pose 派生新 actor
时必须复制位置并新建 `sapien.Pose`。ACT
完成后再从实际 telemetry schema 生成 Tool request，避免为尚未出现的 actor/signal
预写 metric。生成 checker 与 official success 必须并列记录；二者冲突或不可比较时，
Answer 不得把实验语义写成 official benchmark 结论。

若 measurement need 明确要求已声明 semantic trace 的 final/terminal `x/y/z/height`
单信号分量，ToolGen 必须生成 `terminal_signal_component`；若要求 target-vs-distractor
等两信号终态差，则必须生成 `terminal_signal_difference`。两者都不能改用 event time
或 distance 绕过原始问题。动态 VQA 找不到 exact rule 时，先选择该 task 的已审查问题，再退回
`run_local.tracked_object_visible_state_change`；不得继承其他任务的 block/hammer/bell
问题。VQA 与 Rule/checker 冲突时保留 `numeric_consistency=conflict` 和
`evidence_conflict=true`，不能由较高 VLM confidence 覆盖。

同一 evaluation 内的后续 Query 可 exact reuse 其 run-local Tool；跨独立 evaluation
只接受显式 approved 的 reviewed registry 条目。两类检索命中后都必须在当前 episode
上重复执行并做确定性/oracle 校验。普通生成不会自动晋升成跨 evaluation 的可信 Tool。

## 4. 查看证据

最近一次公开索引见
`docs/evidence/current/evidence_bundle_manifest.json`。原始 bundle 根目录由
manifest 的 `source_server_path` 记录。`README.md` 是语义阅读索引；
`evidence_bundle_manifest.json` 只保存 bundle-relative 文件路径、大小与 SHA-256，
不重复嵌入 rounds。阅读顺序：

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

已完成 rollout 的 Tool/VQA 语义审计只使用 `experiments/paper/` 下的 append-only
replay 入口；以各脚本 `--help` 为参数真值。replay 必须记录
`act_rollouts_started=0`，不得覆盖源 manifest/summary/answer，也不能把源
inconclusive 倒改成在线成功。单纯修正 acceptance projection 同样必须写入独立
post-run artifact，并明确它没有增加 policy sample。

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

若 SmolVLA checkpoint 没有可审计的训练 task manifest，必须把 scope 声明为 unknown。
此时 unbound LIBERO 请求应在 rollout 前拒绝；显式 `--bound-task-name` 只授权该次协议，
不证明 checkpoint 的广泛 task scope。direct chain 必须沿用已验证的
顺序 `set_seed → make_env → make_policy → processors → rollout`；paired custom
rollout 在构造 custom env 前恢复捕获的 RNG state，避免把初始化顺序差异误判为 BDDL
效应。

## 6. 测试原则

- 纯 schema、Plan Agent、fixture 和 registry 单测可在服务器快速执行。
- 修改主链后运行相关测试，再运行一次 plan-only。
- 触及 TaskGen/ToolGen、simulator adapter 或 rollout 绑定时，追加一个最小 live smoke。
- 不以固定测试数量为目标；被删除的旧链路测试随实现一起删除。
- 大规模 N、更多 policy 或真实消融须另行预注册，不混入日常 smoke。

包含 LIBERO 测试时必须先加载专用环境路径；否则 upstream `libero` 会交互询问
dataset 路径，不能被解释为代码回归：

```bash
MEA_REPO=/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime
cd "$MEA_REPO"
. /root/autodl-tmp/envs/mea-libero/etc/conda/activate.d/10_mea_libero_paths.sh
/root/autodl-tmp/envs/mea-libero/bin/python -m pytest -q tests/manipeval
```

非交互 shell 若没有执行 conda activation，可一次性安装同一配置作为 upstream 默认值：

```bash
mkdir -p /root/.libero
install -m 600 \
  /root/autodl-tmp/cache/libero/config/config.yaml \
  /root/.libero/config.yaml
```

该文件只记录服务器路径，不含账号或 key；它用于避免非交互测试被 LIBERO 配置询问
阻塞。

最新运行的命令边界、样本数、验收状态和限制统一从
[当前证据](evidence/current/README.md)读取；论文主张的累计覆盖和下一项优先缺口见
[论文 claim 与 gap](paper_claim_gap_zh.md)。运行指南不固化某个旗舰版本的动态数值。
