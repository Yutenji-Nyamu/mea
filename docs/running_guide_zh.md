# 运行指南

所有依赖 simulator、checkpoint、GPU 或 provider 的验证在 canonical AutoDL
`/root/autodl-tmp/mea` 执行。服务器 live 必须使用
`/root/autodl-tmp/conda/envs/RoboTwin/bin/python`；base `python` 只适合不 import
SAPIEN/RoboTwin、也不执行 setup/render/expert/rollout 的纯单测。根 README 不包含
付费运行命令。

## 1. 运行前检查

```bash
cd /root/autodl-tmp/mea
git status --short
git rev-parse HEAD
/root/autodl-tmp/conda/envs/RoboTwin/bin/python -c \
  "import sys, sapien; print(sys.executable)"
nvidia-smi
```

确认要评估的 RoboTwin task、policy checkpoint、seed、最大轮数和 ACT 预算。不要在本地
Windows 下载 checkpoint 或启动 simulator。

batch28 v2 类错误按两层预检归因：base Python 的 import/setup 失败是解释器错误；在
正确解释器下，若生成代码用 `pose.p[i] += ...` 或复用同一 Pose 后场景仍与 control
相同，则是 TaskGen Pose no-op，必须复制位置并新建 `sapien.Pose`。两者都不要归因给
policy、provider 或 VLM；修正后仍须通过 state/checker、render、VLM 与 expert gate。

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

`--auto-route` 的生产默认值就是 ClaimFirst，无需重复选择 Planner。这会先调用一次
inventory-free FreeConcern，再做 official task retrieval、policy compatibility gate 和
QueryContract；`ClaimFirstInitialPlanBuilder` 随后直接写初始计划，不调用
`CatalogPlanAgent` 或任务专属 Planner。它不运行 simulator/ACT，也不是 policy 性能
证据。重点检查 `free_concern.json`、`task_resolution.json` 中的候选、决策、
provider/repair/retry 计数，以及 `manifest.planner.kind` 是否为
`claim_first_direct_initial_v1`、`task_specific_planner_used=false`。

若原始 Query 提出 catalog 外 concern，plan-only 应保存 FreeConcern、
candidate-domain resolution 与 QueryContract，并明确它没有执行 scene/checker/tool
materialization。只有 `control_requirement=required` 才绑定 neutral official control；
`not_required` 的 live 可直接从 FreeConcern 形成首个 Query-derived candidate。因此
no-control plan 的初始 `rounds` 可以为空，状态应为
`awaiting_initial_query_candidate_materialization`；这不是缺少计划，而是避免制造
假的 control 壳。plan-only 不能预测后续一定 exact reuse 或 generation，更不得写成
policy evidence。

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

以
`/root/autodl-tmp/conda/envs/RoboTwin/bin/python scripts/manipeval_agent.py --help`
为当前参数真值。live 命令应显式传入：

- 原始 Query；
- policy 与 checkpoint；
- seed/N 和最大轮数；
- evaluation id；
- live/ACT 授权开关。

`--auto-route` live 的默认 Planner 应为 ClaimFirst；不要在这条生产链重新启用 legacy
task-specific planner、whole-round recovery 或 fault injection。TaskGen/ToolGen 各允许
一次局部修复。执行后至少核对：

1. QueryContract 的 `control_requirement` 与首轮 proposal；
2. scene/checker、render 和 gate；
3. 实际 rollout seed、video、telemetry；
4. Rule/VQA 与 Aggregate 是否消费同一 episode；
5. 下一轮是否由上轮 evidence 产生；
6. stop 是 evidence sufficient、unsupported 还是 budget exhausted；
7. Answer 是否列出 N、未覆盖候选和限制。

open-world round 先由 QueryContract 决定是否需要 neutral official control；仅 Query
需要对照时才执行。随后用 `ExperimentCandidate` 做 Task exact lookup，miss 才调用
provider。scene/checker/tool need 是独立的：Tool-only Query 不启动 TaskGen；
scene/checker need 才走 exact reuse 或 generate。无论 Task 生成还是复用，都必须在当前
seed 重跑 state/checker、render、VLM 与 expert gate；从同一采样 Pose 派生新 actor
时必须复制位置并新建 `sapien.Pose`。ACT
完成后再从实际 telemetry schema 生成 Tool request，避免为尚未出现的 actor/signal
预写 metric。生成 checker 与 official success 必须并列记录；二者冲突或不可比较时，
Answer 不得把实验语义写成 official benchmark 结论。

若 measurement need 明确要求已声明 semantic trace 的 final/terminal `x/y/z/height`
分量，ToolGen 必须生成 `terminal_signal_component`；不能改用 event time 或 distance
绕过原始问题。动态 VQA 找不到 exact rule 时，先选择该 task 的已审查问题，再退回
`run_local.tracked_object_visible_state_change`；不得继承其他任务的 block/hammer/bell
问题。VQA 与 Rule/checker 冲突时保留 `numeric_consistency=conflict` 和
`evidence_conflict=true`，不能由较高 VLM confidence 覆盖。

同一 evaluation 内的后续 Query 可 exact reuse 其 run-local Tool；跨独立 evaluation
只接受显式 approved 的 reviewed registry 条目。两类检索命中后都必须在当前 episode
上重复执行并做确定性/oracle 校验。普通生成不会自动晋升成跨 evaluation 的可信 Tool。

## 4. 查看证据

最近一次公开索引见 `docs/evidence/current/manifest.json`。原始 bundle 根目录由
manifest 的 `source_roots` 记录。阅读顺序：

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

已完成 rollout 的 Tool/VQA 语义修复只能使用 `experiments/paper/` 下的追加式 replay：

```bash
/root/autodl-tmp/conda/envs/RoboTwin/bin/python \
  experiments/paper/manipeval_replay_completed_tool.py \
  --repo-root /root/autodl-tmp/mea \
  --evaluation-id <source-evaluation-id> \
  --round-id <source-round-id> \
  --repair-id <new-repair-id> \
  --tool-request <frozen-tool-request.json> \
  --execution-vqa <append-only-vqa-replay-manifest.json>

# VQA replay 还需要当前进程中的 provider credential；不得写入命令文件或仓库。
/root/autodl-tmp/conda/envs/RoboTwin/bin/python \
  experiments/paper/manipeval_execution_vqa_replay.py \
  --repo-root /root/autodl-tmp/mea \
  --evaluation-id <source-evaluation-id> \
  --round-id <source-round-id> \
  --output-dir <new-append-only-output-dir> \
  --base-url <provider-base-url>
```

两条命令都必须记录 `act_rollouts_started=0`，不得覆盖源 manifest/summary/answer。
replay 可以验证修正后的 Tool/VQA/Aggregate/Planner 行为，但不能把原在线
`budget_exhausted`/inconclusive 倒改成成功旗舰。先独立产生 VQA replay，再由 Tool
replay 的 `--execution-vqa` 显式消费并记录其 path/hash，才构成可审计的 composed
replay；不得手工拼接两个独立 summary。composed replay 若得到
`evidence_conflict`/inconclusive，就是正确的冲突保真结果，不应改成 sufficient。

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
只授权该次协议，不证明 checkpoint 的广泛 task scope。direct chain 必须沿用已验证的
顺序 `set_seed → make_env → make_policy → processors → rollout`；paired custom
rollout 在构造 custom env 前恢复捕获的 RNG state，避免把初始化顺序差异误判为 BDDL
效应。

batch27 `libero_object/task0` 已执行 official-positive/custom-negative 两回合：
2 rollouts、132.698 s，`method_chain_valid=true`、Tool exact reuse，
`query_sufficient=false`。该运行只能称 method-chain smoke；custom failure、单 seed
和四个未测 goal object 决定了它不是 robustness、效率或跨环境一致性证据。batch26
control-failed 结果保留为 parity 修复前的 fail-closed 历史负例。

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

非交互 shell 若没有执行 conda activation，可一次性安装同一配置作为 upstream 默认值：

```bash
mkdir -p /root/.libero
install -m 600 \
  /root/autodl-tmp/cache/libero/config/config.yaml \
  /root/.libero/config.yaml
```

该文件只记录服务器路径，不含账号或 key。本批全量回归正是通过此 fallback 消除了
LIBERO import prompt。

当前公开旗舰见[当前证据](evidence/current/README.md)；batch27 的
catalog-external、第五个 RoboTwin adapter 与 LIBERO 结果见
[`batch27_unified_adapter_libero`](../experiments/paper/results/batch27_unified_adapter_libero/)；
效率、ranking 与 proxy 基线仍由
[`batch26_claim_closure/summary.json`](../experiments/paper/results/batch26_claim_closure/summary.json)
索引。
