# Batch35：通用方法跨任务矩阵与 schema-free TaskGen 补充验收

本目录是冷结果索引。原始 evaluation、视频、render、telemetry 与 provider 记录只保留在
AutoDL；这里仅保存冻结配置、只读摘要和结论边界，不替换当前 v18 旗舰证据。

## 1. 冻结范围

- 方法入口：生产 `Plan Agent → RoundExecutor → TaskGen/ToolGen → Aggregate → Answer`。
- policy：SmolVLA RoboTwin shared checkpoint。
- Query：不给 aspect、template、scene edit、checker 或 metric；每轮依据上一轮 evidence 选下一 sub-aspect。
- 矩阵：5 个任务，seed 1000，每轮 N=1；冻结输入见
  [`batch_config.json`](batch_config.json)。
- 分析：[`generic_method_matrix.py`](../../generic_method_matrix.py) 只读已有 bundle，不启动 rollout。

## 2. 五任务矩阵

服务器原始根目录：
`/root/autodl-tmp/tmp/batch35_generic_method_matrix_9cb13d5`。

| task | rollout | 方法状态 | 主要事实 |
| --- | ---: | --- | --- |
| `press_stapler` | 3 | completed, inconclusive | official + 两个 generated round 均成功；evidence 连续细化位移，第三轮 exact Tool reuse；预算停止 |
| `place_bread_basket` | 1 | completed, inconclusive | official policy negative；不是方法失败 |
| `put_bottles_dustbin` | 1 | completed, inconclusive | official policy negative；不是方法失败 |
| `place_empty_cup` | 1 | completed, inconclusive | official policy negative；不是方法失败 |
| `grab_roller` | 1 | method-system failure | official 成功；随后 TaskGen expert hook 报 `target_pose cannot be None`，generated rollout 未启动 |

合计：5 个 evaluation、7 次 policy rollout、2 个 generated round、3 个 official
policy negative、1 个 TaskGen materialization failure。逐项机器可读结果见
[`summary.json`](summary.json)。这些状态分别表示方法完成度、policy outcome 与 Answer
充分性，不能压成一个“成功率”。

## 3. schema-free scene + checker 补充验收

冻结矩阵后，用另一个不指定具体 aspect/template 的 broad Query 在 `press_stapler`
做独立补充；完整 Query 与结果见
[`supplemental_flagship.json`](supplemental_flagship.json)。

- v1：1 次 official control 成功；provider 两次生成语义正确的
  `bool(official_success and relation)`，但旧 AST validator 误拒，未启动 generated rollout。
- 代码修复后，两个缓存 response 均以 0 rollout 通过；这只验证修复，不回写 v1。
- v2：2 次 rollout。control 成功后，Plan Agent 选择订书机 x 轴 `+0.02 m` 与终端
  TCP–订书机距离关系；generic provider 首次生成 scene/checker 即通过 AST、`2/2`
  fixture、render/VLM、expert 与 preservation gate。
- generated episode 同时满足 official core 与实验 checker；provider Python Tool 从真实
  telemetry 得到 `terminal_min_tcp_stapler_distance = 0.09898103773593903 m` 并进入
  Aggregate。
- 最终 summary 请求遇到瞬时 HTTP 503；0-rollout answer-only finalization 保持三个缓存
  source hash 不变并写出 Answer。

v2 因两轮预算耗尽停止，`evidence_sufficient=false`，最终 verdict 为 `inconclusive`。
Tool 没有独立数值 oracle，也未在本次运行 exact reuse。因此它证明真实 schema-free
TaskGen 执行链，不证明主动充分停止、Tool 长期复用或策略泛化。

## 4. 修复与服务器验证

提交 `7f3e2ec1` 只修方法边界：

- 接受透明 `bool(official_core and relation)`，同时拒绝 `bool` shadowing 与弱 guard；
- 仅把明确的、零 rollout、局部 TaskGen 语义失败投影为 N=0 planning evidence；
- provider/network/IO/unclassified failure 与真实 policy failure 继续 fail closed；
- RoundExecutor 和 Plan Agent 保留真实 observation kind，不把 materialization failure 当 policy failure。

AutoDL 服务器验证：

- focused：`137 passed, 9 subtests passed`；
- default mainline：`326 passed, 21 subtests passed`；
- Windows PC 未运行测试、provider、仿真或 policy inference。

## 5. 复现与边界

矩阵摘要命令：

```bash
/root/autodl-tmp/envs/mea-libero/bin/python experiments/paper/generic_method_matrix.py \
  --repo-root /root/autodl-tmp/mea-worktrees/evidence-refinement-runtime \
  --batch-root /root/autodl-tmp/tmp/batch35_generic_method_matrix_9cb13d5 \
  --config experiments/paper/results/batch35_generic_method_matrix/batch_config.json \
  --output /root/autodl-tmp/tmp/batch35_generic_method_matrix_9cb13d5/summary_v2.json
```

本批 10 次 rollout 恰好达到授权上限：矩阵 7、v1 1、v2 2；缓存校验和 answer-only
finalization 均为 0 rollout。该批不是 preregistered benchmark 实验，不复现论文 Tables
1–2、Table 9、任务平均成功率或统计泛化结论。
