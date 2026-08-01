# Batch33：开放跨任务方法与多任务 policy pilot

本目录是冷结果索引，不是当前接受旗舰。动态 claim 判断仍以
[`docs/paper_claim_gap_zh.md`](../../../../docs/paper_claim_gap_zh.md) 为准。所有 provider、测试、
policy inference 与仿真均在 AutoDL server 执行；Windows 只保存代码与紧凑文档。

## 1. PressStapler 开放方法链

| 运行 | 结果 | 诚实边界 |
| --- | --- | --- |
| `eval_20260801_batch33_press_stapler_open_live_v3` | 三次 SmolVLA policy episode 均成功；evidence 依次产生 `+0.03 m`、`+0.06 m` scene；Python Tool 获得 live finite 值，第三轮命中同一 registration 的 `run_local_reuse` | `N=3` 后因预算停止，`evidence_sufficient=false`；复用 official checker，不是完整 scene+checker 新正例 |
| `eval_20260801_batch33_press_stapler_open_live_v4` | 前三次 policy episode 与 v3 同类成功；Plan Agent 随后提出 `+0.12 m` 候选 | TaskGen expert gate 两次均报 `target_pose cannot be None`；第四次 policy **未启动**。这是不可执行候选，不是 policy failure |
| `eval_20260801_batch33_press_stapler_open_live_v5` | 四次 policy episode 均成功；第四个 vertical Proposal 在 policy 前形成 typed `N=0` planning evidence，Plan Agent 随即切换为 `x +0.20 m` lateral Proposal并成功执行；后两次 metric 均为 `run_local_reuse` | 最终因 5 个 method-step 外部上限停止，`N=4`、单 seed、结论仍为 inconclusive；source answer 将 planning rejection 过度投影成 pipeline invalid，当前代码已修正并由缓存 fixture 验证，尚未重跑 policy |

v5 已真实证明最终 expert-gate 拒绝可表示为 typed、`N=0` planning evidence，且下一 method
step 会读取诊断后更换 Proposal。它证明的是 evidence-conditioned replanning，不是找到了 policy
弱点，也不是 Query 证据充分。

服务器原始目录：

```text
/root/autodl-tmp/mea-worktrees/evidence-refinement-runtime/mea/evaluation_runs/
  eval_20260801_batch33_press_stapler_open_live_v3
  eval_20260801_batch33_press_stapler_open_live_v4
  eval_20260801_batch33_press_stapler_open_live_v5
```

## 2. SmolVLA official breadth N=1

本批固定顺序新增五个 official task，统一 `seed=1000`：

| task | official success | actions / limit | wall time (s) |
| --- | ---: | ---: | ---: |
| `stamp_seal` | false | 400 / 400 | 57.59 |
| `place_a2b_left` | false | 400 / 400 | 55.57 |
| `place_object_stand` | false | 400 / 400 | 56.86 |
| `move_playingcard_away` | true | 124 / 400 | 39.79 |
| `place_empty_cup` | true | 393 / 500 | 70.00 |

新增结果为 `2/5`。连同既有明确 outcome，目前累计覆盖 13 个 official task，为 `8/13`
成功；另有一次 `open_laptop` pipeline/checker error，不计作 policy outcome。这些数字只衡量
official rollout coverage，不代表 13 个任务均跑通 MEA TaskGen/ToolGen/Planner。

服务器原始目录：

```text
/root/autodl-tmp/tmp/eval_20260801_batch33_smolvla_breadth_n1
```

## 3. Hy-VLA official pilot

- official source revision：`8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9`
- checkpoint revision：`bd7bba6f5934ad62293a2a34f74760c6a3ef2ff8`
- offline official-wrapper validation：有限 `(16,)` action，RTX 4090 峰值 CUDA allocation
  约 `9.81 GB`
- `press_stapler / demo_clean / seed=10000` official N=1：success，24/400 actions，
  rollout 41.15 s（不含 164.76 s 模型加载）

服务器原始目录：

```text
/root/autodl-tmp/tmp/batch33_hyvla_validation.json
/root/autodl-tmp/tmp/batch33_hyvla_press_stapler_seed10000
```

完整部署命令、网络问题、版本与 hash 见
[`experiments/paper/robotwin_hyvla/deployment_ledger.md`](../../robotwin_hyvla/deployment_ledger.md)。
该 pilot 只证明第二个多任务 policy adapter 可运行；它不是 MEA round、50-task success、
sample-efficiency 或 policy-ranking 证据。

生产 MEA 接入另有三个诚实负结果：

| run | policy rollout | stop reason |
| --- | ---: | --- |
| **eval_20260801_batch33_hyvla_plan_agent_control_v1** | 0 | schema-less binding 缺少正的 physics timestep；随后在 73d43ed 补齐 runtime metadata |
| **eval_20260801_batch33_hyvla_plan_agent_control_v2** | 0 | provider typed needs 正确；schema-less resolver 仍错误依赖旧 aspect 菜单，admission 拒绝 official candidate；当前代码已修 |
| **eval_20260801_batch33_hyvla_plan_agent_control_v3** | 0 | broad weakness Query 在单轮上限内得到 unsupported_candidate_domain |

三次均在 policy 连接前停止，不能计作 Hy-VLA task failure，也没有形成完整 MEA round。
准确错误、model-load 时间和已采取的修复见冷流水账第 6 节。
