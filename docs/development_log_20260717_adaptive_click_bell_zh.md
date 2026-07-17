# 2026-07-17：click_bell 属性自适应最小闭环

## 目标

把 `click_bell` 从固定 left/right 回归扩展为一个最小的开放查询闭环：Plan Agent 可选择
位置与官方 bell 实例两个方面，真实 ACT、Aggregate 和 Dynamic Execution VQA 证据决定下一轮
深挖、切换方面或停止。实现仍坚持受信 template 与 declarative overlay，不声称已经完成通用
3D 任务生成。

开发基线为 `d342cdb6b1fed928692f99979a786e9bebd95676`。

## 实现

- 新增 `ClickBellAdaptivePlanAgent` 与 `adaptive_properties` profile；保留原 `position_lr` 行为。
- 受信目录包含 `object_position.left_fixed/right_fixed` 和
  `object_instance.base0/base1`。模型选择方面，运行时注入精确 VariantSpec、seed、gate 与 Tool。
- object-instance overlay 先消费官方 pose 与实例随机调用，再覆盖固定 `bell_id`，保持同 seed
  的 RNG 顺序；base0/base1 都使用官方 asset 和现有 `click_bell` ACT checkpoint。
- probe 从 TaskSchema 白名单记录 `task_attributes.bell_id`；位置以 simulator tracked actor XY、
  实例以 simulator task attribute 为真值，VQA 只做视觉可见性与物理合理性检查。
- round observation 新增 `controlled_axis`、variant samples/metrics、`observed_bell_ids` 和
  `bell_instance_id`。
- evidence policy 检查 pipeline、Aggregate input issues/policy cohort coverage、policy success 和
  VQA conflict，并导出唯一 required action/transition/aspect。模型给出相反方向会被拒绝重试。
- 非 JSON proposal/decision 现在会真正重试；规划开始即写 provisional manifest，失败也保留
  可诊断状态，不再留下没有 manifest 的孤儿目录。

## 真实验证

### 官方实例正交 probe

同一 seed `100401` 分别固定 base0 与 base1，均通过 setup、render、rule 和 official expert：

| variant | simulator bell_id | simulator XY | contact Z |
| --- | ---: | --- | ---: |
| base0 | 0 | `[0.2234336883, -0.0804080516]` | `0.7790379971` |
| base1 | 1 | `[0.2234336883, -0.0804080516]` | `0.7667903972` |

相同 XY 与不同 ID/contact height 证明这一轴是“官方实例外观与几何”，不是纯颜色替换。

### 三轮 ACT + VQA smoke

artifact：`mea/evaluation_runs/eval_20260717_click_bell_adaptive_live_2/`

| round | trusted template | ACT | pipeline | Aggregate | Execution VQA |
| ---: | --- | ---: | --- | --- | --- |
| 1 | `object_position.left_fixed` | 0/1 | passed | passed | passed, no conflict |
| 2 | `object_position.right_fixed` | 1/1 | passed | passed | passed, no conflict |
| 3 | `object_instance.base0` | 1/1 | passed | passed | passed, no conflict |

第一轮的有效 policy failure 强制同方面 drill-down；第二轮成功后切换到尚未覆盖的实例方面；
第三轮因 3-round 预算耗尽停止。总 ACT 为 `2/3`，expert solvability gate 为 `3/3`。端到端
命令墙钟约 662 秒。每个 variant 只有一个 seed/episode，因此这些数字只证明调用链与自适应
控制成立，不是泛化结论。

开发中曾在复审发现 evidence 仅被展示、未硬约束方向，以及畸形 scene spec 可空验证通过；
第一次 live 在尚未产生 rollout 前被主动中止并保留为 failed artifact。修复状态机、严格 spec、
反向测试和 JSON retry 后，才使用新 evaluation ID 完成上表实验。

## 验证与边界

- 定向测试：`test_click_bell_generated.py` 17 项、generic schema 11 项、Agent evidence 8 项通过。
- 完整 `tests/manipeval` 共 216 项通过；`git diff --check`、compileall、shell syntax 与 CLI help
  均通过。
- 没有下载新 checkpoint，也没有让 checkpoint、数据集或其他大文件经过本地 Windows/Codex
  工作区。
- 当前只有两个手工属性、四个受信 template；base1 尚未进入本次 ACT smoke。
- generated multi-round 尚未接入论文协议 runner；同 seed 跨 variant 的统计身份应先改为
  `(variant_id, seed)`，否则不能作为论文式有效分母。
