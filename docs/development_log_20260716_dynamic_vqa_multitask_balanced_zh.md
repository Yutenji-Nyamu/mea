# 开发记录：动态 Execution VQA、第二任务与 `balanced_v1`

日期：2026-07-16

## 1. 本批目标与边界

本批完成三个彼此衔接的最小通路：

1. Execution VQA 的视觉问题不再固定为 BeatBlockHammer 三问，而是根据当前
   `task_name`、Plan `template_id/sub_aspect` 与 `ToolSpec.metric` 从受限目录中选择；
2. 增加第二个 RoboTwin 任务 `click_bell`，用通用 `TaskSchema` 驱动 official
   expert、Recorder、Trusted Tool、Aggregate 与 Feedback；
3. 实现 allowlisted `balanced_v1` Recorder，在保留旧产物的同时增加 50 Hz
   dynamics stream。

根 `README.md`、官方 `policy/ACT/eval.sh` 与 RoboTwin 官方任务源码均未修改。
`click_bell` 在当前服务器没有 ACT checkpoint，因此真实验收是 official expert，
不能表述为 `click_bell` ACT policy 已通过。

## 2. 当前参数化调用链

```text
用户 request + --task-name/--task-module/seed/episode/profile
  |
  +-- beat_block_hammer
  |     Plan Agent
  |       -> VariantSpec / ToolSpec
  |       -> TaskGen + Visual Self-Reflection
  |       -> expert gate + ACT rollout
  |
  +-- 其他已有 TaskSchema 的任务（本批为 click_bell）
        OfficialTaskPlanAgent（确定性单轮）
          -> official passthrough（不调用 GPT、不生成场景代码）
          -> official expert probe

两条路径随后汇合：
  Recorder -> Trusted Tools / Auto Tool Router -> Aggregate
           -> Dynamic Execution VQA（有 ACT video 时执行；否则保留 query 并标 skipped）
           -> Feedback Agent -> evaluation_report.md / history database
```

第二任务的完整入口示例：

```bash
export UIUI_API_KEY='...'
python scripts/manipeval_agent.py \
  --request '评估官方 click_bell 任务' \
  --task-name click_bell \
  --task-module envs.click_bell \
  --start-seed 100000 \
  --num-episodes 2 \
  --telemetry-profile balanced_v1 \
  --model-profile economy
```

只检查确定性计划、不调用 UIUI：

```bash
python scripts/manipeval_agent.py \
  --request '评估官方 click_bell 任务' \
  --task-name click_bell \
  --task-module envs.click_bell \
  --plan-only
```

## 3. 动态 Execution VQA

### 3.1 实现

`mea/execution_vqa/query.py` 保存受限的视觉现象目录与选择规则。输入是可信的
task/template/sub-aspect/metric 标识，不执行 ToolSpec 中的自由文本问题。输出固定为：

- `profile=dynamic_v1`；
- 被选择的 `phenomenon_ids` 与选择原因；
- 每个现象的受限问题、视觉范围和 numeric authority；
- 固定 response schema。

当前例子：

- `click_bell + official_check_success` 选择 `bell_visibly_pressed`；
- `beat_block_hammer + pickup_to_first_contact_time` 选择
  `hammer_visibly_lifted` 与 `block_visibly_displaced`；
- 无匹配规则时退回原有 BeatBlockHammer 三问，保持向后兼容。

如果没有完成的 ACT telemetry/video，Agent 仍写
`execution_vqa_query.json`，并在 observation 中明确记录 `skipped` 及原因；它不会拿
expert initial render 冒充 rollout VQA。

### 3.2 真实 Vision 验证

复用了既有蓝色方块 ACT episode，并以
`performance.pickup_to_contact_timing / pickup_to_first_contact_time` 构造问题。
`gpt-5.6-luna` 实际检查了 initial、pickup 前后与 final 关键帧：

- 锤子明显被抬起：`true`，confidence `0.99`；
- 蓝色方块出现可见位移：`true`，confidence `0.99`；
- `numeric_consistency=consistent`；
- `evidence_conflict=false`；
- token usage：prompt 6621、completion 344、total 6965。

服务器产物：

```text
/root/autodl-tmp/setup_logs/dynamic_execution_vqa_live_20260716/
```

## 4. 第二任务与通用 `TaskSchema`

### 4.1 为什么选择 `click_bell`

它只需一个 tracked actor `self.bell`，官方 `check_success()` 明确，且服务器已有
`050_bell` asset，适合作为最低成本的跨任务验证。新增 schema 声明：

- `bell_position`、`bell_contact_position`；
- 左右 TCP；
- tracked actor 的 functional/contact point；
- generic success Tool profile。

Schema validator 会检查 actor、point、semantic field source 与 role 引用；
`TrajectoryView` 按 episode 内的 schema 校验，不再把 hammer/block 当作通用前提。
Tool Retriever 对所有 schema-backed task 提供 `official_check_success` 与
`time_to_success`，并拒绝把 BBH-only Tool 用到 `click_bell`。

### 4.2 official passthrough

非 BBH 的当前最小路线不让 GPT 伪造变式：`OfficialTaskPlanAgent` 生成确定性单轮
计划，`mea/taskgen/official.py` 校验官方 module/class/schema 后生成兼容 run
manifest，并记录：

```text
generation_kind=official_passthrough
provider_called=false
```

这证明 Agent/Recorder/Toolkit/Feedback 已可由任务参数驱动；它尚不等于任意任务的
场景 code generation 已经通用化。

### 4.3 真实 `click_bell` 验证

Evaluation：`eval_20260716_click_bell_balanced_v1`。

- seeds：100000、100001；
- official expert：2/2 setup、render、rule、expert gate 均通过；
- `official_check_success`：2/2 true，rate 1.0；
- `time_to_success`：3.372 s、3.752 s；mean/median 3.562 s，population stddev
  0.190 s；
- Auto Tool Router：`reuse`，未调用 ToolGen GPT；
- TaskGen：official passthrough，未调用 GPT；
- Feedback：`gpt-5.6-luna`；
- Execution VQA：动态问题为 `bell_visibly_pressed`，因 expert probe 没有 ACT
  rollout video 而按设计标记 `skipped`。

完整日志：

```text
/root/autodl-tmp/setup_logs/eval_20260716_click_bell_balanced_v1.log
/root/autodl-tmp/setup_logs/click_bell_balanced_audit_20260716.json
```

## 5. `balanced_v1` Recorder

### 5.1 多频率数据流

`mea/toolkit/profiles.py` 只允许可信 profile：`balanced_v1` 与 `legacy_v1`。
`balanced_v1` 是 additive migration：

- 250 Hz：`semantic_trace.npz` 与 contact/success event monitor；
- 50 Hz（每 5 个 physics steps）：新增 `dynamics_trace.npz`；
- policy boundary：保留 `states.csv`；
- 10 FPS：继续使用现有 H264 video；
- 每个 episode：保存 `telemetry_profile.json`、profile SHA-256 与 stream metadata。

Dynamics 使用 typed NumPy arrays，包含双臂 qpos/qvel、EE/TCP、gripper，以及
TaskSchema tracked actor 的 pose、linear/angular velocity、functional/contact
pose。initial 与 final 强制采样，即使 final 不是 5 的倍数；contact monitor 不随
dynamics 降采样。

### 5.2 真实大小与采样验收

`click_bell` 两个 episode：

| seed | physics rows | dynamics rows | final sampling | telemetry bytes |
| ---: | ---: | ---: | --- | ---: |
| 100000 | 1044 | 210 | `...,1035,1040,1043` | 173,264 |
| 100001 | 1158 | 233 | `...,1150,1155,1157` | 186,708 |

合计 359,972 B。两集都满足 `0,5,10,...,final`，每个 dynamics artifact 包含
20 个 typed arrays。

同 seed 100000 的独立 `legacy_v1` 运行占 45,070 B；`balanced_v1` 为
173,264 B，其中新增 dynamics 为 120,893 B，约为 legacy 的 3.84 倍。两次独立
expert 运行的官方 success、首次 success step（843）与 Trusted Tool 结果相同，
但最终 physics steps 和 semantic arrays 并非逐元素相同。因此这里只确认 outcome/
metric 兼容，不宣称两个独立 rollout 是 bitwise-identical。

对比证据：

```text
/root/autodl-tmp/setup_logs/click_bell_profile_comparison_20260716.json
```

## 6. 测试、变更审计与剩余限制

服务器使用 RoboTwin 环境完成：

- Python test suite：143/143 passed；
- 修改过的 Python 文件 syntax compilation 通过；
- `bash -n policy/ACT/eval_mea.sh` 通过；
- `git diff --check` 通过。

最终服务器测试日志：

```text
/root/autodl-tmp/setup_logs/three_features_final_tests_20260716.log
```

本批仍有明确限制：

1. 当前只有 `beat_block_hammer` 与 `click_bell` 两个 TaskSchema；
2. 非 BBH 路线当前是 official expert，不是通用 VariantSpec/TaskGen codegen；
3. `click_bell` 缺 ACT checkpoint，未做 ACT policy 回归；
4. expert probe 不生成 rollout video，因此 Execution VQA 只能审计地跳过；
5. Recorder 在 `setup_demo()` 完成后 attach，尚未覆盖 setup/stabilization；
6. event-triggered depth/segmentation 与 250 Hz joint/torque profile 尚未实现；
7. 50 Hz dynamics 当前只覆盖 TaskSchema selected actors，而非全场 actor。

## 7. 下一步建议

按论文差距与实现成本排序：

1. 给 official expert 路径增加低成本 rollout video/事件关键帧，使第二任务也能执行
   Dynamic Execution VQA；
2. 用 Offline Extractor 半自动生成第三个 TaskSchema，验证 schema onboarding
   不依赖手写任务分支；
3. 增加读取 `dynamics_trace.npz` 的通用 dynamics Tools，例如 actor speed、TCP
   path、joint velocity peak，并接入 Auto Tool Router；
4. 再扩展非 BBH 的受限 VariantSpec/template，而不是直接开放任意任务代码生成。
