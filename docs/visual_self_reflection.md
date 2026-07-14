# Visual Self-Reflection 原型

## 论文对应关系

论文 TaskGen pipeline 的 Visual Self-Reflection 使用 simulator 渲染任务首帧，与 task
proposal 的 intended vision 比较；发现不可接受偏差后，向代码生成阶段返回 diagnosis
和 revision suggestions。论文将它与 RAG、`README.Agent` 并列为提高 TaskGen 稳定性的
三个关键增强模块。

本实现沿用这一结构，但保持当前 MEA 原型的窄边界：只返修完整 `load_actors()`，不生成
新的 `check_success()`，规范 task identity、checkpoint、官方 success logic 和 protected
files 不变。

## 状态机

```text
initial complete load_actors()
  -> attempt 0 setup-only render
  -> rule check
  -> UIUI Vision structured observation
       aligned / observed_color / unexpected_changes
       diagnosis / suggestions / confidence
  -> passed?
       yes -> expert gate -> ACT
       no  -> repair budget available?
                yes -> Repair Agent generates complete load_actors()
                       -> AST + protected-file validation
                       -> atomic install
                       -> next render/Vision attempt
                no  -> fail run; do not execute expert/ACT
```

默认 `max_reflections=2`，即最多 1 个 initial attempt 加 2 次 repair。允许范围为 0–5，
外层 `scripts/manipeval_agent.py` 和内层 `scripts/manipeval_taskgen.py` 都支持
`--max-reflections`。

## Repair Agent 输入与输出

Repair Agent 接收：

- 原始自然语言请求；
- 已通过 validator 的 `VariantSpec`；
- 当前完整 `load_actors()`；
- 当前 attempt 的 scene/probe 状态；
- Vision diagnosis、unexpected changes 和 suggestions。

它必须返回一个完整 `load_actors()`，不能返回 diff、局部片段或 `super()` 委托。候选
method 安装前重新执行原有 AST contract、literal color、required calls、AST size 和
protected-file hash 检查；编译成功后才原子替换 runtime task module。

## 运行产物

```text
mea/generated_tasks/<run_id>/reflection/
├── summary.json
├── fixture/                         # 仅显式测试时存在
└── attempt_00/
    ├── render.png
    ├── scene.json
    ├── probe.log
    ├── vision_prompt.md
    ├── vision_response.txt
    ├── vision.json
    ├── method_before_repair.py       # 失败并返修时存在
    ├── repair_prompt.md
    ├── repair_response.txt
    ├── candidate_load_actors.py
    └── repair.json
```

`summary.json` 记录全部 attempts、使用的 repair 次数、最终 attempt 与是否通过。外层
evidence bundle 和 `evaluation_report.md` 会继续汇总 `repairs_used` 和 attempt count。

## Mocked 状态机验证

`tests/manipeval/test_visual_reflection.py` 覆盖：

- 首次 observation 失败、一次 repair 后成功；
- repair budget 用尽后明确失败；
- `aligned` 与 expected color 必须同时满足；
- 中文/英文颜色别名 normalization。

加入本功能及 feedback consistency guard 后，provider、TaskGen、Retrieval、Feedback、
Plan Agent 和 reflection 共 18 个 unit tests。

## 真实 fault-injection 验证

提供两个只能显式启用、只作用于 runtime run directory 的测试 fixture，不修改 Git
tracked task 或上游文件：

```bash
python scripts/manipeval_taskgen.py \
  --request '把 beat_block_hammer 的红色方块改成蓝色，其他行为保持不变。' \
  --probe --vision-check --expert \
  --max-reflections 2 \
  --reflection-fixture wrong_color
```

### 颜色 mismatch：成功触发返修

- run id：`run_20260714_161109_visual_reflection_color`；
- attempt 0 真实 render 为红色方块；
- Vision：`aligned=false`、`observed_color=red`、confidence 0.9；
- diagnosis：颜色与蓝色 `VariantSpec` 不一致；
- Repair Agent usage：1238 prompt、379 completion、1617 total tokens；
- repair 生成完整 method，AST 254 nodes、无 `super()`、protected hashes 不变；
- 修订后 literal color 为 `[0.0, 0.2, 1.0]`；
- attempt 1 真实 render 为蓝色；
- Vision：`aligned=true`、`observed_color=blue`、confidence 1.0；
- `repairs_used=1`，expert planning 与 `check_success()` 均通过。

### 尺寸 mismatch：发现 VQA 能力边界

- run id：`run_20260714_155841_visual_reflection_fixture`；
- fixture 把 block half-size 从 0.025 m 放大至 0.06 m；
- 真实图像中方块明显变大，但 Vision 仍回答尺寸正常、`aligned=true`；
- 因而 `repairs_used=0`，没有触发 repair。

这不是状态机错误，而是单张 RGB 图缺乏绝对尺度约束。尺寸、位置、接触、距离等可由
simulator state 精确获得的信号应交给 rule-based Evaluation Toolkit；颜色、材质、
杂物和明显视觉异常更适合 VQA。这个结果直接决定下一阶段的工具分工。

## 正常 Plan Agent + ACT 回归

- evaluation id：`eval_20260714_161708_visual_reflection`；
- child run：`run_20260714_161708_visual_reflection_round_1`；
- 正常生成的蓝色场景在 attempt 0 通过，`repairs_used=0`、attempt count 1；
- Vision：`blue`、`aligned=true`、confidence 1.0、无 unexpected changes；
- expert、ACT pipeline 均通过；
- ACT 从 `2026-07-14 16:19:51 +08:00` 运行至 `16:21:37 +08:00`；
- 视频为 H264、320x240、10 FPS、40 seconds；
- policy result 为 0/1，外层 `pipeline_passed=true`、`policy_success=0.0`；
- outer evidence 和统一报告均记录 reflection passed、0 repair、1 attempt。

首次 Feedback response 曾把 pipeline success 误写为“任务成功完成”，同时又报告
`policy_success=0.0`。因此额外加入 evidence consistency validator：最多要求模型重写
一次；若仍矛盾，程序保留不冲突的 findings，并强制生成
“pipeline 通过、policy 未完成任务”的结论，同时写入
`deterministic_correction=true`。本次最终报告已校正为一致结论。

## 下一阶段推荐路线

按“方法重要性 × 当前容易集成”排序：

1. **Trusted Evaluation Toolkit + Tool Router**：先加入 block pose/size、contact、
   success、steps、duration 等 rule tools，并从视频抽取首/中/末关键帧做 VQA。它能补上
   本轮暴露的尺度盲点，也是论文 Execution/ToolGen 的核心。
2. **Multi-round Plan Agent**：把每轮 evidence 和 tools/VQA observations 送回 Plan
   Agent，支持 `appearance -> position -> summary` 的两到三轮自适应评估和 stop rule。
3. **Episode JSONL + Aggregator**：每 episode 记录 seed、成功、steps、耗时、视频和
   exception，再按 sub-aspect 聚合；这为多轮结论提供可计算证据。
4. **Reuse-first Variant Registry**：登记已通过 AST/render/Vision/expert 的 task variant，
   Plan/TaskGen 先 reuse，找不到再 codegen，对应论文的 reuse-first 原则。
5. **Asset List + Documentation RAG**：在已有 50-task catalog 上增加 asset inventory、
   API/README summaries，约束 proposal 和 CodeGen 不引用不存在的资源。
6. **受限 ToolGen code generation**：在 trusted tools 不够时才生成新 metric，并执行
   AST、simulator smoke test 和注册；其风险与复杂度高于前五项，适合后置。

最推荐下一步先做第 1 项，再做第 2 项。没有结构化 tools，Multi-round Plan Agent 只能
围绕 success/Vision 做浅层推理；先补 signal layer，后续 adaptive planning 才有意义。
