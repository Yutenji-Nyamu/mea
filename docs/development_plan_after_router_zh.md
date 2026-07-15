# Auto Tool Router 之后的开发计划

## 1. 当前方法位置

当前最小闭环已经覆盖：

```text
User Query
→ Plan Agent / ToolSpec
→ Auto Tool Router
   ├─ reuse Trusted Tool
   └─ force_codegen ToolGen
→ trajectory validation
→ Tool observation
→ Feedback Agent / evaluation_report.md
```

它已经对应论文中 Proposal、Tool Retrieval/Generation、Execution 与
Feedback 的一条纵向通路。接下来不应继续堆叠同义的单 episode Tool，
而应优先补齐跨 episode 结论、执行视觉证据、生成 Tool 的复用机制，以及
跨任务可迁移性。

## 2. P0：跨 episode Aggregate Toolkit

### 目标

把多个 episode 的 ToolResult 聚合为可直接回答用户的问题，而不是让
Feedback Agent 自己从若干 JSON 中做数值计算。

最小通路：

```text
episode ToolResult[]
→ deterministic Aggregate Tool
→ aggregate_result.json
→ Feedback Agent
```

第一批 Aggregate Tool 建议包括：

- success count/rate；
- boolean Tool 的 true count/rate；
- numeric Tool 的 mean、median、min、max、standard deviation；
- valid/missing/invalid episode count；
- 按 seed、round、variant、policy/expert 分组；
- 每个统计量保留对应 episode 与 evidence step 链接。

蓝色方块的最小验收可以是：对 3–5 个 episode 聚合 contact rate、pickup
rate、pickup-to-contact duration 与 official success rate。数学统计由可信
代码完成，GPT 只解释结果。缺失值、contact-before-pickup 与 expert
validation 不能被混入 policy 均值。

### 论文差距

论文最终需要基于多次执行给出 generalization 结论。当前单 episode
Feedback 只能证明某一次发生了什么，Aggregate 才能形成 round-level 和
evaluation-level observation，供下一轮 Plan Agent 判断继续、换 sub-aspect
或停止。

## 3. P1：Execution VQA

### 目标

补齐数值 Tool 不擅长判断的执行视觉现象，并与已有 scene render 的 Visual
Self-Reflection 分开：

- scene VQA 检查生成场景是否符合描述；
- execution VQA 检查 policy rollout 中实际发生了什么。

最小实现：

1. 从 video 与 event timeline 选择 initial、pickup 前后、first contact
   前后、final 共 4–8 个关键帧；
2. 把 reference scene、关键帧拼图和结构化 ToolResult 一起交给 Vision
   provider；
3. 只允许输出固定 schema：现象、置信度、对应 frame、是否与 numeric
   evidence 冲突；
4. 冲突时不让 VQA 覆盖 simulator Tool，而是写入
   `evidence_conflict`，交给 Feedback/Plan Agent 决定是否追加测试。

第一条测试可判断：方块是否确为蓝色、锤子是否被明显抬起、最终方块是否
出现可见位移。尺寸、精确距离、接触与冲量仍以 simulator Tool 为准。

### 论文差距

这对应论文 Execution Stage 返回 observation 的视觉分支，也使最终反馈
不只依赖场景生成阶段的一张 render 图。

## 4. P2：run-local Tool 复用与 Promote

### 目标

当前生成 Tool 及 registration 与一个 run 绑定。下一步应让同一 evaluation
中的后续 round 可以直接复用已经验证过的 Tool，并建立受控的长期提升路径。

建议分两层：

- **run-local reuse**：同一 evaluation 内，只要 Tool contract、source hash、
  telemetry schema hash 一致，Router 直接复用，不再次调用 GPT；
- **promote**：经过多个正反例、oracle agreement、determinism 和至少一个
  真实 rollout 验证后，显式提升到 candidate catalog；再通过代码审查与测试
  后进入 Trusted Tool catalog。

每个 registration 至少记录：

- tool id、版本、代码 hash；
- ToolSpec 与 required signals；
- source examples 与 prompt hash；
- telemetry/TaskSchema compatibility；
- validation episodes、oracle、测试结果；
- scope：`run_local`、`candidate` 或 `trusted`。

第一条验收是：Round 1 force-generate pickup-to-contact Tool，Round 2 使用
相同 contract 时 Router 命中 run-local registration，GPT 调用次数为零，
结果与第一次完全一致。

### 论文差距

这补齐论文 Toolkit Retrieval 与 ToolGen 之间的持续积累关系，避免每轮都
把 Generation Stage 当作一次性代码生成。

## 5. P3：第二个 RoboTwin task 的 RAG 与 TaskSchema

### 目标

证明当前 TaskGen、Recorder、Tool Router 不是只对
`beat_block_hammer` 硬编码。

选择第二个任务时优先满足：

- actor 数量少、官方 ACT checkpoint 可用；
- success contract 明确；
- 与现有 pick/contact/place Tool 有部分可复用语义；
- 场景代码和 asset API 容易形成短小 Documentation RAG 条目。

最小迁移步骤：

1. Offline Extractor 生成 task、asset、API 知识条目；
2. 增加 TaskSchema：tracked actors、functional points、contact focus、success
   contract 与 threshold；
3. Recorder 只增加任务 adapter，不复制整套 Recorder；
4. Retriever 根据用户请求选择该任务源码和 3–6 个相关知识条目；
5. 跑通一个静态 appearance variant、expert gate、ACT 1 episode、至少三个
   Trusted Tools 与最终 Feedback。

验收重点不是新场景复杂度，而是同一条 agent/tool 调用链无需为第二任务
重写。

### 论文差距

这开始验证论文 Task Retrieval、TaskGen 和 Evaluation Toolkit 的跨任务
泛化，而不是仅在一个 case study 上增加功能。

## 6. P4：Budget 与效率控制

### 目标

让多轮评估具备可审计的停止条件和成本收益，而不是固定调用所有 GPT、VQA
和 rollout。

建议记录并控制：

- 每个 agent call 的 model、prompt/completion tokens、延迟与失败重试；
- TaskGen、ToolGen、Vision、Feedback 的 cache hit；
- expert 与 ACT rollout 的 wall time、GPU time、episode 数；
- RAG 文档数与 prompt bytes；
- 每轮新增 observation 是否改变结论；
- 总 token、Vision call、episode 与 wall-time budget。

Router 的默认节省顺序：

```text
Trusted Tool reuse
→ run-local Tool reuse
→ deterministic Aggregate
→ 只有缺少 Tool 时才 ToolGen
→ 只有视觉问题时才 Execution VQA
→ 只有证据不足时才追加 episode/round
```

第一版停止规则可以很简单：达到用户指定 episode 数；所有必需 Tool 有有效
结果；Aggregate 置信区间或结论稳定；没有 unresolved evidence conflict；
或者任一预算耗尽。停止原因必须写进 evaluation manifest 和最终报告。

### 论文差距

论文方法包含多 agent、多轮生成与执行，但工程原型必须说明何时调用、何时
复用、何时停止。Budget policy 是把方法从可运行 demo 变成可重复实验系统
所需的约束层。

## 7. 推荐实施批次

建议按以下顺序推进：

1. **Aggregate Toolkit**：纯离线、风险最低，不需要重新跑 ACT；
2. **Execution VQA**：复用现有视频、event 与 UIUI Vision provider；
3. **run-local reuse**：在同一 evaluation 内闭环，再设计 promote；
4. **第二 task**：验证 RAG、TaskSchema、Recorder adapter 与 Tool portability；
5. **Budget policy**：先补全计量，再加入 cache 和停止规则。

可以把前三项作为一个紧凑开发批次：先聚合既有 telemetry，再对关键帧做
一次 execution VQA，最后让下一轮 Router 复用本轮生成 Tool。第二任务与
Recorder `balanced_v1` 应各自独立提交，以便清楚定位跨任务或采样格式导致的
回归。
