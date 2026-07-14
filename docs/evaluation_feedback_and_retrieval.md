# Task Retrieval 与评估反馈原型

## 对应论文图示的位置

这次增加两个最小但完整的能力：

1. `Task Retrieval` 对应论文 TaskGen 图中的 Retrieval Augmented 部分；
2. `Feedback Agent` 对应总架构中从 Observations 回到用户的 Evaluation Feedback。

Visual Self-Reflection 已扩展为有界返修循环：`render -> UIUI Vision -> diagnosis ->
CodeGen repair -> AST/render/Vision revalidation`，通过后才进入 expert/ACT。完整设计和
真实红色→蓝色返修证据见
[`docs/visual_self_reflection.md`](visual_self_reflection.md)。

## 完整调用链

```text
用户自然语言问题
  -> Plan Agent：生成一个受约束的 EvaluationPlan
  -> VariantSpec Agent：把 task instruction 变成场景参数
  -> TaskRetriever：扫描 envs/，给 GPT 50 个 task name
     -> GPT 选择 1–3 个 task
     -> 只读取所选源码
  -> CodeGen Agent：参考规范 task 与检索源码，生成完整 load_actors()
  -> AST / protected-file validation
  -> setup-only render / rule validation
  -> UIUI Vision alignment gate
     -> 若失败：diagnosis + suggestion -> 完整 load_actors() 返修 -> 重新验证
  -> expert solvability gate
  -> ACT 1 episode
  -> deterministic summary + evidence bundle
  -> 独立 Feedback Agent：只根据 evidence 回答用户
  -> evaluation_report.md
```

`Plan Agent`、`VariantSpec Agent`、`TaskRetriever`、`CodeGen Agent`、Vision model 和
`Feedback Agent` 是不同职责的调用。它们可以使用同一个 model id，但 prompt、输入
contract、输出 schema 和运行产物彼此分离，便于检查和后续替换。

## 最小检索实现

`discover_task_catalog()` 使用 AST 扫描 `envs/*.py`，只接受文件名与 class name 一致
的具体任务；框架文件不会进入目录。当前 RoboTwin fork 恰好得到 50 个 concrete
tasks。GPT 先只看到：

- 用户请求；
- 规范 task；
- 已验证 `VariantSpec`；
- 50 个 task name。

它返回严格 `TaskSelection`，规范 task 必须排在第一位，最多再选两项。Validator
拒绝未知名称、重复名称、缺失规范 task 或超过预算的选择。随后 CodeGen prompt 才
加载这些文件的完整源码。这样保留了论文式 retrieval 的结构，又避免把全部 task
source 无差别塞入 prompt。

蓝色方块请求的预期选择是：

- `beat_block_hammer`：权威行为、actor attribute、位置/yaw sampling 与成功逻辑；
- `blocks_ranking_rgb`：颜色 actor construction 的相邻参考。

## Evidence 与 Feedback

外层 runner 在所有执行 gate 结束后创建 `summary/evidence_bundle.json`，内容包括：

- 用户问题、plan、seed 与 episode 数；
- Task Retrieval 选择；
- `VariantSpec` 与完整 method 静态检查；
- render/Vision/expert/ACT observations；
- `pipeline_passed` 与独立的 `policy_success`；
- plan、源码、图像、视频和 result 的路径索引；
- 单 round、单 episode 等限制。

`FeedbackAgent` 只能依据这个 evidence bundle 输出严格 JSON。其规则要求：使用中文、
区分 pipeline 与 policy、不得用一个 episode 推断 generalization、必须说明限制和下一步。
最终 `evaluation_report.md` 把身份、计划、检索、执行观察、GPT 回答和 artifact index
合并成一次评估的单文件入口。

## 主要文件

- `mea/retrieval/task_library.py`：catalog、GPT selection 与 validator；
- `mea/retrieval/README.Agent.md`：retrieval prompt contract；
- `mea/feedback/prototype.py`：Feedback Agent、schema 与 report renderer；
- `mea/feedback/README.Agent.md`：evidence-only feedback contract；
- `mea/taskgen/prototype.py`：在 CodeGen 前接入 retrieval；
- `scripts/manipeval_agent.py`：创建 evidence、调用反馈并写统一报告；
- `tests/manipeval/test_task_retrieval.py`：50-task catalog 和 selection 规则；
- `tests/manipeval/test_feedback_agent.py`：反馈与报告 contract。

## 本轮验证

本轮先执行 syntax、Git diff 和 13 个 mocked unit tests，全部通过；catalog 实测为 50，
并包含 `beat_block_hammer` 与 `blocks_ranking_rgb`。

真实蓝色方块 1 episode 端到端验证：

- evaluation id：`eval_20260714_145830_retrieval_feedback`；
- child run：`run_20260714_145830_retrieval_feedback_round_1`；
- GPT retrieval：从 50 个 task 中选择 `beat_block_hammer` 与
  `blocks_ranking_rgb`；usage 为 802 prompt、40 completion、842 total tokens；
- CodeGen：生成完整 `load_actors()`，AST 254 nodes、无 `super()`，颜色为
  `[0.0, 0.2, 1.0]`；usage 为 4673 prompt、463 completion、5136 total tokens；
- render、rule、Vision、expert gates 均通过；Vision 为 `blue`、confidence 1.0、
  `unexpected_changes=[]`；
- ACT 从 `2026-07-14 14:59:29 +08:00` 运行至 `15:01:13 +08:00`，process
  return 0，policy result 为 0/1；
- Feedback Agent usage 为 1054 prompt、213 completion、1267 total tokens；
- Feedback 结论明确区分：场景和 pipeline 成功，ACT policy 本 episode 未成功；
- 外层 `pipeline_passed=true`，统一报告位于该 evaluation runtime 目录的
  `evaluation_report.md`。

这证明了“用户问题 -> 计划 -> 检索 -> 代码生成 -> 视觉/专家验证 -> policy 执行 ->
证据化反馈”的单轮闭环。一个 episode 仍不能用于判断颜色泛化能力。

## Git 交付

- DCO commit：`a8009817286a6191ee2327c6e8174ca4d6ad2023`；
- commit message：`实现任务检索与评估反馈报告`；
- scope：16 files，982 insertions、20 deletions；
- `2026-07-14 15:15 +08:00` 已确认 `main`、`origin/main` 与该 commit 一致。
