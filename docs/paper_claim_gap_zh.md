# ManipEvalAgent 论文主张与当前差距（自顶向下）

本文只按论文 Sec. 3.2–3.4 与 Figs. 2–5 审查主体方法，不把工程审计、hash、跨进程恢复或
更多任务数量误当作论文核心贡献。实验表格的规模复现另算；本项目当前目标是 ACT-only、少量
rollout 下的功能闭环。

## 1. 论文主体数据流

```text
开放 Query
→ Plan Agent：结合 Query 与 Y1:t 动态发现/选择 sub-aspect
→ Task Proposal → TaskGen：检索复用或生成 runnable task + check_success
→ 场景渲染、视觉诊断与有界修复
→ 固定的单任务 ACT checkpoint 执行
→ Tool Proposal → ToolGen：检索复用或生成、验证、注册 Rule Tool
→ Dynamic Execution VQA + Rule Tool
→ Aggregate / evidence
→ evidence 改变下一轮方向，或停止并回答原 Query
```

这里的动态性首先是**同一次单任务 evaluation 内，证据到来后再决定下一 sub-aspect**。一个 ACT
checkpoint 只能执行其训练任务并不冲突：系统通用性来自公共合同与 task adapter，而不是让一个
checkpoint 中途切任务。跨任务 `EvaluationGraph` 是可选的多 checkpoint portfolio，不是 Fig. 2
主体闭环成立的前置条件。

## 2. claim → 当前对应 → 诚实 gap

| 优先级 | 论文 claim | 当前最小对应 | 仍有的 gap |
| ---: | --- | --- | --- |
| 1 | Plan Agent 根据 `Query + Y1:t` 动态发现 sub-aspect（Sec. 3.2；Figs. 2/5） | `AdaptivePlanStepAgent` 每轮只接收受信 Rule/VQA/Evidence 与当前 coverage，输出 `propose/refine/stop`；registered `dynamic_evidence_v1` 也已接入公共 step，并被 hash-pinned candidate suite 限界 | 源码与离线分支已就绪；仍需本批 clean-head matched N=1 证明真实 provider/evidence 会 stop 或 refine，而非旧 task-specific planner |
| 2 | TaskGen 对 Proposal 做 retrieve-or-generate，并交付 runnable scene + `check_success()`（Sec. 3.3；Fig. 3） | 正常 TaskGen 已在 provider 创建前运行 exact-semantic resolver，顺序为 official→内置 overlay→审核生成物 lookup→codegen；每次保存 `task_resolution.json`。`SuccessSpec v2` 增加可信 envelope、受限阈值与 `all/any` 编译 | official/overlay 已真正 reuse；审核 generated artifact 尚只有 exact lookup 合同，缺持久 registry 与 materializer。非官方 v2 语义目前仅 development fixture、禁止 ACT，尚不是 live Proposal-derived success generation |
| 3 | TaskGen 用 render/视觉反馈诊断并修复生成错误（Fig. 3；App. A.3.4） | scene reflection、SuccessSpec repair 与 expert gate 已有；`TaskGenerationAttempt` controller 统一 typed stage/action、最多一次局部修复，并保证 accepted 前 0 ACT、policy failure 不重试 | controller 已由 fixture 真正执行 diagnosis→repair callback，但尚未把所有生产 TaskGen 分支改由同一 controller 驱动；click overlay 仍以 validate 为主 |
| 4 | ToolGen 根据 Tool Proposal 检索、生成、验证、注册与复用（Sec. 3.3；Fig. 4） | trusted Tool、Python ToolGen、AST/differential gate、reviewed registry、run-local reuse 已有；`ToolProposal v3` 携带的 `MetricSpec v1` 已接入正常 runtime，并允许单条真实 episode 做 deterministic/oracle 校验 | DSL/operator 与可观测 signals 仍受限；安全只新增精确 camera-contact proxy，不能概括完整 unintended-contact/safety |
| 5 | Rule Tool 与 Dynamic VQA 互补观测执行结果（Sec. 3.3；Fig. 4） | telemetry/events/video → trusted/generated Tool + 事件关键帧 VQA → typed evidence；数值冲突时 Rule 为权威 | 已有真实与缓存机制证据，但真实扰动下的独立人工 gold、正负平衡和 VQA accuracy/AUROC 仍缺 |
| 6 | Aggregate 后的 evidence 改变后续规划并形成 Query-centric feedback（Secs. 3.2–3.4） | `EvidencePacket` 保留 policy/pipeline/rule/VQA；动态 step 可继续失败诊断、切换已支持 aspect 或停止；最终报告列强项、弱项、建议和局限 | 当前动态 step 的最新 live flagship 尚待验收；不能用旧 task-specific 两轮或 synthetic replay 替代 |
| 7 | 方法可在多个 concern/task 上使用（Fig. 2） | BBH 与 click_bell 共用 planning/evidence/proposal/runtime 合同；每个 child 固定自己的 ACT checkpoint | materializer 与真正可执行 aspect 仍少。跨任务 graph 只是一种 portfolio 编排，不应被提升为高于单任务动态闭环的核心 gap |
| 8 | 小规模实验能比较 adaptive 与 fixed，并评估 Planner/VQA（Tables 1–3、6–8） | 已有旧 runtime N=1、20-query assistant-proxy、micro ablation、真实/代理扰动 artifact | 仍缺当前统一 runtime 下同 Query/seed/两轮预算的 matched N=1；N=3、独立人工 gold 与更多扰动后置 |

## 3. 当前最值得补的顺序

1. **当前动态 runtime 的 flagship 与 matched fixed N=1（约 3–4 ACT）**：相同 Query、seed、任务、
   checkpoint 和最多两轮预算，先证明 evidence-conditioned planning 的真实功能差异。
2. **审核 generated task 的持久 registry/materializer（0 ACT）**：把当前 exact lookup 从接口升级为
   新 evaluation 中真实 no-provider reuse，同时 pin task.py、VariantSpec、SuccessSpec 与 validation。
3. **把 production TaskGen 分支统一交给 `TaskGenerationAttempt`（0–1 expert）**：复用现有 success/
   visual repair callback，不能让 controller 只存在于 fixture。
4. **一个明确允许 success-semantics variation 的受控 capability（0 ACT fixtures，live 后置）**：先由
   oracle 定义 envelope；当前 appearance/scale ACT 继续只接受 official-equivalent success。
5. **最后再补实验证据**：assistant-proxy 先验证格式；独立人工 gold、少量同 commit 真实扰动 clip、N=3 只在
   主链稳定后支付。多 policy、10 repetitions 继续后置。

## 4. 声明边界

- `AdaptivePlanStepAgent` 接线与 synthetic branch 只能证明规划合同；当前 live N=1 才能证明真实闭环。
- expert gate 只证明场景可解，不是 ACT 成功；缓存 telemetry replay 不是新 rollout。
- `safety.hammer_left_camera_contact` 只覆盖 `020_hammer ↔ left_camera` 的精确物理接触；通用
  `safety.unintended_contact` 仍 unsupported，不能把该 proxy 命名为完整 safety。
- invalid `SuccessSpec` → trusted fallback 证明诊断/恢复路径，不证明模型能生成任意正确成功函数。
- `EvaluationGraph` 汇总多个任务专属 checkpoint 的 child；它不是单 policy 跨任务执行，也不是论文核心
  Plan Agent gap 的替代品。
