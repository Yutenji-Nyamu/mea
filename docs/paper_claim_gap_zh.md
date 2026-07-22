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
| 1 | Plan Agent 根据 `Query + Y1:t` 动态发现 sub-aspect（Sec. 3.2；Figs. 2/5） | `AdaptivePlanStepAgent` 每轮只接收受信 Rule/VQA/Evidence 与当前 coverage，输出 `propose/refine/stop`；源码已补 registered dynamic 的 trusted catalog + provider 成对初始化，并用 candidate allowlist 限界。0-ACT live-provider smoke 在同一真实缓存 evidence 上得到失败→refine、成功→stop | commit `412fe6e` 的 strict dynamic pair 实际仍走旧 task-specific decision path；因此公共 step 虽已有 live-provider plan-only 分支证据，仍缺整合到当前 clean-head ACT runtime 的 N=1。之后才适合讨论 rollout savings |
| 2 | TaskGen 对 Proposal 做 retrieve-or-generate，并交付 runnable scene + `check_success()`（Sec. 3.3；Fig. 3） | 上游可信 capability 先选 official/overlay/codegen route；TaskGen 在 provider 前计算 exact semantic identity 并记录实际 materializer。force-codegen 已有审核生成物 lookup 接口；`SuccessSpec v2` 增加可信 envelope、受限阈值与 `all/any` 编译 | 正常 runtime 尚未配置 reviewed-task registry callback，resolver 还不会跨 artifact 主动 retrieve；成功 run 后才保存 resolution，生成早期失败也无该 artifact。缺持久 registry/materializer 与完整全局 reuse-first selector；非官方 v2 语义仍仅 development fixture、禁止 ACT |
| 3 | TaskGen 用 render/视觉反馈诊断并修复生成错误（Fig. 3；App. A.3.4） | scene reflection、SuccessSpec repair 与 expert gate 已有；`TaskGenerationAttempt` controller 统一 typed stage/action、最多一次局部修复，并保证 accepted 前 0 ACT、policy failure 不重试 | controller 已由 fixture 真正执行 diagnosis→repair callback，但尚未把所有生产 TaskGen 分支改由同一 controller 驱动；click overlay 仍以 validate 为主 |
| 4 | ToolGen 根据 Tool Proposal 检索、生成、验证、注册与复用（Sec. 3.3；Fig. 4） | trusted Tool、Python ToolGen、AST/differential gate、reviewed registry、run-local reuse 已有；`ToolProposal v3` 携带的 `MetricSpec v1` 已接入正常 runtime，并允许单条真实 episode 做 deterministic/oracle 校验 | DSL/operator 与可观测 signals 仍受限；安全只新增精确 camera-contact proxy，不能概括完整 unintended-contact/safety |
| 5 | Rule Tool 与 Dynamic VQA 互补观测执行结果（Sec. 3.3；Fig. 4） | telemetry/events/video → trusted/generated Tool + 事件关键帧 VQA → typed evidence；数值冲突时 Rule 为权威 | 已有真实与缓存机制证据，但真实扰动下的独立人工 gold、正负平衡和 VQA accuracy/AUROC 仍缺 |
| 6 | Aggregate 后的 evidence 改变后续规划并形成 Query-centric feedback（Secs. 3.2–3.4） | `EvidencePacket` 保留 policy/pipeline/rule/VQA；旧 task-specific dynamic run 真实执行 `continue→stop`，最终报告正确区分 base0 成功与 base1 失败 | 该 run 不是公共 step 的 live 证据，且决策受 required candidate coverage 限界；需先做修复后的 clean-head N=1，再用 optional counterfactual 或分支式 suite 验证 evidence 改变资源分配 |
| 7 | 方法可在多个 concern/task 上使用（Fig. 2） | BBH 与 click_bell 共用 planning/evidence/proposal/runtime 合同；每个 child 固定自己的 ACT checkpoint | materializer 与真正可执行 aspect 仍少。跨任务 graph 只是一种 portfolio 编排，不应被提升为高于单任务动态闭环的核心 gap |
| 8 | 小规模实验能比较 adaptive 与 fixed，并评估 Planner/VQA（Tables 1–3、6–8） | 已完成同 Query/seed/checkpoint/candidate suite 的 strict matched N=1：fixed/dynamic 各 2 ACT，exact success agreement=`1.0` | dynamic 是旧决策路径，rollout savings=`0`，N=1 无 trial distribution，`paper_table_eligible=false`；只保留为执行链 pilot。N=3、独立人工 gold 与更多扰动后置 |

## 3. 当前最值得补的顺序

1. **审核 generated task 的持久 registry/materializer（0 ACT）**：把当前 exact lookup 从接口升级为
   新 evaluation 中真实 no-provider reuse，同时 pin task.py、VariantSpec、SuccessSpec 与 validation。
2. **把 production TaskGen 分支统一交给 `TaskGenerationAttempt`（0–1 expert）**：复用现有 success/
   visual repair callback，不能让 controller 只存在于 fixture。
3. **先做修复后公共 step 的 clean-head N=1，再允许 evidence 真正改变预算（约 1–4 ACT）**：先确认
   `plan/adaptive_steps/after_round_01/plan_step_bundle.json` 来自 provider；随后设计 optional counterfactual 或
   分支式 candidate suite；只有 dynamic 可以依据 sufficient/failed evidence 合法 stop 或换方向，fixed
   仍冻结相同最大预算。先检验机制，不预设一定有 savings。
4. **一个明确允许 success-semantics variation 的受控 capability（0 ACT fixtures，live 后置）**：先由
   oracle 定义 envelope；当前 appearance/scale ACT 继续只接受 official-equivalent success。
5. **最后再补实验证据**：assistant-proxy 先验证格式；独立人工 gold、少量同 commit 真实扰动 clip、N=3 只在
   主链稳定后支付。多 policy、10 repetitions 继续后置。

## 4. 声明边界

- `AdaptivePlanStepAgent` 接线与 synthetic/cached branch 只能证明规划合同；修复后的 live N=1 才能证明真实闭环。
- 最新 strict pair 的 ACT/Tool/VQA 是 live，但 planner 仍是旧 task-specific path；exact agreement=`1.0`
  只表示两个重叠样本结论相同。N=1 不等于
  论文统计一致性，rollout savings=`0` 也不支持效率提升 claim。
- expert gate 只证明场景可解，不是 ACT 成功；缓存 telemetry replay 不是新 rollout。
- `safety.hammer_left_camera_contact` 只覆盖 `020_hammer ↔ left_camera` 的精确物理接触；通用
  `safety.unintended_contact` 仍 unsupported，不能把该 proxy 命名为完整 safety。
- invalid `SuccessSpec` → trusted fallback 证明诊断/恢复路径，不证明模型能生成任意正确成功函数。
- `EvaluationGraph` 汇总多个任务专属 checkpoint 的 child；它不是单 policy 跨任务执行，也不是论文核心
  Plan Agent gap 的替代品。
