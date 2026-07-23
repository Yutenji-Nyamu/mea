# ManipEvalAgent 论文主张与当前差距（自顶向下）

本文只按论文 Sec. 3.2–3.4 与 Figs. 2–5 审查主体方法，不把工程审计、hash、跨进程恢复或
更多任务数量误当作论文核心贡献。实验表格的规模复现另算；本项目当前是 ACT-only、两个任务族和
少量可信 capability 内的受限功能原型，不是论文规模或开放世界能力的完整复现。

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
| 1 | Plan Agent 根据 `Query + Y1:t` 动态发现 sub-aspect（Sec. 3.2；Figs. 2/5） | clean-head live v4 已证明真实 Evidence 可触发 `object_position→object_instance`。本批又增加正式 `QuerySufficiencyContract`：分离 claim type、完整 candidate universe、required coverage 与 budget；cached v4 replay 对四候选 suite 得出 `inconclusive/continue`，并保留 `right_fixed/base1` | sufficiency 仍是 0-ACT offline CLI，没有进入 `manipeval_agent.py` 的逐轮 stop；Query、checkpoint、完整 route suite、unsupported axes、CandidateEvidence、P4 answer 与 P3 comparison 也尚未共享同一 hash-bound identity |
| 2 | TaskGen 对 Proposal 做 retrieve-or-generate，并交付 runnable scene + `check_success()`（Sec. 3.3；Fig. 3） | reviewed-task reuse/production acceptance 保持。新增显式 public Proposal v2 已在真实服务器生成 BBH 紫蓝 block scene 与 bounded experimental SuccessSpec，通过 3 provider + 2 simulator probes + 1 expert 的 setup/render/rule/expert/production acceptance，ACT=0；fresh/resume ACT 均由最终 TaskArtifactBundle authority fail closed | experimental v2 仍只覆盖 `beat_block_hammer/object_appearance.color`。official 与 experimental outcome 没有双通道 runtime，故不能执行 ACT；这不是 proposal-derived success 的策略执行证据，也不是通用 RAG/codegen |
| 3 | TaskGen 用 render/视觉反馈诊断并修复生成错误（Fig. 3；App. A.3.4） | scene reflection、SuccessSpec repair 与 expert gate 已有；相关 stage 使用 typed diagnosis/action 做最多一次局部 repair/regenerate，并保证 accepted 前 0 ACT、policy failure 不重试 | 论文使用 stage-specific recovery，并不要求所有阶段共享一个中央 controller。当前真实 visual repair 和 fixture recovery 的覆盖仍窄；click overlay 与大多数正常路径仍以 validate/accept 为主 |
| 4 | ToolGen 根据 Tool Proposal 检索、生成、验证、注册与复用（Sec. 3.3；Fig. 4） | trusted Tool、Python ToolGen、AST/differential gate、reviewed registry、run-local reuse 已有；`ToolProposal v3` 携带的 `MetricSpec v1` 已接入正常 runtime，并允许单条真实 episode 做 deterministic/oracle 校验 | DSL/operator 与可观测 signals 仍受限；安全只新增精确 camera-contact proxy，不能概括完整 unintended-contact/safety |
| 5 | Rule Tool 与 Dynamic VQA 互补观测执行结果（Sec. 3.3；Fig. 4） | telemetry/events/video → trusted/generated Tool + 事件关键帧 VQA → typed evidence；数值冲突时 Rule 为权威。本批增加多人标注、majority/senior tie-break、正负 control 与四条件 accuracy/AUROC 协议 | 发布结果仍是 synthetic proxy；没有 4 名真实机器人 annotator、盲评 clips、足量正负样本或 human gold，不能作为论文 VQA validity 结果 |
| 6 | Aggregate 后的 evidence 改变后续规划并形成 Query-centric feedback（Secs. 3.2–3.4） | clean-head v4 与缓存反事实继续证明 branch sensitivity。`FeedbackAgent` 新增 fail-closed `AnswerScope`，强制 N/seed、tested/untested、unsupported、conflict 和 stop reason；cached v4 projection 为 N=2、同一 seed、两项未测、五类 unsupported、interim answer | P4 guard 已进源码，但 v4 产生于它接线之前；尚无同一次 clean-head live run 同时保存 P1 assessment 与 P4 output。它保证报告诚实，不等于 Plan Agent 已知道何时充分 |
| 7 | 方法可在多个 concern/task 上使用（Fig. 2） | BBH 与 click_bell 共用 planning/evidence/proposal/runtime 合同；每个 child 固定自己的 ACT checkpoint | materializer 与真正可执行 aspect 仍少。跨任务 graph 只是一种 portfolio 编排，不应被提升为高于单任务动态闭环的核心 gap |
| 8 | 小规模实验能比较 adaptive 与 fixed，并评估 Planner/VQA（Tables 1–3、6–8） | 旧 strict live pair 仍为 2 ACT 对 2 ACT、节省 0。本批新增匹配 Query/checkpoint hash/suite/seed/budget/P1 contract 的 preregistration 与分资源账本，并用 synthetic `2:1`、`2:2` 验证 truth table | 新协议尚无真实 pair，synthetic saving 不是 empirical result；也未达到论文 constructed task 默认 5 trials 与 10 agent runs。paper sample count 不能与 ACT starts/steps 混写 |

## 3. 当前最值得补的顺序

1. **闭合同一条 Query identity chain（先 0 ACT replay，再 1–2 ACT）**：hash 绑定原 Query、
   checkpoint、完整 routed suite、预算和 unsupported axes；每轮 Aggregate 转正式 CandidateEvidence，
   P1 assessment 原样进入 Planner stop、P4 answer 与 P3 comparison。不能让各模块自行声明
   `sufficiency_met` 或缩小 candidate universe。
2. **official / experimental 双通道执行（先 cached telemetry，最多 1 ACT）**：并列计算 official
   success 与 compiled experimental SuccessSpec outcome，字段、authority 和 report 完全分离；只有这条
   边界稳定后才解除 experimental v2 的 ACT gate。
3. **真实 matched efficiency pilot（先 N=1）**：同一 structured P1 verdict 下先尝试 fixed 2 +
   adaptive 1；若 adaptive 也需 2，必须报告 2:2、节省为 0。稳定后才扩到论文默认 5 trials/task 与
   10 agent runs，并单独解释 paper sample semantics。
4. **真实独立人工 gold 与 VQA 鲁棒性**：导入有 hash 的 clean/clutter/background texture/lighting
   clips，4 名机器人方向 annotator、majority vote、senior tie-break 和预注册 scorer；不能用
   development-agent/synthetic proxy 填论文 Tables 6–8。
5. **最后扩大任务、policy 与 repetition**：只有前述身份、执行 authority 和统计合同闭合后，才加入
   第三任务、更多 checkpoint/policy、新 metric 与论文规模实验。

## 4. 声明边界

- clean-head v4 是当前公共 Planner 的两轮完成态功能验收：首轮位置成功后的真实 evidence 触发切换
  instance。它不是泛化结论；两个实际样本均成功，也不能代表未测 `right_fixed`、`base1` 或未支持轴。
- batch18 P1/P4 对 v4 的回放是 cached 0-ACT evidence：完整四候选 suite、预算 3、已完成 2，结果为
  `inconclusive/continue`。它没有改变 live Planner，也不是新 ACT/Feedback run。
- batch18 P2 是真实 provider/simulator/expert 的 TaskGen acceptance，但 ACT=0 且
  `experimental_v2_act_runtime_eligible=false`；生成的 experimental `check_success()` 不能报告为
  official policy success。
- batch18 P3/P5 的发布结果均是 synthetic fixture。`2:1` savings、accuracy/AUROC 只证明协议算术，
  不能进入论文表格；现有 development-agent/proxy labels 不是 human gold。
- v4 因 hard cap=2 结束；`completed` 表示执行合同完成，不等于 `evidence_sufficient`。当前首要方法 gap
  是让 Query 量词决定 required coverage，并与所有可选候选和预算上限解耦。
- v4 所有角色显式覆盖为 `gpt-5.6-terra`；它不证明默认多模型 profile 或 Luna 路径稳定。账本精确为
  11 个 logical provider calls、16 个 transport attempts、2 个 ACT starts，且没有 whole-round recovery/
  restart；transport 次数不能与逻辑模型调用数混写。
- v4 的完成态证据本身含 2 ACT，但本批新增 ACT starts 合计 4；clean-head v2/v4 累计 3 ACT，并重复
  左侧 position。不能用最终 bundle 的 2 ACT 代替开发成本，也没有 sampling savings 结论。
- 最新 strict pair 的 ACT/Tool/VQA 是 live，但 planner 仍是旧 task-specific path；exact agreement=`1.0`
  只表示两个重叠样本结论相同。N=1 不等于
  论文统计一致性，rollout savings=`0` 也不支持效率提升 claim。
- expert gate 只证明场景可解，不是 ACT 成功；缓存 telemetry replay 不是新 rollout。
- `safety.hammer_left_camera_contact` 只覆盖 `020_hammer ↔ left_camera` 的精确物理接触；通用
  `safety.unintended_contact` 仍 unsupported，不能把该 proxy 命名为完整 safety。
- invalid `SuccessSpec` → trusted fallback 证明诊断/恢复路径，不证明模型能生成任意正确成功函数。
- reviewed task 由 `development_agent` 审核，固定 `paper_table_eligible=false`；它不是独立人工 gold。
- reviewed reuse 固定的是 task.py、VariantSpec、overlay/load_actors、可选 SuccessSpec 和 validation/static
  等 immutable inputs，以及当前列出的 5 个 Python runtime dependencies。`TaskArtifactBundle` 与
  `SceneCheckSpec` 是 run-local derived rebuild，不能声称整个环境或全部产物按字节复现。
- registry-authoritative variant id 可以与当前请求的 run-local id 不同；复用依据是受信可执行语义和
  provenance，不是宽松字符串相似。Proposal Agent 对 VQA 绑定的有界修复也只是 capability-derived
  结构修复，不是模型完成了新的语义推理。
- partial route 只允许执行同一 task 中已支持的子集，同时把 appearance、mass、scale 等 task-qualified
  gaps 留在最终限制中；它没有把这些 axis 变成新 capability。
- 最终源码已补失败/restart attempt 的账本和 feedback `finally` 收口；v4 没有触发 recovery，不能把
  其 `11 logical / 16 transport / 2 ACT` artifact 当作新 restart 路径的 live 验收。
- `EvaluationGraph` 汇总多个任务专属 checkpoint 的 child；它不是单 policy 跨任务执行，也不是论文核心
  Plan Agent gap 的替代品。
