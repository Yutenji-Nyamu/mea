# ManipEvalAgent 论文主张与当前差距（自顶向下）

本文只按论文 Sec. 3.2–3.4 与 Figs. 2–5 审查主体方法，不把工程审计、hash、跨进程恢复或
更多任务数量误当作论文核心贡献。实验表格的规模复现另算；本项目当前是 ACT-only、两个任务族和
少量可信 capability 内的受限功能原型，不是论文规模或开放世界能力的完整复现。

## 0. 2026-07-23 路线纠偏

上一版路线仍然把 Query identity、AnswerScope、official/experimental authority 和 fail-closed gate
放在最前面。这些工作能改善工程可靠性，但不是论文的科学主张；把它们继续当作主线，会造成
“接口和护栏越来越完整，论文图表仍然没有新证据”的安全性陷阱。

此后路线从论文 Abstract、Introduction contributions、Sec. 4 的三个实验问题以及 Tables 1–9
自顶向下推导。工程护栏只作为 claim 实验的验收条件，不再单独计为复现进展。判断一次开发是否
值得优先做，首先问它会新增哪条论文 claim 的真实可反驳证据，而不是它是否又增加了一层约束。

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
| 1 | Plan Agent 根据 `Query + Y1:t` 动态发现 sub-aspect（Sec. 3.2；Figs. 2/5） | clean-head live v4 已证明真实 Evidence 可触发 `object_position→object_instance`，且两轮均执行真实 ACT/Rule/VQA/Aggregate | position 与 instance 已经同时出现在初始 decomposition/candidate catalog，最终又因 hard cap 停止；尚未证明 Agent 会从开放 Query 与新 evidence 中发现未预编排的 sub-aspect，或在证据充分时自主结束 |
| 2 | TaskGen 对 Proposal 做 retrieve-or-generate，并交付 runnable scene + `check_success()`（Sec. 3.3；Fig. 3） | batch11 的新 scene 已真实执行 ACT；batch18 的单个 BBH Proposal 同时生成 scene 与 experimental `check_success()`，通过 3 provider + 2 simulator probes + 1 expert，ACT=0 | 两半证据尚未在同一个 policy rollout 合一：真实 ACT 仍使用 official success，而生成 checker 没有评价过 ACT；也没有多种 unseen Proposal 或后续 task-library reuse |
| 3 | TaskGen 用 render/视觉反馈诊断并修复生成错误（Fig. 3；App. A.3.4） | scene reflection、SuccessSpec repair 与 expert gate 已有；相关 stage 使用 typed diagnosis/action 做最多一次局部 repair/regenerate，并保证 accepted 前 0 ACT、policy failure 不重试 | 论文使用 stage-specific recovery，并不要求所有阶段共享一个中央 controller。当前真实 visual repair 和 fixture recovery 的覆盖仍窄；click overlay 与大多数正常路径仍以 validate/accept 为主 |
| 4 | ToolGen 根据 Tool Proposal 检索、生成、验证、注册与复用（Sec. 3.3；Fig. 4） | v4 的 generated distance tool 已在真实 ACT/expert telemetry 上执行；AST/differential validation、reviewed registry 与跨 evaluation reuse 均有真实 artifact | 大多数需求仍映射到预定义 MetricSpec target/operator。尚缺一个开放 Query 自然诱发、catalog 中此前不存在的 metric，以及首次生成后在另一 Query 中直接检索复用的完整案例 |
| 5 | Rule Tool 与 Dynamic VQA 互补观测执行结果（Sec. 3.3；Fig. 4） | telemetry/events/video → trusted/generated Tool + 事件关键帧 VQA → typed evidence；数值冲突时 Rule 为权威。本批增加多人标注、majority/senior tie-break、正负 control 与四条件 accuracy/AUROC 协议 | 发布结果仍是 synthetic proxy；没有 4 名真实机器人 annotator、盲评 clips、足量正负样本或 human gold，不能作为论文 VQA validity 结果 |
| 6 | Aggregate 后的 evidence 改变后续规划并形成 Query-centric feedback（Secs. 3.2–3.4） | clean-head v4 证明一类 success evidence 会切换到 instance；旧三轮 click case 也有文字反馈 | 尚无同一开放 Query 下的 live success/failure/ambiguous 三种 evidence 导致 switch/drill-down/change-tool 的因果对照；也没有因证据充分而非 hard cap 结束的真实案例 |
| 7 | 方法可在多个 concern/task 上使用（Fig. 2） | BBH 与 click_bell 共用 planning/evidence/proposal/runtime 合同；每个 child 固定自己的 ACT checkpoint | materializer 与真正可执行 aspect 仍少。跨任务 graph 只是一种 portfolio 编排，不应被提升为高于单任务动态闭环的核心 gap |
| 8 | 少样本结论与 benchmark 一致，并评估 Planner/VQA（Tables 1–3、6–9） | 旧 strict live pair 为 2 ACT 对 2 ACT、节省 0；其余 2:1、Planner/VQA 指标来自 synthetic 或 development proxy | 没有 dense benchmark reference、真实 adaptive saving、5 policy ranking、5 trials/task、10 agent runs、人工 Plan gold、四条件 VQA AUROC 或真实 code-generation ablation |

## 3. 论文真实声称、但当前项目还没有的证据

下表只列论文自己的方法或实验 claim，不把本项目新增的安全合同列为贡献。

| 层级 | 论文 claim | 论文证据 | 当前项目状态 |
| --- | --- | --- | --- |
| 核心结论 | 用显著更少的时间和 samples，仍得到与完整 simulation benchmark 可比的结论 | Abstract；Tables 1–2、4–5；每个 constructed task 默认 5 trials，ManipEvalAgent 完整流程重复 10 次 | **完全缺失**：没有同一 policy/task 上的完整 benchmark 基线与 adaptive 结果；现有 strict pair 为 2 ACT 对 2 ACT |
| 核心结论 | 少样本评估仍能保持五个 policy 的相对排名 | Table 9：ACT、DP、DP3、RDT、π0；RoboTwin/LIBERO clusters；10/20/50 rollouts；Spearman 与 bootstrap CI | **完全缺失**：当前只有 ACT，没有五 policy 排名或 benchmark clusters |
| 核心方法案例 | 开放 Query 在每轮依据 observations 动态选择或细化 sub-aspect，并在需要时现场生成 task/tool | Secs. 3.2–3.4；Figs. 2、5 | **部分存在**：v4 有两轮真实 evidence branch，但候选来自窄 catalog；没有同一次 run 中的三轮细化、现场生成 scene + `check_success()` 和新 Tool 的完整 Fig. 5 案例 |
| 组件有效性 | TaskGen 生成 scene + `check_success()`，RAG、visual self-check、README.Agent 提升生成成功率 | Fig. 3；Table 3：complete 98%，各消融 93–96% | **真实统计缺失**：只有单例生成、deterministic switch 和 4-call development-proxy micro ablation；没有成组 proposal、simulator render 和人工正确性统计 |
| 组件有效性 | ToolGen 能现场生成、验证、注册并复用新 rule tool，RAG 提升成功率 | Fig. 4；Table 3：complete 96%，w/o RAG 92% | **真实统计缺失**：有受限生成/验证 plumbing 和单例，但没有成组新工具、人工/fixture 判定及成功率 |
| 有效性 | Plan Agent 对开放 Query 的 sub-aspect 分解与机器人研究者高度一致 | Table 6；几百条 query dataset；4 名机器人方向 annotator majority、senior tie-break；三种 planner model | **完全缺失论文证据**：当前仅 20 条 development-agent proxy，micro precision 0.571，不是 human gold |
| 有效性 | VQA 在 clean、clutter、background texture、lighting 下与人工一致且 AUROC 稳定 | Tables 7–8；真实 clips；三种 VLM；人工二值 gold | **完全缺失论文证据**：只有 clean/clutter 的真实 N=1 视频和代理标签；无四条件平衡样本、三 VLM、人工 gold 或可计算 AUROC |
| 稳定性 | 整体约 5% 流程受错误影响，并可分解到 Plan/TaskGen/ToolGen/simulator/others | Fig. 6；App. A.1.3 的人工/fixture 错误计数协议 | **完全缺失**：当前记录零散运行失败，没有按统一 case universe 统计系统 error rate 和模块占比 |
| 范围 | multi-task VLA 上仍显著节省时间并保持 benchmark conclusion consistency | Tables 4–5；RDT、π0；RoboTwin 与 LIBERO | **完全缺失**：没有 RDT/π0 checkpoint、多任务 suite 或 LIBERO 实验 |

## 4. Claim-first 候选批次

### A. Fig. 5 完整开放 Query flagship

- **目标 claim**：开放 Query + 中间 observations 真正决定下一轮，并现场生成任务和工具。
- **最小实现**：只给一句无内部路线提示的 object-generalization Query；最多三轮。Planner 自主选择
  初始 sub-aspect；至少一轮生成新的 scene + `check_success()`，至少一轮产生当前 registry 中没有的
  rule metric；上一轮 ambiguous evidence 必须触发更细 sub-aspect。
- **真实预算**：每轮 ACT `N=1`，合计 1–3 ACT；TaskGen/ToolGen provider 和 simulator 正常运行。
- **成功判据**：一个不可拼接的 artifact 同时包含 Query、三轮 Proposal、生成代码/render、ACT、
  Rule/VQA、Aggregate、evidence-conditioned next proposal 和最终回答。
- **地位**：最高优先级；它直接补 Figs. 2/5，而不是再补外围合同。

### B. Proposal-derived TaskGen + `check_success()` 真实 rollout

- **目标 claim**：TaskGen 对一个此前不存在的 Proposal 同时生成 scene 与成功判据，并在视觉修复后
  真正用于 policy evaluation。
- **最小实现**：为 BBH 生成一个含目标 block 与物理 look-alike distractor 的新场景；生成的 checker
  要求击中目标且不误击 distractor；首帧检查能发现重叠、错误 actor 或错误位置。
- **真实预算**：约 4–7 次 provider、3–5 次 simulator/expert probe、1 次 ACT。
- **成功判据**：新 `task.py` 真实实例化两个物理对象；checker 对 expert 正例、误击负例和未击中负例
  判定正确；ACT rollout 直接调用这个生成 checker，而不是继续继承 official success。

### C. Query 触发一个此前不存在的 Tool

- **目标 claim**：ToolGen 从新评估需求生成、验证、注册工具，并在下一 Query 中检索复用。
- **最小实现**：Query 只描述“接触前是否存在明显抖动或急动”，不指定 metric 名；ToolGen 生成
  `precontact_jerk_peak`，用 smooth、oscillatory、missing-contact 三个 oracle trajectory 验证，再在
  真实 telemetry 上执行，并配套一个可见抖动 VQA。
- **真实预算**：先在缓存 telemetry 上 0 ACT 完成 generate/validate/register；再用 1 次 ACT 做 live
  confirmation；第二次 Query 应 0 codegen 直接复用。
- **成功判据**：metric 在 Query 前不在 catalog；生成代码通过独立 oracle；Rule 与 VQA 一同返回
  Planner；下一 Query 从 registry 命中同一实现。

### D. Tables 1–2 的单 policy 小型真比较

- **目标 claim**：少量 adaptive sampling 在成本更低时仍得到与完整 benchmark 相同的能力结论。
- **最小实现**：click_bell 的 left、right、base0、base1 形成 dense reference；fixed 全跑四项，
  adaptive 最多三轮；比较 object-generalization verdict、最弱 axis、rollout 数和 wall-clock。
- **真实预算**：路径验证 `N=1` 为 fixed 4 + adaptive 1–3，共 5–7 ACT；随后放大到 `N=3`
  的 15–21 ACT 和论文 constructed-task 默认 `N=5` 的 25–35 ACT。
- **成功判据**：`N=3/5` 时 adaptive 与 dense reference 对强弱 axis 和总体 verdict 一致，同时真实
  rollout 与 wall-clock 至少降低 25%；若 adaptive 跑满或结论不一致，就判定当前设置未复现该 claim。

### E. Table 6 与 Tables 7–8 的真实 human validity

- **目标 claim**：Plan sub-aspect 和 VQA evidence 都与独立机器人研究者判断一致。
- **最小实现**：Plan 先用五类 taxonomy 各 4–6 条 Query；VQA 为 clean/clutter/background
  texture/lighting 各收集正负 clips；均由 4 名机器人方向 annotator 独立盲标、majority vote、
  senior tie-break。冻结 prompt 后运行三个 planner/VLM model。
- **真实预算**：0 ACT 起步可复用视频；Plan 约 20–30 × 3 provider calls；VQA 先做 8–16 clips，
  不足条件再补 simulator rollout。
- **成功判据**：Plan precision/recall/F1、inter-annotator agreement，以及每个视觉条件的
  accuracy/AUROC；不再使用 development-agent proxy。

### F. Table 3 真实 code-generation ablation

- **目标 claim**：RAG、visual self-check、README.Agent 确实提高 TaskGen/ToolGen 成功率。
- **最小实现**：同一 unseen Proposal 在 complete、base、w/o RAG、w/o visual self-check、
  w/o README.Agent 下真实生成；由不知道 condition 的 reviewer 检查 task code/render/tool oracle。
- **真实预算**：路径验证为 1 proposal × 7 个论文 condition；最小经验批次 `N=5` 约 35 次 generation；
  论文差值仅 3–5 个百分点，真正估计差异可能需要约 20 cases/condition。
- **成功判据**：预先冻结 correct/reasonable/aligned 标准，输出真实分母、成功率、CI 与失败案例；
  “开关能运行”不能代替消融效果。

### G. Table 9 五 policy ranking

- **目标 claim**：agent 少样本评估保持 policy 相对排序。
- **依赖**：必须先具备 DP、DP3、RDT、π0 的可运行 checkpoint 与对应 task/benchmark adapter。
- **真实预算**：最高；先做 ACT/DP/DP3 三 policy pilot，再扩五 policy 和 10/20/50 rollout budget。
- **成功判据**：从真实 scores 计算 Spearman 与 bootstrap CI，不能用合成 ranking。

推荐依赖顺序是 **A → B → C → 把 A/B/C 合成一次完整 Fig. 5 flagship → D (`N=1→3→5`)**；
E、F 可在不大量消耗 ACT 的情况下并行准备，G 在多 policy checkpoint 就绪后执行。科学重要性上
Tables 1–2/9 最高，但在 A/B/C 尚未成为真实方法能力前直接跑效率比较，比较到的仍会是受限脚本而
不是论文的 ManipEvalAgent。

## 5. 声明边界

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
