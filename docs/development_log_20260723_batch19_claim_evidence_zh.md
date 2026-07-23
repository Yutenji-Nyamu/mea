# 2026-07-23：batch19 论文 claim 最小真实证据

本批从论文 claim 反推最小实验，不把接口、guard、plan-only 或 synthetic fixture 计作论文结果。
所有 Python、provider、RoboTwin、expert、ACT 与 DP3 验收均在 canonical AutoDL
`/root/autodl-tmp/mea` 完成；Windows 仅中转文档。根 `README.md` 未修改。

## 1. 本批新增的可执行能力

- `ClaimFirstOpenQueryAgent` 不接收 template ID 或顺序；但它接收的 capability projection 仍公开
  scale、appearance 等受控轴。本批用独立 CLI 手工把每轮真实 evidence 传给下一轮，不是统一
  Agent runtime，也没有集成 VQA、Aggregate 或 Feedback。传入 schema 只有开发者写的
  outcome/summary/limitations，没有 rollout ref、EvidencePacket hash、Tool/VQA 或 Aggregate binding；
  最强结论只是 provider 在阅读人工摘要后改变了 proposal。
- bounded experimental `SuccessSpec` 只在 standalone TaskGen ACT 路径中以
  `generated_check_success` 独立标签执行，不冒充 official success；主 `manipeval_agent.py` 仍继续
  阻止 experimental v2 ACT。
- Query-induced ToolGen v3b 从自然 Query 选择受限 DSL metric
  `precontact_peak_tcp_jerk`，由 runtime 选择 active/right arm、按物理时间计算三阶差分，并完成
  generate/validate/register 与语义 paraphrase reuse。复核发现 v3b 把非物理接触 interval 的起始
  step 当成 target contact，其 43.4348 m/s³ 数值无效。v3c 要求
  `physical_contact=true` 与非空 `first_physical_physics_step`，原 Query/转述 Query 都以 exact reuse
  返回 `value=null, null_reason=no_target_contact_event`。它不是任意 Python codegen，也未接入主
  Agent/VQA 或新 live rollout。
- 论文效率、policy ranking、Plan/VQA validity proxy、proposal prompt ablation 和 error distribution 都有
  fail-closed 的 manifest/result 入口；它们不会把 synthetic 或 development-agent 标注升级成人工 gold。
- DP3 adapter 能启用 RoboTwin 已有的真实 pointcloud observation，并检查 14D joint state、点云形状、
  数值和有限性；未生成 synthetic point cloud。

## 2. 真实实验结果与边界

| 论文 claim | 本批最小真实结果 | 结论边界 |
| --- | --- | --- |
| capability-conditioned 开放 Query | template ID 与顺序未暴露，手工链式调用连续产生 `1.2x scale → 0.8x scale → blue appearance`；后两轮 provider 读取开发者写的前序失败摘要 | 受控轴仍预先暴露；摘要没有 rollout/EvidencePacket/Tool/VQA/Aggregate binding，不是统一 Agent run。post-budget clean seed1000 也失败，property attribution 不成立，原 Query 未回答 |
| scene + 新 `check_success()` 同次 rollout | standalone TaskGen 的新颜色 scene 与受限编译 proximity+contact SuccessSpec 同一 artifact；expert seed 1002 的 `generated_check_success=true`，ACT 为 false | 只有颜色 scene + proximity/contact checker；没有 physical distractor，也没有 controlled mis-hit/miss fixtures。主 Agent 仍阻止 experimental v2，且它不是 official success |
| Query-induced Tool v3b/v3c | v3b 自然 Query 生成、synthetic oracle 验证并 run-local 注册受限 DSL；paraphrase 命中 semantic registry。v3c 修正物理接触 gate 后，原 Query 与 paraphrase exact reuse 均为 null/no-target-contact | `validated_episode_count=0`、threshold=100 未校准；v3b 的 43.4348 数值无效，缓存 ACT 没有 policy jerk evidence。最强 claim 仅是 bounded DSL + synthetic oracle + register/reuse routing |
| 少采样有限域协议 | v4 在两个 outcomes 已观察后，post hoc 把 fixed 首个 rollout 当作 adaptive cached prefix；两者 outcome authority/hash 相同，universal claim 可在首个失败后逻辑停止 | **没有实际 saving 证据**：`act_rollout_saving=null`、wall saving=null。只能说 counterfactual avoidable count=1/fraction=0.5；82.306 s 是估计，不是观测到的 adaptive speedup 或 benchmark claim |
| 多 policy 排名 | ACT 与 DP3 在同一 official BBH `demo_clean`、同一 seed 100000 各跑 1 次，两者均 0/1 | observed ranks 均为 1.5，Spearman 不可定义，结果是 `toy_order_inconclusive_tie`；没有复现 ACT/DP/DP3/RDT/π0 排名 |
| Plan/VQA 有效性 plumbing | Plan n=5 来自 legacy 20260717 taxonomy-routing validation，不是 batch19 ClaimFirst 输出；VQA 四条件共 8 条 heterogeneous propositions 的手选 cached predictions | `preregistered_sample=false`，只算 protocol smoke。AUROC=1.0 不是 robustness estimate；无独立人工 gold |
| Table 3 proposal prompt ablation | TaskGen 5 个 condition + ToolGen 2 个 condition，共 7 次真实 provider proposal 调用，全部通过 JSON/proxy gate | 只测试 proposal prompt 输出，不是 task/tool codegen 成功；每条件 N=1、全部同分，未进入 compile/render/simulator/oracle/ACT |
| 系统错误率与分布 | 对同一冻结的 23-operation universe 做 retrospective status review：v2 inactive-arm 两项从 success 改为 error，结果为 8/23=34.78%；ToolGen 4/4、simulator 4/6 | 只重算冻结 universe，不扩充后续操作；v3 类型错误与 v3b contact gate 假阳性另列为 2 个 post-universe failures，不进分母。不是论文约 5% 的同分母重复实验 |

服务器最终验收：主 RoboTwin 环境完整 `python -m unittest discover tests/manipeval`
为 636/636（54.653 s）；focused suites 为 paper-claim demo 17/17、Query ToolGen 7/7、
Query sufficiency 9/9、toolkit 6/6、TaskGen related 40/40 与 experimental gate 4/4；
DP3 环境 adapter 4/4，deploy/adapter `py_compile` 通过。上述都是服务器验收，不是 Windows
本地测试，也不把测试通过解释成论文经验 claim。

## 3. Open Query 三轮的关键事实

Query：

> How well does this ACT policy generalize across manipulated-object properties, and where is its first likely weakness?

1. 首轮自主选择 1.2x target geometry scale。ACT seed 1000 拿起 hammer，但
   `min_xy=0.01056299 m`、无严格接触、official success=false；matched expert 成功。
2. 看到该失败后，Planner 选择反方向 0.8x scale 作为因果判别。ACT 实际 seed 1001，
   `min_xy=0.00267630 m`、无严格接触、official success=false；expert seed 1000 成功。
   这不是 exact same-seed pair，且首次 0.8x 尝试在 pre-ACT visual repair 阶段失败，不计 policy 结果。
3. 看到两个 scale 方向均失败后，Planner 转向保持 geometry/collision 的 blue appearance-only。
   ACT/expert 均用 seed 1010；ACT `min_xy=0.02819031 m`、无严格接触、official success=false，
   expert 成功。

三轮是 capability-conditioned 的手工链：每次单步 Planner CLI 后，由开发者运行 TaskGen/ACT、整理
evidence JSON，再显式传给下一次 CLI。template ID/顺序虽然隐藏，但 scale/appearance 等轴已暴露；
该 JSON 只有 developer-authored outcome/summary/limitations，没有 rollout ref、EvidencePacket hash、
Tool/VQA/Aggregate。没有统一 runtime artifact 串起 VQA、Aggregate 和 Feedback。三轮后 Planner 请求 official clean
baseline contact control；因此终态是 `budget_exhausted_missing_clean_baseline`，而不是
`evidence_sufficient`。

预算外补跑了一个 exact official clean ACT seed 1000 anchor：eligible/evaluated=1/1，400 policy
steps，official success=false。这与 1.2x seed1000 同为失败，因此 1.2x 的失败不能解释成属性变化造成；
三个 variant failure 也不能确定“最先暴露的属性弱点”。加入该 anchor 后 Planner 仍选择 continue，
要求 official clean seed1010 与 blue seed1010 做 exact paired control。按预注册边界，本批不再增加
ACT。这个 post-budget 结果进一步证明：Planner 没有在三轮预算内及时安排 control，原 Query 未回答。

## 4. DP3 checkpoint 与 exact-seed pilot

- 官方 RoboTwin 2.0 DP3 BBH checkpoint：
  `DP3_ckpt/beat_block_hammer/3000.ckpt`；
  服务器大小 4,199,245,138 bytes，SHA256
  `250500047440d758e5f17f8215c1e9c60622904b9941d60af6623696a4a27fa0`。
- 首次 DP3 尝试在 action 前因 observation 为 list 而停止，属于 deployment/system error，不计性能。
- adapter 修复后，DP3 在 official clean seed 100000 完成 400 policy steps、14,796 physics steps，
  success=false、error=null；ACT exact-seed 对照也完成 400 policy steps、15,070 physics steps，
  success=false、error=null。
- 单 seed 双失败只能说明 adapter 与配对协议跑通，不能支持相对 ranking。RDT/π0 和
  RoboTwin/LIBERO 仍后置。

## 5. 开发失败与 policy failure 分账

以下均不进入 policy success 分母：

- 首次 DP3 run 的 `dp3_observation_type_unsupported`；
- 首次 0.8x TaskGen visual repair 生成非法算术表达式，pre-ACT 失败；
- 三个颜色实验中的 seed mismatch / expert gate rejection；
- ToolGen 初版误把桌面接触当作 target contact 的两次 semantic error；
- Query Tool v2 读取 inactive left arm；v3 首次 prompt 又把 finite-difference order 返回为字符串。
- Query Tool v3b 的 target-contact finder 接受了 `physical_contact=false` 的 interval 起始 step，
  导致 43.4348 m/s³ 假阳性；v3c 修正后返回 no-target-contact null。

失败后修正了 DP3 observation adapter、BBH visual repair prompt 和 target-contact actor filtering。
policy rollout 只有在 seed eligibility 通过且 `policy_executed=true` 时才进入结果。
v3 的字符串类型错误与 v3b 接触 gate 假阳性都发生在 frozen operation universe 之后。v2 audit
现在保持同一 23-operation 分母，只做 retrospective status review，把其中 v2 inactive-arm 两项改为
error，得到 8/23=34.78%；两个后续 failure 在 manifest 中单列，不进入该分母。

## 6. 当前第一性原理排序

1. **在预算内先安排 clean anchor/control，再做 property perturbation。** post-budget clean seed1000
   也失败，Planner 随后仍需要 seed1010 paired control；当前调度没有在三轮预算内获得可归因证据。
2. **把单例 TaskGen 提升为 target + physical distractor。** checker 必须同时判断命中目标与未误击
   distractor，并用 expert 正例、误击/未击中负例和 1 次 ACT 验收。
3. **效率先做预注册独立 arms，再放大到同一 suite 的 N=3。** v4 是 outcomes 已知后的 post-hoc
   cached-prefix counterfactual，没有实际 saving；只有独立执行和重复 seed 后仍与
   dense reference 一致且节省真实成本，才开始接近 Tables 1–2。
4. **policy ranking 先补第三个可运行 policy 或增加预注册 seeds。** 当前 ACT/DP3 tie 对排序没有信息；
   不应为追求排序而 post-hoc 换 seed。DP/DP3 的轻量路径优先，RDT/π0 暂缓。
5. **用独立人工标注替换 development proxy。** 在用户安排多人前，可继续冻结 Query/clip universe
   和盲评表，但不能再用 agent proxy 提升 validity claim。
6. **把有效的 bounded Tool 接入 Agent/VQA，并让 Table 3 进入真实 downstream gate。** v3b 仍是
   standalone cached-real；下一步最多 1 ACT live confirmation，并让 Rule/VQA evidence 进入 Planner。
   proposal ablation 的全部 1/1 通过只能说明开关可运行；下一步至少
   5 个 matched unseen proposals/condition，并纳入 render/simulator/oracle failure。

## 7. 紧凑证据

仓库内发布 [`batch19_claim_evidence/`](evidence_runs/batch19_claim_evidence/)：

- open-query plans、三轮 evidence，以及 post-budget clean anchor/final plan；
- generated scene/checker 的 bundle 与 ACT episode；
- Query-induced ToolGen v3b 的三次 routing 结果、输入/响应与 registry，以及 v3c 的
  no-target-contact null 结果；v2 无效 artifact、v3 类型失败与 v3b 无效数值仅作 failure audit；
- fixed/adaptive efficiency、ACT/DP3 exact-seed ranking；
- proxy validity、proposal prompt ablation 与 frozen error distribution。

原始视频、checkpoint、telemetry arrays、provider prompt/response 和完整 generated task 保留在服务器
ignored artifact 目录，不进入 Git。该 bundle 是可审查摘要，不替代 server machine audit。
