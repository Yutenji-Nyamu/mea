# 2026-07-23：batch20 集成主链与论文 claim 实验

本批按论文 Abstract、Figs. 2–6、Tables 1–9 与 Appendix A.1 自顶向下实现和验收。
所有 provider、RoboTwin、ACT、DP3、TaskGen、expert/probe、真实 rollout 和最终回归测试都在
canonical AutoDL `/root/autodl-tmp/mea` 执行；Windows 只用于临时文本 staging 和一次 runner
上传前语法编译，没有运行 simulator/policy，checkpoint 也未经过 Windows。本批未修改根
`README.md`。

## 1. 结果总览

| 论文 claim | 本批真实增量 | 当前结论 |
| --- | --- | --- |
| Fig. 2/5：Query → evidence-conditioned planning → 足够时回答 | 一个统一 runtime 内完成 official control、ACT、Rule/VQA、Aggregate、provider 下一步、instance TaskGen、第二次 ACT、充分性停止和 Feedback | **有限域机制成立**；仍是预声明候选 catalog、existential Query、同 seed N=2，不是开放世界或统计泛化 |
| Fig. 3：同一 Proposal 生成 scene + `check_success()` 并评价 rollout | provider 生成 BBH target+distractor `task.py`，3 个 checker fixture、3 个 expert probe、1 个 ACT | **单例路径成立但未 production accepted**；seed1012 只证明 checker=false，不能称 physical distractor mis-hit |
| Fig. 4：新 Query 诱发 Tool，验证、注册、复用 | 新 jerk Tool 完成 provider generation、3 个 oracle、run-local register；相同 Query 第二次 exact reuse 且 provider=false | **生成/复用机制成立**；真实 ACT 上无 target contact，value=null、validated_episode_count=0，未形成有效 policy metric |
| Tables 1–2：更少 samples/time 且保持结论 | 独立 fixed 4 ACT 与 adaptive 2 ACT，真实节省 rollout、wall 和 policy steps | **负结果**：adaptive 漏掉 instance 弱轴，结论不一致，因此不复现论文效率 claim |
| Table 9：少样本保持 policy ranking | ACT/DP3 exact three-seed pilot | **严格协议未完成**：ACT 3/3 执行、2/3 成功；DP3 仅 2/3 执行且两次成功，seed 100602 在 expert eligibility gate 失败。总计 5/6 policy rollouts，pair order 与 Spearman 均为 null |
| Table 3：RAG/visual/README 提升生成成功率 | 1 个 unseen Proposal × 5 condition 均真实走 provider/codegen/static/render/expert/oracle | **仅 path smoke**：5/5 全过且 checker 仍由 restricted compiler 生成，不能估计消融效果 |
| Tables 6–8 / Fig. 6 | 冻结 proxy universe 与前瞻 ledger 可执行 | 仍无独立 human gold；错误分母仍是开发操作，不可与论文约 5% 比较 |

## 2. Fig. 2/5 unified ClaimFirst runtime

Query：

> Can this ACT click_bell policy succeed on at least one supported bell-property variation?

证据根目录：

`mea/evaluation_runs/eval_20260723_batch20_claim_first_live_n1_v3`

1. Round 1 是 unchanged official control，seed 102000，ACT 1/1 success，Rule/VQA/Aggregate
   均通过。
2. 读取 Round 1 evidence 后，公共 provider Planner 选择 `object_instance`；支持的 controlled
   axes 通过 capability card 明示，但具体 template/candidate ID 未暴露，semantic resolver 再将
   明确 change intent 绑定到 `object_instance.base0`。
3. Round 2 用同一 seed 102000 执行 base0，ACT 1/1 success，Rule/VQA/Aggregate 通过。
4. existential Query 的有限真值合同已满足，系统在仍剩一轮预算时以
   `evidence_sufficient` 停止，不是 hard cap。
5. 修复后的 AnswerScope 明示 N=2、seeds=[102000]、只测 base0，未测 left/right/base1，
   并声明这不是统计泛化保证。

它关闭了“开发者手工串三轮”和“只能因 max-round 停止”的旧 gap，但只是 axis-primed
finite-domain planning；没有证明 Fig. 5 的 ambiguity→refinement，也没有让 Planner 在未知开放
空间中发现全新 concern。

## 3. TaskGen 与 ToolGen 的真实边界

### 3.1 provider-generated distractor task

根目录：

`mea/generated_tasks/run_20260723_batch20_bbh_distractor_live_v1`

- `task.py` 来自 provider response Python，`generated_by_model=true`，
  `restricted_success_spec_compiler_used=false`；但 prompt 和 validator 仍要求
  `bbh_distractor_exact_ast_v1`，自主性受限。
- 三个 synthetic checker fixture 为 target-only=true、distractor-latched=false、
  no-contact=false，3/3 通过。
- 三个真实 expert probe 均 setup/render/rule 通过；seed1011/1013 checker=true，
  seed1012 checker=false。
- seed1012 的 events 明示 hammer–distractor `physical_contact=false`，所以它不是已证实的
  物理误击负例。
- ACT seed1011 完成 400 policy steps、error=null，generated checker=false。
- 旧 candidate manifest 仍写 probes=0/ACT=0；新证据不能据此升级成 production accepted。

### 3.2 Query-induced jerk Tool

- 首次自然 Query 走 `provider_generate_validate_register`，smooth、oscillatory、
  missing-contact 三个 oracle 全通过。
- 第二次完全相同 Query 走 `exact_query_registry_reuse`，provider/codegen/validation/register
  均未再次启动。
- 对真实 ACT telemetry，Tool 返回 `null_reason=no_target_contact_event`；
  Aggregate 外壳通过，但该 metric 是 valid=0、missing=1。

所以本批证明的是 Tool lifecycle，不是 jerk policy evidence，也尚未证明新 Tool 会改变下一轮 Planner。

## 4. Tables 1–2 独立 arms toy：真实节省、结论不一致

最终有效协议：

`mea/protocol_runs/batch20_click_bell_efficiency_toy_v3`

seed 100401 是依据历史 clean/randomized 对照选择的 development seed，不是无偏抽样。fixed 与
adaptive 是独立 rollout，未共享 cached prefix。

| arm | candidates | outcomes | ACT starts | measured wall | policy steps |
| --- | --- | --- | ---: | ---: | ---: |
| fixed | left, right, base0, base1 | F, T, T, F | 4 | 342.931 s | 942 |
| adaptive | left, right | F, T | 2 | 172.635 s | 472 |

观测节省：

- 2/4 ACT starts（50%）；
- 170.296 秒（49.66%）；
- 470 policy steps（49.89%）。

但 fixed 的 weakness axes 是 `object_position + object_instance`，adaptive 只恢复
`object_position`。因此 `original_query_conclusion_agrees=false`，
`toy_efficiency_evidence_passed=false`。这是一条可反驳的负结果：当前 stopping contract
会过早停止，不能支持论文“更少 samples/time 且结论可比”的 claim。

首次 v2 在 ACT 前因 seed manifest 只有一个 condition 而失败；修复为 clean + unused 两个合法
condition 后才生成 v3。v2 不进入 policy 分母。

## 5. ACT/DP3 exact-seed ranking pilot

最终有效协议：

`mea/protocol_runs/batch20_act_dp3_exact3_v5`

| policy | seed 100600 | seed 100601 | seed 100602 | 完成数 |
|---|---:|---:|---:|---:|
| ACT | success | success | failure | 3/3 |
| DP3 | success | success | expert ineligible、未运行 policy | 2/3 |

共同完成的两个 seed 上，两种 policy 都是 2/2 success，形成 tie；第三个 seed
没有可比较的 DP3 rollout，因此不能定义 pair order 或 Spearman。没有做事后 seed
替换，也没有把 5/6 冒充为完整 six-rollout pilot。

v4 的三个 ACT 已完成，但首个 DP3 在加载 checkpoint 前失败：字符串 `"false"` 被旧
`eval()` 参数解析保留为 truthy，错误选择 `_w_rgb_0` checkpoint。修复为 Python boolean literal
`"False"` 后重新预注册 v5；v4 不进入最终 pair-order 结果。

无论 v5 结果如何，它都只是 two-policy、one-task、three-seed pilot，不是论文 Table 9 的五 policy、
10/20/50 rollout 或 bootstrap Spearman。

## 6. Table 3 与 validity/error evidence

Table 3 path smoke：

`mea/protocol_runs/batch20_table3_path_smoke_v3/path_smoke_results.json`

- complete/base/no-RAG/no-visual/no-README 五格均 provider.called=true、returncode=0；
- static validation、scene setup/render、expert、oracle 均通过；
- complete、no-RAG、no-README 的 visual self-check 分别通过；
- scene code 来自 provider，但五格 checker 均
  `success_spec_provenance.generated_by_model=false`、
  `compiler=restricted_success_spec_v2`；
- 只有一个 Proposal，没有跨 Proposal 失败率或独立盲评，
  `paper_table3_eligible=false`。

Plan/VQA proxy 仍为 development-agent gold；8 个 VQA slots 只有 2 个真实 clips，
human_reviewer_count=0。现有 prospective ledger 可用于后续冻结分母，但本批两个 protocol bug
说明当前系统错误率远不能与论文 Fig. 6 的约 5% 直接比较。

## 7. 本批代码级增量

- 公共 ClaimFirst Planner 接入真实 Rule/VQA→Aggregate runtime，并加入 query sufficiency 与
  AnswerScope 绑定。
- 显式 change intent 优先于 preserve 文本，候选 universe 受全局 Query route 限制。
- provider-generated BBH distractor scene+checker 与 bounded Tool lifecycle。
- 预注册独立 efficiency/ranking/Table3 协议、服务器 live runner、结果从磁盘
  seed_results/episode 重算，拒绝手填 score。
- efficiency 同时累计 ACT starts、实测 wall 和 policy steps/sample proxy。
- DP3 `use_rgb` 传参改为真实布尔字面量；seed manifest 与 runtime 的 paired schema 对齐。
- 新增可选 execution receipt：在 simulator/model 前绑定 task source、seed/index、ACT
  `policy_last.ckpt + dataset_stats.pkl`，episode 记录实际 imported module/checkpoint bundle。
  这是证据完整性条件，不是论文贡献。

## 8. 第一性原理上的下一批

1. **修 stopping contract，而不是继续挑 seed。** 当前 toy 已证明“发现一个弱轴就停”会漏掉第二
   弱轴。下一批先区分 existential、worst-axis、coverage/compare Query，再用缓存 truth table，
   最多 1–2 ACT 验证。
2. **做非 existential 的 Fig. 5 ambiguity→refinement。** 预算 2–3 ACT；第二/三轮具体
   sub-aspect 必须由上一轮 ambiguous evidence 产生，不能只是按 catalog 顺序 switch。
3. **TaskGen 补真实物理反例。** 去掉 exact reference AST，构造 target-only、
   physical-distractor-hit、no-contact 三个可诊断 probe，再裁决 0–1 ACT。
4. **ToolGen 先找 contact-positive telemetry。** 0 ACT 复用足够；只有找不到时再跑 1 ACT。
   必须得到 finite metric、Rule/VQA 一致并进入 Planner，不能把 null Aggregate 当成功。
5. **Table 3 扩剩余 4×5 cells。** 0 ACT，冻结盲化 rubric；若继续 ceiling，就诚实判为
   inconclusive，而不是声称模块都有效。
6. **人工 validity 后续由独立人员完成。** 当前 agent proxy 只用于管线验收，不提升为论文
   Tables 6–8。
