# 2026-07-24：batch21 低成本论文 claim 闭环

本批按论文 Abstract、Figs. 2–6、Tables 1–9 和 Appendix 自顶向下收口，但把成本限制在
ACT 的少量真实 rollout、0-ACT 缓存验证和 development-agent proxy。DP3、RDT、π0、
LIBERO 与大规模重复实验明确后置。所有单元/集成测试、provider、VQA、expert、simulator
和 ACT 均在 canonical AutoDL `/root/autodl-tmp/mea` 执行；Windows 只用于小文件
staging 与进程内密码 SSH。本批没有修改根 `README.md`。

## 1. 本批代码闭环

- `PlanAgentPrototype` 现在真正消费 CLI 的 `start_seed/num_episodes`，并把同一 seed
  override 传播到后续 evidence-conditioned rounds；省略 override 时仍保留各 planner
  原有可信默认值。
- ClaimFirst control 的 official outcome 与辅助 VQA 分层：official success、pipeline
  receipt 和 policy result 决定 control 是否有效；辅助视觉冲突保留为 evidence conflict，
  但不再覆盖数值 authority。
- `AnswerScope` 可 fail-closed 地表达 `control_not_passed`，并禁止在 control 失败时进行
  属性归因。
- official ACT-only TaskGen acceptance 不再伪造缺失的 expert evidence；只有真实要求
  expert 的 generated route 才将 expert 计入 acceptance。
- reviewed Task registry 中的失效条目绝不复用，但会留下
  `invalid_registry_fallback_to_generation` 审计记录，并让无关的新 Proposal 继续受限生成。
- 项目 playbook 已把 policy 边界改为“默认 ACT；用户明确授权时才扩第二策略”。当前只允许
  ACT+DP；DP 未就绪时保持 blocked，禁止用 DP3 代替，也不扩 RDT/π0。

代码修复提交为 `dd350f6`。完整 live v5 在提交前的同一工作树上执行，原始 manifest
仍记录旧 base commit `e5975f5`；运行所用源文件随后未再修改并由 `dd350f6` 封存。因此该
live artifact 是有效的开发验收，但其 manifest 不能被误读为 clean-commit execution receipt。

## 2. Fig. 2/5：统一开放 Query 主链

真实 evaluation：

`mea/evaluation_runs/eval_20260724_batch21_semantic_needs_live_n1_v5`

Query：

> Can this beat_block_hammer ACT policy succeed on at least one bounded novel
> block appearance under an explicit experimental success predicate, and what
> new telemetry observable is needed to judge it?

结果：

1. runtime 先执行不变的 official control，seed 100600，ACT 在 policy step 98 成功。
2. 读取真实 Rule/VQA/Aggregate evidence 后，公共 ClaimFirst Planner 产生新的语义
   sub-aspect `generalization.novel_block_appearance_contact_success`。
3. semantic resolver 将其绑定为 TaskProposal v2
   `object_appearance.color.query_generated_1` 和 ToolProposal v3
   `object_appearance.color.query_metric_1.tool`；binding 明确要求新 scene、实验
   SuccessSpec 和新的 typed metric。
4. TaskGen 生成蓝色 block 场景，通过 render、视觉自检、rule、expert solvability 和
   production acceptance；在 acceptance 前 `act_rollouts_started=0`。
5. 第二个 ACT 在相同 seed 100600 上完成 400 policy steps，但
   `generated_check_success=false`。
6. 新工具 `hammer_block_functional_xy_min_distance` 经 typed MetricSpec 编译和差分 gate，
   在真实 ACT telemetry 上得到 `0.0891936 m`，同时在 expert telemetry 上得到
   `0.00298617 m`，随后进入 Aggregate。
7. Query 是有限候选上的 existential claim。唯一实测 novel candidate 失败且另一个候选
   未测，因此系统因预算耗尽输出 `claim_verdict=inconclusive`，而不是把 N=2 宣称成泛化。

本案例证明的是“真实 control evidence → 模型语义 Proposal → scene/checker/Tool →
第二次 ACT → Aggregate → 受限 Answer”的统一执行链。它不证明开放世界发现能力，也不证明
ACT 对新外观的统计泛化。

## 3. Fig. 3/4：TaskGen 与 ToolGen 的边界

v5 的新 scene 与 checker 由同一 TaskProposal v2 绑定，checker 实际裁决 ACT；不过成功
判据来自受限 SuccessSpec compiler，属于实验语义，不等同于 official benchmark checker。

另外，已有 bounded component run
`run_20260723_batch20_bbh_distractor_live_v1` 保留了更强的 model-code provenance：

- provider 同时写出 target+distractor scene 和 `check_success()`；
- `generated_by_model=true`，未使用 restricted SuccessSpec compiler；
- target-only、distractor-latched 和 no-contact 三个 checker fixture 全部通过；
- 一次 ACT 由该 checker 裁决为 false；
- 自然 Query 生成 `active_tcp_precontact_peak_jerk`，完成 oracle、注册和第二次 exact
  Query 的 provider-free reuse。

该旧 component run 的 jerk 值因真实 episode 没有 target contact 而为 null；v5 的新
XY metric 有真实数值但尚未做第二 Query 精确复用。因此目前两个必要性质分别存在，但还没有
在同一个统一 ClaimFirst run 中同时出现。

## 4. Tables 1–2：真实独立 arms 效率 toy

协议：

`mea/protocol_runs/batch21_position_universal_n1/live_execution/efficiency_result.json`

同一个有限 universal Query、同一 ACT checkpoint 和 seed 100402：

| arm | 实测候选 | outcome | ACT | wall time | policy steps |
| --- | --- | --- | ---: | ---: | ---: |
| fixed | left, right | false, true | 2 | 165.694 s | 470 |
| adaptive | left | false 后按 universal 反例停止 | 1 | 107.543 s | 400 |

fixed 和 adaptive 对原 Query 都得到 `refuted`，adaptive 实际节省 1 个 ACT、58.151 秒和
70 policy steps。这是第一条正向的独立 live mechanism toy，不是缓存反事实。

边界同样明确：候选只有两个位置、只有一个 seed，且 adaptive 的完整 weakness-axis
摘要比 fixed 少；所以 `paper_tables_1_2_eligible=false`，不能替代论文的 dense benchmark、
10 次 agent trial 或多任务结果。

## 5. Table 9：只保留 ACT+DP

readiness artifact：

`mea/protocol_runs/batch21_claim_closure/act_dp_readiness.json`

- ACT checkpoint、dataset stats、RoboTwin 环境均已就绪。
- DP 的 `600.ckpt` 和独立 `RoboTwin-DP` 环境均不存在。
- 预注册了 ACT/DP 各 3 个相同 seeds（100600–100602），但本批启动
  `ACT=0、DP=0、training=0、download=0`。
- 官方数据树提供 ACT、DP3、RDT 权重目录，但没有 DP checkpoint；官方 DP 文档给出的是
  数据处理、600-step 训练和 eval 流程。因此当前正确状态是
  `blocked_missing_prerequisites`，不是用 DP3 冒充 DP。

## 6. Tables 3、6–8 与 Fig. 6 的低成本证据

- Table 3：已有一个 unseen Proposal × complete/base/−RAG/−visual/−README 的五条件
  provider/render/expert path smoke；五格都通过且 checker 仍来自 restricted compiler，
  因而不能估计模块增益。新 evaluator 会拒绝 proposal-only、缺 checker 或虚假 model-code
  provenance；完整 5×5 暂不运行。
- Plan proxy：`query_validation_batch21_claim_closure_n5` 对 5 条分层 Query 做了 5 次真实
  provider 调用，development-agent proxy micro-F1 为 1.0；`human_reviewer_count=0`，
  不能算论文人类一致性。
- VQA proxy：`validation_vqa_proxy_batch21_contact_n3_v1` 使用 3 个缓存真实 montage
  （1 negative、2 positive）派生 clean/clutter-image/texture-image/lighting-image 共 12 个
  case，单一 VLM 的 proxy accuracy/AUROC 均为 1.0。非 clean 条件只是图像变换，v4/v5
  positive 还是相同视觉内容，且没有人工 gold/多 VLM，因此
  `paper_table_eligible=false`。
- Fig. 6：v5 在运行前冻结 5 个 operation；最终 Plan、TaskGen、ToolGen、simulator、
  AnswerScope 全部 terminal passed，得到 0/5。该分母只用于证明 prospective ledger
  会完整计数，不能与论文约 5% 或其模块分布比较。

## 7. 服务器验收

- targeted 组合测试：47/47；registry fallback 定向测试：18/18。
- 最终完整 suite：683/683，55.395 秒，解释器为服务器
  `/root/autodl-tmp/conda/envs/RoboTwin/bin/python`。
- `git diff --check` 和相关 JSON 校验通过。
- checkpoint、视频、telemetry、validation run 与日志全部只留在服务器，未经过 Windows，
  也不进入 Git。
- 根 `README.md` 保持不变。

## 8. 第一性原理后的剩余 gap

1. **结论保持的适用域仍很窄。** universal falsification 可以一次反例早停，因此本次正向
   efficiency toy 是最容易的情形。下一批应比较 existential、worst-case 或 compare Query，
   看自适应停止是否仍保持完整结论；先 3–5 ACT，再决定是否 N=3。
2. **统一主链仍使用受限 SuccessSpec compiler。** provider-written distractor
   scene/checker 在独立 component path 已成立，但还没由公共 ClaimFirst Planner 自动选择并
   进入同一个 evidence bundle。最小集成可复用现有 artifact，预计 0–1 ACT。
3. **“新 Tool 有值”与“第二 Query 复用”尚未同例。** 对 v5 的 XY metric 做缓存 telemetry
   exact reuse 即可，预计 0 ACT；不必再造一个 metric。
4. **ACT/DP ranking 还没有数据。** 若得到匹配 DP checkpoint，先做 3+3 exact-seed
   pair-order；否则不训练、不下载大包。该结果仍不是五 policy Spearman。
5. **生成模块消融缺少统计功效。** 1×5 ceiling path smoke 不能证明 RAG、visual 或
   README.Agent 有增益；应先在真正 provider-generated scene+checker 上做一组 5 条件，
   再决定是否扩到 5×5。
6. **独立有效性仍缺。** Plan 需要机器人研究者 gold；VQA 需要独立人工、真正 simulator
   clutter/texture/lighting 和多个 VLM。当前 proxy 只验证管线和指标计算。
7. **Fig. 6 仍缺稳定的大分母。** 以后每个真实 run 自动加入冻结 roster，累计到有意义的
   分母后再讨论错误率与模块分布。

按科学信息增益排序，下一批建议是：**v5 Tool 的 0-ACT 精确复用 → provider-written
distractor route 接入统一 Planner（0–1 ACT）→ 非 universal 的 3–5 ACT 效率 toy →
DP artifact 就绪后 ACT/DP 3+3**。大策略和 LIBERO 继续后置。
