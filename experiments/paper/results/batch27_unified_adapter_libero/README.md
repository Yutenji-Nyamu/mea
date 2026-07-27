# Batch 27：统一 TaskAdapter、开放 concern 边界与 LIBERO 两回合链

本目录只保留能复核本批结论的紧凑产物。大体积原始 telemetry、逐帧图像和完整运行目录留在服务器，不重复提交。

## 三条数据流

### 1. catalog 外 concern（0 rollout）

`开放 Query → provider FreeConcern → capability inventory → TaskNeed / ToolNeed → fail-closed`

- Query 只询问目标物体质量变化，没有给定已有 aspect。
- Planner 将其解析为 `object_physics.mass`，判定为 `catalog_external`。
- 现有 click_bell catalog 不能执行该变化，因此保留 TaskNeed、ToolNeed，并令 `execution_authorized=false`。
- 本例证明“能发现并明确表达 catalog 外需求”，不证明系统已能执行新 concern。

关键文件：`catalog_external/free_concern.json`、`catalog_external/concern_candidate_resolution.json`。

### 2. 第五个 RoboTwin official adapter

`TaskAdapter → official scene → expert gate → ACT N=1 → Rule / VQA → Aggregate → AnswerScope`

- 新增 `place_phone_stand` 的数据驱动 TaskAdapter、schema 和 ACT checkpoint 绑定。
- expert 在 seed 100501 的 official 任务成功；ACT 在同一 seed 的一次 episode 失败。
- ACT rollout 为 400 policy steps、75.89 s wall time；VQA 与 official checker 均未观察到成功。
- 这只证明第五个任务的 official 评估接口可运行。单个失败 episode 不能说明策略稳定弱点，也不构成第三个深入 TaskGen 案例。

关键文件：`place_phone_stand/evaluation_plan.json`、`scene.png`、`rollout.mp4`、`execution_vqa.json`、`answer.json`。

### 3. LIBERO / SmolVLA 两回合方法链

`Query → official control → Planner → TaskGen prompt/response → generated BDDL → first-frame gate → paired custom rollout → generated Tool → Aggregate → exact reuse → AnswerScope`

- `libero_object/task0` official control 成功。
- Planner 选择新的目标物体；TaskGen 写出 custom BDDL，首帧 gate 通过。
- paired custom rollout 对 `salad_dressing_1` 失败；生成的 goal-predicate Tool 得到非空 `false`，并在第二次 Query 中 exact reuse，未增加 rollout。
- 共 2 rollouts、132.698 s；`method_chain_valid=true`，但仍有四个未测目标，故 `query_contract_sufficient=false`，最终结论为不确定。
- 这是 LIBERO basic adaptation 的结构性 smoke，不是 object-identity robustness、跨模拟器一致性或论文性能结论。

关键文件：

- 模型交互：`libero/planner_prompt.md`、`planner_response.txt`、`taskgen_prompt.md`、`taskgen_response.txt`
- 生成物：`libero/generated_task.bddl`、`generated_tool.py`、`tool_registration.json`
- 视觉与 rollout：`libero/custom_first_frame.png`、`official_rollout.mp4`、`custom_rollout.mp4`
- 证据与回答：`libero/official_evidence.json`、`custom_evidence.json`、`aggregate_summary.json`、`tool_reuse_result.json`、`answer_scope.json`、`compact_result.json`

## 原始服务器目录

- catalog 外 concern：`/root/autodl-tmp/mea/mea/evaluation_runs/eval_20260727_batch27_catalog_external_mass_v2`
- place_phone_stand ACT：`/root/autodl-tmp/mea/mea/evaluation_runs/eval_20260727_batch27_place_phone_act_v1`
- LIBERO：`/root/autodl-tmp/mea/mea/evaluation_runs/eval_20260727_batch27_libero_seed_parity_v3`

统一摘要见 `summary.json`；checkpoint 来源见 `checkpoint_provenance.json`。
