# Batch19 claim-first compact evidence

本目录只收录 2026-07-23 batch19 的小型结构化证据。原始视频、checkpoint、telemetry arrays、
provider prompt/response 与 generated task 仍在 canonical AutoDL ignored artifact 目录。

最重要的边界：

- open Query 是 capability-conditioned 的手工链：template ID/顺序隐藏，但受控轴公开；没有统一
  VQA/Aggregate/Feedback runtime。post-budget clean seed1000 也失败，property-specific weakness
  不可归因；Planner 继续要求 seed1010 clean-vs-blue paired control，原 Query 未回答；
- fixed/adaptive v4 是 outcomes 已观察后的 post-hoc cached-prefix counterfactual；actual rollout/wall
  saving 均为 null。1 次/50% 只是 counterfactual avoidable count，82.306 s 仅为 estimate，不是
  观测到的 adaptive speedup 或 Tables 1–2；
- ACT 与 DP3 同 seed 各 0/1，排序为 tie，Spearman 不可计算；
- Query-induced Tool v3b 只证明受限 DSL、synthetic oracle、run-local register 与
  paraphrase/exact reuse routing。其 43.4348 数值来自 `physical_contact=false` event，已经失效；
  v3c 修正 gate 后，两个 cached-real route 均返回 null/no-target-contact。因此没有 policy jerk
  evidence、校准 threshold、有效 episode、Agent/VQA 集成或新 ACT live confirmation；
- Plan n=5 来自 legacy taxonomy-routing proxy；VQA 是 8 条 heterogeneous、手选、未预注册的
  cached predictions。它们只是 development-agent protocol smoke，不是人工有效性或视觉 robustness；
- proposal prompt ablation 每 condition 只有一次，只验证 structured proposal，不是 codegen；
- error distribution 是冻结 23-operation universe 的 retrospective review：8/23=34.78%，不是论文
  重复实验；两个 post-universe Tool failure 另列且不进入分母。

文件与原服务器来源、证据等级和限制见
[`evidence_bundle_manifest.json`](evidence_bundle_manifest.json)。
