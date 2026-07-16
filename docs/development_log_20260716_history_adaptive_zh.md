# 2026-07-16：History Retrieval 与自适应轮次开发记录

## 修改范围

- 新增 SQLite evaluation history 与 rebuild CLI；
- 将相似 completed planning 注入初始 Plan Agent prompt；
- 新增 deterministic `EvidenceAssessment`；
- 支持同 template、fresh seed、最多一次的 `verify` round；
- history retrieval、assessment 与 stop/verify 原因进入 artifacts 和最终 report；
- 保持根 `README.md` 与官方 `policy/ACT/eval.sh` 不变。

## 兼容性发现

第一次 rebuild 得到 0 条记录，因为旧 evaluations 产生于
`lifecycle_status` 字段加入之前。随后只为满足以下全部条件的旧 run 增加兼容导入：

- manifest `status` 精确为 `completed`；
- 有 `execution_finished_at`；
- plan、evidence bundle 等 required artifacts 存在。

对旧 plan 的 `object_appearance.color` 与 `object_position` 只做明确的一对一
template 映射。最终导入 6 条；缺 evidence、plan-only、手工 control-plane 或无
manifest 的目录均未导入。

## 提交前审查修复

- 修复 `verify` decision 已生成、但未追加到持久化 plan，因而不会真正执行的问题；
- `rebuild` 改为优先读取 canonical `summary/history_record.json`，只在旧 run
  尚无 canonical record 时从原 artifacts 迁移；
- legacy 迁移新增跨 artifact 的 `evaluation_id`、executed round 数量与
  `round_id` 一致性检查；
- 为以上边界新增 3 项回归测试。

## 验证与日志

- 定向测试 24 项：`_ops_logs/history_adaptive_targeted_20260716.log`；
- 初次完整测试 108 项：`_ops_logs/history_adaptive_full_tests_20260716.log`；
- legacy history 回归后 109 项：
  `_ops_logs/history_adaptive_final_tests_20260716.log`；
- 审查修复后的定向测试 28 项：
  `_ops_logs/history_adaptive_postreview_targeted_20260716.log`；
- 审查修复后的最终完整测试 112 项：
  `_ops_logs/history_adaptive_postreview_full_20260716.log`；
- legacy rebuild：`_ops_logs/history_rebuild_legacy_20260716.log`；
- final rebuild：`_ops_logs/history_rebuild_final_20260716.json`；
- canonical 优先规则复验：
  `_ops_logs/history_rebuild_postreview_20260716.json`（6/6 来自 canonical record）；
- live history Plan smoke：
  `_ops_logs/history_retrieval_plan_smoke_v2_20260716.log`；
- adaptive real-artifact smoke：
  `_ops_logs/adaptive_real_artifact_smoke_20260716.json`。

最终关键结果：

- 服务器最终完整测试 112/112；
- history database 6 条 completed records；
- live query 命中 3 条历史；
- clear evidence 为 `sufficient/stop`；
- injected conflict 为 `evidence_conflict/verify`；
- verification route=`reuse`、seed=`100001`、episode=1；
- real-artifact smoke 的 ACT/GPT external calls 均为 0。
