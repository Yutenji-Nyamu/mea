# Batch37 LIBERO basic adaptation

这份证据记录 SmolVLA 在 `libero_object/task0` 上的两回合 basic adaptation：official control 成功后，Plan Agent 根据 evidence 提出 object-identity concern，TaskGen 写出 custom BDDL，第二次 rollout 失败，生成式 predicate Tool 取得 live `false` 并在第二个 Query 中 0-rollout exact reuse。

核心事实：

- Query：`How robust is SmolVLA to task-relevant object identity changes in LIBERO, and where does it first fail?`
- official control：seed 100800，135/280 steps，reward 1.0，success=true
- custom：把目标从 `alphabet_soup_1` 改为场景中已有的 `salad_dressing_1`；280/280 steps，reward 0.0，generated goal predicate=false
- 总计 2 rollouts，elapsed 163.621 s；相同 seed、checkpoint、初始状态协议
- Tool：`libero_goal_predicate_tool`，live non-null `false`；第二个 Query 走 `exact_registry_reuse`，0 新 rollout
- Aggregate：1 个 custom episode，predicate true rate 0/1

重要限制：这是论文所称的 **LIBERO basic adaptation**，不是 RoboTwin 旗舰方法链的等价复现。运行记录中 `method_chain_valid=false`、`plan_agent_active_stop_validated=false`、`scientific_evidence_eligible=false`；最终停止来自有限 diagnostic QueryContract，而不是 Plan Agent 主动 stop。custom checker/predicate 是 experimental semantics，不等价于 official benchmark success；N=2 且只有一个 seed，不能推出一般 object-identity robustness。

- [taskgen_exchange.md](taskgen_exchange.md)：Proposal、TaskGen 约束与响应摘要
- [generated_task.bddl](generated_task.bddl)：实际执行的 provider-written BDDL
- [evidence_summary.json](evidence_summary.json)：Tool、Aggregate、Answer 与限制
- [first_frame.png](first_frame.png)：custom task 首帧
- [official_episode.mp4](official_episode.mp4)、[custom_episode.mp4](custom_episode.mp4)：两次 rollout
