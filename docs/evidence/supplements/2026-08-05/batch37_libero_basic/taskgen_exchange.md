# LIBERO TaskGen provider exchange（精简）

## Query

> How robust is SmolVLA to task-relevant object identity changes in LIBERO, and where does it first fail?

## Evidence-conditioned Proposal

official control 已成功（reward 1.0，135 steps）。Plan Agent 提出最小身份替换：保持 problem/domain、objects、regions、初始状态、camera、workspace、action mode、horizon 和 policy checkpoint，只把 task-relevant object identity 从 `alphabet_soup_1` 改成一个已经存在于场景中的非 basket object，并请求 goal-predicate Tool。

## TaskGen prompt 的关键约束

- 只改 `:language`、`:obj_of_interest` 和 `:goal`。
- 选择恰好一个已有且不是 `alphabet_soup_1`/`basket_1` 的对象。
- sole goal 必须是 `(In <selected_object> basket_1_contain_region)`。
- language、obj_of_interest 与 goal 必须引用同一对象。
- 不得增加 object、region、comment 或代码。

## Provider 响应摘要

```json
{
  "selected_object": "salad_dressing_1",
  "rationale": "Substitutes salad_dressing_1, an existing non-basket object, while preserving all fixtures, objects, regions, and initial predicates."
}
```

完整响应中的 BDDL 与实际执行文件相同，未在这里重复；见 [generated_task.bddl](generated_task.bddl)。
