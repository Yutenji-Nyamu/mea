# Batch30 provider-written Python ToolGen

这是一次 0-ACT 在线方法验收：复用 live13 已完成的真实 `grab_roller`
telemetry，不启动 simulator 或 policy。

第一次运行暴露了两个可迁移失败：返回 NumPy 标量，以及使用 AST allowlist 外的
`.all()`。提示据此补充 JSON 原生标量、纯测量 `passed=None` 和 allowlist 约束。
第二次运行中，首次生成仍错误地把 Query 的 checker 阈值写入 `passed`；一次局部
repair 后通过 AST、双次确定性、独立 MetricSpec oracle 和 artifact 不变性检查，
注册为 run-local Tool。第二个 Query 精确复用同一代码，provider 调用为零。

- [机器摘要](result.json)
- [最终生成 Tool](generated_tool.py)
- 完整 prompt/response/attempt audit 保留在机器摘要记录的服务器路径。

这证明的是 provider-written Python 实现与同 evaluation exact reuse，不是跨
evaluation reviewed reuse；semantic need 仍受五个 typed operator 限制。
