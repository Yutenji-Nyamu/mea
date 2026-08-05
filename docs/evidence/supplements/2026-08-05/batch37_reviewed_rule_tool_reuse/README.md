# Batch37 reviewed Rule Tool reuse

这份补充证据只回答一个方法问题：已审查的生成式 Rule Tool 能否跨 evaluation 精确复用，而不再次调用 provider 或增加 rollout。

- 源 Tool：`terminal_minimum_tcp_to_stapler_distance`
- 复用路由：`reviewed_persistent_reuse`
- 新 evaluation：0 rollout、0 provider call
- Tool 代码哈希：`a4c4c06b48a5ac0ea45c2c676ab8ef43cbffb564459f46084ead3a64adc8982b`，源与复用一致
- 源值：`0.08901826217769938 m`
- 新 episode 上重新验证的值：`0.08965935664029238 m`
- 完整性、telemetry schema、确定性和 oracle agreement gate 均通过

这证明的是 Tool artifact 生命周期中的 reviewed persistent exact reuse；它不增加新的 policy-performance sample，也不证明指标在更多任务或 seed 上泛化。精简的机器可读事实见 [reuse_evidence.json](reuse_evidence.json)。
