# TaskGen global output rules

## Paper-claim execution

- A validated `bbh.experimental_bounded_act` SuccessSpec may evaluate a policy
  rollout after TaskGen acceptance.
- Its outcome must be named `generated_check_success`; never relabel it as
  `official_check_success`.
- Claim evidence requires the generated scene and compiled checker to be bound
  in the same TaskArtifactBundle and exercised by the same recorded rollout.
- Engineering guardrails must protect that evidence path, not silently replace
  the paper claim with a probe-only surrogate.

本文档只记录所有 TaskGen 生成都适用的规则。任务不变量、API 语义与示例代码由 Knowledge Retriever 按需注入，避免 prompt 重复。

- 输出完整 `def load_actors(self):`，不能调用 `super()`。
- 仅修改 validated VariantSpec 明确要求的变量。
- 保持官方 actor attributes、identity、碰撞属性及随机调用顺序。
- 不覆盖 `play_once()` 或 `check_success()`。
- wrapper 已提供 `np`、`sapien`、`create_actor`、`create_box` 与
  `rand_pose`；生成方法不得 import。
- 不访问文件、网络、环境变量、进程或动态执行接口。
- 使用显式 literal 表达 VariantSpec 要求的颜色和尺度，便于 AST 检查。

## 功能验收边界

`scripts/manipeval_taskgen_acceptance.py` 只读核验 official reuse、受限 overlay、真 codegen /
retrieval provenance 和视觉错误修复的既有 artifact。它不调用 provider、simulator 或 ACT，
因此只能声称 cached functional acceptance，不能声称新实验或论文表结果。视觉修复的合格
证据必须包含：正常静态 gate 通过、首帧视觉拒绝、有具体 diagnosis、修复安装后重新通过
静态/protected-file gate、第二次视觉通过；已知漏检的 oversized fixture 不作正证据。
