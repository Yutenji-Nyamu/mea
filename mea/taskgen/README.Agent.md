# TaskGen global output rules

本文档只记录所有 TaskGen 生成都适用的规则。任务不变量、API 语义与示例代码由 Knowledge Retriever 按需注入，避免 prompt 重复。

- 输出完整 `def load_actors(self):`，不能调用 `super()`。
- 仅修改 validated VariantSpec 明确要求的变量。
- 保持官方 actor attributes、identity、碰撞属性及随机调用顺序。
- 不覆盖 `play_once()` 或 `check_success()`。
- wrapper 已提供 `np`、`sapien`、`create_actor`、`create_box` 与
  `rand_pose`；生成方法不得 import。
- 不访问文件、网络、环境变量、进程或动态执行接口。
- 使用显式 literal 表达 VariantSpec 要求的颜色和尺度，便于 AST 检查。
