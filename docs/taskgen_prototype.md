# TaskGen 原型说明

## 目标

当前原型把自然语言场景修改请求变成一个可审计、可执行的 RoboTwin task
variant。它不复制完整 RoboTwin 仓库或官方场景文件，而是在
`mea/generated_tasks/<run_id>/` 下生成薄 subclass。

对于 `beat_block_hammer`，GPT 必须生成完整的 `load_actors()` method body，
不能通过 `super().load_actors()` 委托给现有实现。由可信代码生成的只有固定
module wrapper、class declaration 和运行清单。

## 调用链

```text
自然语言请求
  -> UIUI GPT 生成 VariantSpec
  -> reuse / force_codegen 路由
  -> UIUI GPT 生成完整 load_actors()
  -> AST / import / protected-file 检查
  -> setup-only scene load + render
  -> rule check + UIUI vision check
     -> 失败时 diagnosis + 完整 load_actors() repair（最多 2 次）
     -> AST / render / rule / Vision revalidation
  -> expert gate
  -> ACT 1-episode evaluation
  -> 同一 run_id 下保存全部证据
```

`force_codegen` 是当前主验证路径；即使仓库中已有蓝色方块实现，也要求 GPT
重新生成 method。`reuse` 用于后续复用已审核 variant，当前原型保留路由字段，
但不作为首次垂直验证的重点。

## GPT 上下文与代码契约

代码生成 prompt 会提供：

- 官方 `envs/beat_block_hammer.py` 完整源码；
- 当前可配置的 `mea.tasks.beat_block_hammer` 实现；
- `blocks_ranking_rgb` 和 `stack_blocks_two` 的 actor API 示例；
- `mea/taskgen/README.Agent.md` 中的输出格式与允许 API 约束；
- 已解析的 `VariantSpec`。

模型只能返回一个完整的 `def load_actors(self): ...`。Validator 会检查：

- 顶层只能有一个 `load_actors()`；
- 禁止 import、文件、网络、process、dynamic execution 等调用；
- 禁止 `super()` 委托；
- 保留 BeatBlockHammer 所需的 hammer、block、prohibit area 和 actor 创建调用；
- 生成代码中的 literal color 与 `VariantSpec` 一致；
- 官方 task、官方 `eval.sh` 和参数化 evaluator 的 hash 没有变化。

这不是通用 Python sandbox，而是针对当前 prototype 的窄接口验证。生成文件只在
上述检查通过后才允许 import 和执行。

## 运行目录

每次新请求创建：

```text
mea/generated_tasks/run_<timestamp>_<random>/
├── request.txt
├── prompt/
├── response/
├── spec.json
├── task.py
├── overlay.yml
├── manifest.json
├── validation/
├── evidence/
└── evaluation/
```

这些 `run_*` 目录包含模型原始输出、图像、视频与实验结果，默认被 Git ignore；
TaskGen 源码、测试和说明文档进入 Git。API key 只从环境变量读取，不写入 run
directory。

## 使用方式

一次执行完整蓝色方块垂直链路：

```bash
export UIUI_API_KEY='...'
python scripts/manipeval_taskgen.py \
  --request '把 beat_block_hammer 的红色方块改为蓝色，其他行为保持不变' \
  --mode force_codegen \
  --probe \
  --vision-check \
  --max-reflections 2 \
  --expert \
  --run-act
```

从已生成的 run 继续某个 gate，不会再次调用 code generation：

```bash
python scripts/manipeval_taskgen.py \
  --resume-run run_YYYYMMDD_HHMMSS_xxxxxxxx \
  --probe \
  --expert
```

只有请求 `--vision-check` 时才需要 UIUI key；单纯 resume ACT evaluation 不需要
provider credential。

## 首次真实验证

- run id：`run_20260714_113855_de3af020`
- request：把 `beat_block_hammer` 的红色方块改为蓝色，其他行为保持不变。
- model：`gpt-4o-2024-11-20`
- GPT 生成内容：完整 `load_actors()`，结构与官方 BBH method 一致，只将方块
  color 改为 `[0.0, 0.2, 1.0]`；没有 `super()`。
- AST validation：通过，共 254 nodes；protected-file hash 检查通过。
- setup/render：通过；seed 100000 下成功创建 hammer 和蓝色 block。
- rule check：通过。
- UIUI vision check：识别为 `blue`，未报告 unexpected changes，confidence 1.0。
- expert gate：planning 和 `check_success()` 均通过。
- ACT evaluation：完整执行并正常退出，视频为 320x240、10 FPS、40 seconds；
  policy 在 400 steps 后为 0/1 success。

因此本次结论是：TaskGen 生成、验证与执行链路通过，且场景修改被视觉证据确认；
ACT policy 本身在该 episode 未完成任务。这与先前官方同 seed baseline 的 0/1
结果一致，不能把 policy failure 误判为 TaskGen pipeline failure。
