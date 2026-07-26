# LIBERO / SmolVLA 复现与 MEA 接入

本文是项目中唯一的 LIBERO 文档：先记录已跑通的独立 adapter smoke，再定义
ManipEvalAgent（MEA）迁移到 LIBERO 的最小接口和协议。它不把 LIBERO 的单回合结果
混入 RoboTwin、policy ranking 或采样效率结论。

命令与事实使用三种来源标记：

- **[L 日志确认]**：可由服务器安装元数据、Hub `.metadata`、保留的 runner 或结果直接核实。
- **[I 环境反推]**：当前环境能确定等价命令，但非交互 SSH 没有保存原始 CLI 字符串。
- **[R 建议]**：下一次从零重建或扩展时建议采用的写法，不声称本次原样执行。

官方入口：

- [LeRobot 安装](https://huggingface.co/docs/lerobot/main/installation)
- [LeRobot LIBERO](https://huggingface.co/docs/lerobot/main/libero)
- [LeRobot SmolVLA](https://huggingface.co/docs/lerobot/main/smolvla)
- [HuggingFaceVLA/smolvla_libero](https://huggingface.co/HuggingFaceVLA/smolvla_libero)
- [lerobot/libero-assets](https://huggingface.co/datasets/lerobot/libero-assets)
- [LIBERO upstream](https://github.com/Lifelong-Robot-Learning/LIBERO)

仓库入口：

- [adapter contract gate](../experiments/paper/libero_adapter_smoke.py)
- [紧凑 smoke 结果](../experiments/paper/results/batch23_claim_closure/libero_smolvla_smoke_v1.json)
- [100-step method-chain compact 结果](../experiments/paper/results/batch24_libero_method_chain_v2/compact_result.json)
- [Planner–TaskGen 协议审计](../experiments/paper/results/batch24_libero_method_chain_v2/evidence/protocol_audit.json)
- [batch24 四项结果索引](../experiments/paper/results/batch24_claim_closure/summary.json)

## 1. 已核实环境与产物

| 项目 | 已核实值 |
|---|---|
| OS / GPU | Ubuntu 22.04.4；RTX 4090 24564 MiB；driver 580.105.08 |
| CUDA | driver 报告 13.0；system toolkit 12.4.131；PyTorch build cu130 |
| Python / PyTorch | 3.12.13；2.11.0+cu130 |
| LeRobot / LIBERO | LeRobot 0.6.0；hf-libero 0.1.4；MuJoCo 3.8.1；robosuite 1.4.0 |
| 其他关键包 | torchvision 0.26.0；transformers 5.5.4；huggingface-hub 1.24.0 |
| Conda prefix | `/root/autodl-tmp/envs/mea-libero`，约 7.1 GiB |
| checkpoint | `/root/autodl-tmp/checkpoints/libero/smolvla_libero`，约 1.2 GiB |
| assets | `/root/autodl-tmp/cache/libero/assets`，423,025,310 bytes |
| cache / tmp | `/root/autodl-tmp/cache/{huggingface,pip,libero}`；`/root/autodl-tmp/tmp/mea-libero` |

三种 CUDA 数字并不冲突：驱动的最高兼容级别、系统编译 toolkit 和 PyTorch wheel
自带 runtime 是不同对象。

checkpoint 固定信息：

```text
repo: HuggingFaceVLA/smolvla_libero
revision: 6721902bc4d61e50a3bfdb11dfb4cb626f05d102
model.safetensors bytes: 1,218,047,032
model.safetensors sha256:
71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f
loaded parameter count: 604,934,176
```

8 个 checkpoint `.metadata` 条目都指向上述 revision，且没有 checkpoint
`.incomplete` 文件。assets 固定信息：

```text
repo: lerobot/libero-assets
revision: 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6
on-disk bytes: 423,025,310
non-cache payload files: 586
```

586 个 asset `.metadata` 条目都指向同一 revision。单回合 eval 不需要下载
`HuggingFaceVLA/libero` training dataset；BDDL 与 init states 来自 hf-libero wheel。

## 2. 从零安装

### 2.1 Conda 与依赖

`conda-meta/history` 保存了本次原命令：

```bash
# [L 日志确认]
/root/miniconda3/bin/conda create -y \
  -p /root/autodl-tmp/envs/mea-libero \
  python=3.12 pip
```

LeRobot 是 dist-info 中的直接请求，但 dist-info 不保存原始 extras/index CLI。下面是
与当前依赖闭包一致的等价命令：

```bash
# [I 环境反推]
env \
  PIP_CACHE_DIR=/root/autodl-tmp/cache/pip \
  TMPDIR=/root/autodl-tmp/tmp/mea-libero \
  /root/autodl-tmp/envs/mea-libero/bin/python -m pip install \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    'lerobot[smolvla,libero,evaluation]==0.6.0'
```

本次网络记录：

1. 阿里云 PyPI mirror 缺 `num2words`，切换清华 PyPI 后安装完成。
2. 服务器直连 `huggingface.co` 超时，后续使用 `https://hf-mirror.com`。
3. mirror 的 Xet 路径曾返回 401，设置 `HF_HUB_DISABLE_XET=1` 后完成。
4. assets 默认并发触发 429；以 `--max-workers 1` 断点续传完成。

checkpoint、assets、环境与临时文件始终留在 `/root/autodl-tmp`，没有经过本地
Windows。

### 2.2 activation 与 LIBERO config

当前 activation script：

```bash
# [L 日志确认] 当前文件内容
# /root/autodl-tmp/envs/mea-libero/etc/conda/activate.d/10_mea_libero_paths.sh
export HF_HOME="/root/autodl-tmp/cache/huggingface"
export HUGGINGFACE_HUB_CACHE="/root/autodl-tmp/cache/huggingface/hub"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="1"
export PIP_CACHE_DIR="/root/autodl-tmp/cache/pip"
export TMPDIR="/root/autodl-tmp/tmp/mea-libero"
export XDG_CACHE_HOME="/root/autodl-tmp/cache"
export LIBERO_CONFIG_PATH="/root/autodl-tmp/cache/libero/config"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
```

`MUJOCO_GL=egl` 是 headless server 的渲染设置。当前
`/root/autodl-tmp/cache/libero/config/config.yaml`：

```yaml
assets: /root/autodl-tmp/cache/libero/assets
bddl_files: /root/autodl-tmp/envs/mea-libero/lib/python3.12/site-packages/libero/libero/bddl_files
benchmark_root: /root/autodl-tmp/envs/mea-libero/lib/python3.12/site-packages/libero/libero
datasets: /root/autodl-tmp/cache/libero/datasets
init_states: /root/autodl-tmp/envs/mea-libero/lib/python3.12/site-packages/libero/libero/init_files
```

package assets 入口当前是：

```text
/root/autodl-tmp/envs/mea-libero/lib/python3.12/site-packages/libero/libero/assets
  -> /root/autodl-tmp/cache/libero/assets
```

### 2.3 固定 revision 下载

Hub CLI 不保存原始调用；下面根据当前 local-dir `.metadata` 反推，并保留本次验证过的
镜像和低并发策略：

```bash
# [I 环境反推 + R 低并发]
env \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  HF_HOME=/root/autodl-tmp/cache/huggingface \
  HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/cache/huggingface/hub \
  TMPDIR=/root/autodl-tmp/tmp/mea-libero \
  /root/autodl-tmp/envs/mea-libero/bin/hf download \
    HuggingFaceVLA/smolvla_libero \
    --revision 6721902bc4d61e50a3bfdb11dfb4cb626f05d102 \
    --local-dir /root/autodl-tmp/checkpoints/libero/smolvla_libero \
    --max-workers 1

# [I 环境反推 + R 低并发]
env \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  HF_HOME=/root/autodl-tmp/cache/huggingface \
  HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/cache/huggingface/hub \
  TMPDIR=/root/autodl-tmp/tmp/mea-libero \
  /root/autodl-tmp/envs/mea-libero/bin/hf download \
    lerobot/libero-assets \
    --repo-type dataset \
    --revision 0b3ea86be5fe169d0fd036ae63d1070ec09e90f6 \
    --local-dir /root/autodl-tmp/cache/libero/assets \
    --max-workers 1
```

assets cache 留有一个首次失败下载产生的 0-byte marker：

```text
/root/autodl-tmp/cache/libero/assets/.cache/huggingface/download/scenes/plant/
vGWgFE6ib_8Jk7UJGhUcBnUcGOQ=.63ba9f699457c837b40edd0c46c478c68c822954.24127605.incomplete
```

payload、586 份 metadata 和真实 rollout 均已通过；该 marker 保留，不作为重复下载理由。
当前磁盘空间充足，runtime 所需 env/checkpoint/assets 不清理；约 8 GiB pip cache 仅在不再
需要快速重装时单独处理。

### 2.4 CPU contract gate

下面是与已保存结果一致的等价命令；结果 artifact 是日志确认，原始调用字符串未单独保存：

```bash
# [I 环境反推；输出为 L 日志确认]
/root/autodl-tmp/envs/mea-libero/bin/python \
  /root/autodl-tmp/mea/experiments/paper/libero_adapter_smoke.py \
  --checkpoint /root/autodl-tmp/checkpoints/libero/smolvla_libero \
  --load-model \
  --device cpu \
  --output \
  /root/autodl-tmp/mea/mea/protocol_runs/batch23_libero_smolvla_adapter_smoke/model_load_cpu.json
```

该 gate 验证两路 `[3,256,256]` 图像、8D state、7D action contract，并在 CPU
完整加载 604,934,176 个参数；它不是 rollout。

## 3. 已跑通的 official control

成功 runner 保存了实际命令：

```bash
# [L 日志确认]
run_root=/root/autodl-tmp/mea/mea/protocol_runs/batch23_libero_smolvla_adapter_smoke/live_eval_task0_seed100800_20260726T1242_retry1

env \
  HF_HOME=/root/autodl-tmp/cache/huggingface \
  HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/cache/huggingface/hub \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DISABLE_XET=1 \
  XDG_CACHE_HOME=/root/autodl-tmp/cache \
  PIP_CACHE_DIR=/root/autodl-tmp/cache/pip \
  TMPDIR=/root/autodl-tmp/tmp/mea-libero \
  LIBERO_CONFIG_PATH=/root/autodl-tmp/cache/libero/config \
  MUJOCO_GL=egl \
  /root/autodl-tmp/envs/mea-libero/bin/lerobot-eval \
    --policy.path=/root/autodl-tmp/checkpoints/libero/smolvla_libero \
    --policy.device=cuda \
    --policy.load_vlm_weights=false \
    --policy.n_action_steps=10 \
    --env.type=libero \
    --env.task=libero_object \
    '--env.task_ids=[0]' \
    --env.control_mode=relative \
    --env.max_parallel_tasks=1 \
    --eval.n_episodes=1 \
    --eval.batch_size=1 \
    --eval.recording=false \
    --seed=100800 \
    --output_dir="${run_root}/eval_output"
```

task 0 是 `pick_up_the_alphabet_soup_and_place_it_in_the_basket`。关键边界：

- `load_vlm_weights=false` 使用本地完整 finetuned state，不另取 base VLM。
- checkpoint config 的 `n_action_steps` 默认是 1；本 smoke 显式改为 10。因此结果只验证
  可运行性，不能当作标准 SmolVLA benchmark 数字。
- `relative` 对应 LIBERO 的 7D delta end-effector action。
- 即使 `recording=false`，LeRobot 0.6.0 仍输出了约 67 KiB episode MP4。
- 此 runner 没有显式实施论文的 100-step 对齐 horizon；LeRobot 日志允许最多 280
  steps。因此这次 1/1 结果只是环境/推理 smoke，不是论文对齐实验。

成功产物：

```text
mea/protocol_runs/batch23_libero_smolvla_adapter_smoke/
├── adapter_contract.json
├── model_load_cpu.json
└── live_eval_task0_seed100800_20260726T1242_retry1/
    ├── eval_output/eval_info.json
    ├── eval_output/videos/libero_object_0/eval_episode_0.mp4
    ├── stdout.log
    ├── stderr.log
    ├── gpu_samples.csv
    ├── time.txt
    └── exit_code.txt
```

结果为 1/1 success、reward 1.0、官方 eval 31.13 s、总 wall 89 s、逐秒采样的
GPU 总显存峰值 2512 MiB。它是单回合 feasibility smoke。

首次启动目录 `live_eval_task0_seed100800_20260726T1240/` 保留 exit 127 和错误：

```text
/root/autodl-tmp/tmp/mea-libero/run_libero_smolvla_once.sh:
line 11: /usr/bin/time: No such file or directory
```

失败发生在模型与 simulator 启动前。唯一 retry 只把 `/usr/bin/time` 改为 shell
timestamps；policy、checkpoint、task、seed 和 eval 参数未变，之后没有继续重试。
robosuite private macro 与 `OpenGL_accelerate` 缺失是本次的非致命警告。

## 4. RoboTwin 与 LIBERO 的语义边界

| 维度 | 当前 RoboTwin 主链 | LIBERO / 当前 smoke | 迁移要求 |
|---|---|---|---|
| task representation | Python scene/task code 与 `check_success()` | `.bddl` 本身是 problem definition，并引用已注册 Python Problem/domain 实现 | `TaskContract` 保存 BDDL 定义及实现引用，不虚构第二个 Problem 文件 |
| simulator API | RoboTwin/SAPIEN task 与项目 ACT/DP3 eval 入口 | official suite 可走 `lerobot-eval --task_ids`；custom `.bddl` 需直接构造 `OffScreenRenderEnv`/custom env factory | `LiberoBenchmarkAdapter` 统一 reset/step/render/success，policy processor 继续复用 LeRobot |
| observation/action | task/policy adapter 提供多视角、状态与策略动作 | 两路图像、8D state、7D relative action | `LeRobotPolicyAdapter` 明确映射、归一化与 action chunk |
| success authority | official 或 TaskGen 生成的 `check_success()` | BDDL goal predicates，经 domain predicate evaluator 给 reward/success | 保留 official truth；实验 predicate 必须单独标识 |
| init/randomization | Python scene 参数、seed 与 expert scene | BDDL init、suite/task id、bundled init-state arrays | bundled state 仅在 compatibility probe 通过后复用；记录 state 来源与 hash |
| policy rollout | ACT/DP3 专用 evaluation entry | `lerobot-eval`/LeRobot policy processor | 扩展现有 normalized `episode.json` 为 benchmark-neutral `EpisodeRecord` |
| telemetry/tool hooks | 已有 state、video、telemetry、Rule/VQA hooks | 当前只保存 aggregate、视频与有限 env 输出 | adapter 暴露 predicate/state/contact/action，而非事后猜测 |
| render | SAPIEN 首帧、多相机与 TaskGen visual gate | MuJoCo EGL，agent/wrist cameras | 统一首帧与 clip 引用，但保留 simulator camera 语义 |
| 当前结果 | RoboTwin 的小范围 MEA 证据链 | SmolVLA task0 单回合成功 | task、policy、sim、action、seed 均不同，数值不可直接比较 |

## 5. MEA on LIBERO 设计

继续复用现有方法层：

```text
QueryContract
→ ClaimFirst Planner
→ Proposal
→ Tool registry / VQA
→ Aggregate
→ AnswerScope
```

新增 4 个实现单元和 2 个轻量 schema，不复制第二套 Planner、registry、VQA 或
provenance 系统：

| 单元 | 最小职责 |
|---|---|
| `LiberoBenchmarkAdapter` | official suite/task 与 custom `.bddl` env factory，reset/step/render/close，official reward/success 与 predicate state；`capability_inventory()` 直接生成 Global router 已消费的 schema |
| `LeRobotPolicyAdapter` | checkpoint identity，observation key/shape 映射，pre/postprocessor，action chunk 与 control mode |
| `LiberoTaskGen backend` | Phase 1 生成 state-compatible `.bddl` 并复用注册 class；更改 workspace/camera/execution/check 逻辑时才生成/修改并注册 Python class |
| `LiberoTool/Rule backend` | 只提供 benchmark-specific predicate/state/action extractor 与 `BDDLBaseDomain` binding；复用现有 ToolGen validation、registry 和 Aggregate |
| `TaskContract` schema | `bddl_path/text`、`problem_name`、`domain`、parsed entities/predicates、`python_problem_impl`、`initial_state_source` 及其 refs/hashes |
| `EpisodeRecord` schema | 对现有 normalized `episode.json`/recorder contract 的 benchmark-neutral 扩展，保存 episode、TaskContract、policy、reward、predicate trace 与 artifact refs |

`EpisodeRecord` 投影到当前 telemetry root、video、keyframes、Rule/VQA 和 Aggregate，
不新建第二套 recorder/evidence tree。现有 VQA 继续读取同一 frame/clip；Tool backend
只补 extractor，不新建第二个 registry 或 Aggregate。

### 最小注入点

- `scripts/manipeval_agent.py --benchmark {robotwin,libero}` 只选择现有 adapter factory；
  不复制 Agent CLI、Query router 或 Planner。
- `scripts/manipeval_taskgen.py --benchmark {robotwin,libero}` 只 dispatch 对应 TaskGen
  backend。
- `LiberoBenchmarkAdapter.capability_inventory()` 输出当前 Global router 已消费的
  capability schema；不新增 `CapabilityCatalogAdapter`。
- 当前 episode recorder 写标准 `episode.json`，LIBERO 字段作为 benchmark-neutral
  extension 投影到既有 telemetry/video/keyframes/Rule/VQA/Aggregate 路径。

### TaskGen

论文 Appendix A.3.2 将 LIBERO adaptation 概括为 task BDDL definition 与 Python
Problem class 两部分；这不表示每个生成 task 都必须产生两个独立文件。当前 hf-libero
中，`.bddl` 本身就是 problem definition，其中的 `problem_name`/`domain` 引用已注册
Python Problem/domain 实现：

- **Phase 1**：生成一个 state-compatible `.bddl`，复用已有注册 class。它可以修改兼容的
  init/goal predicates，但不能增删会改变 simulator state 的对象/region，也不能改
  workspace、camera、execution 或 check 逻辑。
- **Phase 2**：对象/region 变化时生成并保存兼容 initial state；只有 Proposal 确实要求
  改变 Python execution/workspace/camera/check 逻辑时，才生成或修改 Python class，
  显式注册后再由 `.bddl` 引用。Phase 2 不在本轮 2-rollout 预算中。

`TaskContract` 至少保存：

```text
bddl_path, bddl_text, bddl_sha256
problem_name, domain
parsed_entities, parsed_predicates
python_problem_impl, python_problem_source_ref, python_problem_source_hash
initial_state_source, initial_state_index, initial_state_hash
```

bundled init state 只能在 state shape、entity handles、reset/set-state 与 render
compatibility probe 全部通过后复用。增删对象/region 时必须生成并保存兼容初始状态，
或像 Phase 1 一样直接限制 variation；不能把 official init array 无条件套到 custom env。
只有 parser、implementation registry、init compatibility、reset/render 与 predicates
fixtures 都通过，variation 才能进入 rollout。

stock `lerobot-eval --env.task_ids` 只会选择 official suite task，不能执行任意 custom
`.bddl`。`LiberoBenchmarkAdapter` 对 custom task 直接以 `.bddl` 构造
`OffScreenRenderEnv` 或已注册 custom env factory，再把 observation/action 交给同一个
`LeRobotPolicyAdapter`，复用 LeRobot policy pre/postprocessor 与 rollout 逻辑。

### ToolGen

ToolGen 应从 Query/Proposal 的 predicate need 出发，并通过 LIBERO
`BDDLBaseDomain` 的 detector / `eval_predicate_fn` 接口生成或复用 detector：

```text
Query need
→ predicate / metric specification
→ BDDLBaseDomain detector or eval_predicate_fn binding
→ positive/negative/missing-state fixtures
→ run-local validation
→ existing Tool registry
→ EpisodeRecord / episode.json
→ Aggregate / Planner
```

official predicate truth 与额外 metric 必须并列保存，不能让模型生成的 detector 静默覆盖
official success。`LiberoTool/Rule backend` 只实现 extractor/binding；Tool spec、
validation、register/reuse 与 Aggregate 继续走现有公共实现。VQA 只补充可见现象，
不能代替不可见 simulator predicate。

## 6. 最小 2-rollout 协议

目标是验证 MEA 方法链能跨到 LIBERO，不验证效率、policy ranking 或跨模拟器一致性。
论文协议默认每个生成 task 运行 5 trials，并让 LIBERO 与 RoboTwin 使用对齐的
100-step horizon。本节为了最低成本只运行 official control N=1 与 generated
variation N=1；实现必须在 adapter 中显式记录并限制每个 episode 为 100 steps。
未实施该 horizon 或未记录实际 steps 时，结果只能叫 environment smoke。

### Gate 0：0 policy rollout

1. 运行 CPU adapter contract：checkpoint、两路图像、8D state、7D action、processor
   和本地完整权重加载全部通过。
2. `LiberoBenchmarkAdapter`、`LeRobotPolicyAdapter`、`TaskContract` 与
   `EpisodeRecord` 通过通用
   construct/serialize/close contract。
3. 一个固定 official fixture 的 `.bddl` 可解析，能解析到已注册 Python
   Problem/domain 实现，并可 reset、EGL render；保存首帧。
4. `LiberoTaskGen backend` 只跑通用 parser、对象/region 引用、init/goal 与正负
   fixture contract；不生成本次 Query 的 variation。
5. `LiberoTool/Rule backend` 只验证通用 `BDDLBaseDomain` detector /
   `eval_predicate_fn` 正负与 missing-state fixtures；不生成本次 Query 的 Tool。

任一 gate 失败都停止，不消耗 policy rollout。

### Rollout 1：official control

- 冻结开放 Query、`QueryContract`、SmolVLA checkpoint、official task、seed、init index、
  relative control、100-step horizon 和 rollout budget，再执行 official task 1 episode。
- `LiberoBenchmarkAdapter` 与 `LeRobotPolicyAdapter` 扩展现有 recorder，写入标准
  `episode.json` / `EpisodeRecord`，并投影到既有 telemetry root、video 与 keyframes。
- Rule/VQA/Aggregate 消费同一 episode。
- control 若失败，不能把 variation 差异归因于 TaskGen；返回 inconclusive。

### Rollout 1→2：evidence-conditioned generation gate

- ClaimFirst Planner 只在收到 control `EpisodeRecord` 后生成具体 Proposal。
- Phase 1 只接受 state-compatible Proposal：`LiberoTaskGen backend` 生成一个新的
  `.bddl`，复用同一已注册 Python Problem/domain class；运行 parser、implementation
  registry、init-state compatibility、reset/render 和 predicate fixtures。
- 若 Proposal 需要增删对象/region，本协议返回 unsupported/deferred；Phase 2 需生成并
  保存兼容 initial state。只有确需改变 Python execution/workspace/camera/check 逻辑时，
  Phase 2 才生成/注册新 class。
- `LiberoTool/Rule backend` 再为该 Query 生成或检索 detector/metric，通过
  `BDDLBaseDomain` detector / `eval_predicate_fn` fixtures 后注册。
- 这一阶段是 **0 additional SmolVLA rollout**；任一 gate 失败都不进入 Rollout 2。

### Rollout 2：generated variation

- `LiberoBenchmarkAdapter` 不走 stock `--task_ids`，而是从 custom `.bddl` 直接构造
  `OffScreenRenderEnv`/custom env factory；同一 `LeRobotPolicyAdapter` 复用 LeRobot
  policy processor/rollout。
- 同一 checkpoint、seed 策略、通过 compatibility probe 的 initial-state source、
  control mode 和 100-step budget 执行 state-compatible variation。
- 至少一个 Query-induced detector/metric 得到非空 live 值，影响 Aggregate/Planner；
  第二次相同需要可在 0 additional SmolVLA rollout replay 中 exact reuse。
- variation 成功或失败都可接受，但 simulator 必须合法完成，official 与 experimental
  semantics 必须分别报告。

### 本批执行结果：机制通过，协议 fail-closed

batch24 v2 真实执行了同一 seed 的 official control 与 custom variation，各 100 steps，
总计 2 episodes。Gate 0、provider-written BDDL、显式 `OffScreenRenderEnv` custom
factory、确定性 predicate MetricSpec adapter 的非空 live 值及 0-rollout exact reuse
均跑通；没有把 custom 文件伪装成 stock task id。该 adapter 来自有界 schema 编译，
不是模型现场生成的新 Tool。

但它不能支持 policy 结论，原因有两层：

1. 未改变任务的 100-step official control 已失败，因此 custom failure 不能归因于
   object identity。
2. ClaimFirst Proposal 明确要求 language-only、semantics-preserving 变化，TaskGen
   却把 goal object 改成 `salad_dressing_1`。`planner_taskgen_alignment=false`。

因此 compact result 为 `completed_with_protocol_violation`，protocol audit 为
`protocol_invalid`，AnswerScope 以 `pipeline_invalid` 停止；
`query_contract_sufficient=false`、`scientific_evidence_eligible=false`。这两回合只能
保留为 component mechanism smoke；Query contract sufficiency 与 scientific evidence
eligibility 是两个独立字段。该历史运行早于 alignment gate；
当前 runtime 已在 TaskGen provider 和 custom rollout 之前拒绝未授权 controlled
change，不再为同类错配消耗第二回合。

### 验收条件

- 两个 episode 都有完整且可对齐的 `TaskContract` 与 `EpisodeRecord`，并进入现有
  `episode.json`/telemetry/video/keyframes/Rule/VQA/Aggregate 路径。
- 第二轮 variation 确由第一轮 evidence 与 Proposal 触发，不是预写 task id 切换。
- Phase 1 variation 是真实 state-compatible `.bddl` 变更并复用已注册 class，不是
  prompt overlay、虚构第二个 Problem 文件或旁路 checker。
- custom rollout 确实通过 `OffScreenRenderEnv`/custom env factory 执行，而不是把
  custom 文件名误传给只支持 official suite 的 stock `--task_ids`。
- Rule/VQA/Aggregate/Planner/AnswerScope 消费同一证据并回应原 Query；证据不足、冲突或
  control 失败时，`answered=false` / `inconclusive` 是合法且优先于强行作答的结果。
- Answer 强制列出每个 condition N=1、总计 2 episodes、100-step limit 与实际 steps、
  单 policy、单 seed、未覆盖 suite/task/属性和停止原因。
- 结论名称只能是 `LIBERO method-chain smoke`；不得写成 Tables 1/2/4/5/9 复现、
  sample saving、policy ranking 或 RoboTwin↔LIBERO 一致性。

## 7. 论文边界

- Appendix A.3.2 对 LIBERO 的描述只支持 **basic adaptation**。它不证明 LIBERO
  已达到论文在 RoboTwin 上的同规模主链完整度。
- Tables 1/2/4/5/9 含 LIBERO 参照结果，但这些表不自动证明 Query→TaskGen→ToolGen→
  evidence-conditioned Planner 的完整主链在两个 simulator 上一致。
- Table 10 明确不保证 absolute correctness；结论仍受 LLM、simulator fidelity、
  observation、predicate/tool 和有限 samples 约束。
- 当前 1/1 SmolVLA success 只证明 official evaluator、LIBERO environment 与
  SmolVLA inference/eval 路径能端到端运行，不证明 MEA adapter。它不能与 RoboTwin
  ACT/DP3 数值直接比较，也不能支持效率、排名或泛化结论。
- batch24 的 2×100-step method-chain 进一步证明 BDDL/custom env/确定性 predicate
  MetricSpec adapter/reuse 机制可执行，但没有证明模型生成新 Tool；control failure
  与 Planner–TaskGen misalignment 使整条科学协议无效，它同样不能支持 SmolVLA
  robustness 或 RoboTwin↔LIBERO 一致性结论。
