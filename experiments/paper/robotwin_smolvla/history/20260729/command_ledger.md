# SmolVLA RoboTwin 服务器部署与五任务最小验证流水账

记录日期：2026-07-29（Asia/Shanghai）

## 1. 范围与执行边界

- 目标服务器：SeetaCloud `connect.bjb2.seetacloud.com:30735`。
- SSH：当前进程内通过 `Paramiko.Transport.start_client()` +
  `auth_password()` 认证；本文不记录密码。
- 所有 checkpoint、metadata、环境克隆、模型加载、processor 探针、仿真和
  rollout 均在服务器完成，没有经过 Windows 本地磁盘。
- 未修改 `/root/autodl-tmp/mea`。最终只读快照：
  - HEAD `1455852adbfdb2931d142739c13c4c6beb3be4c1`
  - `git status --short | wc -l` 为 `351`；这是进入本批前已经存在的 dirty
    工作树，本批没有向其中写文件。
- 所有本批临时脚本和日志位于
  `/root/autodl-tmp/tmp/mea-smolvla-robotwin`。

## 2. 身份、磁盘、GPU 与已有环境

### 2.1 命令

```bash
date -Is
hostname
pwd
id -u
df -h /root/autodl-tmp
nvidia-smi \
  --query-gpu=name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

### 2.2 结果

```text
2026-07-29T15:09:11+08:00
autodl-container-ujxcycmw77-dd0e7d70
/root
0
/dev/md0 879G total, 57G used, 823G available, 7%
NVIDIA GeForce RTX 4090, 24564 MiB, 0 MiB, 0%
```

已有 LeRobot/LIBERO 环境：

```bash
/root/autodl-tmp/envs/mea-libero/bin/python - <<'PY'
import sys
print(sys.version)
for name in ("torch", "transformers", "huggingface_hub", "lerobot"):
    module = __import__(name)
    print(name, module.__version__)
PY
```

结果：

```text
Python 3.12.13
torch 2.11.0+cu130
transformers 5.5.4
huggingface_hub 1.24.0
lerobot 0.6.0
```

已有 RoboTwin 环境：

```bash
/root/autodl-tmp/conda/envs/RoboTwin/bin/python - <<'PY'
import sys, torch, sapien, numpy, gymnasium
print(sys.version)
print(torch.__version__, sapien.__version__, numpy.__version__, gymnasium.__version__)
PY
```

结果：

```text
Python 3.10.20
torch 2.4.1+cu121
sapien 3.0.0b1
numpy 1.26.4
gymnasium 0.29.1
```

结论：磁盘和显存都足够；RoboTwin 仿真与 LeRobot 0.6 不在同一个 Python/ABI
环境中，后续必须隔离。

## 3. Hugging Face 网络诊断

### 3.1 初次 API 查询

在未设置镜像时执行：

```bash
/root/autodl-tmp/envs/mea-libero/bin/python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info("lerobot/smolvla_robotwin", files_metadata=True))
PY
```

结果：120 秒内没有返回，SSH 执行器按超时终止。此时没有下载或创建
checkpoint。

### 3.2 定点连通性检查

```bash
for url in https://huggingface.co https://hf-mirror.com; do
  curl -4 -L -I --connect-timeout 8 --max-time 15 \
    -sS -o /dev/null \
    -w 'code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
    "$url" || echo curl_failed
done
```

结果：

```text
https://huggingface.co: code=000, 4.00s connect timeout
https://hf-mirror.com: code=200, connect=0.116s, total=0.585s
```

诊断：服务器不能稳定直连 `huggingface.co`，但镜像正常。

修复：后续 Hub 查询和下载统一设置：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_DISABLE_XET=1
```

## 4. 固定官方 checkpoint revision

### 4.1 查询命令

```bash
export HF_ENDPOINT=https://hf-mirror.com
/root/autodl-tmp/envs/mea-libero/bin/python - <<'PY'
from huggingface_hub import HfApi
info = HfApi().model_info("lerobot/smolvla_robotwin", files_metadata=True)
print(info.id, info.sha, info.last_modified, info.private, info.gated)
for item in sorted(info.siblings, key=lambda x: x.rfilename):
    print(item.rfilename, item.size, getattr(getattr(item, "lfs", None), "sha256", None))
PY
```

### 4.2 固定值

- Repo：`lerobot/smolvla_robotwin`
- Revision：
  `967623a0f38c7e1236c66b3893c830398d793ff7`
- 公开、非 gated。
- `model.safetensors`：
  - 906,712,520 bytes
  - SHA-256
    `6fbaa809585cb10924351ce101ca2d576787a472cf8c34b5e2f10c84ef8a3134`

注意：模型卡说明训练数据是 `pepijn223/robotwin_unified_v3`，但没有发布
“50 个任务逐任务成功率”。训练覆盖不能解释成 50-task 已验证成功率。

## 5. 服务器下载 checkpoint

### 5.1 命令

下载由服务器后台进程执行，避免 SSH banner/长连接波动影响数据传输：

```bash
run=/root/autodl-tmp/tmp/mea-smolvla-robotwin
target=/root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin
mkdir -p "$run" /root/autodl-tmp/checkpoints/robotwin

nohup env \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_DOWNLOAD_TIMEOUT=60 \
  HF_HUB_DISABLE_XET=1 \
  /root/autodl-tmp/envs/mea-libero/bin/python -c \
  "from huggingface_hub import snapshot_download; \
print(snapshot_download(\
repo_id='lerobot/smolvla_robotwin',\
revision='967623a0f38c7e1236c66b3893c830398d793ff7',\
local_dir='$target'))" \
  > "$run/download.log" 2>&1 < /dev/null &
```

轮询：

```bash
pid=$(cat "$run/download.pid")
kill -0 "$pid"
du -sh "$target"
tail -30 "$run/download.log"
```

### 5.2 结果

- 开始：`2026-07-29T15:14:08+08:00`
- 完成：约 2 分 27 秒
- checkpoint 目录：865 MiB
- 文件数：9
- 日志显示 `Fetching 9 files: 100%`

### 5.3 完整本地校验

```bash
sha256sum /root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin/*
```

重要结果：

```text
6fbaa809585cb10924351ce101ca2d576787a472cf8c34b5e2f10c84ef8a3134  model.safetensors
d606283ad4f5b92495fcd93c51a3209116c4037d4ea9d1d0fe1b87da08409798  policy_preprocessor_step_5_normalizer_processor.safetensors
d606283ad4f5b92495fcd93c51a3209116c4037d4ea9d1d0fe1b87da08409798  policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

完整列表保存在：

```text
/root/autodl-tmp/tmp/mea-smolvla-robotwin/sha256sum.txt
```

## 6. Backbone metadata 的最小离线部署

SmolVLA checkpoint 的 `vlm_model_name` 是
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，模型初始化默认会再次读取其
config、processor 和 tokenizer。完整基础权重为 2,029,990,624 bytes，但
SmolVLA checkpoint 已包含最终 policy 权重。

先固定 backbone revision：

```text
7b375e1b73b11138ff12fe22c8f2822d8fe03467
```

只下载初始化所需 metadata，排除重复权重：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_DISABLE_XET=1

/root/autodl-tmp/envs/mea-libero/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    local_dir="/root/autodl-tmp/checkpoints/robotwin/SmolVLM2-500M-Video-Instruct-metadata",
    ignore_patterns=["*.safetensors", "*.bin", "*.onnx"],
))
PY
```

结果：13 个 metadata/tokenizer 文件，4.8 MiB，约 4 秒。

运行时设置：

```python
config.vlm_model_name = (
    "/root/autodl-tmp/checkpoints/robotwin/"
    "SmolVLM2-500M-Video-Instruct-metadata"
)
config.load_vlm_weights = False
```

这不会改变网络结构；最终用 `strict=True` 从 SmolVLA checkpoint 加载全部
参数，严格加载成功证明没有缺失或多余 key。

## 7. checkpoint 契约审计

### 7.1 静态字段

`config.json`：

```text
3 visual inputs:
  observation.images.camera1 [3,256,256]
  observation.images.camera2 [3,256,256]
  observation.images.camera3 [3,256,256]
observation.state [6]
action [14]
chunk_size=50
n_action_steps=50
max_state_dim=32
max_action_dim=32
```

`policy_preprocessor.json` 中训练数据 rename：

```text
cam_high -> camera1
cam_left_wrist -> camera2
cam_right_wrist -> camera3
```

### 7.2 发现的 state 维度漂移

- `config.json` 和 preprocessor JSON 的 feature 壳写 `state [6]`。
- normalization safetensors 的
  `observation.state.{min,max,mean,std}` 全部是 `[14]`。
- 当前 LeRobot RoboTwin wrapper 明确输出 `agent_pos [14]`。
- 真实 RoboTwin `get_obs()["joint_action"]["vector"]` 也是 `[14]`。

没有修改 checkpoint。用官方 processor 实际执行下面的运行时映射：

```python
rename = {
    "observation.images.head_camera": "observation.images.camera1",
    "observation.images.left_camera": "observation.images.camera2",
    "observation.images.right_camera": "observation.images.camera3",
}
```

真实 processor 探针结果：

```text
before:
  three images [1,3,240,320], CPU float32
  state [1,14], CPU float32
after:
  camera1/2/3 [1,3,240,320], CUDA float32
  state [1,14], CUDA float32, all finite
  language tokens [1,48]
CONTRACT_OK
```

模型在内部把短于 `max_state_dim=32` 的 state pad 到 32；因此运行时 14D
state 与 checkpoint 的 14D normalization stats 一致。`state [6]` 是静态
feature 壳漂移，不是本次人为裁剪输入。

## 8. 严格模型加载与 dummy inference

### 8.1 首次辅助计时问题

命令尝试使用 `/usr/bin/time -v`，服务器没有该文件：

```text
bash: /usr/bin/time: No such file or directory
```

没有开始模型加载。修复为 Python `time.perf_counter()` 和
`resource.getrusage()`。

### 8.2 严格加载核心命令

```python
config = PreTrainedConfig.from_pretrained(checkpoint)
config.pretrained_path = checkpoint
config.vlm_model_name = backbone_metadata
config.load_vlm_weights = False
config.device = "cuda"
policy = SmolVLAPolicy.from_pretrained(
    checkpoint, config=config, strict=True
)
```

结果：

```text
load_seconds=20.64
parameters=450,046,176
CUDA allocated=1,209,838,592 bytes
CUDA peak allocated=1,209,838,592 bytes
process max RSS=3,264,792 KiB
strict load passed
```

### 8.3 Dummy action chunk

输入：

- 三张 `[1,240,320,3]` uint8 图像
- `[1,14]` state
- task text
- 官方 preprocessor、normalizer 和 postprocessor

结果：

```text
output shape [1,50,14]
finite=true
cold chunk latency=0.616s
warm chunk latency=0.233s
CUDA peak allocated=1,270,694,912 bytes
CUDA peak reserved=1,314,914,304 bytes
```

## 9. 仿真环境隔离与 direct evaluator 决策

### 9.1 为何没有污染已有环境

`mea-libero` 能加载 policy，但不能导入：

```text
sapien
curobo
envs.beat_block_hammer
```

RoboTwin Python 3.10 环境能运行 simulator，但缺少 LeRobot、Transformers、
Draccus 和 Safetensors。

为保留可回滚边界，先克隆 RoboTwin：

```bash
/root/miniconda3/bin/conda create -y \
  -p /root/autodl-tmp/envs/mea-robotwin-smolvla \
  --clone /root/autodl-tmp/conda/envs/RoboTwin
```

结果：

```text
Packages: 32
Files: 109231
about 20s
du=8.3G
actual available-space delta=8,690,196,480 bytes
```

克隆没有使用 hardlink，因此回滚是删除这一整个独立路径，不影响原
RoboTwin env。

### 9.2 单进程安装失败与停止

尝试在独立 clone 中执行：

```bash
/root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python -m pip install \
  --no-deps \
  lerobot==0.6.0 \
  transformers==5.5.4 \
  draccus==0.10.0 \
  huggingface-hub==1.24.0 \
  safetensors==0.8.0 \
  num2words==0.5.14 \
  accelerate==1.14.0 \
  tokenizers==0.22.2 \
  regex==2026.7.19 \
  docopt==0.6.2
```

结果：

```text
lerobot 0.6.0 Requires-Python >=3.12
No matching distribution found for lerobot==0.6.0
```

pip 在解析阶段退出，没有安装列表中的任何包。最终检查：

```text
lerobot MISSING
transformers MISSING
torch 2.4.1
numpy 1.26.4
gymnasium 0.29.1
```

没有升级 torch/numpy/gym，也没有破坏 SAPIEN/CuRobo ABI。

决策：停止单进程路线，采用 localhost 双进程：

```text
Python 3.10 RoboTwin simulator
  -- raw uint8 camera bytes + shape, state list -->
Python 3.12 SmolVLA policy server
  -- Python list [50,14] -->
RoboTwin take_action()
```

模型每 50 步重新规划一次；这与 SmolVLA `n_action_steps=50` 队列式执行相同。

## 10. 真实 simulator observation probe

在独立 clone 中，用 `demo_clean.yml`、official embodiment、official
`_eval_step_limit.yml` 构造 BBH seed 100000，只 reset/get_obs，不执行
policy action。

关键结果：

```text
step_limit=400
initial check_success=false
head/left/right camera each [240,320,3] uint8
joint_action.vector [14]
initial vector [0,0,0,0,0,0,1,0,0,0,0,0,0,1]
SIM_PROBE_OK
```

RoboTwin 输出还包含 `front_camera`，但 checkpoint 只训练/使用
head/left/right 三相机，因此没有传 front camera。

## 11. IPC runner

源码：

```text
/root/autodl-tmp/tmp/mea-smolvla-robotwin/policy_server.py
/root/autodl-tmp/tmp/mea-smolvla-robotwin/robotwin_client.py
```

最终 SHA-256：

```text
e6291b936f08b68f371a0421fd577da6bb0f5b829db6f76df152cdb9bbb9f9b0  policy_server.py
1aa619c81e3685f9505303aeca43eb86624b41d59ca9998e2412dae6d56afd56  robotwin_client.py
```

服务器端和客户端分别在自己的服务器环境执行了 `python -m py_compile`，
均通过。

协议只绑定 `127.0.0.1`。Python 3.10/NumPy 1.26 与
Python 3.12/NumPy 2 之间不传 ndarray pickle：

- 图像：raw bytes + public shape list
- state：Python float list
- action chunk：Python nested list

## 12. 两次 0-action 启动失败

这些失败都发生在 rollout action 执行前，不计 episode。

### 12.1 Attempt 1：没有显式 PYTHONPATH

原命令使用绝对 client 脚本路径：

```bash
cd /root/autodl-tmp/RoboTwin
/root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python \
  /root/autodl-tmp/tmp/mea-smolvla-robotwin/robotwin_client.py ...
```

异常：

```text
ModuleNotFoundError: No module named 'envs'
```

原因：Python 把脚本目录而非当前 RoboTwin 目录放到 `sys.path[0]`。

修复：

```bash
export PYTHONPATH=/root/autodl-tmp/RoboTwin
```

日志：

```text
client_attempt1_missing_pythonpath.log
```

### 12.2 Attempt 2：跨 NumPy 私有模块 pickle

环境 reset 和第一个 policy chunk 已成功，但客户端解包 action ndarray 时：

```text
ModuleNotFoundError: No module named 'numpy._core.numeric'
```

原因：服务端 NumPy 2 pickle 使用私有模块名 `numpy._core`，客户端
NumPy 1.26 没有该私有路径。

执行状态：

```text
sim reset: success
policy chunk inference: success, 0.569s
RoboTwin action executed: 0
```

修复：按第 11 节改成 bytes/list 公共协议。

日志：

```text
server_attempt2_numpy_pickle.log
client_attempt2_numpy_pickle.log
```

## 13. BBH N=1 成功

### 13.1 Policy server 命令

```bash
run=/root/autodl-tmp/tmp/mea-smolvla-robotwin

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HOME=/root/autodl-tmp/cache/huggingface \
/root/autodl-tmp/envs/mea-libero/bin/python \
  "$run/policy_server.py" \
  --checkpoint \
    /root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin \
  --backbone-metadata \
    /root/autodl-tmp/checkpoints/robotwin/SmolVLM2-500M-Video-Instruct-metadata \
  --host 127.0.0.1 \
  --port 18771 \
  --seed 1000 \
  --ready-file "$run/server.ready.json" \
  --max-clients 1
```

### 13.2 Simulator client 命令

```bash
cd /root/autodl-tmp/RoboTwin
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/RoboTwin

PYTHONWARNINGS=ignore::UserWarning \
/root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python \
  /root/autodl-tmp/tmp/mea-smolvla-robotwin/robotwin_client.py \
  --host 127.0.0.1 \
  --port 18771 \
  --seed 1000 \
  --task beat_block_hammer \
  --output-dir \
    /root/autodl-tmp/tmp/mea-smolvla-robotwin/rollout_bbh_seed1000
```

### 13.3 结果

```text
task=beat_block_hammer
config=demo_clean
seed=1000
official step_limit=400
success=true
eval_success=true
official check_success=true
actions_executed=234
chunks=5
episode wall=50.857s
chunk latency=[0.555, 0.240, 0.240, 0.241, 0.240]s
GPU peak allocated=1,270,694,912 bytes
GPU peak reserved=1,314,914,304 bytes
```

产物：

```text
rollout_bbh_seed1000/result.json
rollout_bbh_seed1000/initial_head.png
rollout_bbh_seed1000/final_head.png
```

## 14. 五任务累计 N=5

BBH 不重复。通用化 client 后，共用一次模型加载，顺序执行：

```bash
for task in click_bell adjust_bottle grab_roller place_phone_stand; do
  out="/root/autodl-tmp/tmp/mea-smolvla-robotwin/rollout_${task}_seed1000"
  PYTHONWARNINGS=ignore::UserWarning \
  /root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python \
    /root/autodl-tmp/tmp/mea-smolvla-robotwin/robotwin_client.py \
    --host 127.0.0.1 \
    --port 18771 \
    --seed 1000 \
    --task "$task" \
    --output-dir "$out"
done
```

结果：

| task | success | official checker | actions | chunks | episode wall |
|---|---:|---:|---:|---:|---:|
| beat_block_hammer | true | true | 234 | 5 | 50.857s |
| click_bell | true | true | 49 | 1 | 35.869s |
| adjust_bottle | false | false | 400 | 8 | 58.443s |
| grab_roller | true | true | 94 | 2 | 38.787s |
| place_phone_stand | true | true | 134 | 3 | 41.256s |

合计：

```text
tasks=5
successes=4
actions=911
chunks=19
sum episode wall=225.212s
max GPU peak allocated=1,270,694,912 bytes
```

`adjust_bottle` 是 policy 在完整 400 步预算内的有效失败，不是系统异常；
因此保留结果并继续后续任务。

汇总：

```text
/root/autodl-tmp/tmp/mea-smolvla-robotwin/five_task_summary.json
SHA-256 baf64435830fd6853e3849c3da99d2745059494512b27df1966064aeac1286f0
```

每个任务都有：

```text
result.json
initial_head.png
final_head.png
client_<task>.log
```

## 15. 最终资源状态

```text
2026-07-29T15:52:26+08:00
/root/autodl-tmp: 879G total, 66G used, 814G available, 8%
checkpoint: 865M
backbone metadata: 4.8M
isolated simulator env: 8.3G
run logs/artifacts: 764K
GPU: 0 MiB used, 0% utilization
no policy/simulator process remains
```

## 16. 可复现性与结论边界

本批证明：

- 服务器可稳定下载、校验并离线加载该 checkpoint。
- 三相机 + 14D state + 14D action + 50-step chunk 合同真实可执行。
- 独立 Python 环境间可通过 localhost IPC 驱动 RoboTwin。
- 五个代表任务的 single-seed official checker rollout 为 4/5。

本批不证明：

- SmolVLA 在 RoboTwin 50 个任务上的真实总体成功率。
- 训练数据覆盖等价于所有任务成功。
- 单 seed 4/5 能代表统计性能。
- 当前 standalone runner 已经进入 MEA Planner/TaskGen/ToolGen 主链。
- 它与 ACT/DP3 的比较公平；训练数据、模型和协议并未在本批匹配。

下一步应先把同一 IPC runtime 作为数据驱动 policy backend 接入 MEA，再对
3–5 个任务做开放 Query → evidence-conditioned next sub-aspect 的方法验收；
不应直接把 live 预算扩到 50 个任务。

## 17. 回滚说明（本批未执行）

所有新增服务器状态都在以下独立路径：

```text
/root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin
/root/autodl-tmp/checkpoints/robotwin/SmolVLM2-500M-Video-Instruct-metadata
/root/autodl-tmp/envs/mea-robotwin-smolvla
/root/autodl-tmp/tmp/mea-smolvla-robotwin
```

如果以后决定回滚，只需在再次核对绝对路径后移除上述四个独立路径；不会
影响原 RoboTwin、ACT、DP3、LIBERO 或 MEA 工作树。
