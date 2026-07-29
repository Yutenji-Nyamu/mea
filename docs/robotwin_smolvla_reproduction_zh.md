# RoboTwin / SmolVLA 服务器部署、隔离运行与五任务验证

本文记录 2026-07-29 在 SeetaCloud 单卡 RTX 4090 服务器上，下载
`lerobot/smolvla_robotwin`、核对输入输出合同、隔离 RoboTwin/LeRobot 环境，
并完成五个 RoboTwin 任务 N=1 rollout 的全过程。

它是长期复现文档，不保存 SSH 密码、provider key 或 Hugging Face token。
本批所有下载、模型加载、processor 探针和仿真均发生在服务器，没有经过
Windows 本地磁盘，也没有修改服务器上 dirty 的 MEA 工作树。

## 1. 本批结论与边界

固定 checkpoint 后，SmolVLA 在五任务部署 pilot 中得到：

| task | official success | actions | chunks | episode wall |
|---|---:|---:|---:|---:|
| beat_block_hammer | true | 234 | 5 | 50.857s |
| click_bell | true | 49 | 1 | 35.869s |
| adjust_bottle | false | 400 | 8 | 58.443s |
| grab_roller | true | 94 | 2 | 38.787s |
| place_phone_stand | true | 134 | 3 | 41.256s |

合计 4/5、911 个 simulator actions、19 个 action chunks，五个 episode
wall time 之和为 225.212 秒。模型峰值 CUDA allocated 为
1,270,694,912 bytes，reserved 为 1,314,914,304 bytes。

五个 simulator episode 均使用 scene seed=1000；两个 policy server 进程分别以
Torch/NumPy seed=1000 启动。实测由一个 BBH 单任务 session 和一个其余四任务的顺序
session 组成，policy RNG 不会在每个 client 上重新播种，因此第二个 session 的结果还
依赖固定任务顺序。它是接口与可运行性证据，不是每任务独立复现实验，更不是 50-task
成功率。模型训练源覆盖 RoboTwin unified dataset，也不能据此声称五十个任务均成功。

上游口径与来源：

- [LeRobot RoboTwin 2.0 文档](https://huggingface.co/docs/lerobot/main/robotwin)
- [`robotwin_unified` 数据集](https://huggingface.co/datasets/lerobot/robotwin_unified)
- [`lerobot/smolvla_robotwin` checkpoint](https://huggingface.co/lerobot/smolvla_robotwin)

## 2. 服务器目录

```text
/root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin
/root/autodl-tmp/checkpoints/robotwin/SmolVLM2-500M-Video-Instruct-metadata
/root/autodl-tmp/envs/mea-libero
/root/autodl-tmp/envs/mea-robotwin-smolvla
/root/autodl-tmp/tmp/mea-smolvla-robotwin
```

仓库内可复用实验 runner：

```text
experiments/paper/robotwin_smolvla/policy_server.py
experiments/paper/robotwin_smolvla/sim_client.py
```

## 3. 身份、资源和环境探针

SSH 使用进程内 Paramiko 密码认证，关闭 key/agent 探测；认证后执行：

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

初始结果：

```text
2026-07-29T15:09:11+08:00
autodl-container-ujxcycmw77-dd0e7d70
/root
0
/dev/md0: 879G total, 57G used, 823G available
RTX 4090: 24564 MiB total, 0 MiB used
```

LeRobot 环境：

本批复用了此前为 LIBERO 创建的 Python 3.12 policy 环境；若服务器没有该环境，先按
[LIBERO / SmolVLA 复现文档](libero_smolvla_reproduction_zh.md)第 3 节创建，再继续本文。

```bash
/root/autodl-tmp/envs/mea-libero/bin/python - <<'PY'
import sys, torch, transformers, huggingface_hub, lerobot
print(sys.version)
print(torch.__version__)
print(transformers.__version__)
print(huggingface_hub.__version__)
print(lerobot.__version__)
PY
```

```text
Python 3.12.13
torch 2.11.0+cu130
transformers 5.5.4
huggingface_hub 1.24.0
lerobot 0.6.0
```

RoboTwin 环境：

```bash
/root/autodl-tmp/conda/envs/RoboTwin/bin/python - <<'PY'
import sys, torch, sapien, numpy, gymnasium
print(sys.version)
print(torch.__version__, sapien.__version__, numpy.__version__, gymnasium.__version__)
PY
```

```text
Python 3.10.20
torch 2.4.1+cu121
sapien 3.0.0b1
numpy 1.26.4
gymnasium 0.29.1
```

这两个环境不能直接合并：RoboTwin/CuRobo 依赖 Python 3.10 和现有
torch/numpy ABI，而 LeRobot 0.6 的发布包要求 Python >=3.12、torch >=2.7、
numpy >=2。

## 4. Hugging Face 网络问题与修复

第一次直接调用：

```bash
/root/autodl-tmp/envs/mea-libero/bin/python - <<'PY'
from huggingface_hub import HfApi
print(HfApi().model_info("lerobot/smolvla_robotwin", files_metadata=True))
PY
```

120 秒内没有返回，被执行器超时终止。随后做定点检查：

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
huggingface.co: code=000, 4.00s connect timeout
hf-mirror.com: code=200, connect=0.116s, total=0.585s
```

后续统一使用：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_DISABLE_XET=1
```

这只改变下载端点，不改变 repo id、revision 或文件哈希。

## 5. 固定 checkpoint revision 和哈希

查询：

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

固定值：

```text
repo=lerobot/smolvla_robotwin
revision=967623a0f38c7e1236c66b3893c830398d793ff7
model.safetensors bytes=906712520
model.safetensors sha256=
6fbaa809585cb10924351ce101ca2d576787a472cf8c34b5e2f10c84ef8a3134
public=true
gated=false
```

## 6. 服务器下载

下载使用后台进程，避免 SSH 瞬时断开中止传输：

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
echo $! > "$run/download.pid"
```

轮询：

```bash
kill -0 "$(cat "$run/download.pid")"
du -sh "$target"
tail -30 "$run/download.log"
```

下载约 2 分 27 秒，9 个文件，目录占用 865 MiB。完整校验：

```bash
sha256sum "$target"/* | tee "$run/sha256sum.txt"
```

主权重本地 SHA-256 与 Hub LFS 元数据完全一致。

## 7. 最小 backbone metadata

SmolVLA 初始化需要
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 的 config、processor 和
tokenizer。其完整基础权重约 2.03 GB，但最终 SmolVLA checkpoint 已带完整
policy 权重，因此固定 backbone revision 后只下载 metadata：

```bash
export HF_ENDPOINT=https://hf-mirror.com
/root/autodl-tmp/envs/mea-libero/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    revision="7b375e1b73b11138ff12fe22c8f2822d8fe03467",
    local_dir=(
        "/root/autodl-tmp/checkpoints/robotwin/"
        "SmolVLM2-500M-Video-Instruct-metadata"
    ),
    ignore_patterns=["*.safetensors", "*.bin", "*.onnx"],
))
PY
```

得到 13 个文件、4.8 MiB。运行时：

```python
config.vlm_model_name = BACKBONE_METADATA_PATH
config.load_vlm_weights = False
policy = SmolVLAPolicy.from_pretrained(
    CHECKPOINT_PATH, config=config, strict=True
)
```

`strict=True` 成功，证明这一优化没有缺失或多余模型参数。

## 8. 输入输出合同

checkpoint 静态配置：

```text
camera1/2/3: [3,256,256]
state in config/preprocessor JSON: [6]
action: [14]
chunk_size=50
n_action_steps=50
max_state_dim=32
```

发现一处上游 artifact 漂移：

- config/preprocessor JSON 的 state feature 壳写 `[6]`；
- normalizer safetensors 的 state mean/std/min/max 均为 `[14]`；
- 当前 LeRobot RoboTwin wrapper 输出 `agent_pos [14]`；
- 真实 RoboTwin `joint_action.vector` 也是 `[14]`。

checkpoint 保存的训练集相机 rename 是：

```text
cam_high -> camera1
cam_left_wrist -> camera2
cam_right_wrist -> camera3
```

真实 runtime 使用：

```python
rename_map = {
    "observation.images.head_camera": "observation.images.camera1",
    "observation.images.left_camera": "observation.images.camera2",
    "observation.images.right_camera": "observation.images.camera3",
}
```

实际 processor 探针得到：

```text
three images [1,3,240,320], CUDA float32
state [1,14], CUDA float32, all finite
language tokens [1,48]
```

模型将 14D state pad 到 `max_state_dim=32`。没有把真实 state 裁成 6D，
也没有编辑 checkpoint 文件。

## 9. 模型加载与 dummy inference

最初尝试 `/usr/bin/time -v`，服务器没有该文件：

```text
bash: /usr/bin/time: No such file or directory
```

模型尚未启动，因此改用 `time.perf_counter()` 和
`resource.getrusage()`。严格加载结果：

```text
load=20.64s
parameters=450046176
CUDA allocated=1209838592 bytes
max RSS=3264792 KiB
```

三相机 + 14D state dummy inference：

```text
output [1,50,14]
finite=true
cold chunk=0.616s
warm chunk=0.233s
CUDA peak allocated=1270694912 bytes
```

## 10. 独立 RoboTwin 环境

为避免修改原 ACT 环境，执行：

```bash
/root/miniconda3/bin/conda create -y \
  -p /root/autodl-tmp/envs/mea-robotwin-smolvla \
  --clone /root/autodl-tmp/conda/envs/RoboTwin
```

约 20 秒，`du` 为 8.3 GiB，实际新增约 8.69 GB。

随后尝试在 clone 中仅安装 LeRobot 和轻依赖：

```bash
/root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python -m pip install \
  --no-deps \
  lerobot==0.6.0 transformers==5.5.4 draccus==0.10.0 \
  huggingface-hub==1.24.0 safetensors==0.8.0 \
  num2words==0.5.14 accelerate==1.14.0 tokenizers==0.22.2 \
  regex==2026.7.19 docopt==0.6.2
```

pip 在解析阶段拒绝：

```text
lerobot 0.6.0 Requires-Python >=3.12
No matching distribution found for lerobot==0.6.0
```

没有安装任何包。最终 clone 仍为：

```text
lerobot MISSING
transformers MISSING
torch 2.4.1
numpy 1.26.4
gymnasium 0.29.1
```

因此没有为了单进程 evaluator 升级 torch/numpy 或破坏 CuRobo。

## 11. 为什么采用双进程

```text
RoboTwin / Python 3.10 / NumPy 1.26
  -> head,left,right raw bytes + shape; state list ->
SmolVLA / Python 3.12 / NumPy 2.x
  -> Python list [50,14] ->
RoboTwin take_action()
```

服务只绑定 `127.0.0.1`。消息使用 length-prefixed pickle 作为公共 Python
容器 envelope，但绝不在两个 NumPy 大版本间 pickle ndarray：

- 图像是 raw bytes + shape list；
- state 是 14 个 Python float；
- action chunk 是嵌套 Python list。

每次请求返回 50 步，与 checkpoint 的 `n_action_steps=50` 队列式 open-loop
执行一致。

真实 reset-only probe 还确认：

```text
demo_clean
official BBH step_limit=400
head/left/right image=[240,320,3] uint8
joint_action.vector=[14]
initial check_success=false
```

## 12. 两次 0-action 失败及修复

### 12.1 绝对脚本路径缺少 RoboTwin import root

第一次 client：

```text
ModuleNotFoundError: No module named 'envs'
```

原因是 `sys.path[0]` 为脚本目录。修复：

```bash
export PYTHONPATH=/root/autodl-tmp/RoboTwin
```

这次没有创建 simulator scene，也没有执行 action。

### 12.2 NumPy 私有 pickle ABI

第二次已完成 reset 和首个 policy chunk，但 NumPy 1.26 client 解包
NumPy 2 server 的 ndarray 时：

```text
ModuleNotFoundError: No module named 'numpy._core.numeric'
```

当时首个 chunk latency 为 0.569 秒，执行 action 数为 0。修复为第 11 节的
bytes/list 协议，旧日志保存在服务器：

```text
client_attempt1_missing_pythonpath.log
server_attempt2_numpy_pickle.log
client_attempt2_numpy_pickle.log
```

## 13. 复现一个任务

服务器端，在 LeRobot Python 3.12 环境：

```bash
cd /path/to/mea
run=/root/autodl-tmp/tmp/mea-smolvla-robotwin

CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HOME=/root/autodl-tmp/cache/huggingface \
/root/autodl-tmp/envs/mea-libero/bin/python \
  experiments/paper/robotwin_smolvla/policy_server.py \
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

客户端，在 RoboTwin Python 3.10 环境：

```bash
cd /root/autodl-tmp/RoboTwin
export PYTHONPATH=/root/autodl-tmp/RoboTwin
export CUDA_VISIBLE_DEVICES=0

PYTHONWARNINGS=ignore::UserWarning \
/root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python \
  /path/to/mea/experiments/paper/robotwin_smolvla/sim_client.py \
  --host 127.0.0.1 \
  --port 18771 \
  --task beat_block_hammer \
  --seed 1000 \
  --output-dir /path/to/output/beat_block_hammer_seed1000
```

每个 output 目录生成：

```text
result.json
initial_head.png
final_head.png
```

server 的 `--max-clients` 应等于该 policy process 预期接受的 client 数量；最后一个
client 关闭后 server 自动退出并输出 CUDA peak/latency summary。`policy.reset()` 只
清理 action queue，不重置 Torch/NumPy RNG。需要每任务独立可重复时，应为每个任务启动
一个 `--max-clients 1 --seed 1000` 的新 server；顺序 batch 则必须冻结并记录任务顺序。

## 14. 五任务命令

本批不是单次 `--max-clients 5`。真实执行分为：

```text
session A: policy seed 1000, max-clients 1
  beat_block_hammer, simulator seed 1000

session B: policy seed 1000, max-clients 4
  click_bell → adjust_bottle → grab_roller → place_phone_stand
  每个 simulator seed 均为 1000
```

session A 使用第 13 节命令。session B 的 ready/log 与 server log 可重构出以下有效
argv；原始逐字节 shell command 没有保存，因此这里不冒充 verbatim transcript：

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_HOME=/root/autodl-tmp/cache/huggingface \
/root/autodl-tmp/envs/mea-libero/bin/python \
  /root/autodl-tmp/tmp/mea-smolvla-robotwin/policy_server.py \
  --checkpoint /root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin \
  --backbone-metadata \
    /root/autodl-tmp/checkpoints/robotwin/SmolVLM2-500M-Video-Instruct-metadata \
  --host 127.0.0.1 \
  --port 18771 \
  --seed 1000 \
  --ready-file \
    /root/autodl-tmp/tmp/mea-smolvla-robotwin/server_multitask.ready.json \
  --max-clients 4 \
  > /root/autodl-tmp/tmp/mea-smolvla-robotwin/server_multitask.log 2>&1 &
echo $! > /root/autodl-tmp/tmp/mea-smolvla-robotwin/server_multitask.pid
```

port、ready path、log path 与四个 client 均由 raw 文件确认；环境变量和显式
`--seed 1000` 按本批固定启动模板重构。随后严格按以下顺序运行 client：

```bash
for task in \
  click_bell \
  adjust_bottle \
  grab_roller \
  place_phone_stand
do
  out="/path/to/output/${task}_seed1000"
  PYTHONPATH=/root/autodl-tmp/RoboTwin \
  /root/autodl-tmp/envs/mea-robotwin-smolvla/bin/python \
    /path/to/mea/experiments/paper/robotwin_smolvla/sim_client.py \
    --host 127.0.0.1 \
    --port 18771 \
    --task "$task" \
    --seed 1000 \
    --output-dir "$out"
done
```

单 server `--max-clients 5` 是可用的 convenience protocol，但没有产生本表证据；
不能把它与上述 1+4 session 结果混为一谈。实测临时 runner 与完成后整理入仓库的
runner 哈希不同，原因是后者补了 loopback/参数/ready-file 边界与格式清理：

```text
tested temporary policy_server.py sha256:
e6291b936f08b68f371a0421fd577da6bb0f5b829db6f76df152cdb9bbb9f9b0
tested temporary sim_client.py sha256:
1aa619c81e3685f9505303aeca43eb86624b41d59ca9998e2412dae6d56afd56
```

仓库版保持相同 observation/action 与 rollout 合同，并在服务器做 CLI/import 边界
复验；它未被用来重新生成本表五个 episode。

本批真实产物仍只保存在 canonical server：

```text
/root/autodl-tmp/tmp/mea-smolvla-robotwin/command_ledger.md
/root/autodl-tmp/tmp/mea-smolvla-robotwin/five_task_summary.json
/root/autodl-tmp/tmp/mea-smolvla-robotwin/rollout_*_seed1000/
```

raw `command_ledger.md` 保留下载、探针与两次 0-action 失败，但没有保存 session B
逐字节启动命令；本文结合 ready/log/hash 后给出的第 14 节是长期维护的合并流水账。

`five_task_summary.json` SHA-256：

```text
baf64435830fd6853e3849c3da99d2745059494512b27df1966064aeac1286f0
```

## 15. 最终资源和清理

完成后：

```text
/root/autodl-tmp: 879G total, 66G used, 814G available
SmolVLA checkpoint: 865 MiB
backbone metadata: 4.8 MiB
isolated simulator env: 8.3 GiB
run logs/artifacts: 764 KiB
GPU used: 0 MiB
```

若某个 client 在连接前因系统错误退出，server 会继续等待剩余 client。此时应终止该
server，使用新的 ready-file/port 重启尚未完成的协议；系统错误不得计作 episode。
runner 启动会删除同路径的旧 ready-file，调用方仍应核对其中 PID/port 与当前进程。

回滚前必须再次核对绝对路径。本批新增状态都在：

```text
/root/autodl-tmp/checkpoints/robotwin/smolvla_robotwin
/root/autodl-tmp/checkpoints/robotwin/SmolVLM2-500M-Video-Instruct-metadata
/root/autodl-tmp/envs/mea-robotwin-smolvla
/root/autodl-tmp/tmp/mea-smolvla-robotwin
```

删除这四个独立路径即可回滚，不应删除或修改原
`/root/autodl-tmp/conda/envs/RoboTwin`、ACT、DP3、LIBERO 或 MEA 工作树。
本文只记录回滚方法，本批没有执行删除。

## 16. 对 MEA 的意义和限制

SmolVLA 的统一 checkpoint 使 task→checkpoint 的工程绑定从多份单任务模型
缩成一个 policy backend，适合验证 MEA 的跨任务开放 Query 和运行时 task
binding。当前 runner 仍在 `experiments/paper/`，不是第二条生产
Planner/TaskGen/ToolGen orchestration。

后续应复用同一 observation/action contract 接入共享 MethodRuntime，然后用
3–5 个任务验证：

```text
open Query
-> first sub-aspect
-> rollout evidence
-> evidence-conditioned next sub-aspect
-> TaskGen/ToolGen as needed
-> sufficient stop and answer
```

不应把本批 4/5 放大成 50-task 性能结论，也不应未经训练数据和协议匹配就与
单任务 ACT/DP3 计算公平 policy ranking。
