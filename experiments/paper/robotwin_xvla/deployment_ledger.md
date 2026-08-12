# X-VLA RoboTwin2 deployment ledger (cold reference)

Date: 2026-08-12 (Asia/Shanghai)

Purpose: preserve the exact server-side source, checkpoint, environment,
validation commands, observed results, failures, and fixes for
`2toINF/X-VLA-RoboTwin2`. This is cold context; routine MEA development should
read the concise runbook instead. No credential is recorded here. Windows only
held source and documentation; every network download, import, model load, and
synthetic action inference ran on the AutoDL/SeetaCloud server. No simulator
episode was run in this deployment batch.

## 1. Scope and official basis

Official sources:

- code: <https://github.com/2toinf/X-VLA>
- checkpoint: <https://huggingface.co/2toINF/X-VLA-RoboTwin2>
- RoboTwin2 evaluator:
  <https://github.com/2toinf/X-VLA/tree/main/evaluation/robotwin-2.0>

The official repository describes X-VLA as a 0.9B model and the RoboTwin2
checkpoint as trained on the RoboTwin2 dataset with 50 demonstrations per task.
Its official evaluator enumerates 50 task names and reports leaderboard-setting
Easy/Hard aggregate success of 70%/39%. These upstream claims motivate the
deployment; they are not MEA measurements.

This deployment does **not** start a 50-task sweep. Its acceptance boundary is:

1. immutable source and checkpoint revisions;
2. isolated environment;
3. offline import and checkpoint load on one RTX 4090;
4. one bounded action inference and measured peak VRAM;
5. leave simulator integration to the shared policy-backend batch.

## 2. Frozen server state

Connection used the already verified low-level Paramiko password route. The
password was injected only into the current process and is intentionally omitted.

Identity/capacity probe:

```bash
hostname
pwd
id -u
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
df -h /root/autodl-tmp
git -C /root/autodl-tmp/mea-worktrees/evidence-refinement-runtime rev-parse HEAD
git -C /root/autodl-tmp/mea-worktrees/evidence-refinement-runtime status --short
```

Observed:

```text
host       autodl-container-ujxcycmw77-dd0e7d70
pwd        /root
uid        0
GPU        NVIDIA GeForce RTX 4090, 24564 MiB, 0 MiB used
disk       2.4T total, 96G used, 2.3T available
MEA HEAD   1a0841eb0b380f89c81027cbdc30a3ba9e53540a
worktree   clean
```

Chosen isolated paths:

```text
source      /root/autodl-tmp/third_party/X-VLA
checkpoint  /root/autodl-tmp/checkpoints/robotwin/xvla_robotwin2
policy env  /root/autodl-tmp/envs/mea-xvla
HF cache    /root/autodl-tmp/hf-cache-xvla
runtime     /root/autodl-tmp/tmp/mea-xvla-robotwin2
```

Before deployment all three source/checkpoint/environment target paths were
absent. Existing SmolVLA, Hy-VLA, and RoboTwin environments were not modified.

## 3. Network diagnosis and recovery

Default-route bounded probe:

```bash
date -Iseconds
env | grep -iE '^(http|https|all)_proxy=' || true
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 12 \
  -w 'github code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 12 \
  -w 'hf code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://huggingface.co
timeout 20 git ls-remote https://github.com/2toinf/X-VLA.git HEAD
```

At `2026-08-12T16:10:20+08:00`, both main sites failed before HTTP or
authentication: GitHub `code=000` after 7.00 s and Hugging Face `code=000`
after 3.70 s. This was a route timeout, not a repository or credential error.

Only the current remote shell then enabled the AutoDL academic accelerator:

```bash
source /etc/network_turbo
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 12 \
  -w 'github code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://github.com
curl -L -sS -o /dev/null --connect-timeout 7 --max-time 12 \
  -w 'hf code=%{http_code} connect=%{time_connect} total=%{time_total}\n' \
  https://huggingface.co
timeout 20 git ls-remote https://github.com/2toinf/X-VLA.git HEAD
curl -L -sS --connect-timeout 7 --max-time 30 \
  https://huggingface.co/api/models/2toINF/X-VLA-RoboTwin2
```

Observed GitHub HTTP 200 in 4.20 s, Hugging Face HTTP 200 in 1.25 s, and
`git ls-remote` resolved commit
`6bc2513f5f1cbec715cc668b414392a6cae5c671`. No proxy or Git setting was
persisted.

## 4. Source checkout

```bash
source /etc/network_turbo
git clone --no-tags https://github.com/2toinf/X-VLA.git \
  /root/autodl-tmp/third_party/X-VLA
git -C /root/autodl-tmp/third_party/X-VLA \
  checkout --detach 6bc2513f5f1cbec715cc668b414392a6cae5c671
git -C /root/autodl-tmp/third_party/X-VLA rev-parse HEAD
git -C /root/autodl-tmp/third_party/X-VLA status --short
du -sh /root/autodl-tmp/third_party/X-VLA
```

Observed clean detached source at the intended commit, 155 MiB on disk. The
checkout took about 99 s.

The inspected official dependency boundary is Python 3.10, PyTorch 2.1/CUDA
12.1, torchvision 0.16, NumPy 1.26, and Transformers <=4.51.3. The official
FastAPI server loads `XVLA` in float32. The official RoboTwin client sends four
camera views, dual-arm end-effector state, language, and `domain_id=6` and
receives 20-D action chunks.

## 5. Checkpoint metadata

```bash
source /etc/network_turbo
curl -L -sS --connect-timeout 7 --max-time 45 \
  'https://huggingface.co/api/models/2toINF/X-VLA-RoboTwin2?blobs=true' \
  -o /root/autodl-tmp/tmp/xvla_hf_metadata.json
```

Resolved Hugging Face revision:

```text
a157c580cfe6f9f445614490f3bec1b2f9ef9f18
```

The API reports 879,738,545 float32 parameters and 3,519,068,172 bytes of
repository storage. Exact per-file sizes and post-download hashes are recorded
in the next section after the immutable snapshot completes.

## 6. Checkpoint download

The immutable snapshot downloader ran in the existing lightweight downloader
environment, not in Windows or the new policy environment:

```bash
source /etc/network_turbo
export HF_HOME=/root/autodl-tmp/hf-cache-xvla
export HF_HUB_DOWNLOAD_TIMEOUT=120
export HF_HUB_DISABLE_XET=1
nohup /root/autodl-tmp/envs/mea-libero/bin/python \
  experiments/paper/robotwin_xvla/download_checkpoint.py \
  > /root/autodl-tmp/tmp/mea-xvla-robotwin2/download.log 2>&1 \
  < /dev/null &
```

The launched PID was 2422. The downloader uses `snapshot_download` with the
fixed revision and four workers, then writes `mea_download_manifest.json` with
per-file byte counts and SHA-256 for code/config and `model.safetensors`.

An 8 MiB bounded range probe measured about 1.09 MiB/s through the official
host and 1.35 MiB/s through `hf-mirror.com`; the active official-host download
was therefore slow but healthy, reaching about 1.7 GiB after 12 minutes. It was
left running because it is resumable and revision/hash checked. Final size,
elapsed time, and hashes after completion:

```text
elapsed download  approximately 22 minutes including one resumed disconnect
snapshot files    15 files, 3,521,721,099 bytes total
model.safetensors 3,519,068,172 bytes
SHA-256          3f16a4b67a1d2675fc1cb0350c6d5617522e452f47359e729d74d76b8aa4835b
checkpoint disk  3.3 GiB
```

At about 3.2 GiB the remote peer closed one response after 194,817,557 bytes.
`huggingface_hub` logged `Trying to resume download...`, resumed the same fixed
revision, and completed without restarting. The generated manifest records the
required config and main-weight hash; it does not claim a per-file hash for all
15 snapshot files.

## 7. Isolated environment

To avoid re-downloading the existing CUDA 12.1/Torch stack while still
protecting the working simulator environment, a separate clone was made:

```bash
/root/miniconda3/bin/conda create -y \
  -p /root/autodl-tmp/envs/mea-xvla \
  --clone /root/autodl-tmp/envs/mea-robotwin-smolvla
```

The clone completed successfully: Python 3.10.20, Torch 2.4.1+cu121,
torchvision 0.19.1+cu121, NumPy 1.26.4, 8.3 GiB. The official environment asks
for Torch 2.1/CUDA 12.1, but its pip requirements only require Torch >=1.13;
the existing compatible CUDA ABI is tested first instead of replacing it
speculatively. This environment remains independent of the source clone.

First install attempt:

```bash
source /etc/network_turbo
/root/autodl-tmp/envs/mea-xvla/bin/python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --extra-index-url https://pypi.org/simple \
  -r /root/autodl-tmp/third_party/X-VLA/requirements.txt
```

Dependency resolution succeeded but the accelerator made ordinary PyPI wheel
downloads stall at `av==15.0.0`. No pip transaction had begun. PID 3207 was
terminated and the same official requirements were retried without the
accelerator or PyPI fallback:

```bash
env PIP_DEFAULT_TIMEOUT=120 \
    PIP_CACHE_DIR=/root/autodl-tmp/pip-cache-xvla \
  /root/autodl-tmp/envs/mea-xvla/bin/python -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  -r /root/autodl-tmp/third_party/X-VLA/requirements.txt
```

This route immediately sustained approximately 1.1--4.7 MiB/s. The durable
lesson is route-specific: use temporary `network_turbo` for GitHub/Hugging Face,
but use direct Tsinghua PyPI for ordinary Python wheels. Final package versions
and `pip check` after completion:

```text
Python       3.10.20
torch        2.4.1+cu121
torchvision  0.19.1+cu121
transformers 4.51.3
av           15.0.0
numpy        1.26.3
scipy        1.15.0
einops       0.8.1
timm         1.0.12
mmengine     0.10.5
fastapi      0.141.1
uvicorn      0.34.3
json_numpy   2.1.0
pip check    No broken requirements found
```

The clean direct-index install took about eight minutes. A second invocation
was idempotent and completed with all requirements already satisfied.

## 8. Offline import, load, action inference, and VRAM

The first wrapper command used `/usr/bin/time -v`; this server does not provide
that binary, so it failed before Python with `No such file or directory`. No
package was installed merely to add timing: the validator already records
wall-time and CUDA statistics internally.

Successful offline invocation:

```bash
cd /root/autodl-tmp/mea-worktrees/evidence-refinement-runtime
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/root/autodl-tmp/envs/mea-xvla/bin/python \
  /root/autodl-tmp/tmp/mea-xvla-robotwin2/validate_install.py \
  --source /root/autodl-tmp/third_party/X-VLA \
  --checkpoint /root/autodl-tmp/checkpoints/robotwin/xvla_robotwin2 \
  --steps 1 \
  --output /root/autodl-tmp/tmp/mea-xvla-robotwin2/validation.json
```

The validator uses the official `XVLA` and `XVLAProcessor`, float32 as in the
official deploy script, three synthetic zero-valued 224x224 camera images,
20-D dual-arm proprio,
RoboTwin `domain_id=6`, and one denoising step. Result:

```text
load_seconds                 2.2369
one-step inference_seconds   0.4636
action_shape                 [1, 30, 20]
action_finite                true
action_range                 [-0.260929, 0.999971]
CUDA allocated after run     3,529,567,744 bytes
CUDA peak allocated          3,716,609,024 bytes
CUDA peak reserved           3,923,771,392 bytes
```

GPU memory returned to 0 MiB after process exit. Transformers emitted two
upstream compatibility warnings: the saved image processor is slow, and the
custom Florence class does not directly inherit `GenerationMixin`. This action
path calls neither generic text `generate()` nor a fast image processor; output
was finite and correctly shaped, so the official code was not modified merely
to suppress warnings.

## 9. Official RoboTwin adapter/rollout

Not run in this deployment batch. The bounded goal was immutable deployment
plus official model load/action validation; adding a live backend belongs in
the shared `MethodRuntime`, not in this model-specific installation directory.

## 10. Final state and reproducibility boundary

The deployment meets its offline model-load/action acceptance boundary on one
RTX 4090. Final storage:

```text
source       155 MiB
checkpoint   3.3 GiB
policy env   8.6 GiB
pip cache    383 MiB
```

No 50-task sweep and no simulator rollout was run in this deployment step.
X-VLA is therefore **load/action validated**, not yet a validated MEA policy
backend or evidence that any individual RoboTwin task succeeds. A live adapter
must preserve the official four-camera/EE6D preprocessing and should share
MEA's existing `MethodRuntime` rather than copy its outer method loop.
