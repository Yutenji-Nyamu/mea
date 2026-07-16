#!/usr/bin/env bash

set -euo pipefail

policy_name="ACT"

if [ "$#" -lt 6 ] || [ "$#" -gt 15 ]; then
    echo "Usage: $0 TASK CONFIG CKPT EXPERT_NUM SEED GPU [NUM_EPISODES] [TASK_MODULE] [TASK_OVERLAY] [START_SEED] [TELEMETRY_DIR] [TELEMETRY_PROFILE] [SEED_MANIFEST] [SEED_RESULTS_PATH] [OUTPUT_DIR]" >&2
    exit 2
fi

task_name="${1}"
task_config="${2}"
ckpt_setting="${3}"
expert_data_num="${4}"
seed="${5}"
gpu_id="${6}"

num_episodes="${7:-100}"
task_module="${8:-}"
task_overlay="${9:-}"
start_seed="${10:-}"
telemetry_dir="${11:-}"
telemetry_profile="${12:-balanced_v1}"
seed_manifest="${13:-}"
seed_results_path="${14:-}"
output_dir="${15:-}"

export CUDA_VISIBLE_DEVICES="${gpu_id}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo "num_episodes=${num_episodes}"
echo "task_module=${task_module:-<official>}"
echo "task_overlay=${task_overlay:-<none>}"
echo "start_seed=${start_seed:-<official-derived>}"
echo "telemetry_dir=${telemetry_dir:-<disabled>}"
echo "telemetry_profile=${telemetry_profile}"
echo "seed_manifest=${seed_manifest:-<legacy-scan>}"
echo "seed_results_path=${seed_results_path:-<default>}"
echo "output_dir=${output_dir:-<timestamped-default>}"

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"
REPO_ROOT="$(
    cd "${SCRIPT_DIR}/../.."
    pwd
)"
cd "${REPO_ROOT}"

# The server keeps a shared Torch cache beside the repository. Interactive
# shells normally export TORCH_HOME, but subprocess-driven TaskGen runs do not.
if [ -z "${TORCH_HOME:-}" ] && [ -d "${REPO_ROOT}/../cache/torch" ]; then
    export TORCH_HOME="${REPO_ROOT}/../cache/torch"
fi
echo "torch_home=${TORCH_HOME:-<default>}"

OVERRIDES=(
    --task_name "${task_name}"
    --task_config "${task_config}"
    --ckpt_setting "${ckpt_setting}"
    --ckpt_dir "policy/ACT/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num}"
    --seed "${seed}"
    --temporal_agg true
    --num_episodes "${num_episodes}"
)

if [ -n "${task_module}" ]; then
    OVERRIDES+=(--task_module "${task_module}")
fi
if [ -n "${task_overlay}" ]; then
    OVERRIDES+=(--task_overlay "${task_overlay}")
fi
if [ -n "${start_seed}" ]; then
    OVERRIDES+=(--start_seed "${start_seed}")
fi
if [ -n "${telemetry_dir}" ]; then
    OVERRIDES+=(--telemetry_dir "${telemetry_dir}")
    OVERRIDES+=(--telemetry_profile "${telemetry_profile}")
fi
if [ -n "${seed_manifest}" ]; then
    OVERRIDES+=(--seed_manifest "${seed_manifest}")
fi
if [ -n "${seed_results_path}" ]; then
    OVERRIDES+=(--seed_results_path "${seed_results_path}")
fi
if [ -n "${output_dir}" ]; then
    OVERRIDES+=(--output_dir "${output_dir}")
fi

python_bin="${PYTHON_BIN:-python}"

PYTHONWARNINGS=ignore::UserWarning \
"${python_bin}" script/eval_policy.py \
    --config "policy/${policy_name}/deploy_policy.yml" \
    --overrides \
    "${OVERRIDES[@]}"
