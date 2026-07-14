#!/usr/bin/env bash

set -euo pipefail

policy_name="ACT"

if [ "$#" -lt 6 ] || [ "$#" -gt 10 ]; then
    echo "Usage: $0 TASK CONFIG CKPT EXPERT_NUM SEED GPU [NUM_EPISODES] [TASK_MODULE] [TASK_OVERLAY] [START_SEED]" >&2
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

export CUDA_VISIBLE_DEVICES="${gpu_id}"

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo "num_episodes=${num_episodes}"
echo "task_module=${task_module:-<official>}"
echo "task_overlay=${task_overlay:-<none>}"
echo "start_seed=${start_seed:-<official-derived>}"

SCRIPT_DIR="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"
REPO_ROOT="$(
    cd "${SCRIPT_DIR}/../.."
    pwd
)"
cd "${REPO_ROOT}"

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

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
    --config "policy/${policy_name}/deploy_policy.yml" \
    --overrides \
    "${OVERRIDES[@]}"
