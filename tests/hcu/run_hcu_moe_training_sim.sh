#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Framework-independent eight-HCU MoE proxy benchmark.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_path=${1:-"${repo_dir}/hcu_moe_training_sim.log"}
if [[ $# -gt 0 ]]; then
    shift
fi

mkdir -p "$(dirname "${log_path}")"
cd "${repo_dir}"

export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-8}
export ULTRA_EP_GRAD_REDUCE_NUM_SMS=${ULTRA_EP_GRAD_REDUCE_NUM_SMS:-64}
export ULTRA_EP_GRAD_REDUCE_DETERMINISTIC=${ULTRA_EP_GRAD_REDUCE_DETERMINISTIC:-1}
export ULTRA_EP_WEIGHT_SYNC_PLAN_MODE=${ULTRA_EP_WEIGHT_SYNC_PLAN_MODE:-direct}
export ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER=${ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER:-4}
export ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE=${ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE:-thread}
export ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST=${ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST:-1}
export ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK=${ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK:-256}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

sim_preset=${MOE_SIM_PRESET:-small}
case "${sim_preset}" in
    small|moe-50b|moe-100b) ;;
    *)
        echo "MOE_SIM_PRESET must be small, moe-50b, or moe-100b: ${sim_preset}" >&2
        exit 2
        ;;
esac

# Both large presets have four local replica slots. The 50B-class proxy needs
# about 1.125 GiB for their BF16 weights and FP32 gradients; the 100B-class
# proxy needs about 2.7 GiB. Leave an explicit user setting untouched.
case "${sim_preset}" in
    moe-50b)
        export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-4294967296}
        ;;
    moe-100b)
        export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-8589934592}
        ;;
    *)
        export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-1073741824}
        ;;
esac

echo "MoE proxy preset=${sim_preset}, ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE} bytes, Weight Sync plan=${ULTRA_EP_WEIGHT_SYNC_PLAN_MODE}, HIP copy mode=${ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE}, threads/CTA=${ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK}, LDS waves/destination=${ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST}, CTA multiplier=${ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER}"
if [[ "${ULTRA_EP_WEIGHT_SYNC_PLAN_MODE}" != "direct" ]]; then
    export ULTRA_EP_ALLOW_SMALL_DOMAIN_RELAY=${ULTRA_EP_ALLOW_SMALL_DOMAIN_RELAY:-1}
    echo "Relay diagnostic mode enabled for the 8-rank domain. Do not use this override as the production default."
fi


nproc_per_node=${NPROC_PER_NODE:-8}
python -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${nproc_per_node}" \
    tests/hcu/hcu_moe_training_sim.py \
    --preset "${sim_preset}" \
    "$@" \
    |& tee "${log_path}"
