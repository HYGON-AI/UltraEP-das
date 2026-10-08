#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Compare direct and staged relay Weight Sync under a forced fan-out-7 mapping
# within one eight-HCU scale-up domain. This is a performance experiment only;
# it does not alter the production direct-path default.
#
# Usage:
#   bash tests/hcu/run_hcu_high_fanout_relay_sweep.sh [python executable] [log directory]

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_dir=${2:-"${repo_dir}/high_fanout_relay_logs"}


mkdir -p "${log_dir}"
cd "${repo_dir}"

export MAX_NUM_NVL_PEERS=8
export ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE=thread
export ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK=128
export ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER=2
export ULTRA_EP_GRAD_REDUCE_NUM_SMS=64
export ULTRA_EP_GRAD_REDUCE_DETERMINISTIC=1
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-8589934592}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

run_case() {
    local plan=$1
    local run=$2
    local allow_small_domain_relay=0
    if [[ "${plan}" != "direct" ]]; then
        allow_small_domain_relay=1
    fi

    echo "===== forced fan-out 7: plan=${plan}, run=${run} ====="
    ULTRA_EP_WEIGHT_SYNC_PLAN_MODE="${plan}" \
    ULTRA_EP_ALLOW_SMALL_DOMAIN_RELAY="${allow_small_domain_relay}" \
        python -m torch.distributed.run \
        --standalone --nproc_per_node=8 \
        tests/hcu/hcu_weight_sync_fanout.py \
        --plan "${plan}" \
        --fanout 4 \
        --warmup-iters 3 \
        --bench-iters 20 \
        |& tee "${log_dir}/${plan}_run${run}.log"
}

# Three runs make small timing differences distinguishable from normal jitter.
for plan in direct adaptive force_relay; do
    for run in 1 2; do
        run_case "${plan}" "${run}"
    done
done

echo "High-fanout relay sweep complete. Logs: ${log_dir}"
