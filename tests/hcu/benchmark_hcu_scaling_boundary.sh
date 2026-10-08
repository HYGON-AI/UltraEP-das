#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Validate the combined single-node HCU configuration and find
# the load-imbalance range in which UltraEP offsets its control/sync cost.
#
# Usage:
#   bash tests/hcu/benchmark_hcu_scaling_boundary.sh \
#       [all|correctness|boundary] [python executable] [log directory]
#
# Each imbalance ratio runs in a separate process. This keeps the timing result
# attributable to one routing case instead of averaging several cases together.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${1:-all}
log_dir=${3:-"${repo_dir}/hcu_benchmark_logs/scaling_boundary"}

case "${mode}" in
    all|correctness|boundary) ;;
    *)
        echo "Mode must be all, correctness, or boundary: ${mode}" >&2
        exit 2
        ;;
esac

mkdir -p "${log_dir}"
cd "${repo_dir}"

export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-8}
export ULTRA_EP_GRAD_REDUCE_NUM_SMS=64
export ULTRA_EP_GRAD_REDUCE_DETERMINISTIC=1
export ULTRA_EP_WEIGHT_SYNC_PLAN_MODE=direct
export ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE=thread
export ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK=128
export ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER=2
export ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

run_correctness() {
    echo "===== combined configuration correctness ====="
    ROCSHMEM_HEAP_SIZE=2147483648 \
        python -m torch.distributed.run \
        --standalone --nproc_per_node=8 \
        tests/integration/runtime_e2e.py \
        --num-experts 16 \
        --num-redundant-experts-per-rank 2 \
        --tokens-per-rank 1024 \
        --topk 4 \
        --imbalance-ratios 1.0 1.5 2.5 \
        --expert-fc1-numel 1048576 \
        --expert-fc2-numel 524288 \
        --weight-data-bytes 2 \
        --weight-sync-plan-modes direct \
        --warmup-iters 5 \
        --bench-iters 20 \
        |& tee "${log_dir}/correctness.log"
}

run_boundary_case() {
    local preset=$1
    local ratio=$2
    local run=$3
    local ratio_tag=${ratio/./_}
    local name="${preset}_ratio_${ratio_tag}_run${run}"
    local heap_size

    case "${preset}" in
        moe-50b)
            heap_size=2147483648
            ;;
        moe-100b)
            heap_size=8589934592
            ;;
        *)
            echo "Unsupported preset: ${preset}" >&2
            exit 2
            ;;
    esac

    echo "===== ${name} ====="
    MOE_SIM_PRESET="${preset}" \
    ROCSHMEM_HEAP_SIZE="${heap_size}" \
        ./tests/hcu/run_hcu_moe_training_sim.sh \
        "${log_dir}/${name}.log" \
        --imbalance-ratios "${ratio}" \
        --warmup-iters 3 \
        --bench-iters 30 \
        --profile-breakdown
}

run_boundary() {
    # Dense sampling around the expected break-even region, plus balanced and
    # heavily skewed endpoints.
    local ratios=(1.0 1.5 1.75 2.0 2.25 2.5 3.0)
    local preset
    local ratio
    local run

    for preset in moe-50b moe-100b; do
        for ratio in "${ratios[@]}"; do
            for run in 1 2 3; do
                run_boundary_case "${preset}" "${ratio}" "${run}"
            done
        done
    done
}

case "${mode}" in
    correctness)
        run_correctness
        ;;
    boundary)
        run_boundary
        ;;
    all)
        run_correctness
        run_boundary
        ;;
esac

echo "HCU scaling-boundary benchmark ${mode} completed. Logs: ${log_dir}"
