#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Compare the HCU/HIP Weight Sync copy implementations.
#
# Usage:
#   bash tests/hcu/benchmark_hcu_copy_path.sh [all|correctness|sweep|validate] \
#       [python executable] [log directory]
#
# The 100B sweep keeps the total launched threads per CU comparable:
#   thread: 64x4, 128x2, 256x1
#   LDS:    256x1 with 1, 2, or 4 waves per destination
#
# The validation step repeats the two preliminary winners (thread 128x2 and
# LDS 256x1/waves=2) on 50B. Their 100B results are already produced by the
# sweep, so they are not run twice.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${1:-all}
log_dir=${3:-"${repo_dir}/hcu_benchmark_logs/copy_path"}

case "${mode}" in
    all|correctness|sweep|validate) ;;
    *)
        echo "Mode must be all, correctness, sweep, or validate: ${mode}" >&2
        exit 2
        ;;
esac


mkdir -p "${log_dir}"
cd "${repo_dir}"

export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-8}
export ULTRA_EP_GRAD_REDUCE_NUM_SMS=${ULTRA_EP_GRAD_REDUCE_NUM_SMS:-64}
export ULTRA_EP_GRAD_REDUCE_DETERMINISTIC=${ULTRA_EP_GRAD_REDUCE_DETERMINISTIC:-1}
export ULTRA_EP_WEIGHT_SYNC_PLAN_MODE=direct
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

run_correctness_case() {
    local name=$1
    local copy_mode=$2
    local threads=$3
    local multiplier=$4
    local lds_waves=$5

    echo "===== correctness: ${name} ====="
    ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE="${copy_mode}" \
    ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK="${threads}" \
    ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER="${multiplier}" \
    ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST="${lds_waves}" \
    ROCSHMEM_HEAP_SIZE=2147483648 \
        python -m torch.distributed.run \
        --standalone --nproc_per_node=8 \
        tests/integration/runtime_e2e.py \
        --num-experts 16 \
        --num-redundant-experts-per-rank 2 \
        --tokens-per-rank 1024 \
        --topk 4 \
        --imbalance-ratios 2.5 \
        --expert-fc1-numel 1048576 \
        --expert-fc2-numel 524288 \
        --weight-data-bytes 2 \
        --weight-sync-plan-modes direct \
        --warmup-iters 3 \
        --bench-iters 10 \
        |& tee "${log_dir}/correctness_${name}.log"
}

run_proxy_case() {
    local name=$1
    local preset=$2
    local copy_mode=$3
    local threads=$4
    local multiplier=$5
    local lds_waves=$6

    echo "===== performance: ${name} ====="
    MOE_SIM_PRESET="${preset}" \
    ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE="${copy_mode}" \
    ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK="${threads}" \
    ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER="${multiplier}" \
    ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST="${lds_waves}" \
        tests/hcu/run_hcu_moe_training_sim.sh \
        "${log_dir}/${name}.log" \
        --imbalance-ratios 2.5 \
        --warmup-iters 3 \
        --bench-iters 20 \
        --profile-breakdown
}

run_correctness() {
    # Use the preliminary best settings for the implementation-level
    # correctness gate before collecting timing data.
    run_correctness_case thread_128x2 thread 128 2 1
    run_correctness_case lds_256x1_w2 lds 256 1 2
}

run_sweep() {
    local run
    for run in 1 2 3; do
        run_proxy_case "moe-100b_thread_64x4_run${run}"  moe-100b thread 64  4 1
        run_proxy_case "moe-100b_thread_128x2_run${run}" moe-100b thread 128 2 1
        run_proxy_case "moe-100b_thread_256x1_run${run}" moe-100b thread 256 1 1

        run_proxy_case "moe-100b_lds_256x1_w1_run${run}" moe-100b lds 256 1 1
        run_proxy_case "moe-100b_lds_256x1_w2_run${run}" moe-100b lds 256 1 2
        run_proxy_case "moe-100b_lds_256x1_w4_run${run}" moe-100b lds 256 1 4
    done
}

run_validation() {
    local run
    for run in 1 2 3; do
        run_proxy_case "moe-50b_best_thread_run${run}" \
            moe-50b thread 128 2 1
        run_proxy_case "moe-50b_best_lds_run${run}" \
            moe-50b lds 256 1 2
    done
}

case "${mode}" in
    correctness)
        run_correctness
        ;;
    sweep)
        run_sweep
        ;;
    validate)
        run_validation
        ;;
    all)
        run_correctness
        run_sweep
        run_validation
        ;;
esac

echo "HCU copy-path benchmark ${mode} completed. Logs: ${log_dir}"
