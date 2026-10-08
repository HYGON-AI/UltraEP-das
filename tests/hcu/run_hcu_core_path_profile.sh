#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Capture a focused, one-iteration Chrome trace of the single-node UltraEP
# control path.  The timed benchmark that follows provides the corresponding
# synchronized wall-clock breakdown.
#
# Usage:
#   bash tests/hcu/run_hcu_core_path_profile.sh [trace directory]

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
trace_dir=${1:-"${repo_dir}/hcu_core_path_traces/$(date +%Y%m%d_%H%M%S)"}
log_path="${trace_dir}/application.log"

mkdir -p "${trace_dir}"
cd "${repo_dir}"

# Use the current best-known single-node configuration.  These variables stay
# overridable so the same trace can be used to compare a candidate kernel.
export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-8}
export ULTRA_EP_WEIGHT_SYNC_PLAN_MODE=${ULTRA_EP_WEIGHT_SYNC_PLAN_MODE:-direct}
export ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE=${ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE:-thread}
export ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK=${ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK:-128}
export ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER=${ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER:-2}
export ULTRA_EP_GRAD_REDUCE_NUM_SMS=${ULTRA_EP_GRAD_REDUCE_NUM_SMS:-64}
export ULTRA_EP_GRAD_REDUCE_DETERMINISTIC=${ULTRA_EP_GRAD_REDUCE_DETERMINISTIC:-1}
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-8589934592}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

echo "UltraEP single-node core-path profile:"
echo "  traces=${trace_dir}"
echo "  Weight Sync=${ULTRA_EP_WEIGHT_SYNC_PLAN_MODE}/$ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE, threads=${ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK}, CTA multiplier=${ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER}"

python -m torch.distributed.run \
    --standalone --nproc_per_node=8 \
    tests/hcu/hcu_moe_training_sim.py \
    --preset moe-100b \
    --imbalance-ratios 2.5 \
    --warmup-iters 3 \
    --bench-iters 20 \
    --profile-breakdown \
    --core-profile-dir "${trace_dir}" \
    |& tee "${log_path}"

echo "Core-path profile completed. Open core_path_rank*.json in your profiler."
