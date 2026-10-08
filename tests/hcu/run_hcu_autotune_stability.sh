#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Eight-HCU integration validation of Manager autotuning with real placement,
# weight-sync, gradient reduction, and data-result checks.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_path=${1:-"${repo_dir}/hcu_autotune_stability.log"}
if [[ $# -gt 0 ]]; then
    shift
fi

mkdir -p "$(dirname "${log_path}")"
cd "${repo_dir}"

export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-8}
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-2147483648}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
nproc_per_node=${NPROC_PER_NODE:-8}

# Do not set WEIGHT_SYNC_* or GRAD_REDUCE_NUM_SMS here.  If the caller exports
# one, UltraEP treats it as intentionally fixed and prints the locked result.
python -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${nproc_per_node}" \
    tests/hcu/hcu_single_node_stability.py \
    --warmup-iterations 0 \
    --iterations "${AUTOTUNE_STABILITY_ITERATIONS:-22}" \
    --validate-every 1 \
    --report-every 1 \
    --enable-autotune \
    "$@" \
    |& tee "${log_path}"
