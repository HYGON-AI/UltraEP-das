#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Single-node, eight-HCU sensitivity sweep.
#
# Run from the repository root after installing the current UltraEP wheel:
#   bash tests/hcu/benchmark_hcu_sensitivity.sh [python executable] [log directory]
#
# The baseline is intentionally not repeated here.  Compare these logs with
# Compare against a baseline with tokens=1024, topk=4, ratios=1.0/2.0.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_dir=${2:-"${repo_dir}/hcu_benchmark_logs/sensitivity"}

mkdir -p "${log_dir}"
cd "${repo_dir}"

export EP_USE_NVIDIA_TOOLS=1
export MAX_NUM_NVL_PEERS=8

# Keep these fixed across all cases.  The weight tensors total 3 MiB per
# master expert, large enough to expose communication behaviour without making
# the correctness checks impractically long.
common_args=(
    --num-experts 16
    --num-redundant-experts-per-rank 2
    --expert-fc1-numel 1048576
    --expert-fc2-numel 524288
    --weight-data-bytes 2
    --weight-sync-plan-modes direct
    --warmup-iters 10
    --bench-iters 30
)

run_case() {
    local name=$1
    shift
    echo "===== ${name} ====="
    python -m torch.distributed.run \
        --standalone --nproc_per_node=8 \
        tests/integration/runtime_e2e.py "${common_args[@]}" "$@" \
        |& tee "${log_dir}/${name}.log"
}

# Token sweep: compare against the existing 1024-token baseline.
run_case tokens_256  --tokens-per-rank 256  --topk 4 --imbalance-ratios 1.0
run_case tokens_4096 --tokens-per-rank 4096 --topk 4 --imbalance-ratios 1.0

# Routing fan-out sweep: compare against the existing topk=4 baseline.
run_case topk_1 --tokens-per-rank 1024 --topk 1 --imbalance-ratios 1.0
run_case topk_8 --tokens-per-rank 1024 --topk 8 --imbalance-ratios 1.0

# Placement sensitivity: keep the routing shape fixed and increase skew.
run_case imbalance_1_5_2_5 \
    --tokens-per-rank 1024 --topk 4 --imbalance-ratios 1.5 2.5

echo "HCU sensitivity benchmark completed. Logs: ${log_dir}"
