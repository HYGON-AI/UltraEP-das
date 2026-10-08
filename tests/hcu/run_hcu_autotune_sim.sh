#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Real eight-HCU integration test for UltraEP's bounded weight-sync autotuner.
# Unlike run_hcu_moe_training_sim.sh, deliberately do not export any tunable
# ULTRA_EP_* values: an exported value is an explicit user lock by design.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_path=${1:-"${repo_dir}/hcu_autotune_sim.log"}
if [[ $# -gt 0 ]]; then
    shift
fi

mkdir -p "$(dirname "${log_path}")"
cd "${repo_dir}"

export HSA_USE_SVM=${HSA_USE_SVM:-0}
export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-8}
export ROCSHMEM_BACKEND=${ROCSHMEM_BACKEND:-gda}
export ROCSHMEM_GDA_PROVIDER=${ROCSHMEM_GDA_PROVIDER:-shca}
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-8589934592}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

sim_preset=${MOE_SIM_PRESET:-moe-100b}
nproc_per_node=${NPROC_PER_NODE:-8}

echo "UltraEP autotune simulation: tunable environment values left unset unless supplied by caller"
python -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${nproc_per_node}" \
    tests/hcu/hcu_moe_training_sim.py \
    --preset "${sim_preset}" \
    --warmup-iters "${AUTOTUNE_WARMUP_ITERS:-5}" \
    --bench-iters "${AUTOTUNE_BENCH_ITERS:-20}" \
    --enable-autotune \
    --profile-breakdown \
    --imbalance-ratios ${AUTOTUNE_IMBALANCE_RATIOS:-1.5 2.5} \
    "$@" \
    |& tee "${log_path}"
