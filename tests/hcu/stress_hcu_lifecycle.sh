#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Exercise one Manager/rocSHMEM lifecycle per fresh worker group.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cycles=${1:-20}
log_path=${2:-"${repo_dir}/hcu_manager_lifecycle.log"}

if [[ ! "${cycles}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Cycles must be a positive integer: ${cycles}" >&2
    exit 2
fi
mkdir -p "$(dirname "${log_path}")"
cd "${repo_dir}"

export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-8}
export ULTRA_EP_GRAD_REDUCE_NUM_SMS=${ULTRA_EP_GRAD_REDUCE_NUM_SMS:-64}
export ULTRA_EP_GRAD_REDUCE_DETERMINISTIC=${ULTRA_EP_GRAD_REDUCE_DETERMINISTIC:-1}
export ULTRA_EP_WEIGHT_SYNC_PLAN_MODE=${ULTRA_EP_WEIGHT_SYNC_PLAN_MODE:-direct}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

echo "UltraEP stage-4 worker-lifecycle configuration:"
echo "  cycles=${cycles}"
echo "  each cycle uses a fresh 8-rank worker group"

: > "${log_path}"
for ((cycle = 1; cycle <= cycles; ++cycle)); do
    echo "[worker cycle ${cycle}/${cycles}] launching fresh workers" | tee -a "${log_path}"
    python -m torch.distributed.run \
        --standalone \
        --nproc_per_node=8 \
        tests/hcu/hcu_manager_lifecycle.py \
        --worker-cycle "${cycle}" \
        |& tee -a "${log_path}"
    echo "[worker cycle ${cycle}/${cycles}] PASS: worker group exited cleanly" | tee -a "${log_path}"
done
