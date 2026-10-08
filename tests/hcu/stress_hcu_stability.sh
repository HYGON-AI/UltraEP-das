#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Long-running single-node, eight-HCU stability test.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
iterations=${1:-1000}
log_path=${2:-"${repo_dir}/hcu_single_node_stability.log"}
trace_phases=${STABILITY_TRACE_PHASES:-0}
warmup_iterations=${STABILITY_WARMUP_ITERATIONS:-20}
memory_tolerance_mib=${STABILITY_MEMORY_TOLERANCE_MIB:-128}

if [[ ! "${iterations}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Iterations must be a positive integer: ${iterations}" >&2
    exit 2
fi

if (( iterations <= 100 )); then
    default_interval=25
else
    default_interval=100
fi
validate_every=${STABILITY_VALIDATE_EVERY:-${default_interval}}
report_every=${STABILITY_REPORT_EVERY:-${default_interval}}


mkdir -p "$(dirname "${log_path}")"
cd "${repo_dir}"

export EP_USE_NVIDIA_TOOLS=${EP_USE_NVIDIA_TOOLS:-1}
export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-8}
export ULTRA_EP_GRAD_REDUCE_NUM_SMS=${ULTRA_EP_GRAD_REDUCE_NUM_SMS:-64}
export ULTRA_EP_GRAD_REDUCE_DETERMINISTIC=${ULTRA_EP_GRAD_REDUCE_DETERMINISTIC:-1}
export ULTRA_EP_WEIGHT_SYNC_PLAN_MODE=${ULTRA_EP_WEIGHT_SYNC_PLAN_MODE:-direct}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

extra_args=()
if [[ "${trace_phases}" == "1" ]]; then
    extra_args+=(--trace-phases)
fi

echo "UltraEP stage-4 stability configuration:"
echo "  iterations=${iterations}"
echo "  warmup_iterations=${warmup_iterations}"
echo "  validate_every=${validate_every}"
echo "  report_every=${report_every}"
echo "  memory_tolerance_mib=${memory_tolerance_mib}"
echo "  trace_phases=${trace_phases}"

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    tests/hcu/hcu_single_node_stability.py \
    --iterations "${iterations}" \
    --warmup-iterations "${warmup_iterations}" \
    --validate-every "${validate_every}" \
    --report-every "${report_every}" \
    --memory-tolerance-mib "${memory_tolerance_mib}" \
    "${extra_args[@]}" \
    |& tee "${log_path}"
