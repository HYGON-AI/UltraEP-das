#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Capture UltraEP's built-in load profile on one eight-HCU node, then verify
# that reroute conserved tokens and reduced the rank-level imbalance.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
trace_dir=${1:-"${repo_dir}/hcu_load_traces/$(date +%Y%m%d_%H%M%S)"}
if [[ $# -gt 0 ]]; then
    shift
fi
log_path=${LOAD_PROFILE_LOG_PATH:-"${trace_dir}/application.log"}

mkdir -p "${trace_dir}"
if compgen -G "${trace_dir}/*.npz" >/dev/null; then
    echo "Trace directory already contains .npz files; choose a new directory: ${trace_dir}" >&2
    exit 2
fi
cd "${repo_dir}"

export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-8}
export ULTRA_EP_WEIGHT_SYNC_PLAN_MODE=${ULTRA_EP_WEIGHT_SYNC_PLAN_MODE:-direct}
export ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER=${ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER:-2}
export ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE=${ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE:-thread}
export ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK=${ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK:-128}
export ULTRA_EP_LOAD_PROFILING=1
export ULTRA_EP_LOAD_PROFILE_DIR="${trace_dir}"
export ULTRA_EP_LOAD_PROFILE_RECORD_INTERVAL=${ULTRA_EP_LOAD_PROFILE_RECORD_INTERVAL:-1}
export ULTRA_EP_LOAD_PROFILE_FLUSH_INTERVAL=${ULTRA_EP_LOAD_PROFILE_FLUSH_INTERVAL:-16}
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-8589934592}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

echo "UltraEP 8-HCU load-profile run:"
echo "  trace_dir=${trace_dir}"
echo "  profile record/flush interval=${ULTRA_EP_LOAD_PROFILE_RECORD_INTERVAL}/${ULTRA_EP_LOAD_PROFILE_FLUSH_INTERVAL}"
echo "  expected route: ratio=2.5, direct IPC thread-copy"

python -m torch.distributed.run \
    --standalone \
    --nproc_per_node=8 \
    tests/hcu/hcu_moe_training_sim.py \
    --preset moe-100b \
    --imbalance-ratios 2.5 \
    --warmup-iters "${LOAD_PROFILE_WARMUP:-3}" \
    --bench-iters "${LOAD_PROFILE_BENCH:-20}" \
    "$@" \
    |& tee "${log_path}"
python tests/hcu/summarize_hcu_load_profile.py \
    "${trace_dir}" \
    --max-post-rank-imbalance "${LOAD_PROFILE_MAX_POST_IMBALANCE:-1.05}" \
    --require-improvement

echo "Open the visual viewer with:"
echo "  python -m ultra_ep.load_viewer --path ${trace_dir} --host 0.0.0.0 --port 8765"
