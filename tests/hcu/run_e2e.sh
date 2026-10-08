#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Single-node HCU correctness and runtime integration test.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_dir}"

nproc_per_node=${NPROC_PER_NODE:-8}
log_path=${HCU_E2E_LOG_PATH:-"${repo_dir}/hcu_e2e.log"}

export EP_USE_NVIDIA_TOOLS=${EP_USE_NVIDIA_TOOLS:-1}
export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-${nproc_per_node}}

python -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${nproc_per_node}" \
  tests/integration/runtime_e2e.py \
  --num-experts "${HCU_E2E_NUM_EXPERTS:-16}" \
  --num-redundant-experts-per-rank "${HCU_E2E_REDUNDANT_EXPERTS_PER_RANK:-2}" \
  --topk "${HCU_E2E_TOPK:-4}" \
  --tokens-per-rank "${HCU_E2E_TOKENS_PER_RANK:-1024}" \
  --variable-input-tokens \
  --imbalance-ratios ${HCU_E2E_IMBALANCE_RATIOS:-1.0 2.0} \
  --expert-fc1-numel "${HCU_E2E_FC1_NUMEL:-1048576}" \
  --expert-fc2-numel "${HCU_E2E_FC2_NUMEL:-524288}" \
  --weight-data-bytes 2 \
  --weight-sync-plan-modes "${ULTRA_EP_WEIGHT_SYNC_PLAN_MODE:-direct}" \
  --warmup-iters "${HCU_E2E_WARMUP_ITERS:-20}" \
  --bench-iters "${HCU_E2E_BENCH_ITERS:-50}" \
  "$@" |& tee "${log_path}"
