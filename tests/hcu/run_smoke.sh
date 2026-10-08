#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Minimal single-node HCU smoke test using rocSHMEM IPC by default.

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${repo_dir}"

mode=${1:-smoke}
if [[ $# -gt 0 ]]; then
  shift
fi

case "${mode}" in
  smoke) app=tests/hcu/hcu_single_node_smoke.py ;;
  probe) app=tests/hcu/hcu_gda_p2p_probe.py ;;
  *)
    echo "Mode must be smoke or probe: ${mode}" >&2
    exit 2
    ;;
esac

nproc_per_node=${NPROC_PER_NODE:-8}
export HSA_USE_SVM=${HSA_USE_SVM:-0}
export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-${nproc_per_node}}
export EXPECTED_NVL_DOMAIN_SIZE=${EXPECTED_NVL_DOMAIN_SIZE:-${nproc_per_node}}
export ROCSHMEM_BACKEND=${ROCSHMEM_BACKEND:-ipc}
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-2147483648}

python -m torch.distributed.run \
  --standalone \
  --nproc-per-node="${nproc_per_node}" \
  "${app}" "$@"
