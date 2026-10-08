#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Launch the staged UltraEP multi-node HCU validation on every node.
#
# Required on each node:
#   NNODES=2 NODE_RANK=0|1 MASTER_ADDR=<node0-ip> NETWORK_INTERFACE=<ifname>
#
# Examples:
#   bash tests/hcu/run_hcu_multi_node.sh smoke
#   bash tests/hcu/run_hcu_multi_node.sh e2e
#   ULTRA_EP_GDA_PROBE_SENDER=10 ULTRA_EP_GDA_PROBE_TARGET=0 \\
#     bash tests/hcu/run_hcu_multi_node.sh probe

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mode=${1:-smoke}
log_dir=${MULTI_NODE_LOG_DIR:-"${repo_dir}/multi_node_logs"}

case "${mode}" in
    smoke|e2e|probe) ;;
    *)
        echo "Mode must be smoke or e2e or probe: ${mode}" >&2
        exit 2
        ;;
esac

: "${NNODES:?Set NNODES to the number of nodes}"
: "${NODE_RANK:?Set NODE_RANK to this node index}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rendezvous address on node 0}"

export HSA_USE_SVM=${HSA_USE_SVM:-0}

nproc_per_node=${NPROC_PER_NODE:-8}
master_port=${MASTER_PORT:-29500}
world_size=$((NNODES * nproc_per_node))

if (( NNODES < 2 )); then
    echo "Multi-node validation requires NNODES >= 2" >&2
    exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "NODE_RANK=${NODE_RANK} is outside [0, ${NNODES})" >&2
    exit 2
fi

mkdir -p "${log_dir}"
cd "${repo_dir}"

export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-${nproc_per_node}}
export EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-${world_size}}
export ULTRA_EP_VALIDATE_NODE_DOMAINS=${ULTRA_EP_VALIDATE_NODE_DOMAINS:-1}
export ULTRA_EP_REQUIRE_ROCSHMEM_TRANSPORT=${ULTRA_EP_REQUIRE_ROCSHMEM_TRANSPORT:-gda}
export ROCSHMEM_BACKEND=${ROCSHMEM_BACKEND:-gda}
export ROCSHMEM_GDA_PROVIDER=${ROCSHMEM_GDA_PROVIDER:-shca}
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-2147483648}
export ULTRA_EP_WEIGHT_SYNC_PLAN_MODE=${ULTRA_EP_WEIGHT_SYNC_PLAN_MODE:-direct}
export ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE=${ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE:-thread}
export ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK=${ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK:-128}
export ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER=${ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER:-2}
export ULTRA_EP_GRAD_REDUCE_NUM_SMS=${ULTRA_EP_GRAD_REDUCE_NUM_SMS:-64}
export ULTRA_EP_GRAD_REDUCE_DETERMINISTIC=${ULTRA_EP_GRAD_REDUCE_DETERMINISTIC:-1}
export ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA=${ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA:-64}
export EP_USE_NVIDIA_TOOLS=${EP_USE_NVIDIA_TOOLS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1

if [[ -n "${NETWORK_INTERFACE:-}" ]]; then
    export ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=${ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME:-${NETWORK_INTERFACE}}
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE}}
fi

common_torchrun=(
    python -m torch.distributed.run
    --nnodes "${NNODES}"
    --nproc-per-node "${nproc_per_node}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${master_port}"
)

echo "UltraEP multi-node validation:"
echo "  mode=${mode}, node=${NODE_RANK}/${NNODES}, local_ranks=${nproc_per_node}, world=${world_size}"
echo "  rendezvous=${MASTER_ADDR}:${master_port}"
echo "  scale-up-domain=${MAX_NUM_NVL_PEERS}"
echo "  rocSHMEM=${ROCSHMEM_BACKEND}/${ROCSHMEM_GDA_PROVIDER}"
echo "  network-interface=${NETWORK_INTERFACE:-<auto>}"
echo "  SHCA HCA=${ROCSHMEM_USE_IB_HCA:-<auto>}"

if [[ "${mode}" == "smoke" ]]; then
    app=(tests/hcu/hcu_multi_node_smoke.py)
elif [[ "${mode}" == "probe" ]]; then
    app=(tests/hcu/hcu_gda_p2p_probe.py)
else
    num_experts=${MULTI_NODE_NUM_EXPERTS:-$((world_size * 2))}
    tokens_per_rank=${MULTI_NODE_TOKENS_PER_RANK:-512}
    topk=${MULTI_NODE_TOPK:-2}
    warmup_iters=${MULTI_NODE_WARMUP_ITERS:-3}
    bench_iters=${MULTI_NODE_BENCH_ITERS:-10}
    read -r -a imbalance_ratios <<< "${MULTI_NODE_IMBALANCE_RATIOS:-1.5 2.5}"
    if (( num_experts % world_size != 0 )); then
        echo "MULTI_NODE_NUM_EXPERTS=${num_experts} must be divisible by world size ${world_size}" >&2
        exit 2
    fi
    app=(
        tests/integration/runtime_e2e.py
        --num-experts "${num_experts}"
        --num-redundant-experts-per-rank 1
        --tokens-per-rank "${tokens_per_rank}"
        --topk "${topk}"
        --imbalance-ratios "${imbalance_ratios[@]}"
        --expert-fc1-numel 262144
        --expert-fc2-numel 131072
        --weight-data-bytes 2
        --weight-sync-plan-modes direct
        --warmup-iters "${warmup_iters}"
        --bench-iters "${bench_iters}"
    )
fi

"${common_torchrun[@]}" "${app[@]}" \
    |& tee "${log_dir}/${mode}_node${NODE_RANK}.log"
