#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

# Run the framework-independent UltraEP MoE performance proxy on every HCU
# node.  The proxy executes real rocSHMEM global Placement and domain-local
# Weight Sync, while reporting (but deliberately not timing) the token
# dispatch/combine traffic that a framework must add later.
#
# Run this command on every node, changing NODE_RANK:
#   NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0-ip> NETWORK_INTERFACE=<ifname> \
#     bash tests/hcu/run_hcu_multi_node_proxy.sh

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
log_dir=${MULTI_NODE_LOG_DIR:-"${repo_dir}/multi_node_logs"}
sim_preset=${MULTI_NODE_MOE_SIM_PRESET:-moe-100b}

: "${NNODES:?Set NNODES to the number of nodes}"
: "${NODE_RANK:?Set NODE_RANK to this node index}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the rendezvous address on node 0}"

export HSA_USE_SVM=${HSA_USE_SVM:-0}

nproc_per_node=${NPROC_PER_NODE:-8}
master_port=${MASTER_PORT:-29500}
world_size=$((NNODES * nproc_per_node))

if (( NNODES < 2 )); then
    echo "Multi-node proxy requires NNODES >= 2" >&2
    exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NNODES )); then
    echo "NODE_RANK=${NODE_RANK} is outside [0, ${NNODES})" >&2
    exit 2
fi
case "${sim_preset}" in
    small|moe-50b|moe-100b) ;;
    *)
        echo "MULTI_NODE_MOE_SIM_PRESET must be small, moe-50b, or moe-100b" >&2
        exit 2
        ;;
esac

mkdir -p "${log_dir}"
cd "${repo_dir}"

export MAX_NUM_NVL_PEERS=${MAX_NUM_NVL_PEERS:-${nproc_per_node}}
export EXPECTED_WORLD_SIZE=${EXPECTED_WORLD_SIZE:-${world_size}}
export ULTRA_EP_VALIDATE_NODE_DOMAINS=${ULTRA_EP_VALIDATE_NODE_DOMAINS:-1}
export ULTRA_EP_REQUIRE_ROCSHMEM_TRANSPORT=${ULTRA_EP_REQUIRE_ROCSHMEM_TRANSPORT:-gda}
export ROCSHMEM_BACKEND=${ROCSHMEM_BACKEND:-gda}
export ROCSHMEM_GDA_PROVIDER=${ROCSHMEM_GDA_PROVIDER:-shca}
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

# Four BF16 redundant experts plus their FP32 gradients need a larger symmetric
# heap for the model-size proxies.  Preserve an explicit user value.
case "${sim_preset}" in
    moe-100b) export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-8589934592} ;;
    moe-50b)  export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-4294967296} ;;
    *)        export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-1073741824} ;;
esac

if [[ -n "${NETWORK_INTERFACE:-}" ]]; then
    export ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=${ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME:-${NETWORK_INTERFACE}}
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-${NETWORK_INTERFACE}}
fi

num_experts=${MULTI_NODE_MOE_SIM_NUM_EXPERTS:-128}
tokens_per_rank=${MULTI_NODE_MOE_SIM_TOKENS_PER_RANK:-2048}
topk=${MULTI_NODE_MOE_SIM_TOPK:-2}
warmup_iters=${MULTI_NODE_MOE_SIM_WARMUP_ITERS:-5}
bench_iters=${MULTI_NODE_MOE_SIM_BENCH_ITERS:-30}
read -r -a imbalance_ratios <<< "${MULTI_NODE_MOE_SIM_IMBALANCE_RATIOS:-1.0 1.5 2.5}"

if (( num_experts % world_size != 0 )); then
    echo "MULTI_NODE_MOE_SIM_NUM_EXPERTS=${num_experts} must be divisible by world size ${world_size}" >&2
    exit 2
fi

echo "UltraEP multi-node MoE proxy:"
echo "  preset=${sim_preset}, node=${NODE_RANK}/${NNODES}, world=${world_size}, domain=${MAX_NUM_NVL_PEERS}"
echo "  experts=${num_experts}, tokens/rank=${tokens_per_rank}, topk=${topk}, ratios=${imbalance_ratios[*]}"
echo "  warmup/bench=${warmup_iters}/${bench_iters}, rocSHMEM=${ROCSHMEM_BACKEND}/${ROCSHMEM_GDA_PROVIDER}"
echo "  IMPORTANT: token dispatch/combine is reported as traffic but excluded from the timed proxy."

torchrun=(
    python -m torch.distributed.run
    --nnodes "${NNODES}"
    --nproc-per-node "${nproc_per_node}"
    --node-rank "${NODE_RANK}"
    --master-addr "${MASTER_ADDR}"
    --master-port "${master_port}"
)

for ratio in "${imbalance_ratios[@]}"; do
    echo
    echo "================================================================================"
    echo "Independent imbalance case: ratio=${ratio}"
    echo "================================================================================"

    "${torchrun[@]}" \
        tests/hcu/hcu_moe_training_sim.py \
        --preset "${sim_preset}" \
        --num-experts "${num_experts}" \
        --tokens-per-rank "${tokens_per_rank}" \
        --topk "${topk}" \
        --imbalance-ratios "${ratio}" \
        --warmup-iters "${warmup_iters}" \
        --bench-iters "${bench_iters}" \
        --profile-breakdown \
        "$@" |& tee "${log_dir}/proxy_${sim_preset}_ratio${ratio}_node${NODE_RANK}.log"
done
