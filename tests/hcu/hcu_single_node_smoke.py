#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Minimal 8-HCU, single-node UltraEP/RCCL/rocSHMEM smoke test.

Run through build_hcu_shca.sh so the DTK, rocSHMEM and SHCA runtime library
paths are identical to the production launch environment.
"""

import os
import sys
from pathlib import Path


# Exercise the installed wheel rather than an in-tree package or build output.
REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_string = str(REPO_ROOT)
if repo_root_string in sys.path:
    sys.path.remove(repo_root_string)
if "" in sys.path and Path.cwd().resolve() == REPO_ROOT:
    sys.path.remove("")

import torch
import torch.distributed as dist

import ultra_ep
import ultra_ep._C as ext


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")  # RCCL uses PyTorch's nccl name.

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    manager = None
    completed = False

    try:
        if world_size != 8:
            raise RuntimeError(f"This smoke test requires 8 ranks, got {world_size}")

        # First establish that RCCL communication itself is healthy.
        rccl_value = torch.tensor([float(rank + 1)], device="cuda")
        dist.all_reduce(rccl_value, op=dist.ReduceOp.SUM)
        if rccl_value.item() != 36.0:
            raise RuntimeError(f"RCCL all_reduce returned {rccl_value.item()}, expected 36")

        # Manager construction initializes rocSHMEM, allocates symmetric heap
        # buffers, synchronizes all PEs and resolves same-node remote pointers.
        manager = ultra_ep.Manager(
            group=dist.group.WORLD,
            num_layers=1,
            num_local_master_experts=1,
            num_local_redundant_experts=1,
            expert_fc1_numel=8,
            expert_fc2_numel=8,
            is_train=False,
            explicitly_destroy=True,
            weight_data_dtype=torch.bfloat16,
        )

        expected_nvl_domain_size = int(os.getenv("EXPECTED_NVL_DOMAIN_SIZE", str(world_size)))
        if manager.nvl_domain_size != expected_nvl_domain_size:
            raise RuntimeError(
                f"NVL domain size is {manager.nvl_domain_size}, expected "
                f"{expected_nvl_domain_size}. Check peer access and "
                "MAX_NUM_NVL_PEERS."
            )

        # Rank r contributes r+1 tokens to logical expert r. update_placement
        # performs rocSHMEM fcollect/reduction; the resulting global load must
        # therefore be [1, 2, ..., 8] on every HCU.
        routing_map = torch.zeros((rank + 1, world_size), dtype=torch.bool, device="cuda")
        routing_map[:, rank] = True
        manager.update_placement(0, routing_map, verify_reduced_loads=True)
        torch.cuda.synchronize()

        observed_loads = manager.runtime.get_global_logical_expert_loads_tensor().cpu()
        expected_loads = torch.arange(1, world_size + 1, dtype=torch.int32)
        if not torch.equal(observed_loads, expected_loads):
            raise RuntimeError(
                f"rocSHMEM placement reduction mismatch: got {observed_loads.tolist()}, "
                f"expected {expected_loads.tolist()}"
            )

        dist.barrier()
        if rank == 0:
            print(
                "PASS: 8-HCU RCCL all_reduce, rocSHMEM initialization, "
                "symmetric heap/peer pointers, and placement fcollect succeeded."
            )
            print(f"SHMEM backend: {ext.get_shmem_backend_name()}")
        completed = True
    finally:
        # Manager::destroy performs the coordinated rocSHMEM barrier/finalize.
        # Only enter it after every rank has reached the successful path.
        if completed:
            dist.barrier()
            if manager is not None:
                manager.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
