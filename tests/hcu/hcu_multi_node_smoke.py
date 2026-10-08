#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Focused multi-node HCU bootstrap and placement smoke test.

This test intentionally keeps expert buffers small.  It verifies the pieces
that are unique to a multi-node UltraEP world before the full E2E test is run:
RCCL world communication, rocSHMEM GDA/SHCA initialization, world fcollect,
multiple scale-up domains, domain-local replication, and dense reroute.
"""

import os
import sys
from pathlib import Path


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
    dist.init_process_group(backend="nccl")  # PyTorch uses this name for RCCL.

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    domain_size = int(os.getenv("MAX_NUM_NVL_PEERS", "8"))
    expected_world_size = int(os.getenv("EXPECTED_WORLD_SIZE", str(world_size)))
    expected_transport = os.getenv("ULTRA_EP_REQUIRE_ROCSHMEM_TRANSPORT", "gda").lower()
    placement_mode = os.getenv("ULTRA_EP_SMOKE_PLACEMENT_MODE", "quota").strip().lower()
    if placement_mode not in ("quota", "legacy"):
        raise ValueError("ULTRA_EP_SMOKE_PLACEMENT_MODE must be quota or legacy")

    manager = None
    completed = False

    try:
        if world_size != expected_world_size:
            raise RuntimeError(
                f"world size is {world_size}, expected {expected_world_size}"
            )
        if world_size <= domain_size or world_size % domain_size != 0:
            raise RuntimeError(
                f"multi-node smoke requires world_size > domain_size and exact "
                f"division, got world={world_size}, domain={domain_size}"
            )

        # Establish that RCCL reaches every rank before rocSHMEM is involved.
        rccl_value = torch.tensor([rank + 1], dtype=torch.int64, device="cuda")
        dist.all_reduce(rccl_value, op=dist.ReduceOp.SUM)
        expected_sum = world_size * (world_size + 1) // 2
        if int(rccl_value.item()) != expected_sum:
            raise RuntimeError(
                f"RCCL all_reduce returned {int(rccl_value.item())}, expected {expected_sum}"
            )

        # Small loads need a small replica quota to exercise placement.
        os.environ["ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA"] = "1"
        manager = ultra_ep.Manager(
            group=dist.group.WORLD,
            num_layers=1,
            num_local_master_experts=1,
            num_local_redundant_experts=1,
            expert_fc1_numel=8,
            expert_fc2_numel=8,
            is_train=False,
            explicitly_destroy=True,
            legacy_placement=(placement_mode == "legacy"),
            weight_data_dtype=torch.bfloat16,
        )
        runtime_info = dict(ext.get_shmem_runtime_info())
        if manager.nvl_domain_size != domain_size:
            raise RuntimeError(
                f"scale-up domain is {manager.nvl_domain_size}, expected {domain_size}"
            )
        if int(runtime_info["pe"]) != rank or int(runtime_info["num_pes"]) != world_size:
            raise RuntimeError(f"unexpected rocSHMEM PE mapping: {runtime_info}")
        if str(runtime_info["transport"]).lower() != expected_transport:
            raise RuntimeError(
                f"rocSHMEM transport is {runtime_info['transport']}, "
                f"expected {expected_transport}"
            )
        expected_domains = world_size // domain_size
        if int(runtime_info["num_domains"]) != expected_domains:
            raise RuntimeError(
                f"runtime reports {runtime_info['num_domains']} domains, "
                f"expected {expected_domains}"
            )

        # Rank r contributes r+1 tokens to logical expert r.  The rocSHMEM
        # world fcollect must produce [1, 2, ..., world_size] on every rank.
        routing_map = torch.zeros(
            (rank + 1, world_size), dtype=torch.bool, device="cuda"
        )
        routing_map[:, rank] = True
        manager.update_placement(0, routing_map, verify_reduced_loads=False)
        torch.cuda.synchronize()

        expected_reduced_loads = routing_map.sum(dim=0, dtype=torch.int32)
        dist.all_reduce(expected_reduced_loads, group=dist.group.WORLD)

        observed_loads = manager.runtime.get_global_logical_expert_loads_tensor().cpu()
        expected_loads = torch.arange(1, world_size + 1, dtype=torch.int32)
        if not torch.equal(expected_reduced_loads.cpu(), expected_loads):
            raise RuntimeError(
                f"RCCL reduced loads mismatch: got {expected_reduced_loads.cpu().tolist()}, "
                f"expected {expected_loads.tolist()}"
            )
        if not torch.equal(observed_loads, expected_loads):
            raise RuntimeError(
                f"rocSHMEM world fcollect mismatch: got {observed_loads.tolist()}, "
                f"expected {expected_loads.tolist()}"
            )

        # One master and one redundant slot per rank means every physical rank
        # is `physical_id // 2`.  All replicas must stay in the master's
        # contiguous scale-up domain even though planning used global loads.
        counts = manager.logical_replica_counts[0].cpu()
        l2p = manager.logical_to_physical_map[0].cpu()
        replicas_per_domain = [0] * expected_domains
        for logical in range(world_size):
            master_rank = logical
            master_domain = master_rank // domain_size
            count = int(counts[logical].item())
            if count < 1:
                raise RuntimeError(f"logical expert {logical} lost its master")
            for slot in range(count):
                physical = int(l2p[logical, slot].item())
                physical_rank = physical // manager.num_local_physical_experts
                if physical_rank // domain_size != master_domain:
                    raise RuntimeError(
                        f"logical expert {logical} in domain {master_domain} was "
                        f"placed on rank {physical_rank}"
                    )
            replicas_per_domain[master_domain] += count - 1
        if any(count == 0 for count in replicas_per_domain):
            raise RuntimeError(
                f"placement did not exercise replicas in every domain: {replicas_per_domain}"
            )

        # Dense reroute must preserve the logical expert while selecting one of
        # its physical instances.
        probs = routing_map.to(torch.float32)
        expanded_probs, expanded_map = manager.reroute(0, probs, routing_map)
        torch.cuda.synchronize()
        if not torch.equal(expanded_map.sum(dim=1), torch.ones(rank + 1, device="cuda", dtype=torch.int64)):
            raise RuntimeError("dense reroute did not select exactly one physical expert per token")
        selected_physical = expanded_map.to(torch.int32).argmax(dim=1)
        p2l = manager.physical_to_logical_map[0]
        selected_logical = p2l[selected_physical]
        if not torch.equal(selected_logical, torch.full_like(selected_logical, rank)):
            raise RuntimeError(
                f"dense reroute changed logical expert on rank {rank}: "
                f"{selected_logical.cpu().tolist()}"
            )
        selected_probs = expanded_probs.gather(1, selected_physical[:, None]).squeeze(1)
        if not torch.equal(selected_probs, torch.ones_like(selected_probs)):
            raise RuntimeError("dense reroute did not preserve routing probabilities")

        dist.barrier()
        if rank == 0:
            print(
                "PASS: multi-node RCCL, rocSHMEM world fcollect, node-local "
                "replication, and dense reroute succeeded.",
                flush=True,
            )
            print(
                f"Topology: world={world_size}, scale-up-domain={domain_size}, "
                f"domains={expected_domains}, replicas/domain={replicas_per_domain}",
                flush=True,
            )
            print(f"SHMEM runtime: {runtime_info}", flush=True)
        completed = True
    finally:
        # Avoid entering a mismatched collective after a partial bootstrap
        # failure. torchrun will terminate the remaining workers in that case.
        if completed:
            dist.barrier()
            if manager is not None:
                manager.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
