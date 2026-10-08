#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Focused single-node Weight Sync benchmark with a forced replica fan-out.

This intentionally bypasses quota placement after Manager construction.  It is
used to compare direct and relay protocols under an identical, valid mapping:
one logical master expert has one replica on every other rank in the local
scale-up domain.  It does not represent the production placement policy.
"""

import argparse
import os
import sys
import time
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Forced-fan-out UltraEP Weight Sync benchmark"
    )
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-redundant-experts-per-rank", type=int, default=4)
    parser.add_argument("--expert-fc1-numel", type=int, default=4096 * 14336)
    parser.add_argument("--expert-fc2-numel", type=int, default=4096 * 14336)
    parser.add_argument("--fanout", type=int, default=7)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument(
        "--plan",
        choices=("direct", "adaptive", "force_relay"),
        required=True,
    )
    args = parser.parse_args()
    if args.num_experts <= 0 or args.num_experts % 8 != 0:
        parser.error("--num-experts must be positive and divisible by 8")
    if args.num_redundant_experts_per_rank <= 0:
        parser.error("--num-redundant-experts-per-rank must be positive")
    if args.fanout < 1 or args.fanout > 7:
        parser.error("--fanout must be in [1, 7] for an eight-rank domain")
    if args.warmup_iters < 0 or args.bench_iters <= 0:
        parser.error("invalid warmup/bench iteration counts")
    return args


def max_across_ranks(value: float) -> float:
    tensor = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def force_fanout_mapping(manager, fanout: int):
    """Install a valid master-only map plus fanout replicas of logical expert 0."""

    world_size = dist.get_world_size()
    num_local_master = manager.num_local_master_experts
    num_local_physical = manager.num_local_physical_experts
    assert world_size == 8 and manager.nvl_domain_size == world_size

    p2l = manager.physical_to_logical_map[0]
    l2p = manager.logical_to_physical_map[0]
    counts = manager.logical_replica_counts[0]
    p2l.fill_(-1)
    l2p.fill_(-1)
    counts.zero_()

    # Retain every master in its canonical physical master slot.
    for logical_id in range(manager.num_global_logical_experts):
        master_rank = logical_id // num_local_master
        local_master = logical_id % num_local_master
        physical_id = master_rank * num_local_physical + local_master
        p2l[physical_id] = logical_id
        l2p[logical_id, 0] = physical_id
        counts[logical_id] = 1

    # logical expert 0 belongs to rank 0. Put one replica into redundant slot
    # zero of each selected remote rank; all slots are valid symmetric buffers.
    for replica_slot, target_rank in enumerate(range(1, fanout + 1), start=1):
        physical_id = target_rank * num_local_physical + num_local_master
        p2l[physical_id] = 0
        l2p[0, replica_slot] = physical_id
    counts[0] = fanout + 1
    torch.cuda.synchronize()


def fill_master_weights(fc1_weights, fc2_weights, rank: int):
    for local_idx, (fc1, fc2) in enumerate(zip(fc1_weights, fc2_weights)):
        value = float(rank * len(fc1_weights) + local_idx + 1)
        fc1.fill_(value)
        fc2.fill_(-value)


def validate_hotspot_replicas(manager, fanout: int):
    rank = dist.get_rank()
    local_ok = True
    if 1 <= rank <= fanout:
        local_ok = bool(
            (manager.local_replica_fc1_weight_buffer[0] == 1).all().item()
            and (manager.local_replica_fc2_weight_buffer[0] == -1).all().item()
        )
    result = torch.tensor([int(local_ok)], dtype=torch.int32, device="cuda")
    dist.all_reduce(result, op=dist.ReduceOp.MIN)
    if not bool(result.item()):
        raise RuntimeError("forced-fan-out Weight Sync content validation failed")


def main():
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device("cuda", local_rank))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    manager = None

    try:
        if world_size != 8:
            raise RuntimeError(f"this benchmark requires 8 ranks, got {world_size}")
        if args.num_experts % world_size != 0:
            raise RuntimeError("--num-experts must divide the world size")
        num_local_master = args.num_experts // world_size
        manager = ultra_ep.Manager(
            group=dist.group.WORLD,
            num_layers=1,
            num_local_master_experts=num_local_master,
            num_local_redundant_experts=args.num_redundant_experts_per_rank,
            expert_fc1_numel=args.expert_fc1_numel,
            expert_fc2_numel=args.expert_fc2_numel,
            is_train=False,
            explicitly_destroy=True,
            weight_data_dtype=torch.bfloat16,
        )
        if manager.nvl_domain_size != world_size:
            raise RuntimeError(
                f"expected an {world_size}-rank scale-up domain, got {manager.nvl_domain_size}"
            )

        fc1_weights = [
            torch.empty(args.expert_fc1_numel, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_local_master)
        ]
        fc2_weights = [
            torch.empty(args.expert_fc2_numel, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_local_master)
        ]
        fill_master_weights(fc1_weights, fc2_weights, rank)
        manager.construct_local_master_ptr_pool(0, fc1_weights, fc2_weights)
        force_fanout_mapping(manager, args.fanout)
        manager.set_weight_sync_plan_mode(args.plan)

        expert_bytes = (args.expert_fc1_numel + args.expert_fc2_numel) * 2
        if rank == 0:
            print(
                "Forced fan-out Weight Sync benchmark: "
                f"plan={args.plan}, fanout={args.fanout}, payload="
                f"{args.fanout * expert_bytes / (1024 ** 2):.1f} MiB from rank 0, "
                f"warmup/bench={args.warmup_iters}/{args.bench_iters}",
                flush=True,
            )

        # Verify both the forced map and the selected protocol before timing.
        dist.barrier()
        manager.weight_sync(0, async_finish=False)
        torch.cuda.synchronize()
        dist.barrier()
        validate_hotspot_replicas(manager, args.fanout)

        for _ in range(args.warmup_iters):
            dist.barrier()
            event = manager.weight_sync(0, async_finish=True)
            event.current_stream_wait()
            torch.cuda.synchronize()
            dist.barrier()

        times_ms = []
        for _ in range(args.bench_iters):
            dist.barrier()
            start = time.perf_counter()
            event = manager.weight_sync(0, async_finish=True)
            event.current_stream_wait()
            torch.cuda.synchronize()
            dist.barrier()
            times_ms.append(max_across_ranks((time.perf_counter() - start) * 1000.0))

        times_ms.sort()
        if len(times_ms) >= 3:
            times_ms = times_ms[1:-1]
        mean_ms = sum(times_ms) / len(times_ms)
        if rank == 0:
            gib_per_s = (args.fanout * expert_bytes / (1024 ** 3)) / (mean_ms / 1000.0)
            print(
                f"PASS: plan={args.plan}, fanout={args.fanout}, "
                f"phase mean={mean_ms:.3f} ms, "
                f"rank-0 logical payload bandwidth={gib_per_s:.1f} GiB/s",
                flush=True,
            )
    finally:
        if manager is not None:
            manager.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
