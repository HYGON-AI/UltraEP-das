#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""One complete UltraEP worker lifecycle on one eight-HCU node.

The launcher starts a fresh eight-rank worker group for every repetition. This
matches normal training and inference, where UltraEP and rocSHMEM are created
once and live until the worker process exits.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
if "" in sys.path and os.path.abspath(os.getcwd()) == REPO_ROOT:
    sys.path.remove("")

import ultra_ep


def print_rank0(message: str):
    if dist.get_rank() == 0:
        print(message, flush=True)


def setup_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        device_id=torch.device("cuda", local_rank),
    )


def distributed_validate(label: str, validation_fn):
    local_error = None
    try:
        validation_fn()
    except Exception as exc:
        local_error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"

    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    failures = [error for error in errors if error is not None]
    if failures:
        raise AssertionError(f"{label} failed:\n" + "\n".join(failures))


def assert_tensor_scalar(tensor: torch.Tensor, expected: float, label: str):
    if not bool((tensor == expected).all().item()):
        raise AssertionError(
            f"{label} mismatch on rank {dist.get_rank()}: expected {expected}"
        )


def fill_master_tensors(tensors, base: int):
    rank = dist.get_rank()
    for local_idx, tensor in enumerate(tensors):
        logical_id = rank * len(tensors) + local_idx
        tensor.fill_(float(base + logical_id))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one full UltraEP Manager lifecycle on eight HCUs"
    )
    parser.add_argument("--worker-cycle", type=int, default=1)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--num-redundant-experts-per-rank", type=int, default=2)
    parser.add_argument("--tokens-per-rank", type=int, default=1024)
    parser.add_argument("--expert-fc1-numel", type=int, default=262144)
    parser.add_argument("--expert-fc2-numel", type=int, default=131072)
    args = parser.parse_args()

    if args.worker_cycle <= 0:
        parser.error("--worker-cycle must be positive")
    if args.tokens_per_rank <= 0:
        parser.error("--tokens-per-rank must be positive")
    return args


def exercise_one_manager(args):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_local_master = args.num_experts // world_size
    num_local_redundant = args.num_redundant_experts_per_rank
    num_local_physical = num_local_master + num_local_redundant

    manager = None
    try:
        manager = ultra_ep.Manager(
            group=dist.group.WORLD,
            num_layers=1,
            num_local_master_experts=num_local_master,
            num_local_redundant_experts=num_local_redundant,
            expert_fc1_numel=args.expert_fc1_numel,
            expert_fc2_numel=args.expert_fc2_numel,
            is_train=True,
            explicitly_destroy=True,
            weight_data_dtype=torch.bfloat16,
            grad_dtype=torch.float32,
        )
        if manager.nvl_domain_size != world_size:
            raise RuntimeError(
                f"expected NVL domain {world_size}, "
                f"got {manager.nvl_domain_size}"
            )

        fc1_weights = [
            torch.empty(args.expert_fc1_numel, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_local_master)
        ]
        fc2_weights = [
            torch.empty(args.expert_fc2_numel, dtype=torch.bfloat16, device="cuda")
            for _ in range(num_local_master)
        ]
        fc1_grads = [
            torch.empty(args.expert_fc1_numel, dtype=torch.float32, device="cuda")
            for _ in range(num_local_master)
        ]
        fc2_grads = [
            torch.empty(args.expert_fc2_numel, dtype=torch.float32, device="cuda")
            for _ in range(num_local_master)
        ]

        fill_master_tensors(fc1_weights, 1)
        fill_master_tensors(fc2_weights, 33)
        fill_master_tensors(fc1_grads, 65)
        fill_master_tensors(fc2_grads, 97)
        manager.construct_local_master_ptr_pool(
            0,
            fc1_weights,
            fc2_weights,
            fc1_grads,
            fc2_grads,
        )

        # Send every token to logical expert 0. This guarantees a hot expert
        # and therefore exercises replica allocation in every lifecycle.
        routing_map = torch.zeros(
            (args.tokens_per_rank, args.num_experts),
            dtype=torch.bool,
            device="cuda",
        )
        routing_map[:, 0] = True
        probs = routing_map.to(torch.float32)
        manager.update_placement(0, routing_map, verify_reduced_loads=True)
        expanded_probs, expanded_routing = manager.reroute(0, probs, routing_map)

        def validate_reroute():
            if not bool((expanded_routing.sum(dim=1) == 1).all().item()):
                raise AssertionError("reroute did not preserve topk=1")
            replica_count = int(manager.logical_replica_counts[0, 0].item()) - 1
            if replica_count <= 0:
                raise AssertionError("hot expert 0 did not receive a replica")

        distributed_validate("reroute", validate_reroute)

        replica_weights = manager.local_replica_weight_buffer
        replica_fc1_weights = manager.local_replica_fc1_weight_buffer
        replica_fc2_weights = manager.local_replica_fc2_weight_buffer
        replica_grads = manager.local_replica_grad_buffer
        replica_fc1_grads = manager.local_replica_fc1_grad_buffer
        replica_fc2_grads = manager.local_replica_fc2_grad_buffer
        local_replica_phys = (
            rank * num_local_physical
            + num_local_master
            + torch.arange(num_local_redundant, dtype=torch.int64, device="cuda")
        )

        replica_weights.zero_()
        dist.barrier()
        weight_event = manager.weight_sync(0, async_finish=True)
        weight_event.current_stream_wait()
        torch.cuda.synchronize()
        dist.barrier()

        def validate_weight_sync():
            logical_ids = manager.physical_to_logical_map[
                0, local_replica_phys
            ].tolist()
            for replica_idx, logical_id in enumerate(logical_ids):
                if logical_id < 0:
                    continue
                assert_tensor_scalar(
                    replica_fc1_weights[replica_idx],
                    float(logical_id + 1),
                    f"replica FC1 weight {replica_idx}",
                )
                assert_tensor_scalar(
                    replica_fc2_weights[replica_idx],
                    float(logical_id + 33),
                    f"replica FC2 weight {replica_idx}",
                )

        distributed_validate("weight sync", validate_weight_sync)

        fill_master_tensors(fc1_grads, 65)
        fill_master_tensors(fc2_grads, 97)
        for replica_idx in range(num_local_redundant):
            physical_id = rank * num_local_physical + num_local_master + replica_idx
            replica_fc1_grads[replica_idx].fill_(float(physical_id + 129))
            replica_fc2_grads[replica_idx].fill_(float(physical_id + 193))

        torch.cuda.synchronize()
        dist.barrier()
        grad_event = manager.grad_reduce(0, async_finish=True)
        grad_event.current_stream_wait()
        torch.cuda.synchronize()
        dist.barrier()

        def validate_grad_reduce():
            l2p = manager.logical_to_physical_map[0]
            for local_idx in range(num_local_master):
                logical_id = rank * num_local_master + local_idx
                master_physical_id = rank * num_local_physical + local_idx
                physical_ids = [
                    int(value)
                    for value in l2p[logical_id].tolist()
                    if int(value) >= 0
                ]
                replica_ids = [
                    value for value in physical_ids if value != master_physical_id
                ]
                expected_fc1 = float(
                    logical_id + 65 + sum(value + 129 for value in replica_ids)
                )
                expected_fc2 = float(
                    logical_id + 97 + sum(value + 193 for value in replica_ids)
                )
                assert_tensor_scalar(
                    fc1_grads[local_idx],
                    expected_fc1,
                    f"master FC1 grad {local_idx}",
                )
                assert_tensor_scalar(
                    fc2_grads[local_idx],
                    expected_fc2,
                    f"master FC2 grad {local_idx}",
                )

            valid_replicas = (
                manager.physical_to_logical_map[0, local_replica_phys] >= 0
            )
            if bool(valid_replicas.any().item()):
                if bool((replica_grads[valid_replicas] != 0).any().item()):
                    raise AssertionError("valid replica gradients were not cleared")

        distributed_validate("grad reduce", validate_grad_reduce)
        dist.barrier()
    finally:
        if manager is not None and manager.runtime is not None:
            manager.destroy()


def main():
    args = parse_args()
    setup_dist()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"this lifecycle test requires 8 ranks, got {world_size}")
    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by 8")

    print_rank0(
        "UltraEP Manager worker lifecycle test: "
        f"worker_cycle={args.worker_cycle}, "
        f"grad_sms={os.getenv('ULTRA_EP_GRAD_REDUCE_NUM_SMS')}, "
        f"deterministic={os.getenv('ULTRA_EP_GRAD_REDUCE_DETERMINISTIC')}"
    )

    try:
        exercise_one_manager(args)
        dist.barrier()
        print_rank0("PASS: full Manager/rocSHMEM worker lifecycle completed.")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
