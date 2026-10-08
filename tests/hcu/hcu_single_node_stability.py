#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Long-running single-node UltraEP stability test for eight HCUs.

This test intentionally keeps one Manager alive while repeatedly changing the
placement.  It exercises asynchronous weight sync and grad reduce, validates
their results periodically, and watches device free-memory drift. Validation
failures are gathered before raising so every rank preserves the same
collective order during cleanup.
"""

import argparse
import gc
import os
import sys
import time

import torch
import torch.distributed as dist


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT in sys.path:
    sys.path.remove(REPO_ROOT)
if "" in sys.path and os.path.abspath(os.getcwd()) == REPO_ROOT:
    sys.path.remove("")

import ultra_ep

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import generate_routing_map_zipf


def print_rank0(message: str):
    if dist.get_rank() == 0:
        print(message, flush=True)


def setup_dist():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device("cuda", local_rank),
        )


def fill_master_tensors(tensors, base: int):
    for logical_offset, tensor in enumerate(tensors):
        logical_id = dist.get_rank() * len(tensors) + logical_offset
        tensor.fill_(float(base + logical_id))


def assert_tensor_scalar(tensor: torch.Tensor, expected: float, label: str):
    if not bool((tensor == expected).all().item()):
        raise AssertionError(
            f"{label} mismatch on rank {dist.get_rank()}: expected {expected}"
        )


def sample_free_memory() -> tuple[int, int]:
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def max_across_ranks(value: int) -> int:
    tensor = torch.tensor([value], dtype=torch.int64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return int(tensor.item())


def distributed_validate(label: str, validation_fn):
    """Run a local check, then make every rank fail at the same collective point."""
    local_error = None
    try:
        validation_fn()
    except Exception as exc:  # Keep all ranks in the same collective sequence.
        local_error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"

    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    failures = [error for error in errors if error is not None]
    if failures:
        raise AssertionError(f"{label} failed:\n" + "\n".join(failures))


def finish_local_phase_and_barrier(label: str = "", trace_all_ranks: bool = False):
    """Finish this rank's streams before any PE reuses symmetric buffers."""
    rank = dist.get_rank()
    if trace_all_ranks:
        print(f"[rank {rank}] {label}: local synchronize begin", flush=True)
    torch.cuda.synchronize()
    if trace_all_ranks:
        print(
            f"[rank {rank}] {label}: local synchronize complete; barrier begin",
            flush=True,
        )
    dist.barrier()
    if trace_all_ranks:
        print(f"[rank {rank}] {label}: barrier complete", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-node, eight-HCU UltraEP long stability test"
    )
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup-iterations", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--memory-tolerance-mib", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--num-redundant-experts-per-rank", type=int, default=2)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--tokens-per-rank", type=int, default=1024)
    parser.add_argument("--imbalance-ratios", type=float, nargs="+", default=[1.5, 2.5])
    parser.add_argument("--expert-fc1-numel", type=int, default=1048576)
    parser.add_argument("--expert-fc2-numel", type=int, default=524288)
    parser.add_argument("--overlap-numel", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--enable-autotune",
        action="store_true",
        help="run the real weight-sync and grad-reduce autotune integration path",
    )
    parser.add_argument("--autotune-start-iteration", type=int, default=3)
    parser.add_argument(
        "--trace-phases",
        action="store_true",
        help="Print rank-0 progress before and after every communication phase.",
    )
    args = parser.parse_args()

    if args.iterations <= 0 or args.warmup_iterations < 0:
        parser.error(
            "iteration counts must be non-negative and iterations must be positive"
        )
    if args.validate_every <= 0 or args.report_every <= 0:
        parser.error("validation/report intervals must be positive")
    if args.autotune_start_iteration < 1:
        parser.error("--autotune-start-iteration must be positive")
    if any(ratio < 1.0 for ratio in args.imbalance_ratios):
        parser.error("imbalance ratios must be >= 1")
    return args


def main():
    args = parse_args()
    setup_dist()

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError(f"this stability test requires 8 ranks, got {world_size}")
    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must be divisible by 8")

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
            autotune=ultra_ep.AutotuneConfig(
                enabled=args.enable_autotune,
                start_iteration=args.autotune_start_iteration,
            ),
        )

        if manager.nvl_domain_size != 8:
            raise RuntimeError(
                f"expected an 8-rank NVL domain, got {manager.nvl_domain_size}"
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

        replica_weights = manager.local_replica_weight_buffer
        replica_fc1_weights = manager.local_replica_fc1_weight_buffer
        replica_fc2_weights = manager.local_replica_fc2_weight_buffer
        replica_grads = manager.local_replica_grad_buffer
        replica_fc1_grads = manager.local_replica_fc1_grad_buffer
        replica_fc2_grads = manager.local_replica_fc2_grad_buffer
        overlap_tensor = torch.ones(args.overlap_numel, dtype=torch.float32, device="cuda")

        local_replica_phys = (
            rank * num_local_physical
            + num_local_master
            + torch.arange(num_local_redundant, dtype=torch.int64, device="cuda")
        )

        print_rank0(
            "UltraEP single-node stability test: "
            f"iterations={args.iterations}, warmup={args.warmup_iterations}, "
            f"ratios={args.imbalance_ratios}, grad_sms={manager.grad_reduce_num_sms}, "
            f"deterministic={manager.grad_reduce_deterministic}"
        )

        # Computing a target Zipf ratio performs a CPU-side binary search.
        # Repeating it in eight processes on every iteration overwhelms the
        # host and measures routing-data generation rather than UltraEP. Build
        # one immutable input per ratio and alternate them during the stress.
        print_rank0("Preparing reusable routing cases...")
        routing_cases = []
        for case_idx, ratio in enumerate(args.imbalance_ratios):
            case_routing = generate_routing_map_zipf(
                args.tokens_per_rank,
                args.num_experts,
                world_size,
                num_local_master,
                args.topk,
                ratio,
                args.seed + case_idx,
                rank=rank,
            )
            routing_cases.append((case_routing, case_routing.to(torch.float32)))

        def reset_grad_state():
            fill_master_tensors(fc1_grads, 65)
            fill_master_tensors(fc2_grads, 97)
            for replica_idx in range(num_local_redundant):
                physical_id = rank * num_local_physical + num_local_master + replica_idx
                replica_fc1_grads[replica_idx].fill_(float(physical_id + 129))
                replica_fc2_grads[replica_idx].fill_(float(physical_id + 193))

        def validate_weight_sync():
            replica_logical_ids = manager.physical_to_logical_map[
                0, local_replica_phys
            ].tolist()
            for replica_idx, logical_id in enumerate(replica_logical_ids):
                if logical_id < 0:
                    continue
                assert_tensor_scalar(
                    replica_fc1_weights[replica_idx],
                    float(logical_id + 1),
                    f"replica FC1 weight row {replica_idx}",
                )
                assert_tensor_scalar(
                    replica_fc2_weights[replica_idx],
                    float(logical_id + 33),
                    f"replica FC2 weight row {replica_idx}",
                )

        def validate_grad_reduce():
            l2p = manager.logical_to_physical_map[0]
            for local_master_idx in range(num_local_master):
                logical_id = rank * num_local_master + local_master_idx
                master_physical_id = rank * num_local_physical + local_master_idx
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
                    fc1_grads[local_master_idx],
                    expected_fc1,
                    f"master FC1 grad {local_master_idx}",
                )
                assert_tensor_scalar(
                    fc2_grads[local_master_idx],
                    expected_fc2,
                    f"master FC2 grad {local_master_idx}",
                )

            replica_logical_ids = manager.physical_to_logical_map[
                0, local_replica_phys
            ]
            valid_local_replicas = replica_logical_ids >= 0
            if bool(valid_local_replicas.any().item()):
                if bool((replica_grads[valid_local_replicas] != 0).any().item()):
                    raise AssertionError(
                        f"valid replica gradients were not cleared on rank {rank}"
                    )

        def run_iteration(iteration: int, validate: bool):
            def trace(phase: str):
                if args.trace_phases:
                    print_rank0(f"[iteration {iteration + 1:4d}] {phase}")

            routing_map, probs = routing_cases[iteration % len(routing_cases)]

            trace("placement begin")
            manager.update_placement(0, routing_map)
            expanded_probs, expanded_routing = manager.reroute(0, probs, routing_map)
            trace("placement/reroute complete")
            if validate:
                def validate_reroute():
                    if not bool(
                        (expanded_routing.sum(dim=1) == args.topk).all().item()
                    ):
                        raise AssertionError(
                            f"reroute top-k mismatch on rank {rank}"
                        )

                distributed_validate(
                    "reroute",
                    validate_reroute,
                )

            # A sender can write this rank's symmetric replica buffer before
            # this rank reaches the reset below.  Do not launch any remote
            # weight copies until every rank has completed its local reset;
            # otherwise a late zero_() can overwrite a valid peer write.
            replica_weights.zero_()
            finish_local_phase_and_barrier(
                "weight buffer reset", trace_all_ranks=args.trace_phases and validate
            )
            trace("weight sync launch")
            weight_event = manager.weight_sync(0, async_finish=True)
            overlap_tensor.add_(0.0001)
            if args.enable_autotune:
                manager.wait_weight_sync(weight_event, layer_id=0)
            else:
                weight_event.current_stream_wait()
            trace("weight sync event queued; waiting all ranks")
            # A local event only covers this rank's communication stream.
            # Some peer may still be writing our symmetric replica buffer, so
            # do not validate or reuse it until every rank has completed.
            finish_local_phase_and_barrier(
                "weight sync", trace_all_ranks=args.trace_phases and validate
            )
            trace("weight sync complete on all ranks")
            if validate:
                trace("weight validation begin")
                distributed_validate("weight sync", validate_weight_sync)
                trace("weight validation complete")

            reset_grad_state()
            trace("gradient reset queued; waiting all ranks")
            # Grad-reduce masters directly read peer replica buffers.  All PEs
            # must publish their freshly reset gradients before any PE starts
            # consuming them.
            finish_local_phase_and_barrier(
                "gradient reset", trace_all_ranks=args.trace_phases and validate
            )
            trace("gradient reset complete on all ranks")
            trace("grad reduce launch")
            grad_event = manager.grad_reduce(0, async_finish=True)
            overlap_tensor.mul_(0.9999)
            if args.enable_autotune:
                manager.wait_grad_reduce(grad_event, layer_id=0)
            else:
                grad_event.current_stream_wait()
            trace("grad reduce event queued; waiting all ranks")
            # Prevent a faster rank from resetting a symmetric replica buffer
            # for the next iteration while a slower peer still consumes it.
            finish_local_phase_and_barrier(
                "grad reduce", trace_all_ranks=args.trace_phases and validate
            )
            trace("grad reduce complete on all ranks")
            if validate:
                trace("grad validation begin")
                distributed_validate("grad reduce", validate_grad_reduce)
                trace("grad validation complete")

            del grad_event, weight_event
            del expanded_probs, expanded_routing

        for iteration in range(args.warmup_iterations):
            run_iteration(iteration, validate=(iteration + 1 == args.warmup_iterations))
            if (iteration + 1) % 5 == 0 or iteration + 1 == args.warmup_iterations:
                print_rank0(
                    f"[warmup {iteration + 1:4d}/{args.warmup_iterations}] complete"
                )

        dist.barrier()
        baseline_free, total_memory = sample_free_memory()
        start_time = time.monotonic()
        max_drift = 0

        for iteration in range(1, args.iterations + 1):
            validate = iteration == 1 or iteration % args.validate_every == 0
            complete_iteration_start = time.perf_counter()
            run_iteration(args.warmup_iterations + iteration, validate=validate)
            if args.enable_autotune:
                # Equivalent to the full post-optimizer iteration boundary.
                # It is entered by every rank after both asynchronous paths
                # have completed and their correctness has been checked.
                manager.autotune_iteration_end(
                    iteration,
                    iteration_time_ms=(
                        time.perf_counter() - complete_iteration_start
                    ) * 1000.0,
                )

            if iteration % args.report_every == 0 or iteration == args.iterations:
                current_free, _ = sample_free_memory()
                local_drift = max(0, baseline_free - current_free)
                global_max_drift = max_across_ranks(local_drift)
                max_drift = max(max_drift, global_max_drift)
                elapsed = time.monotonic() - start_time
                print_rank0(
                    f"[{iteration:6d}/{args.iterations}] "
                    f"{iteration / elapsed:.2f} iter/s, "
                    f"max free-memory drift={global_max_drift / 2**20:.1f} MiB"
                )

        tolerance_bytes = args.memory_tolerance_mib * 2**20
        if max_drift > tolerance_bytes:
            raise RuntimeError(
                f"device free-memory drift {max_drift / 2**20:.1f} MiB exceeds "
                f"the {args.memory_tolerance_mib} MiB tolerance"
            )

        dist.barrier()
        print_rank0(
            "PASS: asynchronous placement/weight-sync/grad-reduce remained correct; "
            f"maximum free-memory drift was {max_drift / 2**20:.1f} MiB "
            f"of {total_memory / 2**30:.1f} GiB."
        )
        if args.enable_autotune:
            config = manager._autotuner.current_weight_sync
            print_rank0(
                "Autotune integration result: "
                f"copy={config.copy_mode}, tpb={config.threads_per_block}, "
                f"cta={config.cta_multiplier}, lds_waves={config.lds_waves_per_destination}; "
                f"weight_sync_done={manager._autotuner.weight_sync_done}, "
                f"grad_reduce_done={manager._autotuner.grad_done}, "
                f"grad_reduce_sms={manager.grad_reduce_num_sms}."
            )
    finally:
        if manager is not None:
            manager.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
