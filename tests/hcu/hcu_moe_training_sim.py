#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Framework-independent HCU MoE proxy benchmark.

It compares fixed master-expert placement with UltraEP dynamic placement.  The
communication path is real (rocSHMEM global Placement plus domain-local Weight
Sync); expert execution is a
configurable BF16 MLP proxy whose work is proportional to each rank's routed
token count.  This intentionally does not model framework dispatch/combine,
so treat its speedup as a pre-integration decision signal rather than an
end-to-end model throughput claim.

When ``--enable-autotune`` is set, the proxy also executes real UltraEP
grad-reduce during calibration. Its overlap is a configurable GPU tensor
operation, so it validates the tuner state machine and SMS cap but is not a
substitute for Megatron's real attention/router backward overlap.
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

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
    dist.init_process_group(
        backend="nccl", device_id=torch.device("cuda", local_rank)
    )


def finish_all_ranks():
    """Finish local GPU work, then prevent symmetric-buffer reuse races."""
    torch.cuda.synchronize()
    dist.barrier()


def max_across_ranks(value: float) -> float:
    result = torch.tensor([value], dtype=torch.float64, device="cuda")
    dist.all_reduce(result, op=dist.ReduceOp.MAX)
    return float(result.item())


def global_physical_loads(routing: torch.Tensor) -> torch.Tensor:
    """Aggregate a local [tokens, physical-experts] routing map over all PEs."""
    loads = routing.sum(dim=0, dtype=torch.int64)
    dist.all_reduce(loads, op=dist.ReduceOp.SUM)
    return loads


def make_static_physical_loads(
    routing: torch.Tensor, num_local_master: int, num_local_physical: int
) -> torch.Tensor:
    """Map each logical expert to its fixed master physical expert."""
    logical_loads = routing.sum(dim=0, dtype=torch.int64)
    dist.all_reduce(logical_loads, op=dist.ReduceOp.SUM)
    num_experts = logical_loads.numel()
    physical_loads = torch.zeros(
        (dist.get_world_size() * num_local_physical,),
        dtype=torch.int64,
        device="cuda",
    )
    logical_ids = torch.arange(num_experts, device="cuda", dtype=torch.int64)
    master_ranks = logical_ids // num_local_master
    local_master_ids = logical_ids % num_local_master
    master_physical_ids = master_ranks * num_local_physical + local_master_ids
    physical_loads.scatter_add_(0, master_physical_ids, logical_loads)
    return physical_loads


def fill_master_weights(tensors, base: int):
    for local_idx, tensor in enumerate(tensors):
        logical_id = dist.get_rank() * len(tensors) + local_idx
        tensor.fill_(float(base + logical_id))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed HCU UltraEP MoE training proxy benchmark"
    )
    parser.add_argument(
        "--preset",
        choices=("small", "moe-50b", "moe-100b"),
        default="small",
        help=(
            "small is the quick smoke configuration; moe-50b and moe-100b "
            "are 50B- and 100B-class MoE-layer proxies"
        ),
    )
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument(
        "--enable-autotune",
        action="store_true",
        help="exercise the bounded UltraEP autotuner in this real 8-rank proxy run",
    )
    parser.add_argument("--autotune-start-iteration", type=int, default=3)
    parser.add_argument(
        "--grad-reduce-max-sms",
        type=int,
        default=None,
        help=(
            "optional autotune cap for grad-reduce SMS; by default the tuner "
            "uses at most 75%% of device SMS and keeps two SMS reserved"
        ),
    )
    parser.add_argument(
        "--grad-reduce-min-hidden-wait-ms",
        type=float,
        default=0.5,
        help="absolute practical-hidden floor for grad-reduce exposed wait (default: 0.5 ms)",
    )
    parser.add_argument(
        "--grad-reduce-hidden-wait-ratio",
        type=float,
        default=0.01,
        help="practical-hidden target as a fraction of measured overlap (default: 0.01)",
    )
    parser.add_argument(
        "--grad-reduce-overlap-numel",
        type=int,
        default=4 * 1024 * 1024,
        help=(
            "float32 elements in the synthetic grad-reduce overlap operation; "
            "used during --enable-autotune calibration"
        ),
    )
    parser.add_argument("--tokens-per-rank", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument(
        "--routing-pattern",
        choices=("zipf", "hotspot"),
        default="zipf",
        help=(
            "zipf uses the requested rank imbalance; hotspot routes every "
            "token to one logical expert and is used to stress high fan-out."
        ),
    )
    parser.add_argument(
        "--hotspot-expert",
        type=int,
        default=0,
        help="logical expert used as the primary route by --routing-pattern hotspot",
    )
    parser.add_argument(
        "--require-min-fanout",
        type=int,
        default=0,
        help="fail unless the generated placement reaches this Weight Sync fan-out",
    )
    parser.add_argument("--imbalance-ratios", type=float, nargs="+", default=[1.5, 2.5])
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--num-redundant-experts-per-rank", type=int, default=2)
    parser.add_argument("--expert-fc1-numel", type=int, default=1048576)
    parser.add_argument("--expert-fc2-numel", type=int, default=524288)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--ffn-size", type=int, default=512)
    parser.add_argument("--compute-tokens-per-gemm", type=int, default=128)
    parser.add_argument(
        "--activation-bytes-per-token",
        type=int,
        default=0,
        help=(
            "Bytes per routed token for one dispatch or combine direction. "
            "Zero selects BF16 hidden states (2 * --hidden-size). This is "
            "reported as a cross-node traffic envelope; it is not timed."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--profile-breakdown",
        action="store_true",
        help="Report synchronized placement/reroute, Weight Sync, and expert-compute timings.",
    )
    parser.add_argument(
        "--core-profile-dir",
        type=str,
        default="",
        help=(
            "If set, capture one representative placement/reroute/Weight Sync "
            "Chrome trace per rank in this directory."
        ),
    )
    args = parser.parse_args()

    provided_options = {
        item.split("=", 1)[0] for item in sys.argv[1:] if item.startswith("--")
    }
    large_preset_values = {
        # 128 experts x 2 x (3072 x 8192) x 8 MoE layers is about
        # 51.5B expert parameters.
        "moe-50b": {
            "--warmup-iters": ("warmup_iters", 3),
            "--bench-iters": ("bench_iters", 10),
            "--tokens-per-rank": ("tokens_per_rank", 2048),
            "--topk": ("topk", 2),
            "--num-experts": ("num_experts", 128),
            "--num-redundant-experts-per-rank": (
                "num_redundant_experts_per_rank",
                4,
            ),
            "--expert-fc1-numel": ("expert_fc1_numel", 3072 * 8192),
            "--expert-fc2-numel": ("expert_fc2_numel", 3072 * 8192),
            "--hidden-size": ("hidden_size", 3072),
            "--ffn-size": ("ffn_size", 8192),
            "--compute-tokens-per-gemm": ("compute_tokens_per_gemm", 128),
        },
        # 128 experts x 2 x (4096 x 14336) x 8 MoE layers is about
        # 120.3B expert parameters.
        "moe-100b": {
            "--warmup-iters": ("warmup_iters", 3),
            "--bench-iters": ("bench_iters", 10),
            "--tokens-per-rank": ("tokens_per_rank", 2048),
            "--topk": ("topk", 2),
            "--num-experts": ("num_experts", 128),
            "--num-redundant-experts-per-rank": (
                "num_redundant_experts_per_rank",
                4,
            ),
            "--expert-fc1-numel": ("expert_fc1_numel", 4096 * 14336),
            "--expert-fc2-numel": ("expert_fc2_numel", 4096 * 14336),
            "--hidden-size": ("hidden_size", 4096),
            "--ffn-size": ("ffn_size", 14336),
            "--compute-tokens-per-gemm": ("compute_tokens_per_gemm", 128),
        },
    }
    if args.preset in large_preset_values:
        # The benchmark executes one layer, matching UltraEP's per-layer
        # placement/weight-sync lifecycle.  The preset label estimates the
        # aggregate expert parameters across eight such MoE layers.
        preset_values = large_preset_values[args.preset]
        for option, (attribute, value) in preset_values.items():
            if option not in provided_options:
                setattr(args, attribute, value)

    if args.warmup_iters < 0 or args.bench_iters <= 0:
        parser.error("warmup must be non-negative and bench iterations positive")
    if args.autotune_start_iteration < 1:
        parser.error("--autotune-start-iteration must be positive")
    if args.grad_reduce_max_sms is not None and (
        args.grad_reduce_max_sms <= 0 or args.grad_reduce_max_sms % 2
    ):
        parser.error("--grad-reduce-max-sms must be a positive even integer")
    if args.grad_reduce_overlap_numel <= 0:
        parser.error("--grad-reduce-overlap-numel must be positive")
    if args.grad_reduce_min_hidden_wait_ms < 0.0:
        parser.error("--grad-reduce-min-hidden-wait-ms must be non-negative")
    if not 0.0 <= args.grad_reduce_hidden_wait_ratio < 1.0:
        parser.error("--grad-reduce-hidden-wait-ratio must be in [0, 1)")
    if args.num_experts <= 0:
        parser.error("--num-experts must be positive")
    if args.topk <= 0 or args.tokens_per_rank <= 0:
        parser.error("--topk and --tokens-per-rank must be positive")
    if args.topk > args.num_experts:
        parser.error("--topk must not exceed --num-experts")
    if not 0 <= args.hotspot_expert < args.num_experts:
        parser.error("--hotspot-expert must be in [0, --num-experts)")
    if args.require_min_fanout < 0:
        parser.error("--require-min-fanout must be non-negative")
    if args.hidden_size <= 0 or args.ffn_size <= 0 or args.compute_tokens_per_gemm <= 0:
        parser.error("MLP proxy dimensions must be positive")
    if args.activation_bytes_per_token < 0:
        parser.error("--activation-bytes-per-token must be non-negative")
    if any(ratio < 1.0 for ratio in args.imbalance_ratios):
        parser.error("all imbalance ratios must be >= 1")
    return args


class ExpertComputeProxy:
    """Reusable BF16 MLP work proportional to assigned tokens per expert."""

    def __init__(self, num_local_physical: int, hidden_size: int, ffn_size: int, chunk: int):
        self.chunk = chunk
        self.num_local_physical = num_local_physical
        options = {"dtype": torch.bfloat16, "device": "cuda"}
        self.inputs = torch.randn((chunk, hidden_size), **options)
        self.fc1 = [torch.randn((hidden_size, ffn_size), **options) for _ in range(num_local_physical)]
        self.fc2 = [torch.randn((ffn_size, hidden_size), **options) for _ in range(num_local_physical)]
        self.hidden = [torch.empty((chunk, ffn_size), **options) for _ in range(num_local_physical)]
        self.outputs = [torch.empty((chunk, hidden_size), **options) for _ in range(num_local_physical)]

    def run(self, local_loads: list[int]):
        for expert_idx, token_count in enumerate(local_loads):
            repetitions = math.ceil(token_count / self.chunk)
            for _ in range(repetitions):
                torch.mm(self.inputs, self.fc1[expert_idx], out=self.hidden[expert_idx])
                torch.mm(self.hidden[expert_idx], self.fc2[expert_idx], out=self.outputs[expert_idx])


def core_profile_kernel_summary(profiler) -> dict[str, float]:
    """Return CUDA/HIP time totals (ms) for the kernels relevant to tuning."""

    groups = {
        "placement local count": ("rmap_local_sum_kernel",),
        "placement fcollect": ("int32_fcollect_kernel",),
        "placement load reduce": ("reduce_per_rank_loads_kernel",),
        "placement solve": ("quota_placement_solve_kernel",),
        "reroute count": ("reroute_forward_count_kernel",),
        "reroute scatter": (
            "dense_quota_reroute_scatter_kernel",
            "dense_rr_reroute_scatter_kernel",
        ),
        "weight task build": ("build_weight_sync_task_lists_kernel",),
        "weight IPC copy": (
            "weight_sync_thread_copy_kernel",
            "weight_sync_lds_copy_kernel",
        ),
    }
    summary = {label: 0.0 for label in groups}
    for event in profiler.key_averages():
        key = str(event.key)
        device_us = float(
            getattr(event, "cuda_time_total", getattr(event, "device_time_total", 0.0))
        )
        for label, kernels in groups.items():
            if any(kernel in key for kernel in kernels):
                summary[label] += device_us / 1000.0
    return summary


def generate_hotspot_routing_map(
    num_tokens: int,
    num_experts: int,
    topk: int,
    hotspot_expert: int,
    rank: int,
) -> torch.Tensor:
    """Create a deterministic one-hot-expert stress case with valid top-k routes.

    Every token uses ``hotspot_expert`` as its first route.  Remaining top-k
    entries are spread over the other experts, keeping the test representative
    of a valid MoE routing map while forcing the hotspot to its maximum
    domain-local replica fan-out.
    """

    routing = torch.zeros((num_tokens, num_experts), dtype=torch.bool, device="cuda")
    if num_tokens == 0:
        return routing
    token_ids = torch.arange(num_tokens, device="cuda")
    routing[:, hotspot_expert] = True
    if topk == 1:
        return routing

    # Index a ring of all non-hotspot experts. Consecutive offsets are unique
    # for one token because topk <= num_experts.
    offsets = torch.arange(1, topk, device="cuda").unsqueeze(0)
    non_hotspot_ids = (
        token_ids.unsqueeze(1) * (topk - 1) + offsets + rank * (topk - 1)
    ) % (num_experts - 1)
    expert_ids = non_hotspot_ids + (non_hotspot_ids >= hotspot_expert).to(torch.int64)
    routing[token_ids.unsqueeze(1), expert_ids] = True
    return routing


def local_load_list(global_loads: torch.Tensor, num_local_physical: int) -> list[int]:
    start = dist.get_rank() * num_local_physical
    return [int(value) for value in global_loads[start : start + num_local_physical].cpu().tolist()]


def describe_loads(label: str, global_loads: torch.Tensor):
    nonzero = global_loads[global_loads > 0].to(torch.float64)
    maximum = int(global_loads.max().item())
    mean = float(nonzero.mean().item()) if nonzero.numel() else 0.0
    print_rank0(f"  {label:<12} max/mean={maximum}/{mean:.1f}, imbalance={maximum / mean:.3f}")


def describe_rank_loads(label: str, global_loads: torch.Tensor, num_local_physical: int):
    rank_loads = global_loads.view(dist.get_world_size(), num_local_physical).sum(dim=1)
    maximum = int(rank_loads.max().item())
    mean = float(rank_loads.to(torch.float64).mean().item())
    print_rank0(
        f"  {label:<12} rank max/mean={maximum}/{mean:.1f}, imbalance={maximum / mean:.3f}"
    )


def cross_domain_routing_stats(
    expanded_routing: torch.Tensor,
    domain_size: int,
    num_local_physical: int,
    activation_bytes_per_token: int,
) -> dict:
    """Describe traffic a real dispatcher would send outside the source domain.

    The proxy intentionally does not call a framework dispatcher.  The routed
    physical-expert bitmap is nevertheless enough to count assignments whose
    target physical rank belongs to another scale-up domain.  Dispatch and
    combine each move this payload once, so their combined lower-bound payload
    is twice the reported one-way value.
    """

    rank = dist.get_rank()
    target_ranks = torch.arange(
        expanded_routing.size(1), device="cuda", dtype=torch.int64
    ) // num_local_physical
    remote_targets = (target_ranks // domain_size) != (rank // domain_size)
    local_assignments = expanded_routing.sum(dtype=torch.int64)
    local_remote = expanded_routing[:, remote_targets].sum(dtype=torch.int64)
    counts = torch.stack((local_assignments, local_remote))
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    total = int(counts[0].item())
    remote = int(counts[1].item())
    return {
        "total_assignments": total,
        "cross_domain_assignments": remote,
        "cross_domain_fraction": remote / total if total else 0.0,
        "one_way_bytes": remote * activation_bytes_per_token,
    }


def describe_cross_domain_routing(stats: dict, activation_bytes_per_token: int):
    gib = 1024.0**3
    print_rank0(
        "  Dispatcher traffic envelope (not timed by this proxy): "
        f"cross-domain assignments={stats['cross_domain_assignments']}/"
        f"{stats['total_assignments']} ({stats['cross_domain_fraction'] * 100.0:.1f}%); "
        f"one-way dispatch={stats['one_way_bytes'] / gib:.3f} GiB "
        f"({activation_bytes_per_token} B/token), "
        f"dispatch+combine lower bound={2 * stats['one_way_bytes'] / gib:.3f} GiB"
    )


def weight_bytes_per_expert(args) -> int:
    """Bytes copied for one BF16 master expert, excluding optional scales."""
    return (args.expert_fc1_numel + args.expert_fc2_numel) * torch.tensor(
        [], dtype=torch.bfloat16
    ).element_size()


def _relay_count(num_replicas: int, manager) -> int:
    if num_replicas <= 1:
        return 0
    count = math.isqrt(num_replicas)
    count = max(1, min(count, manager.weight_sync_relay_max_relays))
    return min(count, num_replicas - 1)


def collect_weight_sync_traffic(manager, args, num_local_physical: int) -> dict:
    """Model the transfer plan built by weight_sync.cu for this placement.

    Placement storage includes an outer real-layer/microbatch-slot dimension.
    This benchmark uses layer 0, so select that slot before reading the small
    diagnostic maps on the host.  The resulting bytes count payload writes
    only: protocol flags and task metadata are excluded.
    """
    world_size = dist.get_world_size()
    l2p_tensor = manager.logical_to_physical_map
    p2l_tensor = manager.physical_to_logical_map
    replica_counts_tensor = manager.logical_replica_counts
    if l2p_tensor.dim() == 3:
        l2p_tensor = l2p_tensor[0]
    if p2l_tensor.dim() == 2:
        p2l_tensor = p2l_tensor[0]
    if replica_counts_tensor.dim() == 2:
        replica_counts_tensor = replica_counts_tensor[0]
    if l2p_tensor.shape != (args.num_experts, world_size):
        raise RuntimeError(
            "unexpected layer-0 logical_to_physical_map shape: "
            f"{tuple(l2p_tensor.shape)}, expected ({args.num_experts}, {world_size})"
        )
    l2p = l2p_tensor.detach().cpu().tolist()
    p2l = p2l_tensor.detach().cpu().tolist()
    replica_counts = replica_counts_tensor.detach().cpu().tolist()
    expert_bytes = weight_bytes_per_expert(args)
    sent = [0] * world_size
    received = [0] * world_size
    unique_source = [0] * world_size
    max_fanout = [0] * world_size
    transfer_count = 0

    # This mirrors build_weight_sync_task_lists_kernel's ordering and relay
    # selection.  It lets the benchmark report the planned traffic before
    # changing the HIP copy kernel itself.
    use_relay = manager.weight_sync_plan_mode != "direct"
    sender_load = [0] * world_size
    master_order = []
    for master_rank in range(world_size):
        for local_master_idx in range(args.num_experts // world_size):
            physical_id = master_rank * num_local_physical + local_master_idx
            logical_id = int(p2l[physical_id])
            if logical_id >= 0:
                master_order.append((master_rank, logical_id))

    for master_rank, logical_id in master_order:
        replica_phys = [
            int(value)
            for value in l2p[logical_id][1 : int(replica_counts[logical_id])]
        ]
        num_replicas = len(replica_phys)
        if num_replicas == 0:
            continue
        unique_source[master_rank] += expert_bytes

        relay_count = _relay_count(num_replicas, manager)
        relay_allowed = use_relay and relay_count > 0
        if relay_allowed and manager.weight_sync_plan_mode != "forcerelay":
            relay_allowed = num_replicas >= manager.weight_sync_relay_min_replicas
        if relay_allowed and manager.weight_sync_plan_mode != "forcerelay":
            critical_fanout = max(
                relay_count,
                math.ceil((num_replicas - relay_count) / relay_count),
            )
            relay_allowed = (
                num_replicas - critical_fanout
                >= manager.weight_sync_relay_min_fanout_gain
            )

        if not relay_allowed:
            sent[master_rank] += num_replicas * expert_bytes
            sender_load[master_rank] += num_replicas * expert_bytes
            max_fanout[master_rank] = max(max_fanout[master_rank], num_replicas)
            transfer_count += num_replicas
            for physical_id in replica_phys:
                received[physical_id // num_local_physical] += expert_bytes
            continue

        # Select relays using the same least-loaded / distinct-rank preference
        # as the device task builder.
        selected = []
        for _ in range(relay_count):
            candidates = [
                (idx, physical_id)
                for idx, physical_id in enumerate(replica_phys)
                if idx not in {entry[0] for entry in selected}
            ]
            relay_idx, relay_physical = min(
                candidates,
                key=lambda entry: (
                    int(any(
                        existing_physical // num_local_physical
                        == entry[1] // num_local_physical
                        for _, existing_physical in selected
                    )),
                    sender_load[entry[1] // num_local_physical],
                    entry[1] // num_local_physical,
                    entry[0],
                ),
            )
            selected.append((relay_idx, relay_physical))

        children = [[] for _ in selected]
        selected_indices = {idx for idx, _ in selected}
        leaves = [
            (idx, physical_id)
            for idx, physical_id in enumerate(replica_phys)
            if idx not in selected_indices
        ]
        projected = [sender_load[physical_id // num_local_physical] for _, physical_id in selected]
        for leaf_order, leaf in enumerate(leaves):
            if leaf_order < len(selected):
                owner = leaf_order
            else:
                owner = min(
                    range(len(selected)),
                    key=lambda idx: (
                        projected[idx],
                        len(children[idx]),
                        selected[idx][1] // num_local_physical,
                        selected[idx][0],
                    ),
                )
            children[owner].append(leaf)
            projected[owner] += expert_bytes

        sent[master_rank] += len(selected) * expert_bytes
        sender_load[master_rank] += len(selected) * expert_bytes
        max_fanout[master_rank] = max(max_fanout[master_rank], len(selected))
        transfer_count += len(selected)
        for relay_idx, (_, relay_physical) in enumerate(selected):
            relay_rank = relay_physical // num_local_physical
            received[relay_rank] += expert_bytes
            child_count = len(children[relay_idx])
            sent[relay_rank] += child_count * expert_bytes
            sender_load[relay_rank] += child_count * expert_bytes
            max_fanout[relay_rank] = max(max_fanout[relay_rank], child_count)
            transfer_count += child_count
            for _, child_physical in children[relay_idx]:
                received[child_physical // num_local_physical] += expert_bytes

    return {
        "expert_bytes": expert_bytes,
        "sent": sent,
        "received": received,
        "unique_source": unique_source,
        "max_send": max(sent),
        "max_receive": max(received),
        "max_unique_source": max(unique_source),
        "max_fanout": max(max_fanout),
        "transfer_count": transfer_count,
    }


def describe_weight_sync_traffic(stats: dict, plan_mode: str):
    mib = 1024.0 * 1024.0
    reuse = stats["max_send"] / max(stats["max_unique_source"], 1)
    print_rank0(f"  Weight Sync plan={plan_mode} (payload only):")
    print_rank0(
        "    max rank send/receive: "
        f"{stats['max_send'] / mib:.1f}/{stats['max_receive'] / mib:.1f} MiB; "
        f"max fan-out={stats['max_fanout']}; transfers={stats['transfer_count']}"
    )
    print_rank0(
        "    logical source reads issued by current HIP fallback: "
        f"{stats['max_send'] / mib:.1f} MiB; LDS fan-out lower bound: "
        f"{stats['max_unique_source'] / mib:.1f} MiB ({reuse:.2f}x reuse opportunity)"
    )


def main():
    args = parse_args()
    setup_dist()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.num_experts % world_size != 0:
        raise RuntimeError(
            f"--num-experts={args.num_experts} must be divisible by world size {world_size}"
        )

    num_local_master = args.num_experts // world_size
    num_local_physical = num_local_master + args.num_redundant_experts_per_rank
    manager = None
    try:
        manager = ultra_ep.Manager(
            group=dist.group.WORLD,
            num_layers=1,
            num_local_master_experts=num_local_master,
            num_local_redundant_experts=args.num_redundant_experts_per_rank,
            expert_fc1_numel=args.expert_fc1_numel,
            expert_fc2_numel=args.expert_fc2_numel,
            is_train=True,
            explicitly_destroy=True,
            weight_data_dtype=torch.bfloat16,
            grad_dtype=torch.float32,
            autotune=ultra_ep.AutotuneConfig(
                enabled=args.enable_autotune,
                start_iteration=args.autotune_start_iteration,
                tune_grad_reduce=args.enable_autotune,
                grad_reduce_max_sms=args.grad_reduce_max_sms,
                grad_reduce_min_hidden_wait_ms=args.grad_reduce_min_hidden_wait_ms,
                grad_reduce_hidden_wait_ratio=args.grad_reduce_hidden_wait_ratio,
            ),
        )
        if world_size % manager.nvl_domain_size != 0:
            raise RuntimeError(
                f"world size {world_size} must be divisible by scale-up domain "
                f"size {manager.nvl_domain_size}"
            )
        activation_bytes_per_token = (
            args.activation_bytes_per_token or 2 * args.hidden_size
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
        fill_master_weights(fc1_weights, 1)
        fill_master_weights(fc2_weights, 33)
        manager.construct_local_master_ptr_pool(0, fc1_weights, fc2_weights, fc1_grads, fc2_grads)

        estimated_expert_params = ""
        if args.preset in ("moe-50b", "moe-100b"):
            params_b = (
                8
                * args.num_experts
                * (args.expert_fc1_numel + args.expert_fc2_numel)
                / 1.0e9
            )
            estimated_expert_params = (
                f", estimated_expert_params={params_b:.1f}B/8-MoE-layers"
            )

        print_rank0(
            "UltraEP MoE training proxy: "
            f"preset={args.preset}, experts={args.num_experts}, "
            f"tokens/rank={args.tokens_per_rank}, topk={args.topk}, "
            f"routing={args.routing_pattern}, "
            f"ratios={args.imbalance_ratios}, "
            f"world/domain={world_size}/{manager.nvl_domain_size}, "
            f"MLP={args.hidden_size}x{args.ffn_size}x{args.hidden_size}"
            f"{estimated_expert_params}, "
            f"warmup/bench={args.warmup_iters}/{args.bench_iters}, "
            f"autotune={'on' if args.enable_autotune else 'off'}, "
            f"weight_sync_cta_multiplier={os.getenv('ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER', '2')}, "
            f"hip_copy_mode={os.getenv('ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE', 'thread')}, "
            f"weight_sync_threads_per_block={os.getenv('ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK', '128')}, "
            f"lds_waves_per_destination={os.getenv('ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST', '1')}, "
            f"grad_reduce_autotune={'synthetic calibration' if args.enable_autotune else 'off'}, "
            "grad_reduce_hidden_target="
            f"max({args.grad_reduce_min_hidden_wait_ms:.3f} ms, "
            f"{args.grad_reduce_hidden_wait_ratio * 100.0:.1f}% overlap)"
        )

        routing_cases = []
        for case_idx, ratio in enumerate(args.imbalance_ratios):
            if args.routing_pattern == "hotspot":
                routing = generate_hotspot_routing_map(
                    args.tokens_per_rank,
                    args.num_experts,
                    args.topk,
                    args.hotspot_expert,
                    rank,
                )
                case_label = (
                    f"hotspot expert={args.hotspot_expert} "
                    f"(every token primary-routed to it)"
                )
            else:
                routing = generate_routing_map_zipf(
                    args.tokens_per_rank,
                    args.num_experts,
                    world_size,
                    num_local_master,
                    args.topk,
                    ratio,
                    args.seed + case_idx,
                    rank=rank,
                )
                case_label = f"ratio={ratio:.2f}"
            static_loads = make_static_physical_loads(
                routing, num_local_master, num_local_physical
            )

            manager.update_placement(0, routing)
            finish_all_ranks()
            _, expanded_routing = manager.reroute(0, routing.to(torch.float32), routing)
            finish_all_ranks()
            dynamic_loads = global_physical_loads(expanded_routing)
            cross_domain_stats = cross_domain_routing_stats(
                expanded_routing,
                manager.nvl_domain_size,
                num_local_physical,
                activation_bytes_per_token,
            )
            weight_sync_stats = collect_weight_sync_traffic(
                manager, args, num_local_physical
            )
            if (
                args.require_min_fanout
                and weight_sync_stats["max_fanout"] < args.require_min_fanout
            ):
                raise RuntimeError(
                    f"placement fan-out {weight_sync_stats['max_fanout']} is below "
                    f"the required {args.require_min_fanout}; this run is not a valid "
                    "high-fan-out relay comparison"
                )
            routing_cases.append(
                (routing, static_loads, dynamic_loads, local_load_list(static_loads, num_local_physical),
                 local_load_list(dynamic_loads, num_local_physical), weight_sync_stats,
                 cross_domain_stats, case_label)
            )
            if rank == 0:
                print(f"Load case {case_label}:", flush=True)
            describe_loads("static", static_loads)
            describe_loads("UltraEP", dynamic_loads)
            describe_rank_loads("static", static_loads, num_local_physical)
            describe_rank_loads("UltraEP", dynamic_loads, num_local_physical)
            describe_weight_sync_traffic(weight_sync_stats, manager.weight_sync_plan_mode)
            describe_cross_domain_routing(cross_domain_stats, activation_bytes_per_token)
            del expanded_routing

        proxy = ExpertComputeProxy(
            num_local_physical, args.hidden_size, args.ffn_size, args.compute_tokens_per_gemm
        )
        # This is deliberately separate from the forward compute proxy.  It
        # supplies real GPU-stream contention while grad-reduce is in flight,
        # but does not pretend to reproduce Megatron attention/router backward.
        grad_reduce_overlap = torch.ones(
            args.grad_reduce_overlap_numel, dtype=torch.float32, device="cuda"
        )

        def run_static(case):
            proxy.run(case[3])

        def run_ultra(case, profile_breakdown: bool = False, autotune_iteration=None):
            complete_iteration_start = (
                time.perf_counter() if autotune_iteration is not None else None
            )
            routing = case[0]
            placement_start = time.perf_counter()
            manager.update_placement(0, routing)
            _, expanded_routing = manager.reroute(0, routing.to(torch.float32), routing)
            if profile_breakdown:
                torch.cuda.synchronize()
                placement_ms = (time.perf_counter() - placement_start) * 1000.0

            weight_sync_start = time.perf_counter()
            event = manager.weight_sync(0, async_finish=True)
            if args.enable_autotune:
                manager.wait_weight_sync(event, layer_id=0)
            else:
                event.current_stream_wait()
            if profile_breakdown:
                # This completes only this rank's source-side HIP work.  Keep
                # the following process barrier separate: with direct peer
                # stores it is the conservative receiver-completion check.
                torch.cuda.synchronize()
                weight_sync_local_ms = (time.perf_counter() - weight_sync_start) * 1000.0
                barrier_start = time.perf_counter()
                dist.barrier()
                phase_barrier_ms = (time.perf_counter() - barrier_start) * 1000.0
                weight_sync_phase_ms = (time.perf_counter() - weight_sync_start) * 1000.0
            else:
                finish_all_ranks()

            compute_start = time.perf_counter()
            proxy.run(case[4])
            if profile_breakdown:
                torch.cuda.synchronize()
                compute_ms = (time.perf_counter() - compute_start) * 1000.0
            # The normal simulator is a forward proxy and must retain its
            # existing benchmark meaning.  This optional branch exists only
            # to exercise the real grad-reduce tuner during its calibration
            # window, after the weight-sync choice is stable.
            if (
                args.enable_autotune
                and autotune_iteration is not None
                and manager._autotuner.weight_sync_done
                and not manager._autotuner.grad_done
            ):
                grad_event = manager.grad_reduce(0, async_finish=True)
                grad_reduce_overlap.mul_(0.9999)
                manager.wait_grad_reduce(grad_event, layer_id=0)
                del grad_event
            del event, expanded_routing
            if autotune_iteration is not None:
                # Mirrors the post-optimizer/pipeline-flush point in Megatron.
                # All ranks enter together because tuning decisions use max-rank
                # timing reductions.
                finish_all_ranks()
                complete_iteration_ms = (
                    time.perf_counter() - complete_iteration_start
                ) * 1000.0
                manager.autotune_iteration_end(
                    autotune_iteration,
                    iteration_time_ms=complete_iteration_ms,
                )
            if profile_breakdown:
                return (
                    placement_ms,
                    weight_sync_local_ms,
                    phase_barrier_ms,
                    weight_sync_phase_ms,
                    compute_ms,
                )
            return None

        if args.core_profile_dir:
            # Keep this capture outside the timed benchmark.  It is a single,
            # representative iteration of the real control path; no expert
            # MLP is included, so the trace stays focused on UltraEP itself.
            profile_dir = Path(args.core_profile_dir)
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile_case = routing_cases[0]
            finish_all_ranks()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
            ) as core_profiler:
                with torch.profiler.record_function("UltraEP::placement"):
                    manager.update_placement(0, profile_case[0])
                torch.cuda.synchronize()
                with torch.profiler.record_function("UltraEP::reroute"):
                    _, profiled_routing = manager.reroute(
                        0, profile_case[0].to(torch.float32), profile_case[0]
                    )
                torch.cuda.synchronize()
                with torch.profiler.record_function("UltraEP::weight_sync"):
                    profile_event = manager.weight_sync(0, async_finish=True)
                    profile_event.current_stream_wait()
                    torch.cuda.synchronize()
                with torch.profiler.record_function("UltraEP::receiver_barrier"):
                    dist.barrier()
            trace_path = profile_dir / f"core_path_rank{rank}.json"
            core_profiler.export_chrome_trace(str(trace_path))
            kernel_summary = core_profile_kernel_summary(core_profiler)
            labels = list(kernel_summary)
            local_kernel_ms = torch.tensor(
                [kernel_summary[label] for label in labels],
                dtype=torch.float64,
                device="cuda",
            )
            per_rank_kernel_ms = [torch.empty_like(local_kernel_ms) for _ in range(world_size)]
            dist.all_gather(per_rank_kernel_ms, local_kernel_ms)
            if rank == 0:
                profile_matrix = torch.stack(per_rank_kernel_ms).cpu()
                for index, label in enumerate(labels):
                    values = profile_matrix[:, index]
                    print(
                        f"  core profile {label:<18}: "
                        f"min/mean/max={values.min().item():.3f}/"
                        f"{values.mean().item():.3f}/{values.max().item():.3f} ms",
                        flush=True,
                    )
                ipc_index = labels.index("weight IPC copy")
                ipc_per_rank = ", ".join(
                    f"r{idx}={value:.3f}" for idx, value in enumerate(profile_matrix[:, ipc_index].tolist())
                )
                print(f"  core profile IPC copy by rank: {ipc_per_rank} ms", flush=True)
                print(
                    f"Core-path traces written to {profile_dir} "
                    f"(one JSON trace per rank).",
                    flush=True,
                )
            del profile_event, profiled_routing
            finish_all_ranks()

        for iteration in range(args.warmup_iters):
            # Once tuning starts, use one fixed representative workload so an
            # A/B candidate is not accidentally compared on a different
            # imbalance ratio.  The highest requested ratio is the stress case.
            if args.enable_autotune and iteration + 1 >= args.autotune_start_iteration:
                case = routing_cases[-1]
            else:
                case = routing_cases[iteration % len(routing_cases)]
            run_static(case)
            finish_all_ranks()
            run_ultra(case, autotune_iteration=iteration + 1)
            finish_all_ranks()

        # Finish autotune calibration outside the reported benchmark. Weight
        # sync candidates use full proxy iterations. The following grad-reduce
        # phase uses real grad-reduce plus the synthetic overlap operation
        # above, solely to verify the SMS policy.
        calibration_iteration = args.warmup_iters
        if args.enable_autotune and not manager._autotuner.complete:
            print_rank0(
                f"Autotune complete-iteration A/B workload: {routing_cases[-1][7]}"
            )
        while args.enable_autotune and not manager._autotuner.complete:
            calibration_iteration += 1
            finish_all_ranks()
            run_ultra(
                routing_cases[-1],
                profile_breakdown=args.profile_breakdown,
                autotune_iteration=calibration_iteration,
            )
        if args.enable_autotune:
            print_rank0(
                f"Autotune calibration completed after simulated iteration "
                f"{calibration_iteration}; reported benchmark starts afterward."
            )

        # Keep timing samples separate by routing case.  A rank maximum is
        # taken for every individual iteration; the mean below is therefore a
        # mean of slowest-rank times for one particular imbalance ratio, never
        # an accidental mixture of ratios.
        metrics = [
            {
                "static": [], "balanced": [], "ultra": [], "placement": [],
                "weight_local": [], "barrier": [], "weight_phase": [],
                "compute": [], "max_send": [],
            }
            for _ in routing_cases
        ]
        for iteration in range(args.bench_iters):
            case_index = iteration % len(routing_cases)
            case = routing_cases[case_index]
            case_metrics = metrics[case_index]
            finish_all_ranks()
            start = time.perf_counter()
            run_static(case)
            torch.cuda.synchronize()
            case_metrics["static"].append(
                max_across_ranks((time.perf_counter() - start) * 1000.0)
            )

            finish_all_ranks()
            start = time.perf_counter()
            proxy.run(case[4])
            torch.cuda.synchronize()
            case_metrics["balanced"].append(
                max_across_ranks((time.perf_counter() - start) * 1000.0)
            )

            finish_all_ranks()
            start = time.perf_counter()
            breakdown = run_ultra(
                case,
                profile_breakdown=args.profile_breakdown,
                autotune_iteration=calibration_iteration + iteration + 1,
            )
            torch.cuda.synchronize()
            case_metrics["ultra"].append(
                max_across_ranks((time.perf_counter() - start) * 1000.0)
            )
            case_metrics["max_send"].append(case[5]["max_send"])
            if breakdown is not None:
                case_metrics["placement"].append(max_across_ranks(breakdown[0]))
                case_metrics["weight_local"].append(max_across_ranks(breakdown[1]))
                case_metrics["barrier"].append(max_across_ranks(breakdown[2]))
                case_metrics["weight_phase"].append(max_across_ranks(breakdown[3]))
                case_metrics["compute"].append(max_across_ranks(breakdown[4]))

        global_assignments = world_size * args.tokens_per_rank * args.topk
        for case_index, (case, case_metrics) in enumerate(zip(routing_cases, metrics)):
            static_ms = sum(case_metrics["static"]) / len(case_metrics["static"])
            balanced_compute_ms = sum(case_metrics["balanced"]) / len(case_metrics["balanced"])
            ultra_ms = sum(case_metrics["ultra"]) / len(case_metrics["ultra"])
            print_rank0(
                f"\nResults for {case[7]} (slowest rank; {len(case_metrics['ultra'])} samples; "
                "communication included for UltraEP):"
            )
            print_rank0(
                f"  fixed placement: {static_ms:.3f} ms | "
                f"{global_assignments / static_ms * 1000.0:.0f} expert-token/s"
            )
            print_rank0(
                f"  balanced compute: {balanced_compute_ms:.3f} ms | "
                f"pure balance gain {static_ms / balanced_compute_ms:.3f}x"
            )
            print_rank0(
                f"  UltraEP:         {ultra_ms:.3f} ms | "
                f"{global_assignments / ultra_ms * 1000.0:.0f} expert-token/s"
            )
            print_rank0(
                f"  speedup:          {static_ms / ultra_ms:.3f}x "
                f"({(static_ms / ultra_ms - 1.0) * 100.0:+.1f}%)"
            )
            print_rank0(
                f"  UltraEP control/sync cost over balanced compute: "
                f"{ultra_ms - balanced_compute_ms:.3f} ms"
            )
            if case_metrics["placement"]:
                print_rank0("  UltraEP synchronized timing breakdown (slowest rank):")
                print_rank0(f"    placement + reroute: {sum(case_metrics['placement']) / len(case_metrics['placement']):.3f} ms")
                print_rank0(f"    local Weight Sync completion: {sum(case_metrics['weight_local']) / len(case_metrics['weight_local']):.3f} ms")
                print_rank0(f"    receiver barrier wait: {sum(case_metrics['barrier']) / len(case_metrics['barrier']):.3f} ms")
                mean_phase_ms = sum(case_metrics["weight_phase"]) / len(case_metrics["weight_phase"])
                print_rank0(f"    Weight Sync phase makespan: {mean_phase_ms:.3f} ms")
                print_rank0(f"    balanced expert compute: {sum(case_metrics['compute']) / len(case_metrics['compute']):.3f} ms")
                mean_max_send_bytes = sum(case_metrics["max_send"]) / len(case_metrics["max_send"])
                effective_gib_s = mean_max_send_bytes / (mean_phase_ms / 1000.0) / (1024.0**3)
                print_rank0(f"    planned domain-local Weight Sync payload bandwidth: {effective_gib_s:.1f} GiB/s")
        print_rank0(
            "NOTE: global Placement and domain-local Weight Sync are real; token "
            "dispatch/combine and framework overhead are excluded from timing. "
            "Use the printed traffic envelope to judge the required cross-node "
            "dispatcher bandwidth before claiming end-to-end speedup."
        )
        if args.enable_autotune:
            config = manager._autotuner.current_weight_sync
            print_rank0(
                "Autotune integration result: "
                f"weight-sync copy={config.copy_mode}, tpb={config.threads_per_block}, "
                f"cta={config.cta_multiplier}, lds_waves={config.lds_waves_per_destination}; "
                f"weight_sync_done={manager._autotuner.weight_sync_done}; "
                f"grad_reduce_done={manager._autotuner.grad_done}; "
                f"grad_reduce_sms={manager.grad_reduce_num_sms}."
            )
    finally:
        if manager is not None:
            manager.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
