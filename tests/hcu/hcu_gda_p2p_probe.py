#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Isolate one rocSHMEM sender-to-target put path.

All ranks initialize UltraEP/rocSHMEM, but only ``sender_pe`` issues an
8-byte symmetric-heap put to ``target_pe``.  Set
``ULTRA_EP_SHMEM_PROBE_TRANSPORT=ipc`` to test the pure IPC backend and
``ULTRA_EP_SHMEM_PROBE_MODE=wg`` to use the same work-group put primitive as
IPC fcollect.  ``ULTRA_EP_SHMEM_PROBE_CONTEXT=default`` bypasses the
WG-private context; the default ``private`` remains the fcollect-equivalent
path.  Set ``ULTRA_EP_SHMEM_PROBE_OPERATION=barrier`` to isolate the
device-side WG barrier used after fcollect's fan-out puts.
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


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value, 0)


def main() -> None:
    local_rank = _env_int("LOCAL_RANK", 0)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    sender_pe = _env_int("ULTRA_EP_GDA_PROBE_SENDER", 10)
    target_pe = _env_int("ULTRA_EP_GDA_PROBE_TARGET", 0)
    magic = _env_int("ULTRA_EP_GDA_PROBE_MAGIC", 0x554C545241455031)
    domain_size = _env_int("MAX_NUM_NVL_PEERS", 8)
    expected_transport = os.getenv("ULTRA_EP_SHMEM_PROBE_TRANSPORT", "gda").strip().lower()
    probe_mode = os.getenv("ULTRA_EP_SHMEM_PROBE_MODE", "thread").strip().lower()
    probe_context = os.getenv("ULTRA_EP_SHMEM_PROBE_CONTEXT", "private").strip().lower()
    probe_operation = os.getenv("ULTRA_EP_SHMEM_PROBE_OPERATION", "put").strip().lower()
    manager = None

    try:
        if probe_operation not in {"put", "barrier"}:
            raise ValueError("ULTRA_EP_SHMEM_PROBE_OPERATION must be 'put' or 'barrier'")
        if probe_operation == "put" and not (0 <= sender_pe < world_size and 0 <= target_pe < world_size):
            raise ValueError(f"probe PEs must be in [0, {world_size}), got {sender_pe}->{target_pe}")
        if probe_operation == "put" and sender_pe == target_pe:
            raise ValueError("probe sender and target must differ")
        if probe_mode not in {"thread", "wg"}:
            raise ValueError("ULTRA_EP_SHMEM_PROBE_MODE must be 'thread' or 'wg'")
        if probe_context not in {"private", "default"}:
            raise ValueError("ULTRA_EP_SHMEM_PROBE_CONTEXT must be 'private' or 'default'")

        # Manager construction initializes the same rocSHMEM runtime as
        # smoke/E2E, without enqueueing placement/fcollect work.
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
        runtime_info = dict(ext.get_shmem_runtime_info())
        if str(runtime_info["transport"]).lower() != expected_transport:
            raise RuntimeError(
                f"probe requires {expected_transport} transport, got {runtime_info}"
            )

        # Keep all PEs aligned around symmetric allocation and the host-side
        # barriers inside the extension probe.
        dist.barrier()
        if probe_operation == "barrier":
            ext.run_shmem_wg_barrier_probe()
            if rank == 0:
                print(
                    f"PASS: SHMEM {expected_transport} WG barrier probe",
                    flush=True,
                )
            return
        observed = int(
            ext.run_gda_p2p_put_probe(
                sender_pe,
                target_pe,
                magic,
                probe_mode == "wg",
                probe_context == "private",
            )
        )
        local_ok = 1 if rank != target_pe or observed == magic else 0
        result = torch.tensor([local_ok], dtype=torch.int32, device="cuda")
        dist.all_reduce(result, op=dist.ReduceOp.MIN)
        if int(result.item()) != 1:
            raise RuntimeError(
                f"SHMEM {expected_transport}/{probe_context}/{probe_mode} put probe failed: "
                f"sender={sender_pe}, target={target_pe}, "
                f"target observed 0x{observed:016x}, expected 0x{magic:016x}"
            )
        if rank == 0:
            print(
                f"PASS: SHMEM {expected_transport}/{probe_context}/{probe_mode} put probe "
                f"sender={sender_pe} -> target={target_pe}; "
                f"magic=0x{magic:016x}",
                flush=True,
            )
    finally:
        if manager is not None:
            manager.destroy()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
