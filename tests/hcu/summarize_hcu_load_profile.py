#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Summarize UltraEP load-profiler traces and verify rank-level rebalancing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate token conservation and rank-level balance in UltraEP traces"
    )
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument(
        "--max-post-rank-imbalance",
        type=float,
        default=1.05,
        help="maximum allowed post-reroute rank max/mean (default: 1.05)",
    )
    parser.add_argument(
        "--require-improvement",
        action="store_true",
        help="fail unless each record's post-reroute rank imbalance is lower than pre-reroute",
    )
    return parser.parse_args()


def imbalance(values: np.ndarray) -> float:
    mean = float(values.mean())
    return float(values.max()) / mean if mean > 0 else 1.0


def load_chunks(trace_dir: Path):
    chunks = sorted(trace_dir.glob("*.chunk*.npz"))
    if not chunks:
        raise RuntimeError(f"no UltraEP trace chunks found in {trace_dir}")
    for path in chunks:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            yield path, metadata, data["microbatches"], data["pre_logical_loads"], data["post_physical_loads"]


def main():
    args = parse_args()
    if not args.trace_dir.is_dir():
        raise SystemExit(f"trace directory does not exist: {args.trace_dir}")

    records = []
    metadata_ref = None
    for path, metadata, microbatches, pre, post in load_chunks(args.trace_dir):
        if metadata_ref is None:
            metadata_ref = metadata
        num_ranks = int(metadata["ep_size"])
        num_local_master = int(metadata["num_local_master_experts"])
        num_local_physical = int(metadata["num_local_physical_experts"])
        if pre.shape[1] != num_ranks * num_local_master:
            raise RuntimeError(f"{path}: unexpected pre-load shape {pre.shape}")
        if post.shape[1] != num_ranks * num_local_physical:
            raise RuntimeError(f"{path}: unexpected post-load shape {post.shape}")

        for microbatch, pre_loads, post_loads in zip(microbatches, pre, post):
            pre_rank = pre_loads.reshape(num_ranks, num_local_master).sum(axis=1)
            post_rank = post_loads.reshape(num_ranks, num_local_physical).sum(axis=1)
            records.append(
                {
                    "microbatch": int(microbatch),
                    "tokens_pre": int(pre_loads.sum()),
                    "tokens_post": int(post_loads.sum()),
                    "pre_imbalance": imbalance(pre_rank),
                    "post_imbalance": imbalance(post_rank),
                    "pre_max": int(pre_rank.max()),
                    "post_max": int(post_rank.max()),
                }
            )

    if not records:
        raise SystemExit("trace chunks contained no records")

    failures = []
    for record in records:
        if record["tokens_pre"] != record["tokens_post"]:
            failures.append(
                f"microbatch {record['microbatch']}: token conservation failed "
                f"({record['tokens_pre']} -> {record['tokens_post']})"
            )
        if record["post_imbalance"] > args.max_post_rank_imbalance:
            failures.append(
                f"microbatch {record['microbatch']}: post rank imbalance "
                f"{record['post_imbalance']:.3f} exceeds {args.max_post_rank_imbalance:.3f}"
            )
        if args.require_improvement and record["post_imbalance"] >= record["pre_imbalance"]:
            failures.append(
                f"microbatch {record['microbatch']}: rank imbalance did not improve "
                f"({record['pre_imbalance']:.3f} -> {record['post_imbalance']:.3f})"
            )

    pre_values = np.asarray([record["pre_imbalance"] for record in records])
    post_values = np.asarray([record["post_imbalance"] for record in records])
    print("UltraEP load-profile rebalance summary:")
    print(f"  trace directory: {args.trace_dir}")
    print(f"  EP size: {metadata_ref['ep_size']}; records: {len(records)}")
    print(
        "  rank imbalance (max/mean): "
        f"pre mean/max={pre_values.mean():.3f}/{pre_values.max():.3f}, "
        f"post mean/max={post_values.mean():.3f}/{post_values.max():.3f}"
    )
    print(
        "  slowest rank tokens: "
        f"pre max={max(record['pre_max'] for record in records)}, "
        f"post max={max(record['post_max'] for record in records)}"
    )
    for record in records[:8]:
        print(
            f"  microbatch {record['microbatch']:>4}: tokens "
            f"{record['tokens_pre']} -> {record['tokens_post']}, rank imbalance "
            f"{record['pre_imbalance']:.3f} -> {record['post_imbalance']:.3f}"
        )
    if len(records) > 8:
        print(f"  ... {len(records) - 8} additional records omitted")

    if failures:
        print("FAIL: rebalance acceptance check failed:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("PASS: tokens were conserved and all recorded microbatches met the rebalance threshold.")


if __name__ == "__main__":
    main()
