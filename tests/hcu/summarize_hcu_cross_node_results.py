#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Summarize node-0 logs from the HCU cross-node validation suite.

The parser intentionally consumes only rank-0 logs: timings in the benchmarks
are already reduced to the slowest rank, so node-1 copies would duplicate each
result.  It is a reporting aid and never changes test behaviour.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def last_match(pattern: str, text: str):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else None


def parse_proxy(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    ratio = last_match(r"Load case ratio=([0-9.]+):", text)
    static_rank = last_match(r"static\s+rank max/mean=\d+/[0-9.]+, imbalance=([0-9.]+)", text)
    ultra_rank = last_match(r"UltraEP\s+rank max/mean=\d+/[0-9.]+, imbalance=([0-9.]+)", text)
    fixed = last_match(r"fixed placement:\s+([0-9.]+) ms", text)
    balanced = last_match(r"balanced compute:\s+([0-9.]+) ms", text)
    ultra = last_match(r"UltraEP:\s+([0-9.]+) ms", text)
    speedup = last_match(r"speedup:\s+([0-9.]+)x \(([+-][0-9.]+)%\)", text)
    placement = last_match(r"placement \+ reroute:\s+([0-9.]+) ms", text)
    weight = last_match(r"Weight Sync phase makespan \(copy \+ barrier\):\s+([0-9.]+) ms", text)
    traffic = last_match(r"cross-domain assignments=\d+/\d+ \(([0-9.]+)%\)", text)
    return {
        "ratio": ratio or "?",
        "static_rank": static_rank or "?",
        "ultra_rank": ultra_rank or "?",
        "fixed": fixed or "?",
        "balanced": balanced or "?",
        "ultra": ultra or "?",
        "speedup": speedup[0] if speedup else "?",
        "gain": speedup[1] if speedup else "?",
        "placement": placement or "?",
        "weight": weight or "?",
        "traffic": traffic or "?",
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize HCU cross-node test logs")
    parser.add_argument("log_dir", type=Path, help="directory containing *_node0.log files")
    args = parser.parse_args()
    if not args.log_dir.is_dir():
        parser.error(f"not a directory: {args.log_dir}")

    smoke = sorted(args.log_dir.glob("smoke_node0.log"))
    e2e = sorted(args.log_dir.glob("e2e_node0.log"))
    proxies = sorted(args.log_dir.glob("proxy_*_ratio*_node0.log"))

    print("UltraEP HCU cross-node report summary")
    print(f"  log directory: {args.log_dir}")
    print(f"  smoke: {'PASS marker found' if smoke and 'PASS:' in smoke[0].read_text(encoding='utf-8', errors='replace') else 'not found / no PASS marker'}")
    print(f"  e2e:   {'PASS markers found' if e2e and 'PASS' in e2e[0].read_text(encoding='utf-8', errors='replace') else 'not found / inspect log'}")

    if not proxies:
        print("  proxy: no independent ratio logs found")
        return

    rows = sorted((parse_proxy(path) for path in proxies), key=lambda item: float(item["ratio"]) if item["ratio"] != "?" else -1)
    print("\nPerformance proxy (slowest rank; dispatch/combine excluded):")
    print("  ratio | rank imbalance static→UltraEP | fixed | post-placement compute | UltraEP | speedup | place+reroute | Weight Sync | cross-node tokens")
    for row in rows:
        print(
            "  {ratio:>5} | {static_rank:>6} → {ultra_rank:<6} | "
            "{fixed:>6} ms | {balanced:>6} ms | {ultra:>6} ms | "
            "{speedup:>5}x ({gain:>6}%) | {placement:>6} ms | "
            "{weight:>6} ms | {traffic:>5}%".format(**row)
        )


if __name__ == "__main__":
    main()
