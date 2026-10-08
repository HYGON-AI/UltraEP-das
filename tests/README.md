# Testing UltraEP

The test suite is split by the hardware and runtime it needs. This prevents a
normal `pytest` run from accidentally starting distributed GPU jobs.

## CPU unit tests

The unit suite validates policy code without importing the compiled extension.

```bash
python -m pip install -r requirements-test.txt
python -m pytest
```

`pytest.ini` limits automatic discovery to `tests/unit`.

## GPU integration tests

These scripts require an installed UltraEP build. They are executable test
programs rather than pytest cases:

```bash
# Placement and reroute simulation on one GPU
python tests/integration/placement_simulation.py

# Distributed communication and correctness path
torchrun --standalone --nproc-per-node=8 \
  tests/integration/runtime_e2e.py --num-experts 16
```

Use `--help` on either script for the complete argument list. The E2E test
must be launched with a number of ranks compatible with the expert count and
the machine topology.

## HCU validation

HCU launchers, stress tests, and performance experiments live under
`tests/hcu`. Start with:

```bash
bash tests/hcu/run_smoke.sh
bash tests/hcu/run_e2e.sh
bash tests/hcu/run_hcu_autotune_stability.sh
```

Multi-node tests intentionally require `NNODES`, `NODE_RANK`, and
`MASTER_ADDR`; no cluster names or node ranks are embedded in the repository.
See [the HCU guide](../docs/hcu.md) for build and launch examples.

Files prefixed with `benchmark_` report performance without enforcing a
universal pass threshold. Files prefixed with `stress_` run longer lifecycle or
stability validation. The `run_` launchers are the primary smoke, correctness,
autotune, profiling, and multi-node entrypoints.

## Adding tests

- Put CPU-only pytest cases in `tests/unit/test_*.py`.
- Put manually launched CUDA/HIP integration programs in `tests/integration`.
- Put HCU-specific launchers, probes, and benchmarks in `tests/hcu`.
- Keep cluster addresses, device lists, output directories, and rank IDs
  configurable through arguments or environment variables.
- A correctness test must exit nonzero on failure. Performance scripts should
  identify themselves as benchmarks and must not encode a universal threshold.
