# HCU build and validation guide

UltraEP supports HIP-enabled HCU systems through rocSHMEM. The supplied build
helper targets the GDA/SHCA transport and also builds the IPC transport used by
single-node smoke tests.

## Prerequisites

- Linux with a DTK/ROCm-compatible, HIP-enabled PyTorch installation
- Python 3.10 or newer, CMake, Ninja, a C++17 compiler, and `hipcc`
- An initialized compatible rocSHMEM source tree in
  `third-party/rocshmem`, or an installed build provided through
  `ROCSHMEM_DIR`
- SHCA development headers (`infiniband/shca_dv.h`) and `libshca.so`
- RCCL available through the PyTorch/DTK installation
- One process per HCU for distributed tests

Clone submodules before the first build:

```bash
git submodule update --init --recursive
```

The submodule uses the public
[HYGON-AI/rocSHMEM-das](https://github.com/HYGON-AI/rocSHMEM-das) fork and is
pinned to the revision recorded by this repository. Use the pinned revision
for reproducible builds; the configured `develop` branch is only used by
explicit remote-update workflows.

## Build

The defaults assume DTK is installed in `/opt/dtk`. Override them when needed:

```bash
ROCM_PATH=/opt/dtk \
SHCA_ROOT=/opt/shca \
PYTORCH_ROCM_ARCH=gfx942 \
bash build_hcu_shca.sh --wheel

python -m pip install dist/ultra_ep-*.whl
```

Useful build inputs are:

| Variable | Purpose |
| --- | --- |
| `ROCM_PATH` / `ROCM_HOME` | DTK or ROCm root; default `/opt/dtk` |
| `PYTORCH_ROCM_ARCH` | GPU architecture; auto-detected when possible |
| `SHCA_ROOT` | Root containing `include/` and `lib` or `lib64` |
| `SHCA_INCLUDE_DIR` | Direct override for the SHCA include root |
| `SHCA_LIBRARY_DIR` | Direct override for the directory containing `libshca.so` |
| `ROCSHMEM_DIR` | Compatible prebuilt rocSHMEM installation |
| `ROCSHMEM_FORCE_REBUILD=1` | Rebuild the bundled rocSHMEM tree |
| `NETWORK_INTERFACE` | Bootstrap interface for distributed jobs |

`bash build_hcu_shca.sh` builds the extension in place. `--wheel` creates a
wheel, `--force-rocshmem` rebuilds rocSHMEM, and `--no-build -- <command>` runs
a command with the same runtime library environment.

## Single-node validation

Run the smoke test first, then the full correctness path:

```bash
bash build_hcu_shca.sh --no-build -- bash tests/hcu/run_smoke.sh
bash build_hcu_shca.sh --no-build -- bash tests/hcu/run_e2e.sh
```

The launchers default to eight processes. Set `NPROC_PER_NODE` for another
machine shape. Test sizes and iteration counts in `run_e2e.sh` can be changed
through the `HCU_E2E_*` variables documented directly in the launcher.

## Multi-node validation

Run the same command on every node and change only `NODE_RANK`:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.10 NETWORK_INTERFACE=eth0 \
  ROCSHMEM_ALLOWED_IBV_DEVICES=shca_0,shca_1 \
  bash build_hcu_shca.sh --no-build -- \
  bash tests/hcu/run_hcu_multi_node.sh smoke
```

After `smoke` succeeds, repeat with `e2e`. The optional `probe` mode isolates
a sender-to-target GDA operation. The following values are deliberately not
stored in the repository: rendezvous address, node rank, network interface,
and SHCA device list.

## Runtime autotuning

Autotuning is opt-in and bounded. It does not change the number of replica
slots or placement memory allocated by `Manager`; those remain model-level
configuration. It selects the HIP weight-sync launch configuration and, when
enabled, the grad-reduce CU budget from real training observations.

Enable it when constructing the manager:

```python
manager = ultra_ep.Manager(
    ...,
    autotune=ultra_ep.AutotuneConfig(enabled=True),
)
```

During the measured training path, use `manager.wait_weight_sync(event,
layer_id)` and `manager.wait_grad_reduce(event, layer_id)` instead of waiting
on the event directly. After the optimizer step and pipeline flush, call the
following on every rank:

```python
manager.autotune_iteration_end(
    iteration,
    iteration_time_ms=complete_iteration_time_ms,
)
```

Only candidate iterations for which
`manager.autotune_needs_iteration_time(iteration)` is true require a complete
iteration time. Continue calling `autotune_iteration_end` while
`manager.autotune_collecting` is true. Read the committed result with
`manager.get_autotune_result()`.

An explicitly exported tuning variable is a lock: the tuner preserves it. To
let autotune search a parameter, do not export the corresponding
`ULTRA_EP_WEIGHT_SYNC_*` variable or `ULTRA_EP_GRAD_REDUCE_NUM_SMS`.

Use the included validations before integrating with a training framework:

```bash
bash tests/hcu/run_hcu_autotune_stability.sh
bash tests/hcu/run_hcu_autotune_sim.sh
```

## Troubleshooting

- Missing `infiniband/shca_dv.h`: set `SHCA_ROOT` or `SHCA_INCLUDE_DIR`.
- Missing `libshca.so`: set `SHCA_ROOT` or `SHCA_LIBRARY_DIR`.
- Empty `third-party/rocshmem`: initialize submodules or set `ROCSHMEM_DIR`.
- Bootstrap timeout: verify `MASTER_ADDR`, firewall rules, and
  `NETWORK_INTERFACE` on every node.
- Wrong HCA selection: set `ROCSHMEM_ALLOWED_IBV_DEVICES` or
  `ROCSHMEM_USE_IB_HCA` for the local cluster.
- Symmetric heap allocation failure: increase `ROCSHMEM_HEAP_SIZE`; the MoE
  proxy launchers choose larger defaults for model-sized buffers.
