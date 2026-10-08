# UltraEP-das

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Unit tests](https://github.com/HYGON-AI/UltraEP-das/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/HYGON-AI/UltraEP-das/actions/workflows/unit-tests.yml)
[![Upstream](https://img.shields.io/badge/Upstream-UltraEP-blue.svg)](https://github.com/Dots-Infra/UltraEP)

UltraEP-das is the HCU/HIP adaptation of UltraEP, a real-time expert load
balancing library for large-scale Mixture-of-Experts training and inference.
This repository keeps the upstream CUDA implementation and adds the HCU
runtime, rocSHMEM communication, SHCA multi-node support, runtime autotuning,
and HCU validation tools.

For the full upstream design, API examples, performance results, and citation,
see [README_UPSTREAM.md](README_UPSTREAM.md).

## Upstream and modifications

UltraEP-das is based on [Dots-Infra/UltraEP](https://github.com/Dots-Infra/UltraEP)
tag `v1.0.0`, commit
`94cab099b44fffa99a82fea99e7c12d89cf65e4f`, under the MIT License.

Modified by Hygon Information Technology Co., Ltd.

The HCU adaptation provides:

- HCU/HIP runtime and kernel compatibility while preserving the upstream CUDA
  path.
- rocSHMEM IPC communication for single-node deployments.
- rocSHMEM GDA with the SHCA provider for multi-node deployments.
- Bounded runtime autotuning for weight synchronization and gradient reduction.
- HCU smoke, stability, performance, and multi-node validation tools.

The upstream license is preserved in [LICENSE](LICENSE). Component provenance
and dependency notices are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Clone

Clone the repository together with its submodules:

```bash
git clone --recursive https://github.com/HYGON-AI/UltraEP-das.git
cd UltraEP-das
```

If the repository was cloned without submodules, initialize them separately:

```bash
git submodule update --init --recursive
```

## HCU build

The HCU backend requires Python 3.10 or newer, a HIP-enabled PyTorch/DTK
installation, RCCL,
[rocSHMEM-das](https://github.com/HYGON-AI/rocSHMEM-das), and the SHCA
development libraries for GDA/SHCA multi-node communication.

Build a wheel with the supplied helper:

```bash
ROCM_PATH=/opt/dtk \
SHCA_ROOT=/opt/shca \
bash build_hcu_shca.sh --wheel

python -m pip install dist/ultra_ep-*.whl
```

`ROCSHMEM_DIR` can point to a compatible prebuilt rocSHMEM installation. When
it is not set, the build uses `third-party/rocshmem`.

See [docs/hcu.md](docs/hcu.md) for prerequisites, environment variables,
single-node and multi-node launch commands, and troubleshooting.

## Runtime autotuning

Runtime autotuning is opt-in. It keeps the replica budget and memory
allocations fixed, then selects the weight-sync launch configuration and
gradient-reduction CU budget from bounded training samples.

```python
import ultra_ep

manager = ultra_ep.Manager(
    ...,
    autotune=ultra_ep.AutotuneConfig(enabled=True),
)

# Use these methods at the existing dependency joins while tuning is active.
manager.wait_weight_sync(weight_sync_event, layer_id)
manager.wait_grad_reduce(grad_reduce_event, layer_id)

# Call on every rank after the optimizer step and pipeline flush.
manager.autotune_iteration_end(iteration, iteration_time_ms=iteration_ms)

if manager.autotune_complete:
    print(manager.get_autotune_result())
```

Explicit `ULTRA_EP_WEIGHT_SYNC_*` settings and
`ULTRA_EP_GRAD_REDUCE_NUM_SMS` are treated as user locks and are not
overwritten. The complete integration contract is documented in
[the HCU autotuning guide](docs/hcu.md#runtime-autotuning).

## Tests

CPU-only policy tests use pytest and do not require the compiled extension:

```bash
python -m pip install -r requirements-test.txt
python -m pytest
```

HCU validation is launched explicitly so normal pytest discovery never starts
distributed hardware jobs:

```bash
bash tests/hcu/run_smoke.sh
bash tests/hcu/run_e2e.sh
bash tests/hcu/run_hcu_autotune_stability.sh
```

See [tests/README.md](tests/README.md) for the test taxonomy, prerequisites,
integration commands, multi-node variables, and benchmark conventions.

## Documentation map

| Document | Purpose |
| --- | --- |
| [README_UPSTREAM.md](README_UPSTREAM.md) | Upstream design, complete API, performance data, and citation |
| [docs/hcu.md](docs/hcu.md) | HCU build, runtime configuration, validation, and troubleshooting |
| [tests/README.md](tests/README.md) | Unit, integration, HCU, stress, and benchmark test organization |
| [examples/README.md](examples/README.md) | Megatron-LM integration example |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution and validation requirements |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | Upstream and third-party provenance |

## License

UltraEP-das is distributed under the [MIT License](LICENSE). Original upstream
copyright and license terms remain in effect. Hygon-authored source files are
identified with their corresponding copyright and SPDX headers.
