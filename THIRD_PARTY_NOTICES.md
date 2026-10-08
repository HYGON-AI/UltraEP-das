# Third-Party Notices

This document records the upstream source and third-party components used by
UltraEP-das. License texts and copyright notices supplied by each project must
be retained according to their respective terms.

## UltraEP

- Project: UltraEP
- Repository: https://github.com/Dots-Infra/UltraEP
- Tag: `v1.0.0`
- Commit: `94cab099b44fffa99a82fea99e7c12d89cf65e4f`
- License: MIT
- Local use: upstream source base modified to add HCU/HIP support, rocSHMEM
  integration, runtime autotuning, validation, and documentation
- License text: [LICENSE](LICENSE)

## rocSHMEM

- Project: rocSHMEM fork with SHCA support
- Repository: https://github.com/HYGON-AI/rocSHMEM-das
- Local path: `third-party/rocshmem`
- Pinned commit: `029d8f5f128af3173ae55ba412bf40699cf920d9`
- Configured branch: `develop`
- License: MIT
- Copyright: Advanced Micro Devices, Inc.; Hygon Information Technology Co.,
  Ltd.
- Local use: the pinned HCU-adapted fork supplies rocSHMEM IPC and GDA/SHCA
  communication; the superproject does not carry an additional source patch
  for this component
- License text:
  [rocSHMEM-das LICENSE.md](https://github.com/HYGON-AI/rocSHMEM-das/blob/029d8f5f128af3173ae55ba412bf40699cf920d9/LICENSE.md)

## Inherited source acknowledgements

The upstream UltraEP README acknowledges low-level communication-kernel
lineage from [DeepEP](https://github.com/deepseek-ai/DeepEP) and
[HybridEP](https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep). This fork
preserves that acknowledgement in its README. The HCU-specific delta does not
vendor either repository as a separate component; its source base is the
fixed UltraEP revision recorded above.
