#pragma once

#include <cstddef>

namespace ultra_ep::kernels {

using GradScalar = float;

inline constexpr int kMaxNvlDomainSize = 72;
inline constexpr int kMaxGradReduceTaskCount = 256;
inline constexpr int kMaxWeightSyncTaskCount = 256;

inline constexpr std::size_t kNumTMAAlignBytes = 16;

inline constexpr int kGradElementBytes = sizeof(GradScalar);

inline constexpr int kWeightSyncTileSizeBytes = 32 * 1024;
inline constexpr int kWeightSyncThreadsPerBlock = 256;
inline constexpr int kWeightSyncPipelineStages = 2;
inline constexpr int kWeightSyncRelayChunkTiles = 8;

// The CUDA TMA implementation uses a 64 KiB staging tile.  On the target
// HCU, a CU has 64 KiB of LDS in total, so a 64 KiB dynamic allocation leaves
// no space for this kernel's static LDS (metadata, barriers, etc.) and HIP
// rejects the launch with hipErrorOutOfMemory.  Keep a 32 KiB tile on HIP;
// this also leaves headroom for the deterministic fallback's two LDS buffers.
#if defined(ULTRA_EP_USE_HIP)
inline constexpr int kGradReduceTileSizeBytes = 32 * 1024;
#else
inline constexpr int kGradReduceTileSizeBytes = 64 * 1024;
#endif
inline constexpr int kGradReduceTileElements = kGradReduceTileSizeBytes / kGradElementBytes;
inline constexpr int kGradReducePipelineStages = 2;
inline constexpr int kGradReduceThreadsPerBlock = 256;

inline constexpr int kDenseRerouteTileTokens = 128;
// Reroute uses one native warp/wave per expert. CUDA defaults to warp32,
// while the HIP build defines ULTRA_EP_WAVE_SIZE=64 for the target HCU.
#if defined(ULTRA_EP_WAVE_SIZE)
inline constexpr int kDenseRerouteWaveSize = ULTRA_EP_WAVE_SIZE;
#else
inline constexpr int kDenseRerouteWaveSize = 32;
#endif
inline constexpr int kDenseRerouteThreadsPerBlock = 256;
inline constexpr int kDenseRerouteWarpsPerBlock =
    kDenseRerouteThreadsPerBlock / kDenseRerouteWaveSize;
inline constexpr int kDenseRerouteBackwardRowsPerBlock = 4;

static_assert(kDenseRerouteWaveSize == 32 || kDenseRerouteWaveSize == 64);
static_assert(kDenseRerouteThreadsPerBlock % kDenseRerouteWaveSize == 0);

}  // namespace ultra_ep::kernels
