#pragma once

#include <cstdint>

namespace ultra_ep::kernels {

// Some placement algorithms intentionally operate on CUDA-sized logical
// warp32 groups.  A HCU wave64 contains two such groups, so the mask must
// select the current half and every shuffle must explicitly use width 32.
__device__ __forceinline__ auto logical_warp32_mask() {
#if defined(ULTRA_EP_USE_HIP)
    const int native_lane = static_cast<int>(threadIdx.x % warpSize);
    const int base_lane = (native_lane / 32) * 32;
    return uint64_t{0xFFFFFFFFu} << base_lane;
#else
    return uint32_t{0xFFFFFFFFu};
#endif
}

template <typename T>
__device__ __forceinline__ T logical_warp32_shfl(T value, int source_lane) {
    return __shfl_sync(logical_warp32_mask(), value, source_lane, 32);
}

template <typename T>
__device__ __forceinline__ T logical_warp32_shfl_xor(T value, int lane_mask) {
    return __shfl_xor_sync(logical_warp32_mask(), value, lane_mask, 32);
}

__device__ __forceinline__ void logical_warp32_sync() {
    __syncwarp(logical_warp32_mask());
}

}  // namespace ultra_ep::kernels
