// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT

#pragma once

#include <c10/cuda/CUDAStream.h>

#include <cstddef>

#if defined(ULTRA_EP_USE_HIP)
#include <hip/hip_runtime.h>
#elif defined(ULTRA_EP_USE_CUDA)
#include <cuda_runtime.h>
#else
#error "UltraEP requires ULTRA_EP_USE_CUDA or ULTRA_EP_USE_HIP"
#endif

namespace ultra_ep::platform {

// PyTorch intentionally keeps the CUDA device/stream API for ROCm builds.
// Keeping that detail behind this alias prevents it from leaking through the
// UltraEP API and gives the rest of the project backend-neutral terminology.
using DeviceStream = c10::cuda::CUDAStream;

#if defined(ULTRA_EP_USE_HIP)
using DeviceStreamHandle = hipStream_t;
using RuntimeError = hipError_t;
using DeviceProperties = hipDeviceProp_t;
using IpcMemHandle = hipIpcMemHandle_t;

inline constexpr RuntimeError kRuntimeSuccess = hipSuccess;
inline constexpr unsigned int kIpcMemLazyEnablePeerAccess = hipIpcMemLazyEnablePeerAccess;

inline const char* runtime_error_name(RuntimeError error) { return hipGetErrorName(error); }
inline const char* runtime_error_string(RuntimeError error) { return hipGetErrorString(error); }
inline RuntimeError get_device(int* device) { return hipGetDevice(device); }
inline RuntimeError set_device(int device) { return hipSetDevice(device); }
inline RuntimeError get_device_count(int* count) { return hipGetDeviceCount(count); }
inline RuntimeError get_device_properties(DeviceProperties* properties, int device) {
    return hipGetDeviceProperties(properties, device);
}
inline RuntimeError device_malloc(void** ptr, size_t size) { return hipMalloc(ptr, size); }
inline RuntimeError device_free(void* ptr) { return hipFree(ptr); }
inline RuntimeError device_memset(void* ptr, int value, size_t size) { return hipMemset(ptr, value, size); }
inline RuntimeError device_memset_async(void* ptr,
                                        int value,
                                        size_t size,
                                        DeviceStreamHandle stream) {
    return hipMemsetAsync(ptr, value, size, stream);
}
inline RuntimeError device_synchronize() { return hipDeviceSynchronize(); }
inline RuntimeError device_mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    return hipMemGetInfo(free_bytes, total_bytes);
}
inline RuntimeError stream_synchronize(DeviceStreamHandle stream) { return hipStreamSynchronize(stream); }
inline RuntimeError get_last_error() { return hipGetLastError(); }
inline RuntimeError ipc_get_mem_handle(IpcMemHandle* handle, void* ptr) { return hipIpcGetMemHandle(handle, ptr); }
inline RuntimeError ipc_open_mem_handle(void** ptr, IpcMemHandle handle, unsigned int flags) {
    return hipIpcOpenMemHandle(ptr, handle, flags);
}
inline RuntimeError ipc_close_mem_handle(void* ptr) { return hipIpcCloseMemHandle(ptr); }
inline RuntimeError mem_get_address_range(void** base, size_t* size, void* ptr) {
    return hipMemGetAddressRange(base, size, ptr);
}
#else
using DeviceStreamHandle = cudaStream_t;
using RuntimeError = cudaError_t;
using DeviceProperties = cudaDeviceProp;
using IpcMemHandle = cudaIpcMemHandle_t;

inline constexpr RuntimeError kRuntimeSuccess = cudaSuccess;
inline constexpr unsigned int kIpcMemLazyEnablePeerAccess = cudaIpcMemLazyEnablePeerAccess;

inline const char* runtime_error_name(RuntimeError error) { return cudaGetErrorName(error); }
inline const char* runtime_error_string(RuntimeError error) { return cudaGetErrorString(error); }
inline RuntimeError get_device(int* device) { return cudaGetDevice(device); }
inline RuntimeError set_device(int device) { return cudaSetDevice(device); }
inline RuntimeError get_device_count(int* count) { return cudaGetDeviceCount(count); }
inline RuntimeError get_device_properties(DeviceProperties* properties, int device) {
    return cudaGetDeviceProperties(properties, device);
}
inline RuntimeError device_malloc(void** ptr, size_t size) { return cudaMalloc(ptr, size); }
inline RuntimeError device_free(void* ptr) { return cudaFree(ptr); }
inline RuntimeError device_memset(void* ptr, int value, size_t size) { return cudaMemset(ptr, value, size); }
inline RuntimeError device_memset_async(void* ptr,
                                        int value,
                                        size_t size,
                                        DeviceStreamHandle stream) {
    return cudaMemsetAsync(ptr, value, size, stream);
}
inline RuntimeError device_synchronize() { return cudaDeviceSynchronize(); }
inline RuntimeError device_mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    return cudaMemGetInfo(free_bytes, total_bytes);
}
inline RuntimeError stream_synchronize(DeviceStreamHandle stream) { return cudaStreamSynchronize(stream); }
inline RuntimeError get_last_error() { return cudaGetLastError(); }
inline RuntimeError ipc_get_mem_handle(IpcMemHandle* handle, void* ptr) { return cudaIpcGetMemHandle(handle, ptr); }
inline RuntimeError ipc_open_mem_handle(void** ptr, IpcMemHandle handle, unsigned int flags) {
    return cudaIpcOpenMemHandle(ptr, handle, flags);
}
inline RuntimeError ipc_close_mem_handle(void* ptr) { return cudaIpcCloseMemHandle(ptr); }
inline RuntimeError mem_get_address_range(void** base, size_t* size, void* ptr) {
    return cudaMemGetAddressRange(base, size, ptr);
}
#endif

enum class MemcpyKind { HostToDevice, DeviceToHost, DeviceToDevice };

inline RuntimeError device_memcpy(void* dst, const void* src, size_t size, MemcpyKind kind) {
#if defined(ULTRA_EP_USE_HIP)
    const auto native_kind = kind == MemcpyKind::HostToDevice   ? hipMemcpyHostToDevice
                             : kind == MemcpyKind::DeviceToHost ? hipMemcpyDeviceToHost
                                                                : hipMemcpyDeviceToDevice;
    return hipMemcpy(dst, src, size, native_kind);
#else
    const auto native_kind = kind == MemcpyKind::HostToDevice   ? cudaMemcpyHostToDevice
                             : kind == MemcpyKind::DeviceToHost ? cudaMemcpyDeviceToHost
                                                                : cudaMemcpyDeviceToDevice;
    return cudaMemcpy(dst, src, size, native_kind);
#endif
}

inline RuntimeError device_memcpy_async(void* dst,
                                        const void* src,
                                        size_t size,
                                        MemcpyKind kind,
                                        DeviceStreamHandle stream) {
#if defined(ULTRA_EP_USE_HIP)
    const auto native_kind = kind == MemcpyKind::HostToDevice   ? hipMemcpyHostToDevice
                             : kind == MemcpyKind::DeviceToHost ? hipMemcpyDeviceToHost
                                                                : hipMemcpyDeviceToDevice;
    return hipMemcpyAsync(dst, src, size, native_kind, stream);
#else
    const auto native_kind = kind == MemcpyKind::HostToDevice   ? cudaMemcpyHostToDevice
                             : kind == MemcpyKind::DeviceToHost ? cudaMemcpyDeviceToHost
                                                                : cudaMemcpyDeviceToDevice;
    return cudaMemcpyAsync(dst, src, size, native_kind, stream);
#endif
}

template <typename T>
inline RuntimeError device_malloc(T** ptr, size_t size) {
    return device_malloc(reinterpret_cast<void**>(ptr), size);
}

inline DeviceStream get_current_stream() { return c10::cuda::getCurrentCUDAStream(); }
inline DeviceStream get_stream_from_pool(bool high_priority = false) {
    return c10::cuda::getStreamFromPool(high_priority);
}

template <typename Kernel>
inline RuntimeError set_max_dynamic_shared_memory(Kernel kernel, int size) {
#if defined(ULTRA_EP_USE_HIP)
    return hipFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), hipFuncAttributeMaxDynamicSharedMemorySize, size);
#else
    return cudaFuncSetAttribute(
        reinterpret_cast<const void*>(kernel), cudaFuncAttributeMaxDynamicSharedMemorySize, size);
#endif
}

inline RuntimeError launch_kernel(const void* kernel,
                                  dim3 grid,
                                  dim3 block,
                                  void** args,
                                  size_t shared_memory_bytes,
                                  DeviceStreamHandle stream) {
#if defined(ULTRA_EP_USE_HIP)
    return hipLaunchKernel(kernel, grid, block, args, shared_memory_bytes, stream);
#else
    return cudaLaunchKernel(kernel, grid, block, args, shared_memory_bytes, stream);
#endif
}

inline const char* backend_name() {
#if defined(ULTRA_EP_USE_HIP)
    return "HIP";
#else
    return "CUDA";
#endif
}

}  // namespace ultra_ep::platform
