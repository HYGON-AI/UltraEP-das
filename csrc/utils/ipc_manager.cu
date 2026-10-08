#include "ipc_manager.cuh"

#include <cstdio>
#include <unistd.h>

namespace ultra_ep::ipc {

#if defined(ULTRA_EP_USE_HIP)

IpcManager::IpcManager() {
    support_fabric_ = false;
    if (gethostname(hostname_, sizeof(hostname_)) != 0) {
        perror("gethostname");
        std::snprintf(hostname_, sizeof(hostname_), "unknown");
    }

    int device_id = -1;
    DEVICE_RUNTIME_CHECK(platform::get_device(&device_id));
    DEVICE_RUNTIME_CHECK(platform::set_device(device_id));
    malloc(reinterpret_cast<void**>(&test_memory_), 128 * sizeof(int));
    get_handle(&test_mem_handle_, test_memory_);
}

IpcManager::~IpcManager() {
    free(test_memory_);
    test_memory_ = nullptr;
}

bool IpcManager::support_fabric() { return false; }

void IpcManager::malloc(void** ptr, size_t size_raw) {
    DEVICE_RUNTIME_CHECK(platform::device_malloc(ptr, size_raw));
}

void IpcManager::free(void* ptr) {
    if (ptr != nullptr) {
        DEVICE_RUNTIME_CHECK(platform::device_free(ptr));
    }
}

void IpcManager::get_handle(MemHandle* mem_handle, void* ptr) {
    void* allocation_base = nullptr;
    DEVICE_RUNTIME_CHECK(platform::mem_get_address_range(&allocation_base, &mem_handle->size, ptr));
    DEVICE_RUNTIME_CHECK(platform::ipc_get_mem_handle(&mem_handle->inner.hip_ipc_mem_handle, ptr));
    std::strncpy(mem_handle->src_hostname, hostname_, sizeof(mem_handle->src_hostname));
    mem_handle->src_hostname[sizeof(mem_handle->src_hostname) - 1] = '\0';
}

void IpcManager::open_handle(void** ptr, MemHandle* mem_handle) {
    DEVICE_RUNTIME_CHECK(platform::ipc_open_mem_handle(
        ptr, mem_handle->inner.hip_ipc_mem_handle, platform::kIpcMemLazyEnablePeerAccess));
}

void IpcManager::close_handle(void* ptr) {
    DEVICE_RUNTIME_CHECK(platform::ipc_close_mem_handle(ptr));
}

bool IpcManager::is_accessible(MemHandle* mem_handle) {
    // HIP IPC handles are process-local to one host. Inter-node access is
    // provided by rocSHMEM in the next adaptation phase.
    return std::strncmp(mem_handle->src_hostname, hostname_, sizeof(hostname_)) == 0;
}

int IpcManager::detect_accessible_ranks(pybind11::object process_group) {
    auto torch_distributed = py::module_::import("torch.distributed");
    const int world_size = process_group.attr("size")().cast<int>();
    const int current_rank = process_group.attr("rank")().cast<int>();
    auto stream = platform::get_current_stream();

    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    torch::Tensor test_tensor = torch::empty({static_cast<long>(sizeof(MemHandle))}, opts);
    DEVICE_RUNTIME_CHECK(platform::device_memcpy_async(test_tensor.data_ptr(),
                                                       &test_mem_handle_,
                                                       sizeof(MemHandle),
                                                       platform::MemcpyKind::HostToDevice,
                                                       stream.stream()));

    py::list test_handle_list;
    for (int i = 0; i < world_size; ++i) {
        test_handle_list.append(torch::empty_like(test_tensor));
    }
    torch_distributed.attr("all_gather")(test_handle_list, test_tensor, process_group);

    int num_accessible_ranks = 1;
    for (int i = 0; i < world_size; ++i) {
        if (i == current_rank) {
            continue;
        }
        MemHandle test_handle;
        torch::Tensor gathered = test_handle_list[i].cast<torch::Tensor>();
        DEVICE_RUNTIME_CHECK(platform::device_memcpy_async(&test_handle,
                                                           gathered.data_ptr(),
                                                           sizeof(MemHandle),
                                                           platform::MemcpyKind::DeviceToHost,
                                                           stream.stream()));
        DEVICE_RUNTIME_CHECK(platform::stream_synchronize(stream.stream()));
        if (is_accessible(&test_handle)) {
            ++num_accessible_ranks;
        }
    }
    return num_accessible_ranks;
}

#else

// Round-up allocation size to fabric granularity.
size_t inline get_size_align_to_granularity(size_t size_raw, size_t granularity) {
    size_t size = (size_raw + granularity - 1) & ~(granularity - 1);
    if (size == 0)
        size = granularity;
    return size;
}

IpcManager::IpcManager() {
    this->support_fabric_ = support_fabric();
    if (gethostname(hostname_, sizeof(hostname_)) != 0) {
        perror("gethostname");
        std::snprintf(hostname_, sizeof(hostname_), "unknown");
    }

    // It seems a dummy call to set the device. but it is useful to prevent the invalid device context error
    int device_id = -1;
    DEVICE_RUNTIME_CHECK(platform::get_device(&device_id));
    DEVICE_RUNTIME_CHECK(platform::set_device(device_id));

    if (this->support_fabric_) {
        // Get the device context.
        CUDA_DRIVER_CHECK(cuCtxGetDevice(&device_));
        fabric_prop_.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        fabric_prop_.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        fabric_prop_.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
        fabric_prop_.location.id = device_;
        CUDA_DRIVER_CHECK(
            cuMemGetAllocationGranularity(&fabric_granularity_, &fabric_prop_, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc.location.id = device_;
        access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    }

    // Test the fabric support
    // Somtimes support_fabric() returns true, but the fabric can not be used.
    if (this->support_fabric_) {
        size_t size = get_size_align_to_granularity(128, fabric_granularity_);
        CUmemGenericAllocationHandle handle;
        if (CUDA_SUCCESS != cuMemCreate(&handle, size, &fabric_prop_, 0)) {
            this->support_fabric_ = false;
        } else {
            cuMemRelease(handle);
        }
        platform::get_last_error();  // Clear the last error
    }

    this->malloc((void**)&test_memory_, 128 * sizeof(int));
    this->get_handle(&test_mem_handle_, test_memory_);
}

IpcManager::~IpcManager() {
    this->free((void*)test_memory_);
    test_memory_ = nullptr;
}

// Check if the current device supports fabric.
bool IpcManager::support_fabric() {
    int device_count;
    DEVICE_RUNTIME_CHECK(platform::get_device_count(&device_count));

    for (int device = 0; device < device_count; ++device) {
        int support = 0;
        CUDA_DRIVER_CHECK(cuDeviceGetAttribute(&support, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device));
        if (!support) {
            return false;
        }
    }
    return true;
}

void IpcManager::malloc(void** ptr, size_t size_raw) {
    if (support_fabric_) {
        size_t size = get_size_align_to_granularity(size_raw, fabric_granularity_);
        CUmemGenericAllocationHandle handle;
        CUDA_DRIVER_CHECK(cuMemCreate(&handle, size, &fabric_prop_, 0));
        CUDA_DRIVER_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, fabric_granularity_, 0, 0));
        CUDA_DRIVER_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        CUDA_DRIVER_CHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &access_desc, 1));
    } else {
        DEVICE_RUNTIME_CHECK(platform::device_malloc(ptr, size_raw));
    }
}

void IpcManager::free(void* ptr) {
    if (ptr == nullptr) {
        return;
    }
    if (support_fabric_) {
        CUmemGenericAllocationHandle handle;
        CUDA_DRIVER_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
        size_t size = 0;
        CUDA_DRIVER_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
        CUDA_DRIVER_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
        CUDA_DRIVER_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
        CUDA_DRIVER_CHECK(cuMemRelease(handle));
    } else {
        DEVICE_RUNTIME_CHECK(platform::device_free(ptr));
    }
}

void IpcManager::get_handle(MemHandle* mem_handle, void* ptr) {
    size_t size = 0;
    CUDA_DRIVER_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));

    mem_handle->size = size;
    if (support_fabric_) {
        CUmemGenericAllocationHandle handle;
        CUDA_DRIVER_CHECK(cuMemRetainAllocationHandle(&handle, ptr));
        CUDA_DRIVER_CHECK(cuMemExportToShareableHandle(
            &mem_handle->inner.cu_mem_fabric_handle, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0));
    } else {
        DEVICE_RUNTIME_CHECK(platform::ipc_get_mem_handle(&mem_handle->inner.cuda_ipc_mem_handle, ptr));
    }

    // Record the source hostname
    strncpy(mem_handle->src_hostname, hostname_, sizeof(mem_handle->src_hostname));
}

void IpcManager::open_handle(void** ptr, MemHandle* mem_handle) {
    if (support_fabric_) {
        size_t size = mem_handle->size;
        CUmemGenericAllocationHandle handle;
        CUDA_DRIVER_CHECK(cuMemImportFromShareableHandle(
            &handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC));
        CUDA_DRIVER_CHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, 0, 0, 0));
        CUDA_DRIVER_CHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
        CUDA_DRIVER_CHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &access_desc, 1));
    } else {
        DEVICE_RUNTIME_CHECK(platform::ipc_open_mem_handle(
            ptr, mem_handle->inner.cuda_ipc_mem_handle, platform::kIpcMemLazyEnablePeerAccess));
    }
}

void IpcManager::close_handle(void* ptr) {
    if (support_fabric_) {
        size_t size = 0;
        CUDA_DRIVER_CHECK(cuMemGetAddressRange(NULL, &size, (CUdeviceptr)ptr));
        CUDA_DRIVER_CHECK(cuMemUnmap((CUdeviceptr)ptr, size));
        CUDA_DRIVER_CHECK(cuMemAddressFree((CUdeviceptr)ptr, size));
    } else {
        DEVICE_RUNTIME_CHECK(platform::ipc_close_mem_handle(ptr));
    }
}

bool IpcManager::is_accessible(MemHandle* mem_handle) {
    bool accessible = false;
    if (support_fabric_) {
        CUmemGenericAllocationHandle handle;
        auto ret =
            cuMemImportFromShareableHandle(&handle, &mem_handle->inner.cu_mem_fabric_handle, CU_MEM_HANDLE_TYPE_FABRIC);
        accessible = ret == CUDA_SUCCESS;
        if (accessible) {
            cuMemRelease(handle);
        } else {
            if (ret != CUDA_SUCCESS) {
                const char* errStr;
                cuGetErrorString(ret, &errStr);
                fprintf(stderr, "[Error] Failed to import the fabric handle: %s\n", errStr);
                fflush(stderr);
            }
        }
    } else {
        // Check if the source hostname is the same as the current hostname
        accessible = strncmp(mem_handle->src_hostname, hostname_, sizeof(hostname_)) == 0;
    }
    return accessible;
}

int IpcManager::detect_accessible_ranks(pybind11::object process_group) {
    auto torch_distributed = py::module_::import("torch.distributed");
    int world_size = process_group.attr("size")().cast<int>();
    int current_rank = process_group.attr("rank")().cast<int>();
    auto stream = platform::get_current_stream();

    // Put the test memory handle on a CUDA tensor
    auto opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
    torch::Tensor test_tensor = torch::empty({static_cast<long>(sizeof(MemHandle))}, opts);
    DEVICE_RUNTIME_CHECK(platform::device_memcpy_async(test_tensor.data_ptr(),
                                                       &test_mem_handle_,
                                                       sizeof(MemHandle),
                                                       platform::MemcpyKind::HostToDevice,
                                                       stream.stream()));

    // All gather the test memory
    py::list test_handle_list;
    for (int i = 0; i < world_size; i++) {
        test_handle_list.append(torch::empty_like(test_tensor));
    }
    torch_distributed.attr("all_gather")(test_handle_list, test_tensor, process_group);

    // Check if the test memory is accessible on each rank
    int num_accessible_ranks = 1;  // include the current rank
    for (int i = 0; i < world_size; i++) {
        if (i != current_rank) {
            MemHandle test_handle;
            torch::Tensor gathered = test_handle_list[i].cast<torch::Tensor>();
            DEVICE_RUNTIME_CHECK(platform::device_memcpy_async(&test_handle,
                                                               gathered.data_ptr(),
                                                               sizeof(MemHandle),
                                                               platform::MemcpyKind::DeviceToHost,
                                                               stream.stream()));
            DEVICE_RUNTIME_CHECK(platform::stream_synchronize(stream.stream()));
            if (is_accessible(&test_handle)) {
                num_accessible_ranks++;
            }
        }
    }

    return num_accessible_ranks;
}

#endif

}  // namespace ultra_ep::ipc
