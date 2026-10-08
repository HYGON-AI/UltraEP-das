#include "shmem.cuh"

#include <climits>
#include <cstdint>
#include <cstring>

#include "../kernels/launch.cuh"
#include "exception.cuh"

#if defined(ULTRA_EP_USE_ROCSHMEM) && defined(ULTRA_EP_USE_NVSHMEM)
#error "Select exactly one SHMEM backend"
#endif

#if defined(ULTRA_EP_USE_ROCSHMEM)
#include <rocshmem/rocshmem.hpp>
#elif defined(ULTRA_EP_USE_NVSHMEM)
#include <device_host_transport/nvshmem_common_ibgda.h>
#include <non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh>
#include <nvshmem.h>
#include <nvshmemx.h>
#else
#error "UltraEP requires ULTRA_EP_USE_ROCSHMEM or ULTRA_EP_USE_NVSHMEM"
#endif

namespace ultra_ep::shmem {

#if defined(ULTRA_EP_USE_ROCSHMEM)

namespace {

rocshmem::rocshmem_team_t split_team = rocshmem::ROCSHMEM_TEAM_INVALID;
rocshmem::rocshmem_team_config_t split_team_config = {};

__global__ void int32_allreduce_kernel(int32_t* data,
                                       int nelems,
                                       rocshmem::rocshmem_team_t team) {
    __shared__ rocshmem::rocshmem_ctx_t context;
    const int create_status =
        rocshmem::rocshmem_wg_team_create_ctx(team, rocshmem::ROCSHMEM_CTX_WG_PRIVATE, &context);
    EP_DEVICE_ASSERT(create_status == 0);
    // rocSHMEM writes the WG-private context through shared memory.  All
    // threads must observe the initialized handle before entering a WG
    // collective (the rocSHMEM functional testers use the same barrier).
    __syncthreads();
    const int reduce_status = rocshmem::rocshmem_ctx_int_sum_reduce_wg(context, team, data, data, nelems);
    EP_DEVICE_ASSERT(reduce_status == 0);
    __syncthreads();
    rocshmem::rocshmem_wg_ctx_destroy(&context);
    __syncthreads();
}

__global__ void int32_fcollect_kernel(int32_t* dest,
                                      const int32_t* src,
                                      int nelems,
                                      rocshmem::rocshmem_team_t team) {
    __shared__ rocshmem::rocshmem_ctx_t context;
    const int create_status =
        rocshmem::rocshmem_wg_team_create_ctx(team, rocshmem::ROCSHMEM_CTX_WG_PRIVATE, &context);
    EP_DEVICE_ASSERT(create_status == 0);
    // See int32_allreduce_kernel above.  Without this, a wave may consume an
    // incompletely initialized shared context, which surfaces as an HSA
    // VMFault under multi-node GDA pressure.
    __syncthreads();
    rocshmem::rocshmem_ctx_int_fcollect_wg(context, team, dest, src, nelems);
    __syncthreads();
    rocshmem::rocshmem_wg_ctx_destroy(&context);
    __syncthreads();
}

// Diagnostic-only single sender->target transfer.  `use_workgroup` selects
// the same putmem_nbi_wg primitive used by the IPC fcollect implementation;
// the other mode isolates a regular context put on the identical symmetric
// allocation and runtime setup.  `use_private_context` distinguishes the
// WG-private context used by fcollect from the default rocSHMEM context.
__global__ void gda_p2p_put_probe_kernel(uint64_t* slot,
                                         int sender_pe,
                                         int target_pe,
                                         uint64_t magic,
                                         bool use_workgroup,
                                         bool use_private_context,
                                         rocshmem::rocshmem_team_t team) {
    __shared__ rocshmem::rocshmem_ctx_t context;
    if (use_private_context) {
        const int create_status =
            rocshmem::rocshmem_wg_team_create_ctx(team,
                                                   rocshmem::ROCSHMEM_CTX_WG_PRIVATE,
                                                   &context);
        EP_DEVICE_ASSERT(create_status == 0);
    }
    __syncthreads();

    const bool is_sender = rocshmem::rocshmem_my_pe() == sender_pe;
    if (threadIdx.x == 0 && is_sender) {
        *slot = magic;
        __threadfence_system();
    }
    __syncthreads();

    if (use_workgroup) {
        if (is_sender) {
            if (use_private_context) {
                rocshmem::rocshmem_ctx_putmem_nbi_wg(context, slot, slot, sizeof(*slot), target_pe);
            } else {
                rocshmem::rocshmem_putmem_nbi_wg(slot, slot, sizeof(*slot), target_pe);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0 && is_sender) {
            if (use_private_context) {
                rocshmem::rocshmem_ctx_quiet(context);
            } else {
                rocshmem::rocshmem_quiet();
            }
        }
    } else if (threadIdx.x == 0 && is_sender) {
        if (use_private_context) {
            rocshmem::rocshmem_ctx_putmem(context, slot, slot, sizeof(*slot), target_pe);
            rocshmem::rocshmem_ctx_quiet(context);
        } else {
            rocshmem::rocshmem_putmem(slot, slot, sizeof(*slot), target_pe);
            rocshmem::rocshmem_quiet();
        }
    }

    __syncthreads();
    if (use_private_context) {
        rocshmem::rocshmem_wg_ctx_destroy(&context);
    }
    __syncthreads();
}

// fcollect finishes with the same WG barrier below.  Keep this in a separate
// kernel so a fault can be attributed to the IPC work/sync pool rather than
// to symmetric-heap data movement.
__global__ void wg_barrier_probe_kernel(rocshmem::rocshmem_team_t team) {
    __shared__ rocshmem::rocshmem_ctx_t context;
    const int create_status =
        rocshmem::rocshmem_wg_team_create_ctx(team, rocshmem::ROCSHMEM_CTX_WG_PRIVATE, &context);
    EP_DEVICE_ASSERT(create_status == 0);
    __syncthreads();
    rocshmem::rocshmem_ctx_barrier_wg(context, team);
    __syncthreads();
    rocshmem::rocshmem_wg_ctx_destroy(&context);
    __syncthreads();
}

int checked_collective_count(size_t nelems) {
    EP_HOST_ASSERT(nelems <= static_cast<size_t>(INT_MAX));
    return static_cast<int>(nelems);
}

}  // namespace

std::vector<uint8_t> get_unique_id() {
    rocshmem::rocshmem_uniqueid_t unique_id = {};
    EP_HOST_ASSERT(rocshmem::rocshmem_get_uniqueid(&unique_id) == 0);
    return std::vector<uint8_t>(unique_id.begin(), unique_id.end());
}

void* alloc(size_t size, size_t alignment) {
    void* allocation = rocshmem::rocshmem_malloc(size);
    if (allocation != nullptr) {
        EP_HOST_ASSERT(alignment > 0);
        EP_HOST_ASSERT(reinterpret_cast<uintptr_t>(allocation) % alignment == 0);
    }
    return allocation;
}

void free(void* allocation) {
    if (allocation != nullptr) {
        rocshmem::rocshmem_free(allocation);
    }
}

void* ptr(void* local_ptr, int pe) { return rocshmem::rocshmem_ptr(local_ptr, pe); }

void barrier(bool with_cpu_sync, const std::optional<platform::DeviceStreamHandle>& stream) {
    if (with_cpu_sync) {
        DEVICE_RUNTIME_CHECK(platform::device_synchronize());
    }
    if (stream.has_value()) {
        rocshmem::rocshmem_barrier_all_on_stream(stream.value());
    } else {
        rocshmem::rocshmem_barrier_all();
    }
    if (with_cpu_sync) {
        DEVICE_RUNTIME_CHECK(platform::device_synchronize());
    }
}

int init(const std::vector<uint8_t>& root_unique_id,
         int rank,
         int num_ranks,
         int team_split_stride) {
    EP_HOST_ASSERT(root_unique_id.size() == sizeof(rocshmem::rocshmem_uniqueid_t));
    rocshmem::rocshmem_uniqueid_t unique_id = {};
    std::memcpy(unique_id.data(), root_unique_id.data(), root_unique_id.size());

    rocshmem::rocshmem_init_attr_t attributes = {};
    EP_HOST_ASSERT(rocshmem::rocshmem_set_attr_uniqueid_args(rank, num_ranks, &unique_id, &attributes) == 0);
    EP_HOST_ASSERT(rocshmem::rocshmem_init_attr(rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID, &attributes) == 0);

    if (team_split_stride > 0 && num_ranks > team_split_stride) {
        EP_HOST_ASSERT(split_team == rocshmem::ROCSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % team_split_stride == 0);
        EP_HOST_ASSERT(rocshmem::rocshmem_team_split_strided(rocshmem::ROCSHMEM_TEAM_WORLD,
                                                             rank % team_split_stride,
                                                             team_split_stride,
                                                             num_ranks / team_split_stride,
                                                             &split_team_config,
                                                             0,
                                                             &split_team) == 0);
        EP_HOST_ASSERT(split_team != rocshmem::ROCSHMEM_TEAM_INVALID);
    }

    barrier(true);
    return rocshmem::rocshmem_my_pe();
}

void finalize() {
    barrier(true);
    if (split_team != rocshmem::ROCSHMEM_TEAM_INVALID) {
        rocshmem::rocshmem_team_destroy(split_team);
        split_team = rocshmem::ROCSHMEM_TEAM_INVALID;
    }
    rocshmem::rocshmem_finalize();
}

void int32_allreduce(int32_t* data, size_t nelems, platform::DeviceStreamHandle stream) {
    const int count = checked_collective_count(nelems);
    if (count == 0) {
        return;
    }
    kernels::launch_kernel(int32_allreduce_kernel,
                           kernels::make_launch_config(dim3(1), dim3(ULTRA_EP_WAVE_SIZE), stream),
                           data,
                           count,
                           rocshmem::ROCSHMEM_TEAM_WORLD);
}

void int32_fcollect(int32_t* dest,
                    const int32_t* src,
                    size_t nelems_per_pe,
                    platform::DeviceStreamHandle stream) {
    const int count = checked_collective_count(nelems_per_pe);
    if (count == 0) {
        return;
    }
    kernels::launch_kernel(int32_fcollect_kernel,
                           kernels::make_launch_config(dim3(1), dim3(ULTRA_EP_WAVE_SIZE), stream),
                           dest,
                           src,
                           count,
                           rocshmem::ROCSHMEM_TEAM_WORLD);
}

uint64_t gda_p2p_put_probe(int sender_pe,
                           int target_pe,
                           platform::DeviceStreamHandle stream,
                           uint64_t magic,
                           bool use_workgroup,
                           bool use_private_context) {
    const int pe = rocshmem::rocshmem_my_pe();
    const int pes = rocshmem::rocshmem_n_pes();
    EP_HOST_ASSERT(sender_pe >= 0 && sender_pe < pes);
    EP_HOST_ASSERT(target_pe >= 0 && target_pe < pes);
    EP_HOST_ASSERT(sender_pe != target_pe);

    auto* slot = static_cast<uint64_t*>(rocshmem::rocshmem_malloc(sizeof(uint64_t)));
    EP_HOST_ASSERT(slot != nullptr);
    DEVICE_RUNTIME_CHECK(platform::device_memset(slot, 0, sizeof(*slot)));

    // The allocation is symmetric only after every PE has completed the same
    // allocation call.  The host barrier is known to succeed during runtime
    // initialization and deliberately avoids the WG collective under test.
    barrier(true);
    kernels::launch_kernel(gda_p2p_put_probe_kernel,
                           kernels::make_launch_config(dim3(1), dim3(ULTRA_EP_WAVE_SIZE), stream),
                           slot,
                           sender_pe,
                           target_pe,
                           magic,
                           use_workgroup,
                           use_private_context,
                           rocshmem::ROCSHMEM_TEAM_WORLD);
    DEVICE_RUNTIME_CHECK(platform::stream_synchronize(stream));
    barrier(true);

    uint64_t observed = 0;
    if (pe == target_pe) {
        DEVICE_RUNTIME_CHECK(platform::device_memcpy(&observed,
                                                     slot,
                                                     sizeof(observed),
                                                     platform::MemcpyKind::DeviceToHost));
    }

    barrier(true);
    rocshmem::rocshmem_free(slot);
    barrier(true);
    return observed;
}

void wg_barrier_probe(platform::DeviceStreamHandle stream) {
    kernels::launch_kernel(wg_barrier_probe_kernel,
                           kernels::make_launch_config(dim3(1), dim3(ULTRA_EP_WAVE_SIZE), stream),
                           rocshmem::ROCSHMEM_TEAM_WORLD);
    DEVICE_RUNTIME_CHECK(platform::stream_synchronize(stream));
}

const char* backend_name() { return "rocSHMEM"; }

int my_pe() { return rocshmem::rocshmem_my_pe(); }

int num_pes() { return rocshmem::rocshmem_n_pes(); }

const char* transport_name() {
    switch (rocshmem::rocshmem_query_backend_type()) {
        case rocshmem::BackendType::GDA_BACKEND:
            return "gda";
        case rocshmem::BackendType::RO_BACKEND:
            return "ro";
        case rocshmem::BackendType::IPC_BACKEND:
            return "ipc";
    }
    return "unknown";
}

#else

namespace {

nvshmem_team_t split_team = NVSHMEM_TEAM_INVALID;
nvshmem_team_config_t split_team_config = {};

}  // namespace

std::vector<uint8_t> get_unique_id() {
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(unique_id));
    std::memcpy(result.data(), &unique_id, sizeof(unique_id));
    return result;
}

void* alloc(size_t size, size_t alignment) { return nvshmem_align(alignment, size); }

void free(void* allocation) {
    if (allocation != nullptr) {
        nvshmem_free(allocation);
    }
}

void* ptr(void* local_ptr, int pe) { return nvshmem_ptr(local_ptr, pe); }

void barrier(bool with_cpu_sync, const std::optional<platform::DeviceStreamHandle>& stream) {
    if (with_cpu_sync) {
        DEVICE_RUNTIME_CHECK(platform::device_synchronize());
    }
    if (stream.has_value()) {
        nvshmemx_barrier_all_on_stream(stream.value());
    } else {
        nvshmem_barrier_all();
    }
    if (with_cpu_sync) {
        DEVICE_RUNTIME_CHECK(platform::device_synchronize());
    }
}

int init(const std::vector<uint8_t>& root_unique_id,
         int rank,
         int num_ranks,
         int team_split_stride) {
    EP_HOST_ASSERT(root_unique_id.size() == sizeof(nvshmemx_uniqueid_t));
    nvshmemx_uniqueid_t unique_id;
    std::memcpy(&unique_id, root_unique_id.data(), root_unique_id.size());
    nvshmemx_init_attr_t attributes;
    nvshmemx_set_attr_uniqueid_args(rank, num_ranks, &unique_id, &attributes);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attributes);

    if (team_split_stride > 0 && num_ranks > team_split_stride) {
        EP_HOST_ASSERT(split_team == NVSHMEM_TEAM_INVALID);
        EP_HOST_ASSERT(num_ranks % team_split_stride == 0);
        EP_HOST_ASSERT(nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD,
                                                  rank % team_split_stride,
                                                  team_split_stride,
                                                  num_ranks / team_split_stride,
                                                  &split_team_config,
                                                  0,
                                                  &split_team) == 0);
        EP_HOST_ASSERT(split_team != NVSHMEM_TEAM_INVALID);
    }

    barrier(true);
    return nvshmem_my_pe();
}

void finalize() {
    barrier(true);
    if (split_team != NVSHMEM_TEAM_INVALID) {
        nvshmem_team_destroy(split_team);
        split_team = NVSHMEM_TEAM_INVALID;
    }
    nvshmem_finalize();
}

void int32_allreduce(int32_t* data, size_t nelems, platform::DeviceStreamHandle stream) {
    EP_HOST_ASSERT(nvshmemx_int32_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, data, data, nelems, stream) == 0);
}

void int32_fcollect(int32_t* dest,
                    const int32_t* src,
                    size_t nelems_per_pe,
                    platform::DeviceStreamHandle stream) {
    EP_HOST_ASSERT(
        nvshmemx_int32_fcollect_on_stream(NVSHMEM_TEAM_WORLD, dest, src, nelems_per_pe, stream) == 0);
}

const char* backend_name() { return "NVSHMEM"; }

int my_pe() { return nvshmem_my_pe(); }

int num_pes() { return nvshmem_n_pes(); }

const char* transport_name() { return "nvshmem"; }

#endif

}  // namespace ultra_ep::shmem
