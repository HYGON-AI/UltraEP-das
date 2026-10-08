#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "../platform/runtime.hpp"

namespace ultra_ep::shmem {

std::vector<uint8_t> get_unique_id();

void* alloc(size_t size, size_t alignment);

void free(void* ptr);

// Return a directly accessible pointer to a symmetric allocation on PE `pe`.
// A null result means that the selected rocSHMEM/NVSHMEM transport does not
// expose direct load/store access to that PE.
void* ptr(void* local_ptr, int pe);

void barrier(bool with_cpu_sync = false,
             const std::optional<platform::DeviceStreamHandle>& stream = std::nullopt);

int init(const std::vector<uint8_t>& root_unique_id,
         int rank,
         int num_ranks,
         int team_split_stride);

void finalize();

void int32_allreduce(int32_t* ptr, size_t nelems, platform::DeviceStreamHandle stream);

void int32_fcollect(int32_t* dest,
                    const int32_t* src,
                    size_t nelems_per_pe,
                    platform::DeviceStreamHandle stream);

// Diagnostic-only 8-byte put through the selected SHMEM transport.  All PEs
// must call this function in the same order.  With `use_workgroup` it uses the
// same non-blocking work-group put primitive as the IPC fcollect path.
// `use_private_context` selects the WG-private context used by UltraEP's
// fcollect wrapper or rocSHMEM's default context.  It returns the local
// symmetric slot after `sender_pe` has issued a put to `target_pe`; only
// target_pe is expected to return `magic`.
uint64_t gda_p2p_put_probe(int sender_pe,
                           int target_pe,
                           platform::DeviceStreamHandle stream,
                           uint64_t magic,
                           bool use_workgroup,
                           bool use_private_context);

// Diagnostic-only device-side WG barrier.  This isolates rocSHMEM's
// collective synchronization storage from the data-transfer portion of
// fcollect.
void wg_barrier_probe(platform::DeviceStreamHandle stream);

const char* backend_name();

// Runtime diagnostics used by the multi-node bootstrap test.  `transport_name`
// reports the selected rocSHMEM transport (GDA/RO/IPC), not just the compiled
// SHMEM API family.
int my_pe();
int num_pes();
const char* transport_name();

}  // namespace ultra_ep::shmem
