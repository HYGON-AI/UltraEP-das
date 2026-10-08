#pragma once

// Deprecated compatibility include. New code should include shmem.cuh and use
// ultra_ep::shmem so the same source works with NVSHMEM and rocSHMEM.
#include "shmem.cuh"

namespace ultra_ep {
namespace nvshmem = shmem;
}  // namespace ultra_ep
