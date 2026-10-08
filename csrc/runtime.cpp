#include "runtime.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ultra_ep::runtime {

bool is_runtime_initialized = false;

int rank_idx = -1, nvl_rank_idx = -1, rdma_rank_idx = -1;
int num_ranks = 0, num_nvl_ranks = 0, num_rdma_ranks = 0;
int device_id = -1, num_device_sms = 0;

platform::DeviceStream get_global_comm_stream() {
    static std::optional<platform::DeviceStream> comm_stream = std::nullopt;
    if (not comm_stream.has_value())
        comm_stream = platform::get_stream_from_pool(true);
    return comm_stream.value();
}

platform::DeviceStream get_global_relay_stream() {
    static std::optional<platform::DeviceStream> relay_stream = std::nullopt;
    if (not relay_stream.has_value())
        relay_stream = platform::get_stream_from_pool(true);
    return relay_stream.value();
}

pybind11::bytes get_local_shmem_unique_id(const int& rank) {
    EP_HOST_ASSERT(rank == 0 and "Only rank 0 can get SHMEM unique ID");
    const auto unique_id = shmem::get_unique_id();
    return pybind11::bytes(reinterpret_cast<const char*>(unique_id.data()), unique_id.size());
}

void init_runtime(const int& rank_idx_,
                  const int& num_ranks_,
                  const int& max_nvl_peers_,
                  const pybind11::bytes& root_unique_id) {
    std::string root_unique_id_str = root_unique_id;
    std::vector<uint8_t> root_unique_id_bytes(root_unique_id_str.begin(), root_unique_id_str.end());
    EP_HOST_ASSERT(rank_idx_ == shmem::init(root_unique_id_bytes, rank_idx_, num_ranks_, 0));

    // Support both nvl and rdma ranks
    num_ranks = num_ranks_;
    num_nvl_ranks = max_nvl_peers_;
    EP_HOST_ASSERT(num_nvl_ranks <= kernels::kMaxNvlDomainSize);
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0);
    num_rdma_ranks = num_ranks / num_nvl_ranks;
    rank_idx = rank_idx_;
    nvl_rank_idx = rank_idx_ % num_nvl_ranks;
    rdma_rank_idx = rank_idx_ / num_nvl_ranks;

    // Get device info
    DEVICE_RUNTIME_CHECK(platform::get_device(&device_id));
    platform::DeviceProperties device_prop = {};
    DEVICE_RUNTIME_CHECK(platform::get_device_properties(&device_prop, device_id));
    num_device_sms = device_prop.multiProcessorCount;

    // Available to create buffers
    is_runtime_initialized = true;
}

void destroy() {
    EP_HOST_ASSERT(is_runtime_initialized);

    shmem::finalize();

    // Cannot use anymore
    rank_idx = nvl_rank_idx = rdma_rank_idx = -1;
    num_ranks = num_nvl_ranks = num_rdma_ranks = 0;
    is_runtime_initialized = false;
}

void register_apis(pybind11::module_& m) {
    m.def("get_local_shmem_unique_id", &get_local_shmem_unique_id);
    // Keep the old binding for callers compiled against the CUDA-only API.
    m.def("get_local_nvshmem_unique_id", &get_local_shmem_unique_id);
    m.def("get_shmem_backend_name", []() { return shmem::backend_name(); });
    m.def("get_shmem_runtime_info", []() {
        EP_HOST_ASSERT(is_runtime_initialized && "SHMEM runtime is not initialized");
        pybind11::dict info;
        info["backend"] = shmem::backend_name();
        info["transport"] = shmem::transport_name();
        info["pe"] = shmem::my_pe();
        info["num_pes"] = shmem::num_pes();
        info["scale_up_rank"] = nvl_rank_idx;
        info["scale_up_size"] = num_nvl_ranks;
        info["domain_index"] = rdma_rank_idx;
        info["num_domains"] = num_rdma_ranks;
        return info;
    });
    m.def("run_gda_p2p_put_probe",
          [](int sender_pe,
             int target_pe,
             uint64_t magic,
             bool use_workgroup,
             bool use_private_context) {
              EP_HOST_ASSERT(is_runtime_initialized && "SHMEM runtime is not initialized");
              return shmem::gda_p2p_put_probe(sender_pe,
                                               target_pe,
                                               platform::get_current_stream().stream(),
                                               magic,
                                               use_workgroup,
                                               use_private_context);
          },
          pybind11::arg("sender_pe"),
          pybind11::arg("target_pe"),
          pybind11::arg("magic") = UINT64_C(0x554c545241455031),
          pybind11::arg("use_workgroup") = false,
          pybind11::arg("use_private_context") = true);
    m.def("run_shmem_wg_barrier_probe", []() {
        EP_HOST_ASSERT(is_runtime_initialized && "SHMEM runtime is not initialized");
        shmem::wg_barrier_probe(platform::get_current_stream().stream());
    });
    m.def("is_runtime_initialized", []() { return is_runtime_initialized; });
    m.def("init_runtime", &init_runtime);
}

}  // namespace ultra_ep::runtime
