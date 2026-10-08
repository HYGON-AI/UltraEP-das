#include "ultra_ep.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>

namespace ultra_ep {

namespace {

int64_t checked_num_bytes(const int64_t numel, const int elem_bytes) {
    EP_HOST_ASSERT(numel >= 0);
    EP_HOST_ASSERT(elem_bytes > 0);
    return numel * static_cast<int64_t>(elem_bytes);
}

int64_t align_up_bytes(const int64_t value, const int64_t alignment) {
    EP_HOST_ASSERT(value >= 0 && alignment > 0);
    return ((value + alignment - 1) / alignment) * alignment;
}

// rocSHMEM provides an on-stream barrier, but some GDA/SHCA stacks do not
// reliably make progress when it is queued immediately after a device WG
// fcollect.  The host fence first drains the fcollect kernel, then performs a
// host rocSHMEM barrier.  This is intentionally the multi-node default: the
// affected path updates placement infrequently and correctness is more
// important than preserving a fully asynchronous control-plane update.
//
// ULTRA_EP_MULTI_NODE_FCOLLECT_FENCE accepts:
//   host   (default): device synchronize + host barrier + synchronize
//   stream           : previous barrier_all_on_stream implementation
//   none             : diagnostic only; unsafe for normal multi-node use
void finish_multi_node_fcollect(platform::DeviceStreamHandle stream) {
    const char* configured_mode = std::getenv("ULTRA_EP_MULTI_NODE_FCOLLECT_FENCE");
    const char* mode = configured_mode == nullptr ? "host" : configured_mode;
    const bool trace = std::getenv("ULTRA_EP_MULTI_NODE_PLACEMENT_TRACE") != nullptr;

    if (std::strcmp(mode, "host") == 0) {
        if (trace) {
            std::fprintf(stderr, "[UltraEP rank=%d] fcollect queued; waiting for device completion\n", runtime::rank_idx);
            std::fflush(stderr);
        }
        DEVICE_RUNTIME_CHECK(platform::device_synchronize());
        if (trace) {
            std::fprintf(stderr, "[UltraEP rank=%d] fcollect complete; entering host barrier\n", runtime::rank_idx);
            std::fflush(stderr);
        }
        shmem::barrier(false);
        DEVICE_RUNTIME_CHECK(platform::device_synchronize());
        if (trace) {
            std::fprintf(stderr, "[UltraEP rank=%d] host barrier complete\n", runtime::rank_idx);
            std::fflush(stderr);
        }
        return;
    }
    if (std::strcmp(mode, "stream") == 0) {
        shmem::barrier(false, stream);
        return;
    }
    if (std::strcmp(mode, "none") == 0) {
        if (trace) {
            std::fprintf(stderr, "[UltraEP rank=%d] WARNING: fcollect fence disabled\n", runtime::rank_idx);
            std::fflush(stderr);
        }
        return;
    }
    EP_HOST_ASSERT(false && "ULTRA_EP_MULTI_NODE_FCOLLECT_FENCE must be host, stream, or none");
}

void fill_task_build_config(kernels::TaskBuildConfig& config,
                            int num_local_master_experts,
                            int num_local_physical_experts,
                            int num_local_redundant_experts,
                            int64_t expert_fc1_numel,
                            int64_t expert_fc2_numel,
                            int64_t expert_total_numel,
                            int64_t expert_fc1_weight_scale_numel,
                            int64_t expert_fc2_weight_scale_numel,
                            int64_t expert_weight_scale_total_numel,
                            int64_t expert_fc1_weight_scale_bytes,
                            int64_t expert_fc2_weight_scale_bytes,
                            int64_t expert_weight_scale_fc2_offset_bytes,
                            int64_t expert_weight_scale_stride_bytes,
                            int weight_data_element_bytes,
                            int weight_scale_element_bytes,
                            int weight_sync_plan_mode,
                            int weight_sync_relay_min_replicas,
                            int weight_sync_relay_max_relays,
                            int weight_sync_relay_min_fanout_gain) {
    config = {};
    config.rank_idx = runtime::rank_idx;
    config.nvl_rank_idx = runtime::nvl_rank_idx;
    config.num_nvl_ranks = runtime::num_nvl_ranks;
    config.num_local_master_experts = num_local_master_experts;
    config.num_local_physical_experts = num_local_physical_experts;
    config.num_local_redundant_experts = num_local_redundant_experts;
    config.expert_fc1_numel = expert_fc1_numel;
    config.expert_fc2_numel = expert_fc2_numel;
    config.expert_total_numel = expert_total_numel;
    config.expert_fc1_weight_scale_numel = expert_fc1_weight_scale_numel;
    config.expert_fc2_weight_scale_numel = expert_fc2_weight_scale_numel;
    config.expert_weight_scale_total_numel = expert_weight_scale_total_numel;
    config.expert_fc1_weight_scale_bytes = expert_fc1_weight_scale_bytes;
    config.expert_fc2_weight_scale_bytes = expert_fc2_weight_scale_bytes;
    config.expert_weight_scale_fc2_offset_bytes = expert_weight_scale_fc2_offset_bytes;
    config.expert_weight_scale_stride_bytes = expert_weight_scale_stride_bytes;
    config.weight_data_element_bytes = weight_data_element_bytes;
    config.weight_scale_element_bytes = weight_scale_element_bytes;
    config.max_replicas_dim = runtime::num_ranks;
    config.weight_sync_plan_mode = weight_sync_plan_mode;
    config.weight_sync_relay_min_replicas = weight_sync_relay_min_replicas;
    config.weight_sync_relay_max_relays = weight_sync_relay_max_relays;
    config.weight_sync_relay_min_fanout_gain = weight_sync_relay_min_fanout_gain;
}

int weight_sync_num_shards(const int64_t expert_weight_scale_total_numel) {
    return expert_weight_scale_total_numel > 0 ? 4 : 2;
}

int weight_sync_chunks_for_shards(int64_t expert_fc1_numel,
                                  int64_t expert_fc2_numel,
                                  int64_t expert_fc1_weight_scale_bytes,
                                  int64_t expert_fc2_weight_scale_bytes,
                                  int weight_data_element_bytes) {
    return kernels::weight_sync_num_chunks(
               static_cast<size_t>(checked_num_bytes(expert_fc1_numel, weight_data_element_bytes))) +
        kernels::weight_sync_num_chunks(
               static_cast<size_t>(checked_num_bytes(expert_fc2_numel, weight_data_element_bytes))) +
        kernels::weight_sync_num_chunks(static_cast<size_t>(expert_fc1_weight_scale_bytes)) +
        kernels::weight_sync_num_chunks(static_cast<size_t>(expert_fc2_weight_scale_bytes));
}

int weight_sync_tiles_for_shards(int64_t expert_fc1_numel,
                                 int64_t expert_fc2_numel,
                                 int64_t expert_fc1_weight_scale_bytes,
                                 int64_t expert_fc2_weight_scale_bytes,
                                 int weight_data_element_bytes) {
    return kernels::weight_sync_num_tiles(
               static_cast<size_t>(checked_num_bytes(expert_fc1_numel, weight_data_element_bytes))) +
        kernels::weight_sync_num_tiles(
               static_cast<size_t>(checked_num_bytes(expert_fc2_numel, weight_data_element_bytes))) +
        kernels::weight_sync_num_tiles(static_cast<size_t>(expert_fc1_weight_scale_bytes)) +
        kernels::weight_sync_num_tiles(static_cast<size_t>(expert_fc2_weight_scale_bytes));
}

int max_weight_sync_chunks_per_shard(int64_t expert_fc1_numel,
                                     int64_t expert_fc2_numel,
                                     int64_t expert_fc1_weight_scale_bytes,
                                     int64_t expert_fc2_weight_scale_bytes,
                                     int weight_data_element_bytes) {
    int max_chunks = 0;
    const int chunks[4] = {
        kernels::weight_sync_num_chunks(
            static_cast<size_t>(checked_num_bytes(expert_fc1_numel, weight_data_element_bytes))),
        kernels::weight_sync_num_chunks(
            static_cast<size_t>(checked_num_bytes(expert_fc2_numel, weight_data_element_bytes))),
        kernels::weight_sync_num_chunks(static_cast<size_t>(expert_fc1_weight_scale_bytes)),
        kernels::weight_sync_num_chunks(static_cast<size_t>(expert_fc2_weight_scale_bytes)),
    };
    for (int value : chunks) {
        max_chunks = std::max(max_chunks, value);
    }
    return max_chunks;
}

}  // namespace

// ============================================================================
// GlobalExpertPlacement
// ============================================================================

void GlobalExpertPlacement::init(int num_layers, int P, int L, int R, int device_id) {
    num_layers_ = num_layers;
    p2l_numel = P;
    l2p_numel = L * R;
    lcnts_numel = L;
    quota_numel = L * R;
    quota_prefix_numel = L * R;
    rank_quota_numel = L * R;
    per_layer_data_numel = p2l_numel + l2p_numel + lcnts_numel + quota_numel + quota_prefix_numel;
    per_layer_data_bytes = per_layer_data_numel * static_cast<int>(sizeof(int32_t));

    // Align stride so each layer starts on a 256-byte boundary (good for DMA)
    per_layer_stride_bytes = (per_layer_data_bytes + ALIGNMENT_BYTES - 1) / ALIGNMENT_BYTES * ALIGNMENT_BYTES;
    per_layer_stride_numel = per_layer_stride_bytes / static_cast<int>(sizeof(int32_t));
    total_bytes = num_layers * per_layer_stride_bytes;

    // Allocate GPU buffer. Individual placement initializers set active fields.
    DEVICE_RUNTIME_CHECK(platform::device_malloc(&device_buffer, total_bytes));
    DEVICE_RUNTIME_CHECK(platform::device_memset(device_buffer, 0, total_bytes));
    // Create GPU tensor views with strides
    auto device_opts = torch::TensorOptions().dtype(torch::kInt32).device(torch::Device(torch::kCUDA, device_id));
    physical_to_logical_map =
        torch::from_blob(device_buffer, {num_layers, P}, {per_layer_stride_numel, 1}, device_opts);
    logical_to_physical_map =
        torch::from_blob(device_buffer + p2l_numel, {num_layers, L, R}, {per_layer_stride_numel, R, 1}, device_opts);
    logical_replica_counts = torch::from_blob(
        device_buffer + p2l_numel + l2p_numel, {num_layers, L}, {per_layer_stride_numel, 1}, device_opts);
    logical_instance_quota = torch::from_blob(device_buffer + p2l_numel + l2p_numel + lcnts_numel,
                                              {num_layers, L, R},
                                              {per_layer_stride_numel, R, 1},
                                              device_opts);
    logical_instance_quota_prefix = torch::from_blob(device_buffer + p2l_numel + l2p_numel + lcnts_numel + quota_numel,
                                                     {num_layers, L, R},
                                                     {per_layer_stride_numel, R, 1},
                                                     device_opts);
    rank_quota_prefix = torch::zeros({num_layers, L, R}, device_opts);
}

void GlobalExpertPlacement::cleanup() {
    if (device_buffer != nullptr) {
        DEVICE_RUNTIME_CHECK(platform::device_free(device_buffer));
        device_buffer = nullptr;
    }
    physical_to_logical_map = torch::Tensor();
    logical_to_physical_map = torch::Tensor();
    logical_replica_counts = torch::Tensor();
    logical_instance_quota = torch::Tensor();
    logical_instance_quota_prefix = torch::Tensor();
    rank_quota_prefix = torch::Tensor();
}

std::tuple<int32_t*, int32_t*, int32_t*> GlobalExpertPlacement::get_device_ptrs(int layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int32_t* base = device_buffer + layer_id * per_layer_stride_numel;
    return std::make_tuple(base, base + p2l_numel, base + p2l_numel + l2p_numel);
}

std::tuple<int32_t*, int32_t*, int32_t*> GlobalExpertPlacement::get_quota_ptrs(int layer_id) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers_);
    int32_t* base = device_buffer + layer_id * per_layer_stride_numel;
    int32_t* rank_quota_base =
        rank_quota_prefix.data_ptr<int32_t>() + static_cast<int64_t>(layer_id) * rank_quota_numel;
    return std::make_tuple(base + p2l_numel + l2p_numel + lcnts_numel,
                           base + p2l_numel + l2p_numel + lcnts_numel + quota_numel,
                           rank_quota_base);
}

// ============================================================================
// Manager
// ============================================================================

Manager::Manager(const int& num_layers,
                 const int& num_local_master_experts,
                 const int& num_local_redundant_experts,
                 const int64_t& expert_fc1_numel,
                 const int64_t& expert_fc2_numel,
                 const int& weight_data_element_bytes,
                 const int& weight_scale_element_bytes,
                 const int64_t& expert_fc1_weight_scale_numel,
                 const int64_t& expert_fc2_weight_scale_numel,
                 const int& grad_element_bytes,
                 const bool& is_train,
                 const bool& explicitly_destroy,
                 const bool& legacy_placement,
                 const float& balance_threshold,
                 const bool& quota_locality_aware,
                 const int32_t& quota_min_tokens_per_replica,
                 const bool& quota_allow_zero_master_quota,
                 const float& quota_oracle_eps,
                 const int& quota_kernel_stage,
                 const bool& quota_reroute_interleave,
                 const int& grad_reduce_num_sms,
                 const bool& grad_reduce_deterministic,
                 const int& weight_sync_plan_mode,
                 const int& weight_sync_relay_min_replicas,
                 const int& weight_sync_relay_max_relays,
                 const int& weight_sync_relay_min_fanout_gain)
    : num_layers(num_layers),
      num_local_master_experts(num_local_master_experts),
      num_local_redundant_experts(num_local_redundant_experts),
      num_local_physical_experts(num_local_master_experts + num_local_redundant_experts),
      expert_fc1_numel(expert_fc1_numel),
      expert_fc2_numel(expert_fc2_numel),
      expert_total_numel(expert_fc1_numel + expert_fc2_numel),
      expert_fc1_weight_scale_numel(expert_fc1_weight_scale_numel),
      expert_fc2_weight_scale_numel(expert_fc2_weight_scale_numel),
      expert_weight_scale_total_numel(expert_fc1_weight_scale_numel + expert_fc2_weight_scale_numel),
      expert_fc1_weight_scale_bytes(0),
      expert_fc2_weight_scale_bytes(0),
      expert_weight_scale_fc2_offset_bytes(0),
      expert_weight_scale_stride_bytes(0),
      weight_data_element_bytes(weight_data_element_bytes),
      weight_scale_element_bytes(weight_scale_element_bytes),
      grad_element_bytes(grad_element_bytes),
      is_train(is_train),
      explicitly_destroy(explicitly_destroy),
      legacy_placement_(legacy_placement),
      quota_locality_aware_(quota_locality_aware),
      quota_min_tokens_per_replica_(quota_min_tokens_per_replica),
      quota_allow_zero_master_quota_(quota_allow_zero_master_quota),
      quota_oracle_eps_(quota_oracle_eps),
      quota_kernel_stage_(quota_kernel_stage),
      quota_reroute_interleave_(quota_reroute_interleave),
      grad_reduce_num_sms_(grad_reduce_num_sms),
      grad_reduce_deterministic_(grad_reduce_deterministic),
      weight_sync_plan_mode_(weight_sync_plan_mode),
      weight_sync_relay_min_replicas_(weight_sync_relay_min_replicas),
      weight_sync_relay_max_relays_(weight_sync_relay_max_relays),
      weight_sync_relay_min_fanout_gain_(weight_sync_relay_min_fanout_gain),
      balance_threshold_(balance_threshold),
      // PyTorch's stream pool intentionally retains the native HIP streams.
      // Repeated Manager construction must therefore reuse a fixed pair instead
      // of advancing through the pool and growing device-resident stream state.
      comm_stream(runtime::get_global_comm_stream()),
      relay_stream(runtime::get_global_relay_stream()),
      placement_ready_events_(is_train ? num_layers : 1),
      placement_ready_stream_ids_(is_train ? num_layers : 1, -1),
      _placement_versions(num_layers, 0)

{
    // Common checks
    EP_HOST_ASSERT(runtime::is_runtime_initialized and "Runtime must be initialized before creating Manager");
    EP_HOST_ASSERT(weight_data_element_bytes > 0);
    EP_HOST_ASSERT(weight_scale_element_bytes > 0);
    EP_HOST_ASSERT(grad_element_bytes == static_cast<int>(sizeof(float)) &&
                   "UltraEP currently supports only fp32 gradients");
    EP_HOST_ASSERT(expert_fc1_numel >= 0 && expert_fc2_numel >= 0);
    EP_HOST_ASSERT(expert_fc1_weight_scale_numel >= 0 && expert_fc2_weight_scale_numel >= 0);
    const int64_t expert_fc1_weight_bytes = checked_num_bytes(expert_fc1_numel, weight_data_element_bytes);
    const int64_t expert_fc2_weight_bytes = checked_num_bytes(expert_fc2_numel, weight_data_element_bytes);
    EP_HOST_ASSERT(expert_fc1_weight_bytes % kernels::kNumTMAAlignBytes == 0 &&
                   "fc1 weight shard bytes must be 16-byte aligned for TMA weight_sync");
    EP_HOST_ASSERT(expert_fc2_weight_bytes % kernels::kNumTMAAlignBytes == 0 &&
                   "fc2 weight shard bytes must be 16-byte aligned for TMA weight_sync");
    expert_fc1_weight_scale_bytes = checked_num_bytes(expert_fc1_weight_scale_numel, weight_scale_element_bytes);
    expert_fc2_weight_scale_bytes = checked_num_bytes(expert_fc2_weight_scale_numel, weight_scale_element_bytes);
    expert_weight_scale_fc2_offset_bytes = align_up_bytes(expert_fc1_weight_scale_bytes, kernels::kNumTMAAlignBytes);
    expert_weight_scale_stride_bytes = align_up_bytes(
        expert_weight_scale_fc2_offset_bytes + expert_fc2_weight_scale_bytes, kernels::kNumTMAAlignBytes);
    EP_HOST_ASSERT(weight_sync_plan_mode_ >= static_cast<int>(kernels::WeightSyncPlanMode::kDirect) &&
                   weight_sync_plan_mode_ <= static_cast<int>(kernels::WeightSyncPlanMode::kForceRelay));
    EP_HOST_ASSERT(weight_sync_relay_min_replicas_ >= 0);
    EP_HOST_ASSERT(weight_sync_relay_max_relays_ >= 1);
    EP_HOST_ASSERT(weight_sync_relay_min_fanout_gain_ >= 0);
    EP_HOST_ASSERT(grad_reduce_num_sms_ > 0);
    EP_HOST_ASSERT(grad_reduce_num_sms_ % 2 == 0 && "grad_reduce_num_sms must be even");
    grad_reduce_num_sms_ = std::min(grad_reduce_num_sms_, runtime::num_device_sms);
    EP_HOST_ASSERT((quota_kernel_stage_ == 0 || quota_kernel_stage_ == 1) &&
                   "quota kernel_stage supports only {0,1}; stage 2/3 has been removed");
    num_global_physical_experts = num_local_physical_experts * runtime::num_ranks;
    num_global_logical_experts = num_local_master_experts * runtime::num_ranks;
    _weight_sync_task_capacity = num_local_physical_experts *
        weight_sync_chunks_for_shards(expert_fc1_numel,
                                      expert_fc2_numel,
                                      expert_fc1_weight_scale_bytes,
                                      expert_fc2_weight_scale_bytes,
                                      weight_data_element_bytes);

    // Allocate global placement tensors using contiguous per-layer device buffers.
    int num_ranks = runtime::num_ranks;
    int device_id = runtime::device_id;
    placement.init(num_layers,
                   num_global_physical_experts,
                   num_global_logical_experts,
                   num_ranks,  // max_replicas_dim = num_ranks
                   device_id);

    // Allocate local replica weight data buffer via SHMEM symmetric heap.
    // This enables automatic cross-GPU access within NVL domain.
    const int64_t local_replica_weight_bytes = static_cast<int64_t>(num_local_redundant_experts) *
        checked_num_bytes(expert_total_numel, weight_data_element_bytes);

    local_replica_weight_buffer = shmem::alloc(local_replica_weight_bytes, kernels::kNumTMAAlignBytes);
    EP_HOST_ASSERT(local_replica_weight_buffer != nullptr && "Failed to allocate SHMEM weight buffer");

    auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_id));
    const int64_t expert_total_weight_bytes = checked_num_bytes(expert_total_numel, weight_data_element_bytes);
    local_replica_weight_buffer_tensor = torch::from_blob(local_replica_weight_buffer,
                                                          {num_local_redundant_experts, expert_total_weight_bytes},
                                                          {expert_total_weight_bytes, 1},
                                                          byte_opts);
    local_replica_fc1_weight_buffer_tensor = torch::from_blob(local_replica_weight_buffer,
                                                              {num_local_redundant_experts, expert_fc1_weight_bytes},
                                                              {expert_total_weight_bytes, 1},
                                                              byte_opts);
    local_replica_fc2_weight_buffer_tensor =
        torch::from_blob(reinterpret_cast<uint8_t*>(local_replica_weight_buffer) + expert_fc1_weight_bytes,
                         {num_local_redundant_experts, expert_fc2_weight_bytes},
                         {expert_total_weight_bytes, 1},
                         byte_opts);

    if (expert_weight_scale_total_numel > 0) {
        const int64_t local_replica_weight_scale_bytes =
            static_cast<int64_t>(num_local_redundant_experts) * expert_weight_scale_stride_bytes;
        local_replica_weight_scale_buffer =
            shmem::alloc(local_replica_weight_scale_bytes, kernels::kNumTMAAlignBytes);
        EP_HOST_ASSERT(local_replica_weight_scale_buffer != nullptr &&
                       "Failed to allocate SHMEM weight-scale buffer");
        local_replica_weight_scale_buffer_tensor =
            torch::from_blob(local_replica_weight_scale_buffer,
                             {num_local_redundant_experts, expert_weight_scale_stride_bytes},
                             {expert_weight_scale_stride_bytes, 1},
                             byte_opts);
        local_replica_fc1_weight_scale_buffer_tensor =
            torch::from_blob(local_replica_weight_scale_buffer,
                             {num_local_redundant_experts, expert_fc1_weight_scale_bytes},
                             {expert_weight_scale_stride_bytes, 1},
                             byte_opts);
        local_replica_fc2_weight_scale_buffer_tensor = torch::from_blob(
            reinterpret_cast<uint8_t*>(local_replica_weight_scale_buffer) + expert_weight_scale_fc2_offset_bytes,
            {num_local_redundant_experts, expert_fc2_weight_scale_bytes},
            {expert_weight_scale_stride_bytes, 1},
            byte_opts);
    } else {
        auto byte_opts = torch::TensorOptions().dtype(torch::kUInt8).device(torch::Device(torch::kCUDA, device_id));
        local_replica_weight_scale_buffer_tensor = torch::empty({num_local_redundant_experts, 0}, byte_opts);
        local_replica_fc1_weight_scale_buffer_tensor = torch::empty({num_local_redundant_experts, 0}, byte_opts);
        local_replica_fc2_weight_scale_buffer_tensor = torch::empty({num_local_redundant_experts, 0}, byte_opts);
    }

    const int max_relay_chunks_per_shard = max_weight_sync_chunks_per_shard(expert_fc1_numel,
                                                                            expert_fc2_numel,
                                                                            expert_fc1_weight_scale_bytes,
                                                                            expert_fc2_weight_scale_bytes,
                                                                            weight_data_element_bytes);
    const int64_t local_ready_flag_count = static_cast<int64_t>(num_local_redundant_experts) *
        weight_sync_num_shards(expert_weight_scale_total_numel) * max_relay_chunks_per_shard;
    local_weight_sync_ready_flags = reinterpret_cast<uint64_t*>(
        shmem::alloc(local_ready_flag_count * sizeof(uint64_t), kernels::kNumTMAAlignBytes));
    EP_HOST_ASSERT(local_weight_sync_ready_flags != nullptr &&
                   "Failed to allocate SHMEM ready-flag buffer for relay weight sync");
    if (local_ready_flag_count > 0) {
        DEVICE_RUNTIME_CHECK(platform::device_memset(local_weight_sync_ready_flags, 0, local_ready_flag_count * sizeof(uint64_t)));
    }

    // Grad buffer only needed for training
    if (is_train) {
        int64_t local_replica_grad_bytes = static_cast<int64_t>(num_local_redundant_experts) *
            checked_num_bytes(expert_total_numel, grad_element_bytes);
        local_replica_grad_buffer = shmem::alloc(local_replica_grad_bytes, kernels::kNumTMAAlignBytes);
        EP_HOST_ASSERT(local_replica_grad_buffer != nullptr && "Failed to allocate SHMEM grad buffer");
        auto grad_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(torch::kCUDA, device_id));
        local_replica_grad_buffer_tensor = torch::from_blob(local_replica_grad_buffer,
                                                            {num_local_redundant_experts, expert_total_numel},
                                                            {expert_total_numel, 1},
                                                            grad_opts);
        local_replica_fc1_grad_buffer_tensor = torch::from_blob(local_replica_grad_buffer,
                                                                {num_local_redundant_experts, expert_fc1_numel},
                                                                {expert_total_numel, 1},
                                                                grad_opts);
        local_replica_fc2_grad_buffer_tensor =
            torch::from_blob(reinterpret_cast<float*>(local_replica_grad_buffer) + expert_fc1_numel,
                             {num_local_redundant_experts, expert_fc2_numel},
                             {expert_total_numel, 1},
                             grad_opts);
        local_replica_grad_buffer_tensor.zero_();
    }

    // Synchronize all PEs to ensure buffers are allocated on all ranks
    shmem::barrier(true);

    // Obtain remote pointers via shmem::ptr() for all NVL ranks
    int num_nvl_ranks = runtime::num_nvl_ranks;
    int rdma_rank_idx = runtime::rdma_rank_idx;
    for (int i = 0; i < num_nvl_ranks; ++i) {
        int target_rank = rdma_rank_idx * num_nvl_ranks + i;
        global_replica_weight_buffer_ptrs[i] = shmem::ptr(local_replica_weight_buffer, target_rank);
        EP_HOST_ASSERT(global_replica_weight_buffer_ptrs[i] != nullptr &&
                       "shmem::ptr failed for weight buffer - target PE may not be in same NVL domain");
        if (expert_weight_scale_total_numel > 0) {
            global_replica_weight_scale_buffer_ptrs[i] = shmem::ptr(local_replica_weight_scale_buffer, target_rank);
            EP_HOST_ASSERT(global_replica_weight_scale_buffer_ptrs[i] != nullptr &&
                           "shmem::ptr failed for weight-scale buffer - target PE may not be in same NVL domain");
        }
        global_weight_sync_ready_flag_ptrs[i] =
            reinterpret_cast<uint64_t*>(shmem::ptr(local_weight_sync_ready_flags, target_rank));
        EP_HOST_ASSERT(global_weight_sync_ready_flag_ptrs[i] != nullptr &&
                       "shmem::ptr failed for ready-flag buffer - target PE may not be in same NVL domain");
        if (is_train) {
            global_replica_grad_buffer_ptrs[i] = shmem::ptr(local_replica_grad_buffer, target_rank);
            EP_HOST_ASSERT(global_replica_grad_buffer_ptrs[i] != nullptr &&
                           "shmem::ptr failed for grad buffer - target PE may not be in same NVL domain");
        }
    }

    DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_remote_ready_flag_ptrs, kernels::kMaxNvlDomainSize * sizeof(uint64_t*)));
    DEVICE_RUNTIME_CHECK(platform::device_memcpy(_remote_ready_flag_ptrs,
                                  global_weight_sync_ready_flag_ptrs,
                                  kernels::kMaxNvlDomainSize * sizeof(uint64_t*),
                                  platform::MemcpyKind::HostToDevice));

    // Allocate intermediate buffers for task-build and persistent kernels.
    DEVICE_RUNTIME_CHECK(
        platform::device_malloc((void**)&_grad_reduce_tasks, kernels::kMaxGradReduceTaskCount * sizeof(kernels::GradReduceTask)));
    DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_global_task_or_tile_counter, sizeof(int)));
    const int shared_task_capacity = std::max(kernels::kMaxGradReduceTaskCount, _weight_sync_task_capacity);
    DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_task_tile_offsets, (shared_task_capacity + 1) * sizeof(int)));

    DEVICE_RUNTIME_CHECK(
        platform::device_malloc((void**)&_weight_sync_tasks, _weight_sync_task_capacity * sizeof(kernels::WeightSyncTask)));
    DEVICE_RUNTIME_CHECK(
        platform::device_malloc((void**)&_weight_sync_task_remaining_tiles, _weight_sync_task_capacity * sizeof(int)));
    DEVICE_RUNTIME_CHECK(
        platform::device_malloc((void**)&_relay_weight_sync_tasks, _weight_sync_task_capacity * sizeof(kernels::WeightSyncTask)));
    DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_relay_task_tile_offsets, (_weight_sync_task_capacity + 1) * sizeof(int)));
    DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_relay_task_metadata, 2 * sizeof(int)));
    DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_relay_global_tile_counter, sizeof(int)));
    DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_task_metadata, 2 * sizeof(int)));

    kernels::TaskBuildConfig config_cpu = {};
    fill_task_build_config(config_cpu,
                           num_local_master_experts,
                           num_local_physical_experts,
                           num_local_redundant_experts,
                           expert_fc1_numel,
                           expert_fc2_numel,
                           expert_total_numel,
                           expert_fc1_weight_scale_numel,
                           expert_fc2_weight_scale_numel,
                           expert_weight_scale_total_numel,
                           expert_fc1_weight_scale_bytes,
                           expert_fc2_weight_scale_bytes,
                           expert_weight_scale_fc2_offset_bytes,
                           expert_weight_scale_stride_bytes,
                           weight_data_element_bytes,
                           weight_scale_element_bytes,
                           weight_sync_plan_mode_,
                           weight_sync_relay_min_replicas_,
                           weight_sync_relay_max_relays_,
                           weight_sync_relay_min_fanout_gain_);
    DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_task_build_config, sizeof(kernels::TaskBuildConfig)));
    DEVICE_RUNTIME_CHECK(
        platform::device_memcpy(_task_build_config, &config_cpu, sizeof(kernels::TaskBuildConfig), platform::MemcpyKind::HostToDevice));

    DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_remote_weight_ptrs, kernels::kMaxNvlDomainSize * sizeof(void*)));
    DEVICE_RUNTIME_CHECK(platform::device_memcpy(_remote_weight_ptrs,
                                  global_replica_weight_buffer_ptrs,
                                  kernels::kMaxNvlDomainSize * sizeof(void*),
                                  platform::MemcpyKind::HostToDevice));
    if (expert_weight_scale_total_numel > 0) {
        DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_remote_weight_scale_ptrs, kernels::kMaxNvlDomainSize * sizeof(void*)));
        DEVICE_RUNTIME_CHECK(platform::device_memcpy(_remote_weight_scale_ptrs,
                                      global_replica_weight_scale_buffer_ptrs,
                                      kernels::kMaxNvlDomainSize * sizeof(void*),
                                      platform::MemcpyKind::HostToDevice));
    }
    if (is_train) {
        DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_remote_grad_ptrs, kernels::kMaxNvlDomainSize * sizeof(void*)));
        DEVICE_RUNTIME_CHECK(platform::device_memcpy(_remote_grad_ptrs,
                                      global_replica_grad_buffer_ptrs,
                                      kernels::kMaxNvlDomainSize * sizeof(void*),
                                       platform::MemcpyKind::HostToDevice));
    }

    const int max_stage_tiles_per_expert = weight_sync_tiles_for_shards(expert_fc1_numel,
                                                                        expert_fc2_numel,
                                                                        expert_fc1_weight_scale_bytes,
                                                                        expert_fc2_weight_scale_bytes,
                                                                        weight_data_element_bytes);
    _max_ws_total_tiles = num_local_physical_experts * max_stage_tiles_per_expert;

    DEVICE_RUNTIME_CHECK(platform::device_malloc((void**)&_reroute_sparse_counters, num_global_logical_experts * sizeof(int)));

    global_logical_expert_loads =
        reinterpret_cast<int*>(shmem::alloc(num_global_logical_experts * sizeof(int), kernels::kNumTMAAlignBytes));
    expert_loads_per_rank = reinterpret_cast<int32_t*>(
        shmem::alloc(static_cast<size_t>(runtime::num_ranks) * num_global_logical_experts * sizeof(int32_t),
                       kernels::kNumTMAAlignBytes));
    // Initialize default placement (master-only) for all layers so sparse reroute
    // remains valid before the first placement update.
    for (int lid = 0; lid < num_layers; ++lid) {
        auto [p2l_ptr, l2p_ptr, lcnts_ptr] = placement.get_device_ptrs(lid);
        auto [quota_ptr, quota_prefix_ptr, rank_quota_prefix_ptr] = placement.get_quota_ptrs(lid);
        kernels::init_master_placement(p2l_ptr,
                                       l2p_ptr,
                                       lcnts_ptr,
                                       quota_ptr,
                                       quota_prefix_ptr,
                                       rank_quota_prefix_ptr,
                                       platform::get_current_stream().stream(),
                                       num_global_physical_experts,
                                       num_global_logical_experts,
                                       runtime::num_ranks,
                                       num_local_master_experts,
                                       num_local_redundant_experts,
                                       runtime::num_ranks);
    }
    DEVICE_RUNTIME_CHECK(platform::stream_synchronize(platform::get_current_stream().stream()));

    // Ready to use (no IPC handle exchange needed with SHMEM)
    _available = true;
}

Manager::~Manager() noexcept(false) {
    if (!explicitly_destroy) {
        if (_available) {
            destroy();
        }
    } else if (_available) {
        printf("WARNING: destroy() was not called before UltraEP manager destruction, which can leak resources.\n");
        fflush(stdout);
    }
}

void Manager::destroy() {
    EP_HOST_ASSERT(is_available());
    // Synchronize all PEs before cleanup
    shmem::barrier(true);

    // Free SHMEM symmetric heap buffers
    shmem::free(local_replica_weight_buffer);
    local_replica_weight_buffer = nullptr;
    if (local_replica_weight_scale_buffer != nullptr) {
        shmem::free(local_replica_weight_scale_buffer);
        local_replica_weight_scale_buffer = nullptr;
    }
    if (local_weight_sync_ready_flags != nullptr) {
        shmem::free(local_weight_sync_ready_flags);
        local_weight_sync_ready_flags = nullptr;
    }
    if (local_replica_grad_buffer != nullptr) {
        shmem::free(local_replica_grad_buffer);
        local_replica_grad_buffer = nullptr;
    }
    shmem::free(global_logical_expert_loads);
    global_logical_expert_loads = nullptr;
    if (expert_loads_per_rank != nullptr) {
        shmem::free(expert_loads_per_rank);
        expert_loads_per_rank = nullptr;
    }

    // Clear remote pointers
    for (int i = 0; i < runtime::num_nvl_ranks; ++i) {
        global_replica_weight_buffer_ptrs[i] = nullptr;
        global_replica_weight_scale_buffer_ptrs[i] = nullptr;
        global_replica_grad_buffer_ptrs[i] = nullptr;
        global_weight_sync_ready_flag_ptrs[i] = nullptr;
    }

    // Free intermediate CUDA buffers
    DEVICE_RUNTIME_CHECK(platform::device_free(_grad_reduce_tasks));
    DEVICE_RUNTIME_CHECK(platform::device_free(_global_task_or_tile_counter));
    DEVICE_RUNTIME_CHECK(platform::device_free(_task_tile_offsets));
    _grad_reduce_tasks = nullptr;
    _global_task_or_tile_counter = nullptr;
    _task_tile_offsets = nullptr;

    // Free weight sync buffers
    DEVICE_RUNTIME_CHECK(platform::device_free(_weight_sync_tasks));
    DEVICE_RUNTIME_CHECK(platform::device_free(_weight_sync_task_remaining_tiles));
    DEVICE_RUNTIME_CHECK(platform::device_free(_relay_weight_sync_tasks));
    DEVICE_RUNTIME_CHECK(platform::device_free(_relay_task_tile_offsets));
    DEVICE_RUNTIME_CHECK(platform::device_free(_relay_task_metadata));
    DEVICE_RUNTIME_CHECK(platform::device_free(_relay_global_tile_counter));
    _weight_sync_tasks = nullptr;
    _weight_sync_task_remaining_tiles = nullptr;
    _relay_weight_sync_tasks = nullptr;
    _relay_task_tile_offsets = nullptr;
    _relay_task_metadata = nullptr;
    _relay_global_tile_counter = nullptr;
    _weight_sync_task_capacity = 0;
    _weight_sync_epoch = 0;

    // Free task metadata buffer
    DEVICE_RUNTIME_CHECK(platform::device_free(_task_metadata));
    _task_metadata = nullptr;

    // Free device task build buffers
    if (_task_build_config) {
        DEVICE_RUNTIME_CHECK(platform::device_free(_task_build_config));
        _task_build_config = nullptr;
    }
    if (_remote_weight_ptrs) {
        DEVICE_RUNTIME_CHECK(platform::device_free(_remote_weight_ptrs));
        _remote_weight_ptrs = nullptr;
    }
    if (_remote_weight_scale_ptrs) {
        DEVICE_RUNTIME_CHECK(platform::device_free(_remote_weight_scale_ptrs));
        _remote_weight_scale_ptrs = nullptr;
    }
    if (_remote_grad_ptrs) {
        DEVICE_RUNTIME_CHECK(platform::device_free(_remote_grad_ptrs));
        _remote_grad_ptrs = nullptr;
    }
    if (_remote_ready_flag_ptrs) {
        DEVICE_RUNTIME_CHECK(platform::device_free(_remote_ready_flag_ptrs));
        _remote_ready_flag_ptrs = nullptr;
    }

    // Free sparse reroute counters
    DEVICE_RUNTIME_CHECK(platform::device_free(_reroute_sparse_counters));
    _reroute_sparse_counters = nullptr;

    // Free contiguous placement buffers (CPU pinned + GPU)
    placement.cleanup();

    // Free SHMEM runtime
    runtime::destroy();

    // Ready to destroy
    _available = false;
}

void Manager::record_placement_ready(const int layer_id, const platform::DeviceStream& stream) {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    const int slot = placement_sync_slot(layer_id);
    placement_ready_events_[slot] = EventHandle(stream);
    placement_ready_stream_ids_[slot] = static_cast<int64_t>(stream.id());
}

void Manager::wait_for_placement_ready(const int layer_id, const platform::DeviceStream& stream) const {
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    const int slot = placement_sync_slot(layer_id);
    if (!placement_ready_events_[slot].has_value()) {
        return;
    }
    if (placement_ready_stream_ids_[slot] == static_cast<int64_t>(stream.id())) {
        return;
    }
    stream_wait(stream, placement_ready_events_[slot].value());
}

void Manager::update_placement(const int& layer_id, torch::Tensor& routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(routing_map.dim() == 2 && routing_map.size(1) == num_global_logical_experts &&
                   routing_map.dtype() == torch::kBool);

    auto curr_stream = platform::get_current_stream();

    kernels::rmap_local_sum(routing_map.size(0),
                            num_global_logical_experts,
                            routing_map.data_ptr<bool>(),
                            global_logical_expert_loads,
                            curr_stream.stream());

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    auto [logical_instance_quota, logical_instance_quota_prefix, rank_quota_prefix] =
        placement.get_quota_ptrs(layer_id);

    if (legacy_placement_) {
        shmem::int32_allreduce(global_logical_expert_loads, num_global_logical_experts, curr_stream.stream());
        kernels::legacy::solve_placement(global_logical_expert_loads,
                                         nullptr,
                                         physical_to_logical_map,
                                         logical_to_physical_map,
                                         logical_replica_counts,
                                         logical_instance_quota,
                                         logical_instance_quota_prefix,
                                         rank_quota_prefix,
                                         curr_stream.stream(),
                                         num_global_logical_experts,
                                         runtime::num_ranks,
                                         num_local_master_experts,
                                         num_local_redundant_experts,
                                         runtime::num_nvl_ranks,
                                         runtime::num_ranks,
                                         balance_threshold_,
                                         quota_min_tokens_per_replica_,
                                         quota_allow_zero_master_quota_,
                                         quota_locality_aware_,
                                         quota_oracle_eps_,
                                         quota_kernel_stage_);
    } else {
        shmem::int32_fcollect(
            expert_loads_per_rank, global_logical_expert_loads, num_global_logical_experts, curr_stream.stream());
        // The WG fcollect has no host-stream counterpart.  Complete it before
        // consuming its output on a multi-node HCU world.
        if (runtime::num_ranks > runtime::num_nvl_ranks) {
            finish_multi_node_fcollect(curr_stream.stream());
        }
        kernels::reduce_per_rank_loads(expert_loads_per_rank,
                                       global_logical_expert_loads,
                                       runtime::num_ranks,
                                       num_global_logical_experts,
                                       curr_stream.stream());
        kernels::solve_placement(global_logical_expert_loads,
                                 expert_loads_per_rank,
                                 physical_to_logical_map,
                                 logical_to_physical_map,
                                 logical_replica_counts,
                                 logical_instance_quota,
                                 logical_instance_quota_prefix,
                                 rank_quota_prefix,
                                 curr_stream.stream(),
                                 num_global_logical_experts,
                                 runtime::num_ranks,
                                 num_local_master_experts,
                                 num_local_redundant_experts,
                                 runtime::num_nvl_ranks,
                                 runtime::num_ranks,
                                 balance_threshold_,
                                 quota_min_tokens_per_replica_,
                                 quota_allow_zero_master_quota_,
                                 quota_locality_aware_,
                                 quota_oracle_eps_,
                                 quota_kernel_stage_);
    }
    ++_placement_versions[layer_id];
    record_placement_ready(layer_id, curr_stream);
}

void Manager::update_placement_sparse(const int& layer_id, torch::Tensor& topk_ids) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(topk_ids.is_cuda() && topk_ids.dtype() == torch::kInt64);
    EP_HOST_ASSERT(topk_ids.dim() == 2);

    int T = topk_ids.size(0);
    int K = topk_ids.size(1);

    // Use comm_stream for histogram + allreduce (same pattern as update_placement)
    auto compute_stream = platform::get_current_stream();
    stream_wait(comm_stream, compute_stream);

    kernels::topk_local_sum(topk_ids.data_ptr<int64_t>(),
                            T,
                            K,
                            num_global_logical_experts,
                            global_logical_expert_loads,
                            comm_stream.stream());

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    auto [logical_instance_quota, logical_instance_quota_prefix, rank_quota_prefix] =
        placement.get_quota_ptrs(layer_id);

    if (legacy_placement_) {
        shmem::int32_allreduce(global_logical_expert_loads, num_global_logical_experts, comm_stream.stream());
        kernels::legacy::solve_placement(global_logical_expert_loads,
                                         nullptr,
                                         physical_to_logical_map,
                                         logical_to_physical_map,
                                         logical_replica_counts,
                                         logical_instance_quota,
                                         logical_instance_quota_prefix,
                                         rank_quota_prefix,
                                         comm_stream.stream(),
                                         num_global_logical_experts,
                                         runtime::num_ranks,
                                         num_local_master_experts,
                                         num_local_redundant_experts,
                                         runtime::num_nvl_ranks,
                                         runtime::num_ranks,
                                         balance_threshold_,
                                         quota_min_tokens_per_replica_,
                                         quota_allow_zero_master_quota_,
                                         quota_locality_aware_,
                                         quota_oracle_eps_,
                                         quota_kernel_stage_);
    } else {
        shmem::int32_fcollect(
            expert_loads_per_rank, global_logical_expert_loads, num_global_logical_experts, comm_stream.stream());
        if (runtime::num_ranks > runtime::num_nvl_ranks) {
            finish_multi_node_fcollect(comm_stream.stream());
        }
        kernels::reduce_per_rank_loads(expert_loads_per_rank,
                                       global_logical_expert_loads,
                                       runtime::num_ranks,
                                       num_global_logical_experts,
                                       comm_stream.stream());
        kernels::solve_placement(global_logical_expert_loads,
                                 expert_loads_per_rank,
                                 physical_to_logical_map,
                                 logical_to_physical_map,
                                 logical_replica_counts,
                                 logical_instance_quota,
                                 logical_instance_quota_prefix,
                                 rank_quota_prefix,
                                 comm_stream.stream(),
                                 num_global_logical_experts,
                                 runtime::num_ranks,
                                 num_local_master_experts,
                                 num_local_redundant_experts,
                                 runtime::num_nvl_ranks,
                                 runtime::num_ranks,
                                 balance_threshold_,
                                 quota_min_tokens_per_replica_,
                                 quota_allow_zero_master_quota_,
                                 quota_locality_aware_,
                                 quota_oracle_eps_,
                                 quota_kernel_stage_);
    }
    ++_placement_versions[layer_id];
    record_placement_ready(layer_id, comm_stream);
}

void Manager::reroute_sparse(const int& layer_id, torch::Tensor& topk_ids) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(layer_id >= 0 && layer_id < num_layers);
    EP_HOST_ASSERT(topk_ids.is_cuda() && topk_ids.dtype() == torch::kInt64);
    EP_HOST_ASSERT(topk_ids.dim() == 2);

    int T = topk_ids.size(0);
    int K = topk_ids.size(1);
    auto stream = platform::get_current_stream();
    wait_for_placement_ready(layer_id, stream);

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    (void)physical_to_logical_map;

    if (!legacy_placement_) {
        const int32_t* rank_quota_prefix = std::get<2>(placement.get_quota_ptrs(layer_id));
        kernels::run_sparse_reroute_quota(topk_ids.data_ptr<int64_t>(),
                                          logical_to_physical_map,
                                          logical_replica_counts,
                                          rank_quota_prefix,
                                          _reroute_sparse_counters,
                                          T,
                                          K,
                                          num_global_logical_experts,
                                          runtime::num_ranks,
                                          stream);
    } else {
        kernels::run_sparse_reroute_round_robin(topk_ids.data_ptr<int64_t>(),
                                                logical_to_physical_map,
                                                logical_replica_counts,
                                                _reroute_sparse_counters,
                                                T,
                                                K,
                                                num_global_logical_experts,
                                                runtime::num_ranks,
                                                stream);
    }
}

std::tuple<torch::Tensor, torch::Tensor> Manager::dense_reroute_forward(const int& layer_id,
                                                                        torch::Tensor& probs,
                                                                        torch::Tensor& routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(routing_map.is_cuda() && probs.is_cuda());
    EP_HOST_ASSERT(routing_map.dtype() == torch::kBool);

    probs = probs.contiguous();
    routing_map = routing_map.contiguous();

    const int T = routing_map.size(0);
    const int L = routing_map.size(1);
    const int P = num_global_physical_experts;
    auto device = routing_map.device();
    auto stream = platform::get_current_stream();
    wait_for_placement_ready(layer_id, stream);

    auto expanded_probs = torch::zeros({T, P}, torch::TensorOptions().dtype(probs.scalar_type()).device(device));
    auto expanded_rmap = torch::zeros({T, P}, torch::TensorOptions().dtype(torch::kBool).device(device));
    void* expand_probs_ptr = expanded_probs.data_ptr();
    bool* expand_rmap_ptr = expanded_rmap.data_ptr<bool>();

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    (void)physical_to_logical_map;

    if (T > 0 && L > 0) {
        constexpr int TILE_T = kernels::kDenseRerouteTileTokens;
        const int num_tiles = (T + TILE_T - 1) / TILE_T;

        int64_t tile_count_numel = static_cast<int64_t>(L) * num_tiles;
        torch::Tensor tile_counts_tensor =
            torch::empty({tile_count_numel}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
        int32_t* tile_counts_ptr = tile_counts_tensor.data_ptr<int32_t>();

        EP_HOST_ASSERT(probs.scalar_type() == torch::kFloat32);

        if (!legacy_placement_) {
            const int32_t* rank_quota_prefix = std::get<2>(placement.get_quota_ptrs(layer_id));
            kernels::run_dense_reroute_forward_quota(routing_map.data_ptr<bool>(),
                                                     probs.data_ptr(),
                                                     logical_to_physical_map,
                                                     logical_replica_counts,
                                                     rank_quota_prefix,
                                                     expand_rmap_ptr,
                                                     expand_probs_ptr,
                                                     tile_counts_ptr,
                                                     T,
                                                     L,
                                                     P,
                                                     runtime::num_ranks,
                                                     quota_reroute_interleave_,
                                                     stream);
        } else {
            kernels::run_dense_reroute_forward_round_robin(routing_map.data_ptr<bool>(),
                                                           probs.data_ptr(),
                                                           logical_to_physical_map,
                                                           logical_replica_counts,
                                                           expand_rmap_ptr,
                                                           expand_probs_ptr,
                                                           tile_counts_ptr,
                                                           T,
                                                           L,
                                                           P,
                                                           runtime::num_ranks,
                                                           stream);
        }
    }

    return std::make_tuple(expanded_probs, expanded_rmap);
}

torch::Tensor Manager::dense_reroute_backward(const int& layer_id,
                                              torch::Tensor& grad_expanded_probs,
                                              torch::Tensor& routing_map,
                                              torch::Tensor& expanded_routing_map) {
    EP_HOST_ASSERT(is_available());
    EP_HOST_ASSERT(grad_expanded_probs.is_cuda() && routing_map.is_cuda());

    grad_expanded_probs = grad_expanded_probs.contiguous();
    routing_map = routing_map.contiguous();

    const int T = routing_map.size(0);
    const int L = routing_map.size(1);
    const int P = grad_expanded_probs.size(1);
    auto device = routing_map.device();
    auto stream = platform::get_current_stream();
    auto grad_probs = torch::zeros({T, L}, torch::TensorOptions().dtype(grad_expanded_probs.dtype()).device(device));
    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    (void)physical_to_logical_map;
    const bool* rerouted_map = expanded_routing_map.data_ptr<bool>();

    if (T > 0 && L > 0) {
        EP_HOST_ASSERT(grad_expanded_probs.scalar_type() == torch::kFloat32);

        kernels::run_dense_reroute_backward(grad_expanded_probs.data_ptr(),
                                            routing_map.data_ptr<bool>(),
                                            rerouted_map,
                                            logical_to_physical_map,
                                            logical_replica_counts,
                                            grad_probs.data_ptr(),
                                            T,
                                            L,
                                            P,
                                            runtime::num_ranks,
                                            stream);
    }

    return grad_probs;
}

void Manager::set_weight_sync_plan_mode(const int& plan_mode) {
    EP_HOST_ASSERT(plan_mode >= static_cast<int>(kernels::WeightSyncPlanMode::kDirect) &&
                   plan_mode <= static_cast<int>(kernels::WeightSyncPlanMode::kForceRelay));
    DEVICE_RUNTIME_CHECK(platform::device_synchronize());
    weight_sync_plan_mode_ = plan_mode;
    _weight_sync_task_cache_valid = false;
    auto stream = platform::get_current_stream();
    if (local_weight_sync_ready_flags != nullptr) {
        const int max_relay_chunks_per_shard = max_weight_sync_chunks_per_shard(expert_fc1_numel,
                                                                                expert_fc2_numel,
                                                                                expert_fc1_weight_scale_bytes,
                                                                                expert_fc2_weight_scale_bytes,
                                                                                weight_data_element_bytes);
        const int64_t local_ready_flag_count = static_cast<int64_t>(num_local_redundant_experts) *
            weight_sync_num_shards(expert_weight_scale_total_numel) * max_relay_chunks_per_shard;
        DEVICE_RUNTIME_CHECK(platform::device_memset_async(
            local_weight_sync_ready_flags, 0, local_ready_flag_count * sizeof(uint64_t), stream.stream()));
    }
    kernels::TaskBuildConfig config_cpu = {};
    fill_task_build_config(config_cpu,
                           num_local_master_experts,
                           num_local_physical_experts,
                           num_local_redundant_experts,
                           expert_fc1_numel,
                           expert_fc2_numel,
                           expert_total_numel,
                           expert_fc1_weight_scale_numel,
                           expert_fc2_weight_scale_numel,
                           expert_weight_scale_total_numel,
                           expert_fc1_weight_scale_bytes,
                           expert_fc2_weight_scale_bytes,
                           expert_weight_scale_fc2_offset_bytes,
                           expert_weight_scale_stride_bytes,
                           weight_data_element_bytes,
                           weight_scale_element_bytes,
                           weight_sync_plan_mode_,
                           weight_sync_relay_min_replicas_,
                           weight_sync_relay_max_relays_,
                           weight_sync_relay_min_fanout_gain_);
    DEVICE_RUNTIME_CHECK(platform::device_memcpy_async(
        _task_build_config, &config_cpu, sizeof(kernels::TaskBuildConfig), platform::MemcpyKind::HostToDevice, stream.stream()));
    DEVICE_RUNTIME_CHECK(platform::stream_synchronize(stream.stream()));
}

void Manager::set_weight_sync_launch_config(const int& hip_copy_mode,
                                            const int& hip_threads_per_block,
                                            const int& hip_cta_multiplier,
                                            const int& hip_lds_waves_per_destination) {
    EP_HOST_ASSERT(hip_copy_mode == 0 || hip_copy_mode == 1);
    EP_HOST_ASSERT(hip_threads_per_block == 64 || hip_threads_per_block == 128 || hip_threads_per_block == 256);
    EP_HOST_ASSERT(hip_cta_multiplier > 0);
    EP_HOST_ASSERT(hip_lds_waves_per_destination == 1 || hip_lds_waves_per_destination == 2 ||
                   hip_lds_waves_per_destination == 4);
    // The Python API only invokes this at a completed iteration boundary.
    DEVICE_RUNTIME_CHECK(platform::device_synchronize());
    weight_sync_runtime_config_.enabled = true;
    weight_sync_runtime_config_.hip_copy_mode = hip_copy_mode;
    weight_sync_runtime_config_.hip_threads_per_block = hip_threads_per_block;
    weight_sync_runtime_config_.hip_cta_multiplier = hip_cta_multiplier;
    weight_sync_runtime_config_.hip_lds_waves_per_destination = hip_lds_waves_per_destination;
}

void Manager::set_grad_reduce_deterministic(const bool& deterministic, const int& grad_reduce_num_sms) {
    EP_HOST_ASSERT(grad_reduce_num_sms > 0);
    EP_HOST_ASSERT(grad_reduce_num_sms % 2 == 0 && "grad_reduce_num_sms must be even");
    grad_reduce_deterministic_ = deterministic;
    grad_reduce_num_sms_ = std::min(grad_reduce_num_sms, runtime::num_device_sms);
}

std::optional<EventHandle> Manager::grad_reduce(const int& layer_id,
                                                torch::Tensor& local_master_fc1_grad_ptr_tensor,
                                                torch::Tensor& local_master_fc2_grad_ptr_tensor,
                                                std::optional<EventHandle>& previous_event,
                                                bool async) {
    EP_HOST_ASSERT(is_available());

    auto compute_stream = platform::get_current_stream();
    std::optional<EventHandle> event;
    // Wait for previous event to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    EP_HOST_ASSERT(local_master_fc1_grad_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc2_grad_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc1_grad_ptr_tensor.numel() == num_local_master_experts);
    EP_HOST_ASSERT(local_master_fc2_grad_ptr_tensor.numel() == num_local_master_experts);

    EP_HOST_ASSERT(local_master_fc1_grad_ptr_tensor.is_cuda());
    EP_HOST_ASSERT(local_master_fc2_grad_ptr_tensor.is_cuda());
    int64_t* local_master_fc1_grad_ptrs = local_master_fc1_grad_ptr_tensor.data_ptr<int64_t>();
    int64_t* local_master_fc2_grad_ptrs = local_master_fc2_grad_ptr_tensor.data_ptr<int64_t>();

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    kernels::build_grad_reduce_tasks(_task_build_config,
                                     physical_to_logical_map,
                                     logical_to_physical_map,
                                     logical_replica_counts,
                                     _remote_grad_ptrs,
                                     local_master_fc1_grad_ptrs,
                                     local_master_fc2_grad_ptrs,
                                     _grad_reduce_tasks,
                                     _task_tile_offsets,
                                     _task_metadata,
                                     _global_task_or_tile_counter,
                                     comm_stream);

    kernels::run_grad_reduce(_grad_reduce_tasks,
                             _task_tile_offsets,
                             _task_metadata,
                             _global_task_or_tile_counter,
                             comm_stream,
                             grad_reduce_num_sms_,
                             grad_reduce_deterministic_);

    // Wait streams
    if (async) {
        event = EventHandle(comm_stream);
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    return event;
}

std::optional<EventHandle> Manager::weight_sync(const int& layer_id,
                                                torch::Tensor& local_master_fc1_weight_ptr_tensor,
                                                torch::Tensor& local_master_fc2_weight_ptr_tensor,
                                                torch::Tensor& local_master_fc1_weight_scale_ptr_tensor,
                                                torch::Tensor& local_master_fc2_weight_scale_ptr_tensor,
                                                std::optional<EventHandle>& previous_event,
                                                bool async) {
    EP_HOST_ASSERT(is_available());

    auto compute_stream = platform::get_current_stream();
    std::optional<EventHandle> event;
    // Wait for previous event to be finished
    const bool enable_relay_stages = weight_sync_plan_mode_ != static_cast<int>(kernels::WeightSyncPlanMode::kDirect) &&
        num_local_redundant_experts > 0 && runtime::num_nvl_ranks > 2;
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
        if (enable_relay_stages) {
            stream_wait(relay_stream, previous_event.value());
        }
    } else {
        stream_wait(comm_stream, compute_stream);
        if (enable_relay_stages) {
            stream_wait(relay_stream, compute_stream);
        }
    }

    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.dtype() == torch::kInt64);
    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.numel() == num_local_master_experts);
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.numel() == num_local_master_experts);
    if (expert_weight_scale_total_numel > 0) {
        EP_HOST_ASSERT(local_master_fc1_weight_scale_ptr_tensor.dtype() == torch::kInt64);
        EP_HOST_ASSERT(local_master_fc2_weight_scale_ptr_tensor.dtype() == torch::kInt64);
        EP_HOST_ASSERT(local_master_fc1_weight_scale_ptr_tensor.numel() == num_local_master_experts);
        EP_HOST_ASSERT(local_master_fc2_weight_scale_ptr_tensor.numel() == num_local_master_experts);
        EP_HOST_ASSERT(local_master_fc1_weight_scale_ptr_tensor.is_cuda());
        EP_HOST_ASSERT(local_master_fc2_weight_scale_ptr_tensor.is_cuda());
    }
    const uint64_t current_epoch = ++_weight_sync_epoch;
    bool launched_stage2 = false;

    EP_HOST_ASSERT(local_master_fc1_weight_ptr_tensor.is_cuda());
    EP_HOST_ASSERT(local_master_fc2_weight_ptr_tensor.is_cuda());
    int64_t* local_master_fc1_weight_ptrs = local_master_fc1_weight_ptr_tensor.data_ptr<int64_t>();
    int64_t* local_master_fc2_weight_ptrs = local_master_fc2_weight_ptr_tensor.data_ptr<int64_t>();
    int64_t* local_master_fc1_weight_scale_ptrs =
        expert_weight_scale_total_numel > 0 ? local_master_fc1_weight_scale_ptr_tensor.data_ptr<int64_t>() : nullptr;
    int64_t* local_master_fc2_weight_scale_ptrs =
        expert_weight_scale_total_numel > 0 ? local_master_fc2_weight_scale_ptr_tensor.data_ptr<int64_t>() : nullptr;

    auto [physical_to_logical_map, logical_to_physical_map, logical_replica_counts] =
        placement.get_device_ptrs(layer_id);
    const bool can_reuse_task_plan = _weight_sync_task_cache_valid &&
        _weight_sync_task_cache_layer_id == layer_id &&
        _weight_sync_task_cache_placement_version == _placement_versions[layer_id] &&
        _weight_sync_task_cache_fc1_ptrs == local_master_fc1_weight_ptrs &&
        _weight_sync_task_cache_fc2_ptrs == local_master_fc2_weight_ptrs &&
        _weight_sync_task_cache_fc1_scale_ptrs == local_master_fc1_weight_scale_ptrs &&
        _weight_sync_task_cache_fc2_scale_ptrs == local_master_fc2_weight_scale_ptrs;
    if (can_reuse_task_plan) {
        kernels::reset_weight_sync_task_state(_weight_sync_tasks,
                                              _task_metadata,
                                              _weight_sync_task_remaining_tiles,
                                              _global_task_or_tile_counter,
                                              _relay_global_tile_counter,
                                              comm_stream);
    } else {
        kernels::build_weight_sync_task_lists(_task_build_config,
                                              physical_to_logical_map,
                                              logical_to_physical_map,
                                              logical_replica_counts,
                                              _remote_weight_ptrs,
                                              _remote_weight_scale_ptrs,
                                              local_master_fc1_weight_ptrs,
                                              local_master_fc2_weight_ptrs,
                                              local_master_fc1_weight_scale_ptrs,
                                              local_master_fc2_weight_scale_ptrs,
                                              reinterpret_cast<uint8_t*>(local_replica_weight_buffer),
                                              reinterpret_cast<uint8_t*>(local_replica_weight_scale_buffer),
                                              _weight_sync_tasks,
                                              _task_tile_offsets,
                                              _task_metadata,
                                              _weight_sync_task_remaining_tiles,
                                              _global_task_or_tile_counter,
                                              _relay_weight_sync_tasks,
                                              _relay_task_tile_offsets,
                                              _relay_task_metadata,
                                              _relay_global_tile_counter,
                                              comm_stream);
        _weight_sync_task_cache_valid = true;
        _weight_sync_task_cache_layer_id = layer_id;
        _weight_sync_task_cache_placement_version = _placement_versions[layer_id];
        _weight_sync_task_cache_fc1_ptrs = local_master_fc1_weight_ptrs;
        _weight_sync_task_cache_fc2_ptrs = local_master_fc2_weight_ptrs;
        _weight_sync_task_cache_fc1_scale_ptrs = local_master_fc1_weight_scale_ptrs;
        _weight_sync_task_cache_fc2_scale_ptrs = local_master_fc2_weight_scale_ptrs;
    }
    EventHandle task_build_ready(comm_stream);

    kernels::run_weight_sync(_weight_sync_tasks,
                             _task_tile_offsets,
                             _task_metadata,
                             _global_task_or_tile_counter,
                             _weight_sync_task_remaining_tiles,
                             local_weight_sync_ready_flags,
                             _remote_ready_flag_ptrs,
                             current_epoch,
                             comm_stream,
                             runtime::num_device_sms,
                             runtime::num_nvl_ranks,
                             _max_ws_total_tiles,
                             2,
                             weight_sync_runtime_config_);

    if (enable_relay_stages) {
        stream_wait(relay_stream, task_build_ready);
        kernels::run_weight_sync(_relay_weight_sync_tasks,
                                 _relay_task_tile_offsets,
                                 _relay_task_metadata,
                                 _relay_global_tile_counter,
                                 nullptr,
                                 local_weight_sync_ready_flags,
                                 _remote_ready_flag_ptrs,
                                 current_epoch,
                                 relay_stream,
                                 runtime::num_device_sms,
                                 runtime::num_nvl_ranks,
                                 _max_ws_total_tiles,
                                 1,
                                 weight_sync_runtime_config_);
        launched_stage2 = true;
    }

    if (launched_stage2) {
        stream_wait(comm_stream, relay_stream);
    }

    // Wait streams
    if (async) {
        event = EventHandle(comm_stream);
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    return event;
}

}  // namespace ultra_ep
