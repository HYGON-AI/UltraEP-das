import os
import random
import statistics

import torch
import torch.distributed as dist
from typing import List, Optional

import ultra_ep._C as _C
from .config import load_tuning_from_env
from .runtime import init_runtime
from .event import EventHandle
from .autotune import AutotuneConfig, RuntimeAutotuner, WeightSyncLaunchConfig
from .profiling import ExpertLoadProfiler, load_profile_config
from .reroute import _DenseRerouteFunction
from .util import (
    profile_physical_loads_from_quota_triton,
    check_dtype,
    dtype_element_size,
)


class Manager:
    def __init__(
        self,
        group: dist.ProcessGroup,
        num_layers: int,
        num_local_master_experts: int,
        num_local_redundant_experts: int,
        expert_fc1_numel: int,
        expert_fc2_numel: int,
        is_train: bool = True,
        explicitly_destroy: bool = False,
        max_microbatches: int = 1,
        legacy_placement: bool = False,
        weight_data_dtype: torch.dtype = torch.bfloat16,
        weight_scale_dtype: torch.dtype = torch.float32,
        expert_fc1_weight_scale_numel: int = 0,
        expert_fc2_weight_scale_numel: int = 0,
        grad_dtype: torch.dtype = torch.float32,
        autotune: Optional[AutotuneConfig] = None,
    ):
        self._load_profile_config = load_profile_config()
        if self._load_profile_config.enabled and legacy_placement:
            raise AssertionError(
                "UltraEP load profiling is only supported by quota placement; "
                "disable ULTRA_EP_LOAD_PROFILING or legacy_placement."
            )

        self.nvl_domain_size = init_runtime(group)

        self.group = group
        self.id = id(group)
        self.device = torch.cuda.current_device()
        self.num_device_sms = torch.cuda.get_device_properties(self.device).multi_processor_count
        self.rank = group.rank()
        self.num_ranks = group.size()

        self.num_layers = num_layers
        self.num_local_master_experts = num_local_master_experts
        self.num_local_redundant_experts = num_local_redundant_experts
        self.expert_fc1_numel = expert_fc1_numel
        self.expert_fc2_numel = expert_fc2_numel
        self.expert_total_numel = expert_fc1_numel + expert_fc2_numel
        self.expert_fc1_weight_scale_numel = expert_fc1_weight_scale_numel
        self.expert_fc2_weight_scale_numel = expert_fc2_weight_scale_numel
        self.expert_weight_scale_total_numel = (
            expert_fc1_weight_scale_numel + expert_fc2_weight_scale_numel
        )
        self.weight_data_dtype = check_dtype("weight_data_dtype", weight_data_dtype)
        self.weight_scale_dtype = check_dtype("weight_scale_dtype", weight_scale_dtype)
        self.grad_dtype = check_dtype("grad_dtype", grad_dtype)
        if self.grad_dtype is not torch.float32:
            raise ValueError("UltraEP currently supports only fp32 gradients")
        self.weight_data_element_bytes = dtype_element_size(self.weight_data_dtype)
        self.weight_scale_element_bytes = dtype_element_size(self.weight_scale_dtype)
        self.grad_element_bytes = dtype_element_size(self.grad_dtype)
        self.num_local_physical_experts = (
            num_local_master_experts + num_local_redundant_experts
        )
        self.num_global_physical_experts = (
            self.num_local_physical_experts * self.num_ranks
        )
        self.num_global_logical_experts = num_local_master_experts * self.num_ranks
        self.is_train = is_train

        # PP/VPP support: multiple micro-batches may be in-flight simultaneously.
        # Each (real_layer, microbatch_slot) pair gets a unique "virtual layer ID"
        # so that placement maps and reroute buffers don't collide across micro-batches.
        # For DDP-only (max_microbatches=1) the virtual ID equals the real layer ID.
        self.max_microbatches = max(1, max_microbatches)
        self.real_num_alloc_layers = num_layers + 3  # padding for 1-indexed layer IDs
        self.num_alloc_layers = self.real_num_alloc_layers * self.max_microbatches

        # Master weight/grad pointer pools, indexed by real layer.
        self.local_master_fc1_weight_ptr_pool = [None] * self.real_num_alloc_layers
        self.local_master_fc2_weight_ptr_pool = [None] * self.real_num_alloc_layers
        self.local_master_fc1_weight_scale_ptr_pool = [
            None
        ] * self.real_num_alloc_layers
        self.local_master_fc2_weight_scale_ptr_pool = [
            None
        ] * self.real_num_alloc_layers
        self.local_master_fc1_grad_ptr_pool = [None] * self.real_num_alloc_layers
        self.local_master_fc2_grad_ptr_pool = [None] * self.real_num_alloc_layers

        # Per-real-layer micro-batch slot counters (wraps modulo max_microbatches)
        self._mb_counters = [0] * self.real_num_alloc_layers

        tuning = load_tuning_from_env()
        self.grad_reduce_num_sms = tuning.grad_reduce_num_sms
        self.grad_reduce_deterministic = tuning.grad_reduce_deterministic

        # Create cpp handle
        self.explicitly_destroy = explicitly_destroy
        self.legacy_placement = legacy_placement
        self.quota_kernel_stage = tuning.quota_kernel_stage
        self.quota_reroute_interleave = tuning.quota_reroute_interleave
        weight_sync_plan_mode = tuning.weight_sync_plan_mode
        weight_sync_plan_mode_id = tuning.weight_sync_plan_mode_id
        # Direct IPC fan-out is the production default for a small scale-up
        # domain.  Relay remains available behind an explicit opt-in solely for
        # topology/performance experiments; it must not silently alter normal
        # single-node behavior.
        allow_small_domain_relay = os.getenv(
            "ULTRA_EP_ALLOW_SMALL_DOMAIN_RELAY", "0"
        ).lower() in ("1", "true", "yes", "on")
        if (
            self.nvl_domain_size <= 8
            and weight_sync_plan_mode != "direct"
            and not allow_small_domain_relay
        ):
            if self.rank == 0:
                print(
                    "UltraEP: NVL domain size <= 8; forcing weight_sync "
                    f"plan mode to direct (was {weight_sync_plan_mode}).",
                    flush=True,
                )
            weight_sync_plan_mode = "direct"
            weight_sync_plan_mode_id = 0
        self.weight_sync_plan_mode = weight_sync_plan_mode
        self.weight_sync_relay_min_replicas = tuning.weight_sync_relay_min_replicas
        self.weight_sync_relay_max_relays = tuning.weight_sync_relay_max_relays
        self.weight_sync_relay_min_fanout_gain = (
            tuning.weight_sync_relay_min_fanout_gain
        )
        self.reroute_mode = "round_robin" if legacy_placement else "quota"
        self.runtime = _C.Manager(
            self.num_alloc_layers,
            num_local_master_experts,
            num_local_redundant_experts,
            expert_fc1_numel,
            expert_fc2_numel,
            self.weight_data_element_bytes,
            self.weight_scale_element_bytes,
            expert_fc1_weight_scale_numel,
            expert_fc2_weight_scale_numel,
            self.grad_element_bytes,
            is_train,
            explicitly_destroy,
            legacy_placement,
            tuning.balance_threshold,
            tuning.quota_locality_aware,
            tuning.quota_min_tokens_per_replica,
            tuning.quota_allow_zero_master_quota,
            tuning.quota_oracle_eps,
            tuning.quota_kernel_stage,
            tuning.quota_reroute_interleave,
            self.grad_reduce_num_sms,
            self.grad_reduce_deterministic,
            weight_sync_plan_mode_id,
            tuning.weight_sync_relay_min_replicas,
            tuning.weight_sync_relay_max_relays,
            tuning.weight_sync_relay_min_fanout_gain,
        )
        assert self.runtime.is_available()

        # Inert unless explicitly enabled: no native setter, timing event, or
        # synchronization is added to the original Manager path by default.
        self._autotuner = RuntimeAutotuner(autotune or AutotuneConfig())
        self._autotune_ws_wait_events = {}
        self._autotune_grad_wait_events = []

        # Placement maps are device-resident. Tests and diagnostics should reduce
        # metrics on GPU and only materialize small scalar summaries for printing.
        self._physical_to_logical_map: torch.Tensor = (
            self.runtime.get_physical_to_logical_map_tensor()
        )
        self._logical_to_physical_map: torch.Tensor = (
            self.runtime.get_logical_to_physical_map_tensor()
        )
        self._logical_replica_counts: torch.Tensor = (
            self.runtime.get_logical_replica_counts_tensor()
        )
        self._logical_instance_quota: torch.Tensor = (
            self.runtime.get_logical_instance_quota_tensor()
        )
        self._logical_instance_quota_prefix: torch.Tensor = (
            self.runtime.get_logical_instance_quota_prefix_tensor()
        )
        self._rank_quota_prefix: torch.Tensor = (
            self.runtime.get_rank_quota_prefix_tensor()
        )
        # Full buffers keep the original contiguous expert layout for backward
        # compatibility. The fc1/fc2 tensors below are strided views into these
        # buffers: each expert row is contiguous, while rows use the full-expert stride.
        self.local_replica_weight_buffer: torch.Tensor = (
            self.runtime.get_local_replica_weight_buffer_tensor()
            .view(self.weight_data_dtype)
            .view(self.num_local_redundant_experts, self.expert_total_numel)
        )
        self.local_replica_fc1_weight_buffer: torch.Tensor = (
            self.runtime.get_local_replica_fc1_weight_buffer_tensor()
            .view(self.weight_data_dtype)
            .view(self.num_local_redundant_experts, self.expert_fc1_numel)
        )
        self.local_replica_fc2_weight_buffer: torch.Tensor = (
            self.runtime.get_local_replica_fc2_weight_buffer_tensor()
            .view(self.weight_data_dtype)
            .view(self.num_local_redundant_experts, self.expert_fc2_numel)
        )
        self.local_replica_weight_scale_buffer_raw: torch.Tensor = (
            self.runtime.get_local_replica_weight_scale_buffer_tensor()
        )
        raw_fc1_scale_buffer = (
            self.runtime.get_local_replica_fc1_weight_scale_buffer_tensor()
        )
        raw_fc2_scale_buffer = (
            self.runtime.get_local_replica_fc2_weight_scale_buffer_tensor()
        )
        if self.expert_weight_scale_total_numel > 0:
            self.local_replica_fc1_weight_scale_buffer: (
                torch.Tensor
            ) = raw_fc1_scale_buffer.view(self.weight_scale_dtype).view(
                self.num_local_redundant_experts,
                self.expert_fc1_weight_scale_numel,
            )
            self.local_replica_fc2_weight_scale_buffer: (
                torch.Tensor
            ) = raw_fc2_scale_buffer.view(self.weight_scale_dtype).view(
                self.num_local_redundant_experts,
                self.expert_fc2_weight_scale_numel,
            )
        else:
            self.local_replica_fc1_weight_scale_buffer = torch.empty(
                (self.num_local_redundant_experts, 0),
                dtype=self.weight_scale_dtype,
                device=raw_fc1_scale_buffer.device,
            )
            self.local_replica_fc2_weight_scale_buffer = torch.empty(
                (self.num_local_redundant_experts, 0),
                dtype=self.weight_scale_dtype,
                device=raw_fc2_scale_buffer.device,
            )
        # Grad buffer only available in training mode
        if self.is_train:
            self.local_replica_grad_buffer: torch.Tensor = (
                self.runtime.get_local_replica_grad_buffer_tensor()
            )
            self.local_replica_fc1_grad_buffer: torch.Tensor = (
                self.runtime.get_local_replica_fc1_grad_buffer_tensor()
            )
            self.local_replica_fc2_grad_buffer: torch.Tensor = (
                self.runtime.get_local_replica_fc2_grad_buffer_tensor()
            )
        else:
            self.local_replica_grad_buffer = None
            self.local_replica_fc1_grad_buffer = None
            self.local_replica_fc2_grad_buffer = None

        self.check_tensors_blob_from_cpp()
        self._load_profiler = ExpertLoadProfiler(
            config=self._load_profile_config,
            group_rank=self.rank,
            global_rank=dist.get_rank() if dist.is_initialized() else self.rank,
            metadata={
                "ep_group_id": f"{self.id:x}",
                "ep_rank": self.rank,
                "ep_size": self.num_ranks,
                "num_layers": self.num_layers,
                "max_microbatches": self.max_microbatches,
                "num_local_master_experts": self.num_local_master_experts,
                "num_local_redundant_experts": self.num_local_redundant_experts,
                "num_local_physical_experts": self.num_local_physical_experts,
                "num_global_logical_experts": self.num_global_logical_experts,
                "num_global_physical_experts": self.num_global_physical_experts,
                "nvl_domain_size": self.nvl_domain_size,
                "placement_mode": self.reroute_mode,
            },
        )

    @property
    def autotune_enabled(self) -> bool:
        """Whether one-shot runtime autotuning was requested for this Manager."""
        return self._autotuner.enabled

    @property
    def autotune_collecting(self) -> bool:
        """Whether the framework must still feed real iteration observations."""
        return self._autotuner.collecting

    @property
    def autotune_complete(self) -> bool:
        """Whether all enabled runtime tuning phases have committed a result."""
        return self._autotuner.enabled and self._autotuner.complete

    @property
    def autotune_start_iteration(self) -> int:
        """One-based, process-local training iteration at which tuning starts."""
        return self._autotuner.config.start_iteration

    def autotune_needs_iteration_time(self, iteration: int) -> bool:
        """Return whether ``iteration`` runs a real end-to-end candidate."""
        return self._autotuner.needs_iteration_time(iteration)

    def get_autotune_result(self) -> dict:
        """Return a stable public snapshot without exposing RuntimeAutotuner internals."""
        weight_sync = self._autotuner.current_weight_sync
        return {
            "enabled": self._autotuner.enabled,
            "complete": self._autotuner.complete,
            "weight_sync_done": self._autotuner.weight_sync_done,
            "grad_reduce_done": self._autotuner.grad_done,
            "weight_sync": {
                "copy_mode": weight_sync.copy_mode,
                "threads_per_block": weight_sync.threads_per_block,
                "cta_multiplier": weight_sync.cta_multiplier,
                "lds_waves_per_destination": weight_sync.lds_waves_per_destination,
            },
            "grad_reduce_num_sms": self.grad_reduce_num_sms,
        }

    @property
    def physical_to_logical_map(self) -> torch.Tensor:
        return self._physical_to_logical_map

    @property
    def logical_to_physical_map(self) -> torch.Tensor:
        return self._logical_to_physical_map

    @property
    def logical_replica_counts(self) -> torch.Tensor:
        return self._logical_replica_counts

    @property
    def logical_instance_quota(self) -> torch.Tensor:
        return self._logical_instance_quota

    @property
    def logical_instance_quota_prefix(self) -> torch.Tensor:
        return self._logical_instance_quota_prefix

    @property
    def rank_quota_prefix(self) -> torch.Tensor:
        return self._rank_quota_prefix

    def get_quota_tensor(self, layer_id: int) -> torch.Tensor:
        return self._logical_instance_quota[layer_id]

    def get_quota_prefix_tensor(self, layer_id: int) -> torch.Tensor:
        return self._logical_instance_quota_prefix[layer_id]

    def get_rank_quota_prefix_tensor(self, layer_id: int) -> torch.Tensor:
        return self._rank_quota_prefix[layer_id]

    def destroy(self):
        assert self.explicitly_destroy

        self._load_profiler.close()
        if self.runtime is not None:
            self.runtime.destroy()
        self.runtime = None

    def allocate_microbatch_slot(self, real_layer_id: int) -> int:
        """Allocate the next virtual layer ID for this real layer.

        Maps ``(real_layer_id, mb_slot)`` → ``virtual_layer_id`` using:
            ``virtual = real_layer_id * max_microbatches + (counter % max_microbatches)``

        The counter wraps modulo ``max_microbatches``, so once a micro-batch's
        backward completes and frees a slot, that slot can be reused by a later
        micro-batch.  The caller must ensure ``max_microbatches`` is at least as
        large as the peak number of in-flight micro-batches per layer.

        For DDP-only (``max_microbatches == 1``), the virtual ID equals the real
        layer ID and the counter overhead is a single integer increment.
        """
        assert real_layer_id < self.real_num_alloc_layers
        mb_slot = self._mb_counters[real_layer_id] % self.max_microbatches
        self._mb_counters[real_layer_id] += 1
        return real_layer_id * self.max_microbatches + mb_slot

    def _real_layer_id(self, virtual_layer_id: int) -> int:
        """Map a virtual layer ID back to the real layer ID."""
        return virtual_layer_id // self.max_microbatches

    def _profile_physical_loads_from_quota(
        self, layer_id: int, fused: bool = True
    ) -> torch.Tensor:
        l2p = self._logical_to_physical_map[layer_id]
        quota = self._logical_instance_quota[layer_id]
        if fused:
            return profile_physical_loads_from_quota_triton(
                self._physical_to_logical_map[layer_id],
                l2p,
                quota,
            )

        flat_l2p = l2p.reshape(-1)
        flat_quota = quota.reshape(-1)
        valid = flat_l2p >= 0
        post_loads = torch.zeros(
            (self.num_global_physical_experts,),
            dtype=torch.int32,
            device=quota.device,
        )
        post_loads.scatter_add_(
            0,
            flat_l2p.clamp_min(0),
            torch.where(valid, flat_quota, torch.zeros_like(flat_quota)),
        )
        return post_loads

    def construct_local_master_ptr_pool(
        self,
        layer_id: int,
        fc1_weights: List[torch.Tensor],
        fc2_weights: List[torch.Tensor],
        fc1_grads: Optional[List[torch.Tensor]] = None,
        fc2_grads: Optional[List[torch.Tensor]] = None,
        fc1_weight_scales: Optional[List[torch.Tensor]] = None,
        fc2_weight_scales: Optional[List[torch.Tensor]] = None,
    ):
        assert layer_id < self.real_num_alloc_layers
        assert len(fc1_weights) == self.num_local_master_experts
        assert len(fc2_weights) == self.num_local_master_experts

        def check_tensors_dtype(tensors: List[torch.Tensor], dtype: torch.dtype):
            for t in tensors:
                assert (
                    t.dtype == dtype
                ), f"Expected weight/grad dtype {dtype}, got {t.dtype}"

        check_tensors_dtype(fc1_weights, self.weight_data_dtype)
        check_tensors_dtype(fc2_weights, self.weight_data_dtype)

        def _to_dataptr_tensor(tensors: List[torch.Tensor], device) -> torch.Tensor:
            return torch.tensor(
                [t.data_ptr() for t in tensors], dtype=torch.int64, device=device
            ).contiguous()

        weight_or_grad_device = torch.device("cuda", self.device)
        self.local_master_fc1_weight_ptr_pool[layer_id] = _to_dataptr_tensor(
            fc1_weights, device=weight_or_grad_device
        )
        self.local_master_fc2_weight_ptr_pool[layer_id] = _to_dataptr_tensor(
            fc2_weights, device=weight_or_grad_device
        )

        if self.expert_weight_scale_total_numel > 0:
            if fc1_weight_scales is None or fc2_weight_scales is None:
                raise AssertionError(
                    "Weight scale tensors are required when scale numel is non-zero"
                )
            assert len(fc1_weight_scales) == self.num_local_master_experts
            assert len(fc2_weight_scales) == self.num_local_master_experts
            check_tensors_dtype(fc1_weight_scales, self.weight_scale_dtype)
            check_tensors_dtype(fc2_weight_scales, self.weight_scale_dtype)
            self.local_master_fc1_weight_scale_ptr_pool[layer_id] = _to_dataptr_tensor(
                fc1_weight_scales, device=weight_or_grad_device
            )
            self.local_master_fc2_weight_scale_ptr_pool[layer_id] = _to_dataptr_tensor(
                fc2_weight_scales, device=weight_or_grad_device
            )
        else:
            empty_ptrs = torch.empty(0, dtype=torch.int64, device=weight_or_grad_device)
            self.local_master_fc1_weight_scale_ptr_pool[layer_id] = empty_ptrs
            self.local_master_fc2_weight_scale_ptr_pool[layer_id] = empty_ptrs

        if self.is_train:
            assert (
                fc1_grads is not None and fc2_grads is not None
            ), "Grad tensors required in training mode"
            assert len(fc1_grads) == self.num_local_master_experts
            assert len(fc2_grads) == self.num_local_master_experts
            check_tensors_dtype(fc1_grads, self.grad_dtype)
            check_tensors_dtype(fc2_grads, self.grad_dtype)
            self.local_master_fc1_grad_ptr_pool[layer_id] = _to_dataptr_tensor(
                fc1_grads, device=weight_or_grad_device
            )
            self.local_master_fc2_grad_ptr_pool[layer_id] = _to_dataptr_tensor(
                fc2_grads, device=weight_or_grad_device
            )

        # Task plans embed the registered master addresses. Re-registering a
        # layer must invalidate the native cache even if the device allocator
        # happens to reuse the same pointer-tensor storage address.
        self.runtime.invalidate_weight_sync_task_cache()

    def grad_reduce(
        self,
        layer_id: int,
        previous_event: Optional[EventHandle] = None,
        async_finish: bool = False,
    ):
        """Aggregate replica gradients to masters.

        Args:
            layer_id: Virtual layer ID (encodes both real layer and micro-batch
                slot).  Used for placement map lookup in C++.  Master pointer
                pools are looked up by the real layer ID derived from this.

        Notes:
            The grad-reduce SM budget is controlled globally via the
            ``ULTRA_EP_GRAD_REDUCE_NUM_SMS`` environment variable.
            Set ``ULTRA_EP_GRAD_REDUCE_DETERMINISTIC=1`` to use the deterministic
            non-atomic path.
        """
        assert layer_id < self.num_alloc_layers
        real_lid = self._real_layer_id(layer_id)
        assert (
            self.local_master_fc1_grad_ptr_pool[real_lid] is not None
            and self.local_master_fc2_grad_ptr_pool[real_lid] is not None
        )
        event = EventHandle(self.runtime.grad_reduce(
            layer_id,
            self.local_master_fc1_grad_ptr_pool[real_lid],
            self.local_master_fc2_grad_ptr_pool[real_lid],
            getattr(previous_event, "event", None),
            async_finish,
        ))
        if self._autotuner.collecting and event.event is not None:
            # This event is on the framework's current stream.  The elapsed
            # time until wait_grad_reduce() is the real available overlap
            # window, including any attention/router backward work enqueued by
            # the framework on that stream.
            event._autotune_grad_overlap_start = torch.cuda.Event(enable_timing=True)
            event._autotune_grad_overlap_start.record()
        return event

    def weight_sync(
        self,
        layer_id: int,
        previous_event: Optional[EventHandle] = None,
        async_finish: bool = False,
    ):
        """
        Synchronize master weights to replicas.

        The runtime derives a deterministic communication plan from the current
        placement. Mild cases stay on the flat direct fan-out path; extreme hot
        masters may use a staged relay plan to reduce source-side bottlenecks.

        Args:
            layer_id: Virtual layer ID.  Used for placement map lookup in C++.
                Master pointer pools are looked up by the derived real layer ID.
            previous_event: Optional event to wait for before starting.
            async_finish: If True, return immediately with an event handle.

        Returns:
            EventHandle if async_finish=True, else None.
        """
        assert layer_id < self.num_alloc_layers
        real_lid = self._real_layer_id(layer_id)
        assert (
            self.local_master_fc1_weight_ptr_pool[real_lid] is not None
            and self.local_master_fc2_weight_ptr_pool[real_lid] is not None
            and self.local_master_fc1_weight_scale_ptr_pool[real_lid] is not None
            and self.local_master_fc2_weight_scale_ptr_pool[real_lid] is not None
        )
        with torch.cuda.nvtx.range(f"Launch weight_sync (layer {layer_id})"):
            event = self.runtime.weight_sync(
                layer_id,
                self.local_master_fc1_weight_ptr_pool[real_lid],
                self.local_master_fc2_weight_ptr_pool[real_lid],
                self.local_master_fc1_weight_scale_ptr_pool[real_lid],
                self.local_master_fc2_weight_scale_ptr_pool[real_lid],
                getattr(previous_event, "event", None),
                async_finish,
            )
            return EventHandle(event)

    def wait_weight_sync(self, event: EventHandle, layer_id: int) -> None:
        """Wait for weight-sync and, only when enabled, record exposed delay."""
        if not self._autotuner.collecting or event.event is None:
            event.current_stream_wait()
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        event.current_stream_wait()
        end.record()
        self._autotune_ws_wait_events.setdefault(layer_id, []).append((start, end))

    def wait_grad_reduce(self, event: EventHandle, layer_id: Optional[int] = None) -> None:
        """Wait for grad-reduce and, only when enabled, record exposed delay."""
        if not self._autotuner.collecting or event.event is None:
            event.current_stream_wait()
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        event.current_stream_wait()
        end.record()
        overlap_start = getattr(event, "_autotune_grad_overlap_start", None)
        if overlap_start is not None:
            self._autotune_grad_wait_events.append((overlap_start, start, end))

    @staticmethod
    def _p95(values):
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    def _harvest_autotune_waits(self):
        """Summarize waits once per rank, then select the cross-rank maximum."""
        ws = torch.zeros(self.num_alloc_layers, dtype=torch.float32, device=self.device)
        for layer_id, events in self._autotune_ws_wait_events.items():
            values = []
            for start, end in events:
                end.synchronize()
                values.append(start.elapsed_time(end))
            if values:
                ws[layer_id] = self._p95(values)
        grad_values = []
        grad_overlap_values = []
        for overlap_start, start, end in self._autotune_grad_wait_events:
            end.synchronize()
            grad_values.append(start.elapsed_time(end))
            grad_overlap_values.append(overlap_start.elapsed_time(start))
        grad = torch.tensor([self._p95(grad_values)], dtype=torch.float32, device=self.device)
        grad_overlap = torch.tensor(
            [self._p95(grad_overlap_values)], dtype=torch.float32, device=self.device
        )
        has_grad_samples = bool(grad_values)
        self._autotune_ws_wait_events.clear()
        self._autotune_grad_wait_events.clear()
        dist.all_reduce(ws, op=dist.ReduceOp.MAX, group=self.group)
        dist.all_reduce(grad, op=dist.ReduceOp.MAX, group=self.group)
        has_grad = torch.tensor([int(has_grad_samples)], dtype=torch.int32, device=self.device)
        dist.all_reduce(has_grad, op=dist.ReduceOp.MIN, group=self.group)
        has_global_grad_sample = bool(has_grad.item())
        if has_global_grad_sample:
            # The rank with the shortest usable window determines whether a
            # reduction is safely hidden for the whole EP group.
            dist.all_reduce(grad_overlap, op=dist.ReduceOp.MIN, group=self.group)
        return (
            {i: [float(v)] for i, v in enumerate(ws.tolist()) if v > 0},
            [float(grad.item())] if has_global_grad_sample else [],
            [float(grad_overlap.item())] if has_global_grad_sample else [],
        )

    def _apply_weight_sync_launch_config(self, config: WeightSyncLaunchConfig) -> None:
        self.runtime.set_weight_sync_launch_config(
            0 if config.copy_mode == "thread" else 1,
            config.threads_per_block,
            config.cta_multiplier,
            config.lds_waves_per_destination,
        )

    def _benchmark_weight_sync_configs_for_layers(
        self,
        layer_ids,
        configs,
        warmups: int,
        repeats: int,
    ):
        """Benchmark aggregate Weight Sync phases across real-layer workloads.

        Each layer phase is independently reduced with MAX across EP ranks,
        then those layer makespans are summed into one sample. Repeated
        aggregate samples are scored by their median in RuntimeAutotuner.
        """
        layer_ids = tuple(layer_ids)
        configs = tuple(configs)
        if not layer_ids:
            raise ValueError("weight-sync autotune requires at least one layer")
        if not configs:
            raise ValueError("weight-sync autotune requires at least one configuration")

        def run_phase(layer_id: int, measure: bool) -> float | None:
            """Run one Weight Sync phase without a device-wide synchronization.

            Weight Sync executes on the runtime communication stream.  Recording
            the start on the current stream, waiting for the returned completion
            event, and recording the end on that same stream measures the actual
            dependency chain used by training.  In particular, it does not fold
            a trailing process-group barrier or unrelated GPU work into the
            candidate score.
            """
            start = torch.cuda.Event(enable_timing=measure)
            end = torch.cuda.Event(enable_timing=measure)
            start.record()
            completion = self.weight_sync(layer_id, async_finish=True)
            completion.current_stream_wait()
            end.record()
            end.synchronize()
            return start.elapsed_time(end) if measure else None

        for config in configs:
            self._apply_weight_sync_launch_config(config)
            for _ in range(warmups):
                for layer_id in layer_ids:
                    dist.barrier(group=self.group)
                    run_phase(layer_id, measure=False)

        # The same order is generated on every rank.  Rotation places each
        # candidate at well-separated positions across repeated samples.
        order = list(configs)
        random.Random(20260902).shuffle(order)
        samples = {config: [] for config in configs}
        for sample_index in range(repeats):
            offset = sample_index * len(order) // repeats
            round_order = order[offset:] + order[:offset]
            for config in round_order:
                self._apply_weight_sync_launch_config(config)
                aggregate_ms = 0.0
                for layer_id in layer_ids:
                    # Keep only the entry alignment barrier.  A trailing barrier
                    # used to be inside the timed interval and made the scout
                    # score depend on barrier/arrival jitter rather than solely
                    # on Weight Sync completion.
                    dist.barrier(group=self.group)
                    local_elapsed = run_phase(layer_id, measure=True)
                    elapsed = torch.tensor(
                        [local_elapsed],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=self.group)
                    aggregate_ms += float(elapsed.item())
                samples[config].append(aggregate_ms)

        return {config: statistics.median(values) for config, values in samples.items()}

    def autotune_iteration_end(
        self, iteration: int, iteration_time_ms: Optional[float] = None
    ) -> None:
        """Call on every rank after optimizer step and pipeline flush.

        ``iteration_time_ms`` is this rank's complete iteration time, excluding
        this bookkeeping call.  The Manager reduces it with MAX so final
        weight-sync selection follows the actual slowest-rank makespan.
        """
        if not self._autotuner.collecting:
            return
        if iteration < self._autotuner.config.start_iteration:
            # Initial training iterations are intentionally excluded from all
            # tuning observations because their cache/initialization behavior
            # is not representative of steady-state execution.
            self._autotune_ws_wait_events.clear()
            self._autotune_grad_wait_events.clear()
            return
        ws_waits, grad_waits, grad_overlaps = self._harvest_autotune_waits()
        global_iteration_time_ms = None
        if iteration_time_ms is not None:
            elapsed = torch.tensor(
                [float(iteration_time_ms)], dtype=torch.float64, device=self.device
            )
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX, group=self.group)
            global_iteration_time_ms = float(elapsed.item())
        self._autotuner.on_iteration_end(
            self,
            iteration,
            ws_waits,
            grad_waits,
            global_iteration_time_ms,
            grad_overlaps,
        )

    def set_grad_reduce_num_sms(self, num_sms: int) -> None:
        if num_sms <= 0:
            raise ValueError("grad-reduce NUM_SMS must be positive")
        num_sms = min(num_sms, self.num_device_sms)
        num_sms -= num_sms % 2
        if num_sms <= 0:
            raise ValueError("grad-reduce NUM_SMS must be even")
        self.grad_reduce_num_sms = num_sms
        self.runtime.set_grad_reduce_deterministic(
            self.grad_reduce_deterministic, self.grad_reduce_num_sms
        )

    def set_weight_sync_plan_mode(self, plan_mode: str):
        normalized = plan_mode.lower().replace("_", "")
        mode_ids = {"direct": 0, "adaptive": 1, "adaptiverelay": 1, "forcerelay": 2}
        if normalized not in mode_ids:
            raise ValueError(
                "plan_mode must be one of: direct, adaptive_relay, force_relay"
            )
        self.weight_sync_plan_mode = normalized
        self.runtime.set_weight_sync_plan_mode(mode_ids[normalized])

    def set_grad_reduce_deterministic(self, deterministic: bool):
        self.grad_reduce_deterministic = bool(deterministic)
        self.runtime.set_grad_reduce_deterministic(
            self.grad_reduce_deterministic, self.grad_reduce_num_sms
        )

    def update_placement(
        self,
        layer_id: int,
        routing_map: torch.Tensor,
        verify_reduced_loads: bool = False,
    ):
        """
        Update expert placement for a single layer based on real-time load statistics.

        Runs the default device placement algorithm and only falls back to the
        legacy CPU implementation when ``legacy_placement=True``:
          1. Masters remain fixed at their pre-assigned positions.
          2. Per NVL domain, greedily replicates the most loaded experts.
          3. Per NVL domain, packs replicas to GPU slots via LPT bin-packing,
             ensuring replicas are never placed on the same GPU as their master.

        Deterministic: all ranks compute identical results, no broadcast needed.

        Args:
            layer_id: The MoE layer index to update.
            routing_map: [num_tokens, num_global_logical_experts] bool tensor, logical routing map.
        """
        assert layer_id < self.num_alloc_layers
        with torch.cuda.nvtx.range(f"Update placement (layer {layer_id})"):
            self.runtime.update_placement(layer_id, routing_map)
        self._load_profiler.stage_pre(
            layer_id,
            self._real_layer_id(layer_id),
            self.runtime.get_global_logical_expert_loads_tensor(),
        )
        if verify_reduced_loads:
            # The HCU multi-node path may have just submitted a rocSHMEM WG
            # collective.  Do not interleave its completion with this optional
            # RCCL-only correctness check.
            torch.cuda.synchronize()
            global_logical_expert_loads = routing_map.sum(dim=0, dtype=torch.int32)
            dist.all_reduce(global_logical_expert_loads, group=self.group)
            assert torch.equal(
                global_logical_expert_loads,
                self.runtime.get_global_logical_expert_loads_tensor(),
            )

    def update_placement_sparse(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
    ):
        assert layer_id < self.num_alloc_layers
        self.runtime.update_placement_sparse(layer_id, topk_ids)
        with torch.cuda.stream(self.get_comm_stream()):
            self._load_profiler.stage_pre(
                layer_id,
                self._real_layer_id(layer_id),
                self.runtime.get_global_logical_expert_loads_tensor(),
            )

    def reroute(
        self,
        layer_id: int,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        backend: str = "cuda",
    ):
        """
        Expand routing from logical experts to physical experts using deterministic
        round-robin dispatch.

        For each logical expert l with C_l = lcnts[l] physical instances,
        the k-th token (by global token index) routed to l is dispatched to
        physical expert l2p[l, k % C_l].

        Args:
            layer_id: The MoE layer index to reroute.
            probs: [num_tokens, num_logical_experts] float tensor, routing probabilities.
            routing_map: [num_tokens, num_logical_experts] bool tensor, logical routing map.
        Returns:
            expanded_probs: [num_tokens, num_physical_experts] float tensor, expanded probabilities.
            expanded_routing_map: [num_tokens, num_physical_experts] bool tensor, physical routing map.
        """

        if backend != "cuda":
            raise ValueError(
                "Only backend='cuda' is supported; CPU reroute has been removed"
            )

        expanded_probs, expanded_routing_map = self._dense_reroute(
            layer_id, probs, routing_map
        )
        if self._load_profiler.has_staged(layer_id):
            self._load_profiler.record_post(
                layer_id, self._profile_physical_loads_from_quota(layer_id)
            )

        return expanded_probs, expanded_routing_map

    def _dense_reroute(
        self,
        layer_id: int,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
    ):
        """Dense reroute through the fused device kernel."""
        if self.is_train and probs.requires_grad:
            expanded_probs, expanded_routing_map = _DenseRerouteFunction.apply(
                probs,
                routing_map,
                self.runtime,
                layer_id,
            )
        else:
            with torch.cuda.nvtx.range(f"Dense reroute forward (layer {layer_id})"):
                expanded_probs, expanded_routing_map = (
                    self.runtime.dense_reroute_forward(layer_id, probs, routing_map)
                )

        return expanded_probs, expanded_routing_map

    def reroute_sparse(
        self,
        layer_id: int,
        topk_ids: torch.Tensor,
    ):
        assert layer_id < self.num_alloc_layers

        self.runtime.reroute_sparse(layer_id, topk_ids)
        if self._load_profiler.has_staged(layer_id):
            self._load_profiler.record_post(
                layer_id, self._profile_physical_loads_from_quota(layer_id)
            )

    def check_tensors_blob_from_cpp(self):
        assert (
            self._physical_to_logical_map.device == torch.device("cuda", self.device)
            and self._logical_to_physical_map.device
            == torch.device("cuda", self.device)
            and self._logical_replica_counts.device == torch.device("cuda", self.device)
            and self._logical_instance_quota.device == torch.device("cuda", self.device)
            and self._logical_instance_quota_prefix.device
            == torch.device("cuda", self.device)
        )
        assert (
            self._physical_to_logical_map.dtype == torch.int32
            and self._logical_to_physical_map.dtype == torch.int32
            and self._logical_replica_counts.dtype == torch.int32
            and self._logical_instance_quota.dtype == torch.int32
            and self._logical_instance_quota_prefix.dtype == torch.int32
        )
        assert (
            self._physical_to_logical_map.shape
            == (
                self.num_alloc_layers,
                self.num_global_physical_experts,
            )
            and self._logical_to_physical_map.shape
            == (self.num_alloc_layers, self.num_global_logical_experts, self.num_ranks)
            and self._logical_replica_counts.shape
            == (
                self.num_alloc_layers,
                self.num_global_logical_experts,
            )
            and self._logical_instance_quota.shape
            == (self.num_alloc_layers, self.num_global_logical_experts, self.num_ranks)
            and self._logical_instance_quota_prefix.shape
            == (self.num_alloc_layers, self.num_global_logical_experts, self.num_ranks)
        )
        assert self._rank_quota_prefix.device == torch.device("cuda", self.device)
        assert self._rank_quota_prefix.dtype == torch.int32
        assert self._rank_quota_prefix.shape == (
            self.num_alloc_layers,
            self.num_global_logical_experts,
            self.num_ranks,
        )
        assert self.local_replica_weight_buffer.device == torch.device(
            "cuda", self.device
        )
        assert self.local_replica_weight_buffer.dtype == self.weight_data_dtype
        assert self.local_replica_weight_buffer.shape == (
            self.num_local_redundant_experts,
            self.expert_total_numel,
        )
        assert self.local_replica_fc1_weight_buffer.device == torch.device(
            "cuda", self.device
        )
        assert self.local_replica_fc1_weight_buffer.dtype == self.weight_data_dtype
        assert self.local_replica_fc1_weight_buffer.shape == (
            self.num_local_redundant_experts,
            self.expert_fc1_numel,
        )
        assert self.local_replica_fc2_weight_buffer.device == torch.device(
            "cuda", self.device
        )
        assert self.local_replica_fc2_weight_buffer.dtype == self.weight_data_dtype
        assert self.local_replica_fc2_weight_buffer.shape == (
            self.num_local_redundant_experts,
            self.expert_fc2_numel,
        )
        assert self.local_replica_weight_scale_buffer_raw.device == torch.device(
            "cuda", self.device
        )
        assert self.local_replica_weight_scale_buffer_raw.dtype == torch.uint8
        assert self.local_replica_fc1_weight_scale_buffer.device == torch.device(
            "cuda", self.device
        )
        assert (
            self.local_replica_fc1_weight_scale_buffer.dtype == self.weight_scale_dtype
        )
        assert self.local_replica_fc1_weight_scale_buffer.shape == (
            self.num_local_redundant_experts,
            self.expert_fc1_weight_scale_numel,
        )
        assert self.local_replica_fc2_weight_scale_buffer.device == torch.device(
            "cuda", self.device
        )
        assert (
            self.local_replica_fc2_weight_scale_buffer.dtype == self.weight_scale_dtype
        )
        assert self.local_replica_fc2_weight_scale_buffer.shape == (
            self.num_local_redundant_experts,
            self.expert_fc2_weight_scale_numel,
        )
        if self.is_train:
            assert self.local_replica_grad_buffer.device == torch.device(
                "cuda", self.device
            )
            assert self.local_replica_grad_buffer.dtype == self.grad_dtype
            assert self.local_replica_grad_buffer.shape == (
                self.num_local_redundant_experts,
                self.expert_total_numel,
            )
            assert self.local_replica_fc1_grad_buffer.device == torch.device(
                "cuda", self.device
            )
            assert self.local_replica_fc1_grad_buffer.dtype == self.grad_dtype
            assert self.local_replica_fc1_grad_buffer.shape == (
                self.num_local_redundant_experts,
                self.expert_fc1_numel,
            )
            assert self.local_replica_fc2_grad_buffer.device == torch.device(
                "cuda", self.device
            )
            assert self.local_replica_fc2_grad_buffer.dtype == self.grad_dtype
            assert self.local_replica_fc2_grad_buffer.shape == (
                self.num_local_redundant_experts,
                self.expert_fc2_numel,
            )

    def get_comm_stream(self) -> torch.Stream:
        ts: torch.Stream = self.runtime.get_comm_stream()
        return torch.cuda.Stream(
            stream_id=ts.stream_id,
            device_index=ts.device_index,
            device_type=ts.device_type,
        )
