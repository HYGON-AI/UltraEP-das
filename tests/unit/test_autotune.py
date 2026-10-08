# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""CPU-only checks for the autotuner policy and backward-compatible default."""

import importlib.util
from pathlib import Path
import sys


_PATH = Path(__file__).resolve().parents[2] / "ultra_ep" / "autotune.py"
_SPEC = importlib.util.spec_from_file_location("ultra_ep_autotune_test", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

AutotuneConfig = _MODULE.AutotuneConfig
RuntimeAutotuner = _MODULE.RuntimeAutotuner
WeightSyncLaunchConfig = _MODULE.WeightSyncLaunchConfig


def test_default_weight_sync_sampling_is_robust_but_bounded():
    config = AutotuneConfig()
    assert config.weight_sync_warmups == 1
    assert config.weight_sync_repeats == 5
    assert config.weight_sync_scout_real_layers == 6
    assert config.weight_sync_scout_hot_layers == 2
    assert config.weight_sync_layer_tie_tolerance == 0.05
    assert config.weight_sync_scout_tie_tolerance == 0.01
    assert config.weight_sync_observation_iterations == 3
    assert config.weight_sync_e2e_repeats_per_candidate == 3
    assert config.grad_reduce_samples_per_candidate == 3
    assert config.grad_reduce_iteration_tie_tolerance == 0.02
    assert config.grad_reduce_wait_tie_tolerance == 0.05
    assert config.grad_reduce_worst_wait_tie_tolerance == 0.05


def test_disabled_autotune_is_inert():
    tuner = RuntimeAutotuner(AutotuneConfig())
    # A disabled controller must not access the manager or make a decision.
    tuner.on_iteration_end(None, 3, {0: [1.0]}, [1.0])
    assert not tuner.weight_sync_done
    assert tuner._grad_reduce_active_sms is None


def test_iteration_timing_starts_only_after_weight_sync_scout():
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    assert not tuner.needs_iteration_time(1)
    assert not tuner.needs_iteration_time(3)
    tuner._weight_sync_active_probe = tuner.current_weight_sync
    assert tuner.needs_iteration_time(6)


def test_weight_sync_observes_iterations_three_through_five_before_scout():
    class Manager:
        @staticmethod
        def _real_layer_id(virtual_layer_id):
            return virtual_layer_id

    manager = Manager()
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    scouted = []
    tuner._shortlist_weight_sync = lambda manager, layers: scouted.append(layers) or ()
    tuner._start_weight_sync_e2e_probes = lambda manager, finalists: None

    # Initialization iterations must not contaminate real-layer history.
    tuner.on_iteration_end(manager, 1, {99: [100.0]}, [], None, [])
    tuner.on_iteration_end(manager, 2, {99: [100.0]}, [], None, [])
    for iteration in (3, 4):
        tuner.on_iteration_end(
            manager, iteration, {3: [3.0], 4: [4.0], 5: [5.0]}, [], None, []
        )
        assert not scouted
    tuner.on_iteration_end(
        manager, 5, {3: [3.0], 4: [4.0], 5: [5.0]}, [], None, []
    )

    assert tuner._weight_sync_observation_count == 3
    assert 99 not in tuner._weight_sync_real_layer_wait_ms
    assert scouted == [(3, 4, 5)]


def test_explicit_environment_settings_are_locks(monkeypatch):
    monkeypatch.setenv("ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE", "lds")
    monkeypatch.setenv("ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK", "256")
    monkeypatch.setenv("ULTRA_EP_GRAD_REDUCE_NUM_SMS", "40")
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    assert tuner.current_weight_sync.copy_mode == "lds"
    assert tuner.current_weight_sync.threads_per_block == 256
    assert "ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE" in tuner.locked_environment_variables
    assert "ULTRA_EP_GRAD_REDUCE_NUM_SMS" in tuner.locked_environment_variables


def test_grad_reduce_is_not_changed_without_an_observation():
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))

    class Manager:
        grad_reduce_num_sms = 40
        num_device_sms = 80

        def set_grad_reduce_num_sms(self, value):
            raise AssertionError("must not tune without wait observations")

    tuner.weight_sync_done = True
    tuner.on_iteration_end(Manager(), 3, {}, [])
    assert tuner._grad_reduce_active_sms is None


def test_grad_reduce_stops_at_first_hidden_candidate_and_reserves_sms():
    """A 64-SMS device must stop at 48 and finish as soon as 24 is hidden."""

    class Manager:
        rank = 1
        num_device_sms = 64
        grad_reduce_num_sms = 40

        def __init__(self):
            self.applied = []

        def set_grad_reduce_num_sms(self, value):
            self.applied.append(value)

    manager = Manager()
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    tuner.weight_sync_done = True

    # First observed iteration installs the smallest candidate; the following
    # three are the real-backward samples for that candidate.
    tuner.on_iteration_end(manager, 3, {}, [0.01], 10.0, [1.0])
    tuner.on_iteration_end(manager, 4, {}, [0.01], 10.0, [1.0])
    tuner.on_iteration_end(manager, 5, {}, [0.01], 10.0, [1.0])
    tuner.on_iteration_end(manager, 6, {}, [0.01], 10.0, [1.0])

    assert tuner._grad_reduce_candidates == (24, 32, 40, 48)
    assert manager.applied == [24, 24]
    assert tuner.grad_done
    assert tuner._grad_reduce_active_sms is None


def test_grad_reduce_fallback_uses_complete_iteration_not_smallest_wait():
    """When no candidate hides, complete-iteration time selects the SMS."""

    class Manager:
        rank = 1
        num_device_sms = 64
        grad_reduce_num_sms = 40

        def __init__(self):
            self.applied = []

        def set_grad_reduce_num_sms(self, value):
            self.applied.append(value)
            self.grad_reduce_num_sms = value

    manager = Manager()
    tuner = RuntimeAutotuner(
        AutotuneConfig(
            enabled=True,
            grad_reduce_candidates=(24, 32),
            grad_reduce_max_sms=32,
            grad_reduce_min_hidden_wait_ms=0.5,
        )
    )
    tuner.weight_sync_done = True

    # 32 has the lower exposed wait *and* a materially better complete
    # iteration.  Neither can be hidden because 5/4 ms > 1% of 100 ms.
    tuner.on_iteration_end(manager, 3, {}, [5.0], 100.0, [100.0])
    tuner.on_iteration_end(manager, 4, {}, [5.0], 100.0, [100.0])
    tuner.on_iteration_end(manager, 5, {}, [5.0], 101.0, [100.0])
    tuner.on_iteration_end(manager, 6, {}, [5.0], 100.5, [100.0])
    tuner.on_iteration_end(manager, 7, {}, [4.0], 90.0, [100.0])
    tuner.on_iteration_end(manager, 8, {}, [4.0], 91.0, [100.0])
    tuner.on_iteration_end(manager, 9, {}, [4.0], 90.5, [100.0])

    assert tuner.grad_done
    assert manager.grad_reduce_num_sms == 32


def test_representative_weight_sync_layers_are_distinct_and_use_history():
    class Manager:
        @staticmethod
        def _real_layer_id(virtual_layer_id):
            return virtual_layer_id // 3

    manager = Manager()
    tuner = RuntimeAutotuner(
        AutotuneConfig(enabled=True, weight_sync_scout_real_layers=2)
    )
    # Real layer 0 has two rotating virtual slots. Its per-iteration maximum
    # is aggregated into one real-layer history instead of counting it twice.
    tuner._record_weight_sync_observations(
        manager, {0: [8.0], 1: [10.0], 3: [7.0], 6: [1.0]}
    )
    tuner._record_weight_sync_observations(
        manager, {2: [9.0], 4: [8.0], 7: [2.0]}
    )

    selected = tuner._representative_layers()
    assert selected == (2, 4)
    assert [manager._real_layer_id(value) for value in selected] == [0, 1]


def test_weight_sync_scout_uses_all_layers_when_stage_is_small():
    class Manager:
        @staticmethod
        def _real_layer_id(virtual_layer_id):
            return virtual_layer_id // 3

    manager = Manager()
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    tuner._record_weight_sync_observations(manager, {0: [4.0], 3: [3.0]})
    # Layer zero is absent from the later iteration but must remain eligible.
    tuner._record_weight_sync_observations(manager, {6: [2.0], 9: [1.0]})

    selected = tuner._representative_layers()
    assert selected == (0, 3, 6, 9)
    assert [manager._real_layer_id(value) for value in selected] == [0, 1, 2, 3]


def test_weight_sync_scout_combines_fixed_anchors_and_stable_hot_layers():
    class Manager:
        @staticmethod
        def _real_layer_id(layer_id):
            return layer_id

    manager = Manager()
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    samples = {layer_id: [float(layer_id)] for layer_id in range(10)}
    samples[7] = [96.0]
    samples[8] = [100.0]
    tuner._record_weight_sync_observations(manager, samples)

    # Four deterministic position anchors are 0/3/6/9. Layers 7 and 8 are
    # within the 5% hot-layer tie band, so lower ID 7 is chosen first.
    assert tuner._representative_layers() == (0, 3, 6, 9, 7, 8)


def test_weight_sync_scout_uses_compact_core_and_boundary_expansion():
    class Manager:
        rank = 1

        def __init__(self):
            self.calls = []

        @staticmethod
        def _real_layer_id(layer_id):
            return layer_id

        def _benchmark_weight_sync_configs_for_layers(
            self, layer_ids, candidates, warmups, repeats
        ):
            self.calls.append((tuple(layer_ids), tuple(candidates)))
            scores = {}
            for candidate in candidates:
                if candidate.copy_mode == "thread":
                    scores[candidate] = (
                        0.97
                        if candidate.threads_per_block == 128
                        and candidate.cta_multiplier == 8
                        else 0.98
                        if candidate.threads_per_block == 128
                        and candidate.cta_multiplier == 4
                        else 1.0
                        if candidate.threads_per_block == 128
                        and candidate.cta_multiplier == 2
                        else 2.0
                    )
                else:
                    scores[candidate] = (
                        0.97
                        if candidate.cta_multiplier == 8
                        and candidate.lds_waves_per_destination == 2
                        else 0.98
                        if candidate.cta_multiplier == 4
                        and candidate.lds_waves_per_destination == 2
                        else 1.0
                        if candidate.cta_multiplier == 2
                        and candidate.lds_waves_per_destination == 2
                        else 2.0
                    )
            return scores

    manager = Manager()
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    finalists = tuner._shortlist_weight_sync(manager, (1, 2, 3, 4))

    assert len(manager.calls[0][1]) == 8
    assert len(manager.calls[1][1]) == 4
    assert len(manager.calls[2][1]) == 2
    assert len(finalists) <= 2
    assert all(layer_ids == (1, 2, 3, 4) for layer_ids, _ in manager.calls)


def test_weight_sync_scout_keeps_one_stable_finalist_per_mode():
    class Manager:
        rank = 1

        @staticmethod
        def _real_layer_id(layer_id):
            return layer_id

        @staticmethod
        def _benchmark_weight_sync_configs_for_layers(
            layer_ids, candidates, warmups, repeats
        ):
            scores = {candidate: 2.0 for candidate in candidates}
            for candidate in candidates:
                if candidate.copy_mode == "thread":
                    if candidate.threads_per_block == 256 and candidate.cta_multiplier == 2:
                        scores[candidate] = 1.0
                    elif candidate.threads_per_block == 128 and candidate.cta_multiplier == 2:
                        scores[candidate] = 1.005
                elif candidate.cta_multiplier == 2 and candidate.lds_waves_per_destination == 1:
                    scores[candidate] = 1.0
                elif candidate.cta_multiplier == 1 and candidate.lds_waves_per_destination == 1:
                    scores[candidate] = 1.005
            return scores

    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    finalists = tuner._shortlist_weight_sync(Manager(), (1, 2))

    assert len(finalists) == 2
    assert WeightSyncLaunchConfig("thread", 128, 2, 1) in finalists
    assert WeightSyncLaunchConfig("lds", 128, 1, 1) in finalists
    assert WeightSyncLaunchConfig("thread", 256, 2, 1) not in finalists
    assert WeightSyncLaunchConfig("lds", 128, 2, 1) not in finalists


def test_grad_reduce_near_best_prefers_shorter_median_wait():
    GradReduceMeasurement = _MODULE.GradReduceMeasurement

    class Manager:
        rank = 1
        grad_reduce_num_sms = 40

        def set_grad_reduce_num_sms(self, value):
            self.grad_reduce_num_sms = value

    manager = Manager()
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    tuner._grad_reduce_candidates = (40, 48, 56)
    tuner._grad_reduce_baseline_sms = 40
    tuner._grad_reduce_samples = {
        40: [GradReduceMeasurement(2.0, 1.0, 100.0) for _ in range(3)],
        48: [GradReduceMeasurement(1.0, 1.0, 99.5) for _ in range(3)],
        56: [GradReduceMeasurement(1.02, 1.0, 99.8) for _ in range(3)],
    }

    tuner._select_grad_reduce_sms(manager)

    # 48 and 56 have waits within 5%; prefer the smaller of those two rather
    # than the smallest SMS (40) in the iteration-time near-best set.
    assert manager.grad_reduce_num_sms == 48


def test_grad_reduce_uses_worst_wait_after_median_wait_tie():
    GradReduceMeasurement = _MODULE.GradReduceMeasurement

    class Manager:
        rank = 1
        grad_reduce_num_sms = 40

        def set_grad_reduce_num_sms(self, value):
            self.grad_reduce_num_sms = value

    manager = Manager()
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    tuner._grad_reduce_candidates = (40, 48)
    tuner._grad_reduce_baseline_sms = 40
    tuner._grad_reduce_samples = {
        40: [
            GradReduceMeasurement(1.0, 1.0, 100.0),
            GradReduceMeasurement(1.0, 1.0, 100.0),
            GradReduceMeasurement(5.0, 1.0, 100.0),
        ],
        48: [
            GradReduceMeasurement(1.02, 1.0, 100.0),
            GradReduceMeasurement(1.02, 1.0, 100.0),
            GradReduceMeasurement(1.10, 1.0, 100.0),
        ],
    }

    tuner._select_grad_reduce_sms(manager)

    assert manager.grad_reduce_num_sms == 48


def test_grad_reduce_iteration_guard_rejects_short_wait_with_regression():
    GradReduceMeasurement = _MODULE.GradReduceMeasurement

    class Manager:
        rank = 1
        grad_reduce_num_sms = 40

        def set_grad_reduce_num_sms(self, value):
            self.grad_reduce_num_sms = value

    manager = Manager()
    tuner = RuntimeAutotuner(AutotuneConfig(enabled=True))
    tuner._grad_reduce_candidates = (40, 56)
    tuner._grad_reduce_baseline_sms = 40
    tuner._grad_reduce_samples = {
        40: [GradReduceMeasurement(2.0, 1.0, 100.0) for _ in range(3)],
        56: [GradReduceMeasurement(1.0, 1.0, 103.0) for _ in range(3)],
    }

    tuner._select_grad_reduce_sms(manager)

    assert manager.grad_reduce_num_sms == 40
