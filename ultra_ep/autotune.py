# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Bounded runtime autotuning for a fixed UltraEP replica budget.

This module deliberately does not tune placement or the number of replica
slots.  Those values determine Manager allocation and must be fixed before
construction.  Explicit ULTRA_EP_* environment settings are hard locks.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .manager import Manager


@dataclass(frozen=True)
class AutotuneConfig:
    enabled: bool = False
    start_iteration: int = 3
    weight_sync_warmups: int = 1
    # The scout uses five timed samples so the median is a real middle
    # observation rather than the arithmetic mean produced by two samples.
    weight_sync_repeats: int = 5
    # Use four deterministic layer-position anchors plus two stable hot layers
    # by default. This bounds tuning cost without returning to a noisy top-3.
    weight_sync_scout_real_layers: int = 6
    weight_sync_scout_hot_layers: int = 2
    weight_sync_layer_tie_tolerance: float = 0.05
    # Ignore iterations before start_iteration, then collect this many stable
    # real-layer observations before running the Weight Sync microbenchmark.
    weight_sync_observation_iterations: int = 3
    # Three complete iterations per finalist provide a robust median while
    # adding only one training step per candidate over the old default.
    weight_sync_e2e_repeats_per_candidate: int = 3
    minimum_weight_sync_speedup: float = 0.01
    weight_sync_scout_tie_tolerance: float = 0.01
    weight_sync_mode_tie_tolerance: float = 0.01
    tune_grad_reduce: bool = True
    # A reduction is practically hidden when its exposed tail is below both a
    # small absolute floor and a fraction of the *measured* overlap window.
    # This is intentionally not a fixed 50-us target.
    grad_reduce_min_hidden_wait_ms: float = 0.5
    grad_reduce_hidden_wait_ratio: float = 0.01
    # Complete iteration is a safety guard rather than the primary Grad Reduce
    # objective. Reject candidates with more than 2% end-to-end regression.
    grad_reduce_iteration_tie_tolerance: float = 0.02
    # Median exposed wait is the primary objective after the iteration guard.
    grad_reduce_wait_tie_tolerance: float = 0.05
    # Use worst exposed wait as the second objective; only fall back to the
    # smaller SMS budget when both wait metrics are effectively tied.
    grad_reduce_worst_wait_tie_tolerance: float = 0.05
    grad_reduce_candidates: tuple[int, ...] = (24, 32, 40, 48, 56, 64)
    # Use an odd sample count so complete-iteration scoring has a true median
    # and one noisy training step cannot dominate a candidate's result.
    grad_reduce_samples_per_candidate: int = 3
    # Leave headroom for the backward kernels that overlap with grad-reduce.
    # A caller may lower this with grad_reduce_max_sms, but automatic tuning
    # never consumes every physical SMS.
    grad_reduce_max_sms: int | None = None
    grad_reduce_max_sms_fraction: float = 0.75


@dataclass(frozen=True)
class WeightSyncLaunchConfig:
    copy_mode: str = "thread"
    threads_per_block: int = 128
    cta_multiplier: int = 2
    lds_waves_per_destination: int = 1


@dataclass(frozen=True)
class GradReduceMeasurement:
    exposed_wait_ms: float
    overlap_window_ms: float
    iteration_ms: float | None


class RuntimeAutotuner:
    """One-shot, low-cost tuner driven by complete training iterations."""

    _LOCKED_WEIGHT_SYNC = (
        "ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE",
        "ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK",
        "ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER",
        "ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST",
    )

    def __init__(self, config: AutotuneConfig) -> None:
        if config.start_iteration < 1:
            raise ValueError("start_iteration must be positive")
        if (
            config.weight_sync_warmups < 0
            or config.weight_sync_repeats < 1
            or config.weight_sync_scout_real_layers < 1
            or config.weight_sync_scout_hot_layers < 0
            or not 0.0 <= config.weight_sync_layer_tie_tolerance < 1.0
            or config.weight_sync_observation_iterations < 1
            or config.weight_sync_e2e_repeats_per_candidate < 1
            or not 0.0 <= config.minimum_weight_sync_speedup < 1.0
            or not 0.0 <= config.weight_sync_scout_tie_tolerance < 1.0
            or not 0.0 <= config.weight_sync_mode_tie_tolerance < 1.0
            or config.grad_reduce_min_hidden_wait_ms < 0.0
            or not 0.0 <= config.grad_reduce_hidden_wait_ratio < 1.0
            or not 0.0 <= config.grad_reduce_iteration_tie_tolerance < 1.0
            or not 0.0 <= config.grad_reduce_wait_tie_tolerance < 1.0
            or not 0.0 <= config.grad_reduce_worst_wait_tie_tolerance < 1.0
            or config.grad_reduce_samples_per_candidate < 1
            or not config.grad_reduce_candidates
            or any(value <= 0 or value % 2 for value in config.grad_reduce_candidates)
            or (
                config.grad_reduce_max_sms is not None
                and (config.grad_reduce_max_sms <= 0 or config.grad_reduce_max_sms % 2)
            )
            or not 0.0 < config.grad_reduce_max_sms_fraction <= 1.0
        ):
            raise ValueError("invalid autotune configuration")
        self.config = config
        # Do not even parse autotune-only policy when disabled.  This keeps the
        # pre-existing runtime/environment behaviour byte-for-byte in charge.
        self.locked_environment_variables = (
            frozenset(
                name for name in (*self._LOCKED_WEIGHT_SYNC, "ULTRA_EP_GRAD_REDUCE_NUM_SMS") if name in os.environ
            )
            if config.enabled
            else frozenset()
        )
        self.current_weight_sync = (
            self._weight_sync_from_environment() if config.enabled else WeightSyncLaunchConfig()
        )
        self.weight_sync_done = config.enabled and all(
            name in self.locked_environment_variables
            for name in self._LOCKED_WEIGHT_SYNC
        )
        self._weight_sync_probe_sequence: list[WeightSyncLaunchConfig] = []
        self._weight_sync_probe_index = 0
        self._weight_sync_active_probe: WeightSyncLaunchConfig | None = None
        self._weight_sync_e2e_samples: dict[WeightSyncLaunchConfig, list[float]] = {}
        self._weight_sync_real_layer_wait_ms: dict[int, list[float]] = {}
        self._weight_sync_latest_virtual_by_real: dict[int, tuple[int, float]] = {}
        self._weight_sync_observation_count = 0
        self._grad_reduce_candidates: tuple[int, ...] = ()
        self._grad_reduce_probe_index = 0
        self._grad_reduce_active_sms: int | None = None
        self._grad_reduce_samples: dict[int, list[GradReduceMeasurement]] = {}
        self._grad_reduce_baseline_sms: int | None = None
        self.grad_done = not config.tune_grad_reduce

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def complete(self) -> bool:
        return self.weight_sync_done and self.grad_done

    @property
    def collecting(self) -> bool:
        return self.enabled and not self.complete

    def needs_iteration_time(self, iteration: int) -> bool:
        """Whether the next train step is an active end-to-end candidate."""
        if not self.collecting or iteration < self.config.start_iteration:
            return False
        if not self.weight_sync_done:
            return self._weight_sync_active_probe is not None
        if self.grad_done or "ULTRA_EP_GRAD_REDUCE_NUM_SMS" in self.locked_environment_variables:
            return False
        return self._grad_reduce_active_sms is not None

    def on_iteration_end(
        self,
        manager: "Manager",
        iteration: int,
        weight_sync_wait_ms: dict[int, list[float]],
        grad_reduce_wait_ms: list[float],
        iteration_time_ms: float | None = None,
        grad_reduce_overlap_ms: list[float] | None = None,
    ) -> None:
        if not self.enabled or iteration < self.config.start_iteration:
            return

        if not self.weight_sync_done:
            if self._weight_sync_active_probe is not None:
                if iteration_time_ms is None:
                    raise RuntimeError(
                        "weight-sync autotune requires the complete iteration time; "
                        "pass iteration_time_ms to Manager.autotune_iteration_end()"
                    )
                self._weight_sync_e2e_samples[self._weight_sync_active_probe].append(
                    iteration_time_ms
                )
                self._weight_sync_probe_index += 1
                if self._weight_sync_probe_index < len(self._weight_sync_probe_sequence):
                    self._weight_sync_active_probe = self._weight_sync_probe_sequence[
                        self._weight_sync_probe_index
                    ]
                    manager._apply_weight_sync_launch_config(self._weight_sync_active_probe)
                    return
                self.current_weight_sync = self._select_weight_sync_from_e2e(manager)
                self.weight_sync_done = True
                self._weight_sync_active_probe = None
            else:
                # Build a stable real-layer history before changing any launch
                # setting. A virtual ID contains a rotating microbatch slot,
                # so observations are aggregated by real layer.
                if weight_sync_wait_ms:
                    self._record_weight_sync_observations(
                        manager, weight_sync_wait_ms
                    )
                    self._weight_sync_observation_count += 1
                if (
                    self._weight_sync_observation_count
                    < self.config.weight_sync_observation_iterations
                ):
                    return
                layer_ids = self._representative_layers()
                if not layer_ids:
                    return
                finalists = self._shortlist_weight_sync(manager, layer_ids)
                self._start_weight_sync_e2e_probes(manager, finalists)
                return

        if not self.weight_sync_done:
            return

        if "ULTRA_EP_GRAD_REDUCE_NUM_SMS" in self.locked_environment_variables:
            self.grad_done = True
            return
        if self.grad_done:
            return

        # The framework must use Manager.wait_grad_reduce() to expose an
        # overlap measurement.  Without one, guessing an SMS budget would be
        # less safe than preserving the established environment/default value.
        if not grad_reduce_wait_ms or not grad_reduce_overlap_ms:
            return

        observed_wait_ms = self._p95(grad_reduce_wait_ms)
        observed_overlap_ms = self._p95(grad_reduce_overlap_ms)
        if self._grad_reduce_active_sms is None:
            self._start_grad_reduce_probes(manager)
            return

        self._grad_reduce_samples[self._grad_reduce_active_sms].append(
            GradReduceMeasurement(
                exposed_wait_ms=observed_wait_ms,
                overlap_window_ms=observed_overlap_ms,
                iteration_ms=iteration_time_ms,
            )
        )
        if len(self._grad_reduce_samples[self._grad_reduce_active_sms]) < self.config.grad_reduce_samples_per_candidate:
            return

        current_samples = self._grad_reduce_samples[self._grad_reduce_active_sms]
        if self._grad_reduce_is_hidden(current_samples):
            # Candidates are sorted from small to large.  The first one whose
            # real exposed wait is consistently hidden is therefore the
            # smallest safe SMS budget; do not spend further training steps
            # probing larger allocations.
            self._finish_grad_reduce_sms(
                manager,
                self._grad_reduce_active_sms,
                stopped_early=True,
            )
            return

        self._grad_reduce_probe_index += 1
        if self._grad_reduce_probe_index < len(self._grad_reduce_candidates):
            self._grad_reduce_active_sms = self._grad_reduce_candidates[self._grad_reduce_probe_index]
            manager.set_grad_reduce_num_sms(self._grad_reduce_active_sms)
            return

        self._select_grad_reduce_sms(manager)
        self.grad_done = True

    def _start_grad_reduce_probes(self, manager: "Manager") -> None:
        # The percentage cap protects overlap: allowing grad-reduce to claim
        # every SMS can make its own wait look small while starving attention
        # and router backward kernels.  Keep at least two hardware SMS free
        # even when a caller asks for a 100% fraction.
        fraction_cap = int(manager.num_device_sms * self.config.grad_reduce_max_sms_fraction)
        fraction_cap -= fraction_cap % 2
        physical_cap = max(0, manager.num_device_sms - 2)
        physical_cap -= physical_cap % 2
        configured_cap = (
            self.config.grad_reduce_max_sms
            if self.config.grad_reduce_max_sms is not None
            else physical_cap
        )
        effective_cap = min(fraction_cap, configured_cap, physical_cap)
        candidates = tuple(
            value
            for value in self.config.grad_reduce_candidates
            if value <= effective_cap
        )
        if not candidates:
            # This is only expected on unusually small devices.  Keep the
            # established launch value rather than inventing an unsupported one.
            self.grad_done = True
            return
        self._grad_reduce_candidates = candidates
        self._grad_reduce_samples = {value: [] for value in candidates}
        self._grad_reduce_baseline_sms = manager.grad_reduce_num_sms
        self._grad_reduce_probe_index = 0
        self._grad_reduce_active_sms = candidates[0]
        manager.set_grad_reduce_num_sms(self._grad_reduce_active_sms)
        if manager.rank == 0:
            print(
                "UltraEP grad-reduce SMS probe started: "
                f"candidates={list(candidates)}, "
                f"effective max={effective_cap}/{manager.num_device_sms} SMS, "
                f"samples/candidate={self.config.grad_reduce_samples_per_candidate}, "
                "hidden target=max("
                f"{self.config.grad_reduce_min_hidden_wait_ms:.3f} ms, "
                f"{self.config.grad_reduce_hidden_wait_ratio * 100.0:.1f}% of measured overlap)",
                flush=True,
            )
            print(
                "  exposed fallback policy: complete-iteration regression guard="
                f"{self.config.grad_reduce_iteration_tie_tolerance * 100.0:.1f}%, "
                "median-wait tie="
                f"{self.config.grad_reduce_wait_tie_tolerance * 100.0:.1f}%, "
                "worst-wait tie="
                f"{self.config.grad_reduce_worst_wait_tie_tolerance * 100.0:.1f}%",
                flush=True,
            )

    def _select_grad_reduce_sms(self, manager: "Manager") -> None:
        # No candidate was practically hidden. Complete iteration is only a
        # regression guard because different training batches add substantial
        # noise. Within that safe set, optimize the Grad Reduce signal itself:
        # median exposed wait, then worst wait, then the smaller SMS budget.
        complete_iteration_scores = {
            sms: self._median([sample.iteration_ms for sample in samples])
            for sms, samples in self._grad_reduce_samples.items()
            if all(sample.iteration_ms is not None for sample in samples)
        }
        if len(complete_iteration_scores) == len(self._grad_reduce_candidates):
            best_ms = min(complete_iteration_scores.values())
            iteration_safe = [
                sms
                for sms in self._grad_reduce_candidates
                if complete_iteration_scores[sms]
                <= best_ms * (1.0 + self.config.grad_reduce_iteration_tie_tolerance)
            ]
            median_wait_scores = {
                sms: self._median(
                    [sample.exposed_wait_ms for sample in self._grad_reduce_samples[sms]]
                )
                for sms in iteration_safe
            }
            best_wait_ms = min(median_wait_scores.values())
            median_wait_safe = [
                sms
                for sms in iteration_safe
                if median_wait_scores[sms]
                <= best_wait_ms * (1.0 + self.config.grad_reduce_wait_tie_tolerance)
            ]
            worst_wait_scores = {
                sms: max(
                    sample.exposed_wait_ms
                    for sample in self._grad_reduce_samples[sms]
                )
                for sms in median_wait_safe
            }
            best_worst_wait_ms = min(worst_wait_scores.values())
            worst_wait_safe = [
                sms
                for sms in median_wait_safe
                if worst_wait_scores[sms]
                <= best_worst_wait_ms
                * (1.0 + self.config.grad_reduce_worst_wait_tie_tolerance)
            ]
            selected = min(worst_wait_safe)
            reason = (
                "no candidate was practically hidden; complete iteration was used "
                "only as a regression guard, then selection minimized median wait, "
                "worst wait, and finally SMS usage"
            )
        else:
            # This should only happen when an integration omits
            # iteration_time_ms.  Do not incorrectly prefer a larger SMS
            # budget merely because it lowers the wait measurement.
            selected = self._grad_reduce_baseline_sms
            if selected not in self._grad_reduce_candidates:
                selected = self._grad_reduce_candidates[0]
            reason = "complete iteration time was unavailable; restored the conservative baseline SMS"
        self._finish_grad_reduce_sms(
            manager, selected, stopped_early=False, selection_reason=reason
        )

    def _finish_grad_reduce_sms(
        self,
        manager: "Manager",
        selected: int,
        *,
        stopped_early: bool,
        selection_reason: str | None = None,
    ) -> None:
        """Commit one SMS budget and report only real probes that ran."""
        manager.set_grad_reduce_num_sms(selected)
        self._grad_reduce_active_sms = None
        self.grad_done = True
        if manager.rank == 0:
            print("UltraEP grad-reduce SMS results (slowest rank exposed wait):", flush=True)
            for sms in self._grad_reduce_candidates[: self._grad_reduce_probe_index + 1]:
                samples = self._grad_reduce_samples[sms]
                worst_wait = max(sample.exposed_wait_ms for sample in samples)
                median_wait = self._median(
                    [sample.exposed_wait_ms for sample in samples]
                )
                shortest_overlap = min(sample.overlap_window_ms for sample in samples)
                allowed_wait = min(self._grad_reduce_hidden_limit(sample) for sample in samples)
                marker = "  <-- selected" if sms == selected else ""
                status = "hidden" if self._grad_reduce_is_hidden(samples) else "exposed"
                iteration_text = ""
                iteration_values = [sample.iteration_ms for sample in samples]
                if all(value is not None for value in iteration_values):
                    iteration_text = (
                        f" median_iteration={self._median(iteration_values):.3f} ms"
                    )
                print(
                    f"  sms={sms:2d} median_wait={median_wait:.3f} ms "
                    f"worst_wait={worst_wait:.3f} ms "
                    f"min_overlap={shortest_overlap:.3f} ms "
                    f"target<={allowed_wait:.3f} ms {status}{iteration_text}{marker}",
                    flush=True,
                )
            if stopped_early:
                print(
                    "  Stopped at the first consistently hidden candidate; "
                    "larger SMS budgets were not probed.",
                    flush=True,
                )
            elif selection_reason is not None:
                print(f"  {selection_reason}.", flush=True)

    def _grad_reduce_hidden_limit(self, sample: GradReduceMeasurement) -> float:
        return max(
            self.config.grad_reduce_min_hidden_wait_ms,
            self.config.grad_reduce_hidden_wait_ratio * sample.overlap_window_ms,
        )

    def _grad_reduce_is_hidden(self, samples: list[GradReduceMeasurement]) -> bool:
        # Every real sample must meet its own overlap-derived target.
        return all(
            sample.exposed_wait_ms <= self._grad_reduce_hidden_limit(sample)
            for sample in samples
        )

    def _shortlist_weight_sync(
        self, manager: "Manager", layer_ids: tuple[int, ...]
    ) -> tuple[WeightSyncLaunchConfig, ...]:
        current = self.current_weight_sync

        # Scan eight core points on a stable representative layer set. Outer
        # values are explored one boundary step at a time.
        copy_modes = (
            (current.copy_mode,)
            if "ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE" in self.locked_environment_variables
            else ("thread", "lds")
        )
        thread_tpbs = (
            (current.threads_per_block,)
            if "ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK" in self.locked_environment_variables
            else (128, 256)
        )
        cta_multipliers = (
            (current.cta_multiplier,)
            if "ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER" in self.locked_environment_variables
            else (1, 2)
        )
        lds_waves = (
            (current.lds_waves_per_destination,)
            if "ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST" in self.locked_environment_variables
            else (1, 2)
        )

        core_candidates = {current}
        if "thread" in copy_modes:
            for tpb in thread_tpbs:
                for cta in cta_multipliers:
                    core_candidates.add(WeightSyncLaunchConfig(
                        "thread", tpb, cta, current.lds_waves_per_destination
                    ))
        if "lds" in copy_modes:
            for cta in cta_multipliers:
                for waves in lds_waves:
                    core_candidates.add(WeightSyncLaunchConfig(
                        "lds", current.threads_per_block, cta, waves
                    ))

        def ordered(values):
            return sorted(values, key=lambda value: (
                value.copy_mode, value.threads_per_block, value.cta_multiplier,
                value.lds_waves_per_destination,
            ))

        def benchmark(candidates):
            candidates = ordered(candidates)
            if not candidates:
                return {}
            return manager._benchmark_weight_sync_configs_for_layers(
                layer_ids,
                candidates,
                self.config.weight_sync_warmups,
                self.config.weight_sync_repeats,
            )

        ordered_core = ordered(core_candidates)
        scout_scores = benchmark(ordered_core)
        core_winners = {
            copy_mode: min(
                (candidate for candidate in ordered_core if candidate.copy_mode == copy_mode),
                key=lambda candidate: scout_scores[candidate],
            )
            for copy_mode in copy_modes
        }

        first_expansion: set[WeightSyncLaunchConfig] = set()
        tpb_locked = (
            "ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK"
            in self.locked_environment_variables
        )
        cta_locked = (
            "ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER"
            in self.locked_environment_variables
        )
        waves_locked = (
            "ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST"
            in self.locked_environment_variables
        )
        thread_winner = core_winners.get("thread")
        if thread_winner is not None:
            if not tpb_locked and thread_winner.threads_per_block == 128:
                first_expansion.add(WeightSyncLaunchConfig(
                    "thread", 64, thread_winner.cta_multiplier,
                    current.lds_waves_per_destination,
                ))
            if not cta_locked and thread_winner.cta_multiplier == 2:
                first_expansion.add(WeightSyncLaunchConfig(
                    "thread", thread_winner.threads_per_block, 4,
                    current.lds_waves_per_destination,
                ))
        lds_winner = core_winners.get("lds")
        if lds_winner is not None:
            if not cta_locked and lds_winner.cta_multiplier == 2:
                first_expansion.add(WeightSyncLaunchConfig(
                    "lds", current.threads_per_block, 4,
                    lds_winner.lds_waves_per_destination,
                ))
            if not waves_locked and lds_winner.lds_waves_per_destination == 2:
                first_expansion.add(WeightSyncLaunchConfig(
                    "lds", current.threads_per_block,
                    lds_winner.cta_multiplier, 4,
                ))
        first_expansion.difference_update(core_candidates)
        scout_scores.update(benchmark(first_expansion))

        # CTA=8 is only worth probing when CTA=4 beats the core CTA=2 winner
        # by the same meaningful-improvement threshold used for final safety.
        second_expansion: set[WeightSyncLaunchConfig] = set()
        for winner in (thread_winner, lds_winner):
            if winner is None or cta_locked or winner.cta_multiplier != 2:
                continue
            cta4 = WeightSyncLaunchConfig(
                winner.copy_mode,
                winner.threads_per_block,
                4,
                winner.lds_waves_per_destination,
            )
            if (
                cta4 in scout_scores
                and scout_scores[cta4]
                <= scout_scores[winner] * (1.0 - self.config.minimum_weight_sync_speedup)
            ):
                second_expansion.add(WeightSyncLaunchConfig(
                    winner.copy_mode,
                    winner.threads_per_block,
                    8,
                    winner.lds_waves_per_destination,
                ))
        second_expansion.difference_update(scout_scores)
        scout_scores.update(benchmark(second_expansion))

        ordered_candidates = ordered(scout_scores)
        mode_finalists = []
        for copy_mode in copy_modes:
            mode_candidates = [
                candidate
                for candidate in ordered_candidates
                if candidate.copy_mode == copy_mode
            ]
            raw_best = min(mode_candidates, key=lambda candidate: scout_scores[candidate])
            best_ms = scout_scores[raw_best]
            equivalent = [
                candidate for candidate in mode_candidates
                if scout_scores[candidate] <= best_ms * (
                    1.0 + self.config.weight_sync_scout_tie_tolerance
                )
            ]

            def stable_key(candidate: WeightSyncLaunchConfig):
                if candidate.copy_mode == "thread":
                    return (
                        candidate.threads_per_block != 128,
                        abs(candidate.cta_multiplier - 2),
                        candidate.threads_per_block,
                        candidate.cta_multiplier,
                        scout_scores[candidate],
                    )
                return (
                    candidate.cta_multiplier,
                    candidate.lds_waves_per_destination,
                    scout_scores[candidate],
                )

            mode_finalists.append(min(equivalent, key=stable_key))

        finalists = tuple(dict.fromkeys(mode_finalists))
        scout_measurements = [
            (candidate, scout_scores[candidate])
            for candidate in ordered_candidates
        ]
        expanded_candidates = first_expansion | second_expansion
        real_layer_ids = list(dict.fromkeys(
            manager._real_layer_id(layer_id) for layer_id in layer_ids
        ))
        if manager.rank == 0:
            print(
                "UltraEP weight-sync autotune representative-layer scout: "
                f"virtual_layers={list(layer_ids)}, "
                f"real_layers={real_layer_ids}, "
                f"observation_iterations={self._weight_sync_observation_count}, "
                f"core_candidates={len(ordered_core)}, "
                f"expanded_candidates={len(expanded_candidates)}, "
                f"total_candidates={len(scout_measurements)}, "
                f"warmups/config={self.config.weight_sync_warmups}, "
                f"samples/config={self.config.weight_sync_repeats}",
                flush=True,
            )
            for candidate, elapsed_ms in scout_measurements:
                marker = "  <-- mode finalist" if candidate in finalists else ""
                tpb_text = (
                    f"{candidate.threads_per_block:3d}"
                    if candidate.copy_mode == "thread"
                    else "  -"
                )
                waves_text = (
                    "-"
                    if candidate.copy_mode == "thread"
                    else str(candidate.lds_waves_per_destination)
                )
                print(
                    f"  copy={candidate.copy_mode:<6} tpb={tpb_text} "
                    f"cta={candidate.cta_multiplier:2d} waves={waves_text} "
                    f"slowest-rank median representative-layer phase={elapsed_ms:.3f} ms{marker}",
                    flush=True,
                )
            print(
                "  One stable finalist per mode is marked above; "
                "final selection uses complete-iteration A/B.",
                flush=True,
            )
        return finalists

    def _start_weight_sync_e2e_probes(
        self,
        manager: "Manager",
        finalists: tuple[WeightSyncLaunchConfig, ...],
    ) -> None:
        # Include the original launch as a real-time baseline even when the
        # thread-mode microbenchmark found a different TPB/CTA pair.
        candidates = [self.current_weight_sync]
        for candidate in finalists:
            if candidate not in candidates:
                candidates.append(candidate)
        sequence = []
        for repeat in range(self.config.weight_sync_e2e_repeats_per_candidate):
            sequence.extend(candidates if repeat % 2 == 0 else reversed(candidates))
        self._weight_sync_probe_sequence = list(sequence)
        self._weight_sync_e2e_samples = {candidate: [] for candidate in candidates}
        self._weight_sync_probe_index = 0
        self._weight_sync_active_probe = self._weight_sync_probe_sequence[0]
        manager._apply_weight_sync_launch_config(self._weight_sync_active_probe)
        if manager.rank == 0:
            print(
                "UltraEP complete-iteration A/B started: "
                f"candidates={len(candidates)}, "
                f"samples/candidate={self.config.weight_sync_e2e_repeats_per_candidate}",
                flush=True,
            )

    def _select_weight_sync_from_e2e(self, manager: "Manager") -> WeightSyncLaunchConfig:
        medians = {
            candidate: self._median(values)
            for candidate, values in self._weight_sync_e2e_samples.items()
        }
        raw_best = min(medians, key=medians.get)
        raw_best_ms = medians[raw_best]

        # Treat modes within the configured tolerance as equivalent and prefer
        # the original mode.  This removes the old deterministic LDS tie bias.
        near_best = [
            candidate
            for candidate, elapsed_ms in medians.items()
            if elapsed_ms <= raw_best_ms * (1.0 + self.config.weight_sync_mode_tie_tolerance)
        ]
        same_mode = [
            candidate
            for candidate in near_best
            if candidate.copy_mode == self.current_weight_sync.copy_mode
        ]
        proposed = min(same_mode or near_best, key=lambda candidate: medians[candidate])
        baseline_ms = medians[self.current_weight_sync]
        speedup = (baseline_ms - medians[proposed]) / max(baseline_ms, 1e-9)
        selected = proposed if speedup >= self.config.minimum_weight_sync_speedup else self.current_weight_sync
        manager._apply_weight_sync_launch_config(selected)

        if manager.rank == 0:
            print("UltraEP complete-iteration A/B results (slowest rank):", flush=True)
            for candidate, elapsed_ms in sorted(medians.items(), key=lambda item: item[1]):
                marker = "  <-- selected" if candidate == selected else ""
                tpb_text = f"{candidate.threads_per_block:3d}" if candidate.copy_mode == "thread" else "  -"
                waves_text = "-" if candidate.copy_mode == "thread" else str(candidate.lds_waves_per_destination)
                print(
                    f"  copy={candidate.copy_mode:<6} tpb={tpb_text} "
                    f"cta={candidate.cta_multiplier:2d} waves={waves_text} "
                    f"median complete iteration={elapsed_ms:.3f} ms{marker}",
                    flush=True,
                )
            print(
                f"  baseline={baseline_ms:.3f} ms, selected={medians[selected]:.3f} ms, "
                f"required speedup={self.config.minimum_weight_sync_speedup * 100.0:.1f}%, "
                f"mode tie tolerance={self.config.weight_sync_mode_tie_tolerance * 100.0:.1f}%",
                flush=True,
            )
        return selected

    @staticmethod
    def _median(values: list[float]) -> float:
        ordered = sorted(values)
        size = len(ordered)
        middle = size // 2
        return ordered[middle] if size % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0

    def _record_weight_sync_observations(
        self,
        manager: "Manager",
        samples: dict[int, list[float]],
    ) -> None:
        """Aggregate rotating virtual slots into one observation per real layer."""
        current_by_real: dict[int, tuple[int, float]] = {}
        for virtual_layer_id, values in samples.items():
            if not values:
                continue
            score = self._p95(values)
            real_layer_id = manager._real_layer_id(virtual_layer_id)
            previous = current_by_real.get(real_layer_id)
            if previous is None or score > previous[1]:
                current_by_real[real_layer_id] = (virtual_layer_id, score)
        # A pipeline schedule may not expose every real layer in every one of
        # the observation iterations. Preserve the latest valid virtual slot
        # for every layer seen anywhere in the stable window.
        self._weight_sync_latest_virtual_by_real.update(current_by_real)
        for real_layer_id, (_, score) in current_by_real.items():
            self._weight_sync_real_layer_wait_ms.setdefault(real_layer_id, []).append(score)

    def _representative_layers(self) -> tuple[int, ...]:
        """Choose deterministic position anchors plus stable hot real layers."""
        available = sorted(
            real_layer_id
            for real_layer_id in self._weight_sync_real_layer_wait_ms
            if real_layer_id in self._weight_sync_latest_virtual_by_real
        )
        limit = self.config.weight_sync_scout_real_layers
        if len(available) <= limit:
            selected_real = available
        else:
            hot_count = min(self.config.weight_sync_scout_hot_layers, limit)
            anchor_count = limit - hot_count
            if anchor_count == 1:
                anchor_indices = [(len(available) - 1) // 2]
            elif anchor_count > 1:
                anchor_indices = [
                    round(index * (len(available) - 1) / (anchor_count - 1))
                    for index in range(anchor_count)
                ]
            else:
                anchor_indices = []
            anchors = list(dict.fromkeys(available[index] for index in anchor_indices))

            remaining = [layer_id for layer_id in available if layer_id not in anchors]
            hot_layers = []
            while remaining and len(hot_layers) < hot_count:
                scores = {
                    layer_id: self._median(self._weight_sync_real_layer_wait_ms[layer_id])
                    for layer_id in remaining
                }
                best_score = max(scores.values())
                equivalent = [
                    layer_id
                    for layer_id, score in scores.items()
                    if score >= best_score * (
                        1.0 - self.config.weight_sync_layer_tie_tolerance
                    )
                ]
                chosen = min(equivalent)
                hot_layers.append(chosen)
                remaining.remove(chosen)
            selected_real = anchors + hot_layers

        return tuple(
            self._weight_sync_latest_virtual_by_real[real_layer_id][0]
            for real_layer_id in selected_real
        )

    @staticmethod
    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

    @staticmethod
    def _integer_environment(name: str, default: int, valid: tuple[int, ...]) -> int:
        value = os.getenv(name)
        if value is None:
            return default
        parsed = int(value)
        if parsed not in valid:
            raise ValueError(f"{name} has unsupported value {parsed}")
        return parsed

    def _weight_sync_from_environment(self) -> WeightSyncLaunchConfig:
        copy_mode = os.getenv("ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE", "thread").lower()
        if copy_mode not in ("thread", "lds"):
            raise ValueError("ULTRA_EP_WEIGHT_SYNC_HIP_COPY_MODE must be thread or lds")
        return WeightSyncLaunchConfig(
            copy_mode=copy_mode,
            threads_per_block=self._integer_environment(
                "ULTRA_EP_WEIGHT_SYNC_THREADS_PER_BLOCK", 128, (64, 128, 256)
            ),
            cta_multiplier=self._integer_environment(
                "ULTRA_EP_WEIGHT_SYNC_CTA_MULTIPLIER", 2, tuple(range(1, 17))
            ),
            lds_waves_per_destination=self._integer_environment(
                "ULTRA_EP_WEIGHT_SYNC_LDS_WAVES_PER_DEST", 1, (1, 2, 4)
            ),
        )
