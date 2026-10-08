from dataclasses import dataclass
import math

from .util import read_bool_env, read_float_env, read_int_env, read_str_env


_WEIGHT_SYNC_PLAN_MODE_IDS = {
    "direct": 0,
    "adaptive": 1,
    "forcerelay": 2,
}

# These are the public UltraEP defaults documented in README.md and docs/hcu.md.
# Keep the C++ constructor and pybind defaults in csrc/ultra_ep.hpp aligned with
# this table so direct native users and Python users observe identical behavior.
DEFAULT_BALANCE_THRESHOLD = 1.0
DEFAULT_QUOTA_LOCALITY_AWARE = True
DEFAULT_QUOTA_MIN_TOKENS_PER_REPLICA = 1024
DEFAULT_QUOTA_ALLOW_ZERO_MASTER_QUOTA = False
DEFAULT_GRAD_REDUCE_NUM_SMS = 24
DEFAULT_GRAD_REDUCE_DETERMINISTIC = True
DEFAULT_QUOTA_ORACLE_EPS = 0.01
DEFAULT_QUOTA_KERNEL_STAGE = 1
DEFAULT_QUOTA_REROUTE_INTERLEAVE = True
DEFAULT_WEIGHT_SYNC_PLAN_MODE = "adaptive"
DEFAULT_WEIGHT_SYNC_RELAY_MIN_REPLICAS = 4
DEFAULT_WEIGHT_SYNC_RELAY_MAX_RELAYS = 8
DEFAULT_WEIGHT_SYNC_RELAY_MIN_FANOUT_GAIN = 2


def _normalize_weight_sync_plan_mode(value: str) -> str:
    normalized = value.lower().replace("_", "")
    if normalized not in _WEIGHT_SYNC_PLAN_MODE_IDS:
        raise ValueError(
            "ULTRA_EP_WEIGHT_SYNC_PLAN_MODE must be one of: direct, adaptive, force_relay"
        )
    return normalized


@dataclass(frozen=True)
class UltraEPTuning:
    balance_threshold: float
    quota_locality_aware: bool
    quota_min_tokens_per_replica: int
    quota_allow_zero_master_quota: bool
    grad_reduce_num_sms: int
    grad_reduce_deterministic: bool
    quota_oracle_eps: float
    quota_kernel_stage: int
    quota_reroute_interleave: bool
    weight_sync_plan_mode: str
    weight_sync_plan_mode_id: int
    weight_sync_relay_min_replicas: int
    weight_sync_relay_max_relays: int
    weight_sync_relay_min_fanout_gain: int


def load_tuning_from_env() -> UltraEPTuning:
    weight_sync_plan_mode = _normalize_weight_sync_plan_mode(
        read_str_env("ULTRA_EP_WEIGHT_SYNC_PLAN_MODE", DEFAULT_WEIGHT_SYNC_PLAN_MODE)
    )
    grad_reduce_num_sms = read_int_env(
        "ULTRA_EP_GRAD_REDUCE_NUM_SMS", DEFAULT_GRAD_REDUCE_NUM_SMS
    )
    if grad_reduce_num_sms <= 0:
        raise ValueError("ULTRA_EP_GRAD_REDUCE_NUM_SMS must be positive")
    if grad_reduce_num_sms % 2 != 0:
        raise ValueError("ULTRA_EP_GRAD_REDUCE_NUM_SMS must be even")
    grad_reduce_deterministic = read_bool_env(
        "ULTRA_EP_GRAD_REDUCE_DETERMINISTIC", DEFAULT_GRAD_REDUCE_DETERMINISTIC
    )

    balance_threshold = read_float_env(
        "ULTRA_EP_BALANCE_THRESHOLD", DEFAULT_BALANCE_THRESHOLD
    )
    if not math.isfinite(balance_threshold) or balance_threshold < 1.0:
        raise ValueError("ULTRA_EP_BALANCE_THRESHOLD must be finite and >= 1.0")

    quota_min_tokens_per_replica = read_int_env(
        "ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA",
        DEFAULT_QUOTA_MIN_TOKENS_PER_REPLICA,
    )
    if quota_min_tokens_per_replica <= 0:
        raise ValueError("ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA must be positive")

    quota_oracle_eps = read_float_env(
        "ULTRA_EP_QUOTA_ORACLE_EPS", DEFAULT_QUOTA_ORACLE_EPS
    )
    if not math.isfinite(quota_oracle_eps) or quota_oracle_eps < 0.0:
        raise ValueError("ULTRA_EP_QUOTA_ORACLE_EPS must be finite and non-negative")

    quota_kernel_stage = read_int_env(
        "ULTRA_EP_QUOTA_KERNEL_STAGE", DEFAULT_QUOTA_KERNEL_STAGE
    )
    if quota_kernel_stage not in (0, 1):
        raise ValueError("ULTRA_EP_QUOTA_KERNEL_STAGE supports only 0 or 1")

    relay_min_replicas = read_int_env(
        "ULTRA_EP_WEIGHT_SYNC_RELAY_MIN_REPLICAS",
        DEFAULT_WEIGHT_SYNC_RELAY_MIN_REPLICAS,
    )
    if relay_min_replicas < 0:
        raise ValueError("ULTRA_EP_WEIGHT_SYNC_RELAY_MIN_REPLICAS must be non-negative")

    relay_max_relays = read_int_env(
        "ULTRA_EP_WEIGHT_SYNC_RELAY_MAX_RELAYS", DEFAULT_WEIGHT_SYNC_RELAY_MAX_RELAYS
    )
    if relay_max_relays < 1:
        raise ValueError("ULTRA_EP_WEIGHT_SYNC_RELAY_MAX_RELAYS must be positive")

    relay_min_fanout_gain = read_int_env(
        "ULTRA_EP_WEIGHT_SYNC_RELAY_MIN_FANOUT_GAIN",
        DEFAULT_WEIGHT_SYNC_RELAY_MIN_FANOUT_GAIN,
    )
    if relay_min_fanout_gain < 0:
        raise ValueError(
            "ULTRA_EP_WEIGHT_SYNC_RELAY_MIN_FANOUT_GAIN must be non-negative"
        )

    return UltraEPTuning(
        balance_threshold=balance_threshold,
        quota_locality_aware=read_bool_env(
            "ULTRA_EP_QUOTA_LOCALITY_AWARE", DEFAULT_QUOTA_LOCALITY_AWARE
        ),
        quota_min_tokens_per_replica=quota_min_tokens_per_replica,
        quota_allow_zero_master_quota=read_bool_env(
            "ULTRA_EP_QUOTA_ALLOW_ZERO_MASTER_QUOTA",
            DEFAULT_QUOTA_ALLOW_ZERO_MASTER_QUOTA,
        ),
        grad_reduce_num_sms=grad_reduce_num_sms,
        grad_reduce_deterministic=grad_reduce_deterministic,
        quota_oracle_eps=quota_oracle_eps,
        quota_kernel_stage=quota_kernel_stage,
        quota_reroute_interleave=read_bool_env(
            "ULTRA_EP_QUOTA_REROUTE_INTERLEAVE", DEFAULT_QUOTA_REROUTE_INTERLEAVE
        ),
        weight_sync_plan_mode=weight_sync_plan_mode,
        weight_sync_plan_mode_id=_WEIGHT_SYNC_PLAN_MODE_IDS[weight_sync_plan_mode],
        weight_sync_relay_min_replicas=relay_min_replicas,
        weight_sync_relay_max_relays=relay_max_relays,
        weight_sync_relay_min_fanout_gain=relay_min_fanout_gain,
    )
