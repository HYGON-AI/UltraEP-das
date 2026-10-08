import os
import socket
import torch
import torch.distributed as dist
from .util import print_rank_0

# noinspection PyUnresolvedReferences
import ultra_ep._C as _C

_group = None
_nvl_domain_size = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _validate_node_aligned_domains(
    group: dist.ProcessGroup, domain_size: int
) -> None:
    """Validate the contiguous 8-rank-per-node layout used by the HCU port."""

    group_size = group.size()
    if group_size <= domain_size or torch.version.hip is None:
        return
    if not _env_flag("ULTRA_EP_VALIDATE_NODE_DOMAINS", default=True):
        return

    # A custom EP subgroup can legally use a different rank ordering. Require
    # explicit opt-in before applying the torchrun WORLD-layout check to it.
    if group != dist.group.WORLD:
        if _env_flag("ULTRA_EP_VALIDATE_CUSTOM_GROUP_DOMAINS"):
            raise RuntimeError(
                "ULTRA_EP_VALIDATE_CUSTOM_GROUP_DOMAINS is not implemented; "
                "provide a WORLD-aligned EP group or disable node-domain validation"
            )
        return

    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "0"))
    if local_world_size and local_world_size != domain_size:
        raise RuntimeError(
            "HCU multi-node mode expects one scale-up domain per node: "
            f"LOCAL_WORLD_SIZE={local_world_size}, MAX_NUM_NVL_PEERS={domain_size}. "
            "Set ULTRA_EP_VALIDATE_NODE_DOMAINS=0 only for a verified custom topology."
        )

    record = (socket.gethostname(), int(os.getenv("LOCAL_RANK", "-1")))
    records = [None] * group_size
    dist.all_gather_object(records, record, group=group)
    domain_hosts = []
    expected_local_ranks = list(range(domain_size))
    for start in range(0, group_size, domain_size):
        domain = records[start : start + domain_size]
        hosts = {item[0] for item in domain}
        if len(hosts) != 1:
            raise RuntimeError(
                "EP ranks are not grouped into contiguous node-local scale-up domains: "
                f"ranks [{start}, {start + domain_size}) report hosts {sorted(hosts)}"
            )
        local_ranks = sorted(item[1] for item in domain)
        if local_ranks[0] >= 0 and local_ranks != expected_local_ranks:
            raise RuntimeError(
                f"ranks [{start}, {start + domain_size}) report LOCAL_RANK values "
                f"{local_ranks}, expected {expected_local_ranks}"
            )
        domain_hosts.append(next(iter(hosts)))
    if len(set(domain_hosts)) != len(domain_hosts):
        raise RuntimeError(
            "Multiple contiguous scale-up domains resolved to the same hostname: "
            f"{domain_hosts}"
        )


def init_runtime(group: dist.ProcessGroup):
    global _group, _nvl_domain_size
    if _C.is_runtime_initialized():
        assert group == _group, "All EP buffers should share the same process group"
        return _nvl_domain_size

    if torch.version.hip is None:
        # NVSHMEM initialization requires at least 256 MiB. Disable NVLink
        # SHArP because UltraEP obtains direct remote pointers itself.
        os.environ["NVSHMEM_DISABLE_NVLS"] = "1"
        os.environ["NVSHMEM_CUMEM_GRANULARITY"] = f"{2 ** 29}"
        os.environ.setdefault("NVSHMEM_DISABLE_NCCL", "1")
    else:
        # The bundled rocSHMEM defaults to a 1 GiB symmetric heap. Users can
        # increase this before initialization for larger expert buffers.
        os.environ.setdefault("ROCSHMEM_HEAP_SIZE", str(2 ** 30))

    # Support both scale-up and RDMA ranks by setting MAX_NUM_NVL_PEERS.
    # When the topology is explicitly configured, do not create IpcManager
    # merely to verify it. On HIP, IpcManager exports a temporary allocation
    # with hipIpcGetMemHandle; some runtimes retain export-registration state
    # until process exit even after hipFree, contaminating repeated lifecycle
    # tests and serving no purpose when the domain size is already known.
    max_nvl_peers = os.getenv("MAX_NUM_NVL_PEERS")
    if max_nvl_peers is not None:
        max_nvl_peers = int(max_nvl_peers)
        if max_nvl_peers <= 0 or max_nvl_peers > group.size():
            raise ValueError(
                "MAX_NUM_NVL_PEERS must be in [1, group.size()], "
                f"got {max_nvl_peers} for group size {group.size()}"
            )
        print_rank_0(
            f"[INFO] Use configured scale-up domain: {max_nvl_peers} ranks "
            "(IPC topology probe skipped)"
        )
    else:
        _ipc_manager = _C.IpcManager()
        print_rank_0(
            f"[INFO] Use MNNVL fabric: {_ipc_manager.is_fabric_supported()}"
        )
        detected_ranks = _ipc_manager.detect_accessible_ranks(group)
        del _ipc_manager
        max_nvl_peers = detected_ranks
        print_rank_0(
            f"[WARN] MAX_NUM_NVL_PEERS is not set. Using detected value {detected_ranks}."
        )

    if group.size() % max_nvl_peers != 0:
        raise ValueError(
            f"EP group size {group.size()} must be divisible by the scale-up "
            f"domain size {max_nvl_peers}"
        )
    _validate_node_aligned_domains(group, max_nvl_peers)

    # Synchronize the selected SHMEM runtime's unique ID only after topology
    # validation, so a bad multi-node rank layout fails before rocSHMEM starts.
    root_unique_id = None
    if group.rank() == 0:
        root_unique_id = _C.get_local_shmem_unique_id(group.rank())
    shmem_unique_ids = [None] * group.size()
    dist.all_gather_object(shmem_unique_ids, root_unique_id, group)
    root_unique_id = shmem_unique_ids[0]

    print_rank_0(f"[INFO] SHMEM backend: {_C.get_shmem_backend_name()}")
    _C.init_runtime(group.rank(), group.size(), max_nvl_peers, root_unique_id)
    runtime_info = _C.get_shmem_runtime_info()
    if runtime_info["pe"] != group.rank() or runtime_info["num_pes"] != group.size():
        raise RuntimeError(
            "rocSHMEM PE mapping differs from the EP process group: "
            f"PE {runtime_info['pe']}/{runtime_info['num_pes']} versus "
            f"group rank {group.rank()}/{group.size()}"
        )
    if group.size() > max_nvl_peers and runtime_info["transport"] == "ipc":
        raise RuntimeError(
            "rocSHMEM selected the IPC-only transport for a multi-node EP group. "
            "Build the gda_shca preset and set ROCSHMEM_BACKEND=gda and "
            "ROCSHMEM_GDA_PROVIDER=shca."
        )
    required_transport = os.getenv("ULTRA_EP_REQUIRE_ROCSHMEM_TRANSPORT")
    if required_transport and runtime_info["transport"] != required_transport.lower():
        raise RuntimeError(
            f"rocSHMEM selected transport={runtime_info['transport']}, expected "
            f"{required_transport.lower()}"
        )
    print_rank_0(
        "[INFO] SHMEM runtime: "
        f"backend={runtime_info['backend']}, transport={runtime_info['transport']}, "
        f"PEs={runtime_info['num_pes']}, scale-up-domain={runtime_info['scale_up_size']}, "
        f"domains={runtime_info['num_domains']}"
    )

    # Remember the EP group, which can not be changed anymore
    _group = group
    _nvl_domain_size = max_nvl_peers

    return max_nvl_peers
