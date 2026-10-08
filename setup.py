import ast
import importlib.util
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import setuptools
import torch

# torch.utils.cpp_extension decides whether CUDAExtension should use its HIP
# path when the module is imported.  Some DTK PyTorch builds only consult
# ROCM_HOME (rather than ROCM_PATH) during that one-time detection.  Normalize
# the vendor ROCm/DTK root before importing cpp_extension so a HIP PyTorch does
# not fall through to CUDA SM architecture discovery.
if torch.version.hip is not None:
    _configured_rocm_root = (
        os.getenv("ROCM_HOME")
        or os.getenv("ROCM_PATH")
        or os.getenv("HIP_PATH")
    )
    if _configured_rocm_root:
        os.environ.setdefault("ROCM_HOME", _configured_rocm_root)

from torch.utils import cpp_extension

BuildExtension = cpp_extension.BuildExtension
CUDAExtension = cpp_extension.CUDAExtension


CURRENT_DIR = Path(__file__).resolve().parent
BUNDLED_ROCSHMEM_SOURCE = CURRENT_DIR / "third-party" / "rocshmem"
SUPPORTED_BACKENDS = {"cuda", "hip"}


def _split_env_flags(name):
    value = os.getenv(name, "").strip()
    return shlex.split(value) if value else []


def _first_env(*names):
    for name in names:
        value = os.getenv(name)
        if value:
            return Path(value).expanduser().resolve()
    return None


def _existing_dirs(paths):
    result = []
    for path in paths:
        path = Path(path)
        if path.is_dir() and str(path) not in result:
            result.append(str(path))
    return result


def _find_library(root, names):
    root = Path(root)
    for lib_dir_name in ("lib", "lib64"):
        lib_dir = root / lib_dir_name
        for name in names:
            candidate = lib_dir / name
            if candidate.is_file():
                return candidate.resolve()
            if name.endswith(".so"):
                versioned = sorted(lib_dir.glob(f"{name}.*"))
                if versioned:
                    return versioned[0].resolve()
    return None


def _library_dirs(root):
    return _existing_dirs((Path(root) / "lib", Path(root) / "lib64"))


def _env_enabled(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def detect_backend():
    requested = os.getenv("ULTRA_EP_BACKEND", "auto").strip().lower()
    if requested not in SUPPORTED_BACKENDS | {"auto"}:
        raise RuntimeError(
            "ULTRA_EP_BACKEND must be one of: auto, cuda, hip; "
            f"got {requested!r}"
        )

    detected = None
    if torch.version.hip is not None:
        detected = "hip"
    elif torch.version.cuda is not None:
        detected = "cuda"

    if detected is None:
        raise RuntimeError(
            "Unable to detect a GPU backend from this PyTorch installation. "
            "Install a CUDA/HIP-enabled PyTorch build; ULTRA_EP_BACKEND cannot "
            "cross-build against a CPU-only PyTorch installation."
        )
    backend = detected if requested == "auto" else requested
    if detected is not None and backend != detected:
        raise RuntimeError(
            f"ULTRA_EP_BACKEND={backend!r} conflicts with the PyTorch backend "
            f"({detected!r}). Use a matching PyTorch installation."
        )
    return backend


def get_package_version():
    init_file = CURRENT_DIR / "ultra_ep" / "__init__.py"
    with init_file.open("r", encoding="utf-8") as file:
        version_match = re.search(
            r"^__version__\s*=\s*(.*)$", file.read(), re.MULTILINE
        )
    if version_match is None:
        raise RuntimeError(f"Unable to find __version__ in {init_file}")
    public_version = ast.literal_eval(version_match.group(1))

    try:
        status_output = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=CURRENT_DIR
        ).decode("ascii").strip()
        if status_output:
            print(
                "Warning: Git working directory is not clean. "
                f"Uncommitted changes:\n{status_output}"
            )
        revision = "+" + subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=CURRENT_DIR
        ).decode("ascii").rstrip()
    except (OSError, subprocess.SubprocessError):
        revision = "+local"
    return f"{public_version}{revision}"


def find_cpp_gpu_sources(root_dir="csrc"):
    valid_exts = {".cpp", ".cc", ".cu", ".hip"}
    root = CURRENT_DIR / root_dir
    return sorted(
        str(path.relative_to(CURRENT_DIR))
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in valid_exts
        # PyTorch hipify emits foo.hip beside foo.cu.  On the next build that
        # generated file must not be treated as an additional source; foo.cu
        # remains canonical and will be hipified again.
        and not (path.suffix == ".hip" and path.with_suffix(".cu").is_file())
    )


def get_nvshmem_host_lib_name(base_dir):
    for lib_dir in _library_dirs(base_dir):
        unversioned = Path(lib_dir) / "libnvshmem_host.so"
        if unversioned.is_file():
            return unversioned.name
        versioned = sorted(Path(lib_dir).glob("libnvshmem_host.so.*"))
        if versioned:
            return versioned[0].name
    raise ModuleNotFoundError(
        f"libnvshmem_host.so was not found below {base_dir}/lib or {base_dir}/lib64"
    )


def find_nvshmem():
    nvshmem_dir = _first_env("NVSHMEM_DIR", "NVSHMEM_HOME")
    if nvshmem_dir is None:
        spec = importlib.util.find_spec("nvidia.nvshmem")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(
                "Unable to locate NVSHMEM. Install nvidia-nvshmem-cu12/13 "
                "or set NVSHMEM_DIR."
            )
        nvshmem_dir = Path(spec.submodule_search_locations[0]).resolve()
    if not nvshmem_dir.is_dir():
        raise RuntimeError(f"NVSHMEM directory does not exist: {nvshmem_dir}")
    return nvshmem_dir, get_nvshmem_host_lib_name(nvshmem_dir)


def find_rocm_root():
    configured = _first_env("ROCM_PATH", "ROCM_HOME", "HIP_PATH")
    if configured is not None:
        return configured

    hipcc = shutil.which("hipcc")
    if hipcc:
        hipcc_path = Path(hipcc).resolve()
        root = hipcc_path.parent.parent
        return root.parent if root.name == "hip" else root

    default_root = Path("/opt/rocm")
    if default_root.is_dir():
        return default_root
    raise RuntimeError(
        "Unable to locate the HIP/ROCm (or DTK) root. Set ROCM_PATH or HIP_PATH."
    )


def build_bundled_rocshmem(source_dir, rocm_root):
    source_dir = Path(source_dir).resolve()
    build_dir = Path(
        os.getenv("ROCSHMEM_BUILD_DIR", source_dir / "build-ultra-ep")
    ).expanduser().resolve()
    install_dir = Path(
        os.getenv("ROCSHMEM_INSTALL_DIR", source_dir / "install-ultra-ep")
    ).expanduser().resolve()

    backend = os.getenv("ULTRA_EP_ROCSHMEM_BACKEND", "ipc").strip().lower()
    backend_options = {
        "ipc": (False, True, False, True),
        "ro": (True, False, False, False),
        "ro_ipc": (True, True, False, False),
        "gda": (False, False, True, False),
        # GDA over the Dawning/SHCA provider, with IPC compiled in so the
        # rocSHMEM runtime can still use direct same-node transport.
        "gda_shca": (False, True, True, False),
    }
    if backend not in backend_options:
        raise RuntimeError(
            "ULTRA_EP_ROCSHMEM_BACKEND must be one of: "
            "ipc, ro, ro_ipc, gda, gda_shca"
        )
    use_ro, use_ipc, use_gda, single_node = backend_options[backend]
    gpu_targets = os.getenv("PYTORCH_ROCM_ARCH", "").strip()
    extra_cmake_args = _split_env_flags("ROCSHMEM_CMAKE_ARGS")
    build_signature = (
        f"backend={backend}\nrocm={rocm_root}\n"
        f"arch={gpu_targets or '<local>'}\ncmake_args={extra_cmake_args!r}\n"
    )
    signature_file = install_dir / ".ultra_ep_rocshmem_build"
    installed_library = _find_library(
        install_dir, ("librocshmem.a", "librocshmem.so")
    )
    installed_header = install_dir / "include" / "rocshmem" / "rocshmem.hpp"
    if (
        installed_library is not None
        and installed_header.is_file()
        and signature_file.is_file()
        and signature_file.read_text(encoding="utf-8") == build_signature
        and not _env_enabled("ROCSHMEM_FORCE_REBUILD")
    ):
        return install_dir, installed_library

    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError(
            "cmake is required to build third-party/rocshmem. Install cmake, "
            "or set ROCSHMEM_DIR to a prebuilt rocSHMEM installation."
        )

    configure_command = [
        cmake,
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        f"-DCMAKE_INSTALL_PREFIX={install_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
        f"-DROCM_PATH={rocm_root}",
        f"-DUSE_RO={'ON' if use_ro else 'OFF'}",
        f"-DUSE_IPC={'ON' if use_ipc else 'OFF'}",
        f"-DUSE_GDA={'ON' if use_gda else 'OFF'}",
        f"-DGDA_SHCA={'ON' if backend == 'gda_shca' else 'OFF'}",
        f"-DUSE_SINGLE_NODE={'ON' if single_node else 'OFF'}",
        "-DUSE_EXTERNAL_MPI=OFF",
        "-DBUILD_EXAMPLES=OFF",
        "-DBUILD_FUNCTIONAL_TESTS=OFF",
        "-DBUILD_UNIT_TESTS=OFF",
        "-DBUILD_TOOLS=OFF",
    ]
    hipcc = Path(rocm_root) / "bin" / "hipcc"
    if hipcc.is_file():
        configure_command.append(f"-DCMAKE_CXX_COMPILER={hipcc}")

    if gpu_targets:
        configure_command.append(
            f"-DGPU_TARGETS={gpu_targets.replace(',', ';')}"
        )
    else:
        configure_command.append("-DBUILD_LOCAL_GPU_TARGET_ONLY=ON")
    configure_command.extend(extra_cmake_args)

    print(
        f"Building bundled rocSHMEM ({backend} backend) from {source_dir} "
        f"into {install_dir}"
    )
    subprocess.run(configure_command, check=True, cwd=CURRENT_DIR)
    build_jobs = os.getenv("MAX_JOBS", str(os.cpu_count() or 8))
    subprocess.run(
        [
            cmake,
            "--build",
            str(build_dir),
            "--target",
            "install",
            "--parallel",
            build_jobs,
        ],
        check=True,
        cwd=CURRENT_DIR,
    )

    installed_library = _find_library(
        install_dir, ("librocshmem.a", "librocshmem.so")
    )
    if installed_library is None or not installed_header.is_file():
        raise RuntimeError(
            f"Bundled rocSHMEM build completed but its install is incomplete: {install_dir}"
        )
    signature_file.write_text(build_signature, encoding="utf-8")
    return install_dir, installed_library


def find_rocshmem(rocm_root):
    configured = _first_env("ROCSHMEM_DIR", "ROCSHMEM_HOME")
    if configured is None and BUNDLED_ROCSHMEM_SOURCE.is_dir():
        return build_bundled_rocshmem(BUNDLED_ROCSHMEM_SOURCE, rocm_root)

    roots = [configured] if configured is not None else []
    roots.extend((Path(rocm_root), Path(rocm_root) / "rocshmem"))

    explicit_library = os.getenv("ROCSHMEM_LIBRARY")
    for root in roots:
        if root is None or not root.is_dir():
            continue
        include_dir = root / "include"
        if not include_dir.is_dir():
            continue

        if explicit_library:
            library = Path(explicit_library).expanduser().resolve()
            if not library.is_file():
                raise RuntimeError(
                    f"ROCSHMEM_LIBRARY does not exist: {library}"
                )
        else:
            # rocSHMEM applications are normally linked to the static archive.
            # Accept a shared library as a DTK/vendor packaging fallback.
            library = _find_library(root, ("librocshmem.a", "librocshmem.so"))
        if library is not None:
            return root.resolve(), library

        # ROCSHMEM_DIR may point at a source checkout rather than an install.
        if (root / "CMakeLists.txt").is_file() and (
            root / "include" / "rocshmem" / "rocshmem.hpp"
        ).is_file():
            return build_bundled_rocshmem(root, rocm_root)

    raise RuntimeError(
        "Unable to locate rocSHMEM headers, a rocSHMEM source checkout, or "
        "librocshmem. Set ROCSHMEM_DIR or ROCSHMEM_LIBRARY."
    )


def find_rccl(rocm_root):
    configured = _first_env("RCCL_DIR", "RCCL_HOME")
    roots = [configured] if configured is not None else []
    roots.append(Path(rocm_root))
    for root in roots:
        if root is None or not root.is_dir():
            continue
        library = _find_library(root, ("librccl.so", "librccl.a"))
        if library is not None:
            return root.resolve(), library
    raise RuntimeError(
        "Unable to locate RCCL. Set RCCL_DIR to an installation prefix "
        "containing include/ and lib/librccl.so (or lib64/librccl.so)."
    )


def configure_cuda_build(cxx_flags, gpu_flags):
    nvshmem_dir, nvshmem_host_lib = find_nvshmem()
    include_dirs = _existing_dirs((CURRENT_DIR / "csrc", nvshmem_dir / "include"))
    library_dirs = _library_dirs(nvshmem_dir)

    cxx_flags.extend(("-DULTRA_EP_USE_CUDA=1", "-DULTRA_EP_USE_NVSHMEM=1"))
    gpu_flags.extend(
        (
            "-DULTRA_EP_USE_CUDA=1",
            "-DULTRA_EP_USE_NVSHMEM=1",
            "-rdc=true",
            "--ptxas-options=--register-usage-level=10",
        )
    )

    if "TORCH_CUDA_ARCH_LIST" not in os.environ:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        compute_cap = result.stdout.strip().splitlines()[0].strip()
        sm = int(compute_cap.replace(".", ""))
        if sm not in (90, 100):
            raise RuntimeError(
                f"Unsupported CUDA compute capability: {compute_cap} (SM{sm}). "
                "Only SM90 and SM100 are supported. Set TORCH_CUDA_ARCH_LIST "
                "manually to override."
            )
        os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0" if sm == 90 else "10.0"

    if torch.version.cuda and torch.version.cuda.startswith("13."):
        cccl_dir = Path(os.getenv("CUDA_HOME", "/usr/local/cuda")) / "include" / "cccl"
        include_dirs.extend(_existing_dirs((cccl_dir,)))

    dlink_flags = ["-dlink", f"-L{library_dirs[0]}", "-lnvshmem_device"]
    link_args = [
        "-lcuda",
        f"-l:{nvshmem_host_lib}",
        "-l:libnvshmem_device.a",
        f"-Wl,-rpath,{library_dirs[0]}",
        "-Wl,--allow-multiple-definition",
    ]
    return {
        "backend": "cuda",
        "arch": os.environ["TORCH_CUDA_ARCH_LIST"],
        "shmem": nvshmem_dir,
        "rccl": None,
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "dlink_flags": dlink_flags,
        "link_args": link_args,
    }


def configure_hip_build(cxx_flags, gpu_flags):
    rocm_root = find_rocm_root()
    hipcc = Path(rocm_root) / "bin" / "hipcc"
    if not hipcc.is_file():
        configured_hipcc = shutil.which("hipcc")
        if configured_hipcc is None:
            raise RuntimeError(f"hipcc was not found below {rocm_root} or PATH")
        hipcc = Path(configured_hipcc).resolve()

    # The final shared-object link must also be driven by hipcc.  This is
    # required for --hip-link to consume relocatable device code from the
    # UltraEP objects and the static rocSHMEM archive.
    os.environ["CC"] = str(hipcc)
    os.environ["CXX"] = str(hipcc)

    rocshmem_dir, rocshmem_library = find_rocshmem(rocm_root)
    rccl_dir, rccl_library = find_rccl(rocm_root)

    include_dirs = _existing_dirs(
        (
            CURRENT_DIR / "csrc",
            Path(rocm_root) / "include",
            rocshmem_dir / "include",
            rccl_dir / "include",
        )
    )
    library_dirs = _existing_dirs(
        (
            rocshmem_library.parent,
            rccl_library.parent,
            Path(rocm_root) / "lib",
            Path(rocm_root) / "lib64",
        )
    )

    common_defines = (
        "-DULTRA_EP_USE_HIP=1",
        "-DULTRA_EP_USE_ROCSHMEM=1",
        "-DULTRA_EP_USE_RCCL=1",
        "-DULTRA_EP_WAVE_SIZE=64",
    )
    cxx_flags.extend(common_defines)
    gpu_flags.extend((*common_defines, "-fgpu-rdc"))

    # CUDAExtension uses PyTorch's HIP build path when torch.version.hip is set.
    # PYTORCH_ROCM_ARCH is consumed by torch.utils.cpp_extension; if it is not
    # set, PyTorch compiles for all visible device architectures.
    configured_arch = os.getenv("PYTORCH_ROCM_ARCH", "").strip()
    arch = configured_arch or "<all visible HIP devices>"
    link_arches = [
        value
        for value in re.split(r"[;,\s]+", configured_arch)
        if value
    ]
    link_arch_flags = [f"--offload-arch={value}" for value in link_arches]

    link_args = [
        "-fgpu-rdc",
        "--hip-link",
        *link_arch_flags,
        str(rocshmem_library),
        str(rccl_library),
        "-lamdhip64",
        "-lhsa-runtime64",
        "-Wl,--allow-multiple-definition",
    ]
    link_args.extend(f"-Wl,-rpath,{path}" for path in library_dirs)
    link_args.extend(_split_env_flags("ROCSHMEM_EXTRA_LINK_ARGS"))
    link_args.extend(_split_env_flags("RCCL_EXTRA_LINK_ARGS"))

    # Do not use PyTorch's nvcc_dlink channel for HIP.  In the DTK PyTorch
    # build that path enters CUDA SM discovery even when the ordinary source
    # compilation path correctly recognizes HIP.  hipcc performs the device
    # link as part of the final shared-object link above.
    link_args.extend(_split_env_flags("ROCSHMEM_DLINK_FLAGS"))

    return {
        "backend": "hip",
        "arch": arch,
        "shmem": rocshmem_dir,
        "rccl": rccl_dir,
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "dlink_flags": [],
        "link_args": link_args,
    }


def make_extension():
    backend = detect_backend()
    if backend == "hip" and not getattr(
        cpp_extension, "IS_HIP_EXTENSION", False
    ):
        raise RuntimeError(
            "PyTorch reports a HIP build, but torch.utils.cpp_extension did "
            "not enable its HIP extension path. Set ROCM_HOME and ROCM_PATH "
            "to the DTK root (normally /opt/dtk) before running setup.py. "
            f"ROCM_HOME={os.getenv('ROCM_HOME')!r}, "
            f"ROCM_PATH={os.getenv('ROCM_PATH')!r}, "
            f"torch.version.hip={torch.version.hip!r}."
        )
    cxx_flags = [
        "-O3",
        "-Wno-deprecated-declarations",
        "-Wno-unused-variable",
        "-Wno-sign-compare",
        "-Wno-reorder",
        "-Wno-attributes",
    ]
    gpu_flags = ["-O3"]

    if backend == "cuda":
        gpu_flags.extend(("-Xcompiler", "-O3"))
        config = configure_cuda_build(cxx_flags, gpu_flags)
    else:
        config = configure_hip_build(cxx_flags, gpu_flags)

    if int(os.getenv("DISABLE_AGGRESSIVE_PTX_INSTRS", "1")):
        cxx_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")
        gpu_flags.append("-DDISABLE_AGGRESSIVE_PTX_INSTRS")

    cxx_flags.extend(_split_env_flags("ULTRA_EP_EXTRA_CXX_FLAGS"))
    gpu_flags.extend(_split_env_flags("ULTRA_EP_EXTRA_GPU_FLAGS"))
    config["link_args"].extend(_split_env_flags("ULTRA_EP_EXTRA_LINK_ARGS"))

    extra_compile_args = {"cxx": cxx_flags, "nvcc": gpu_flags}
    if config["dlink_flags"]:
        extra_compile_args["nvcc_dlink"] = config["dlink_flags"]

    sources = find_cpp_gpu_sources()
    print("Build summary:")
    print(f" > Backend: {config['backend']}")
    print(f" > PyTorch CUDA version: {torch.version.cuda}")
    print(f" > PyTorch HIP version: {torch.version.hip}")
    print(f" > Architecture: {config['arch']}")
    print(f" > Sources: {sources}")
    print(f" > Includes: {config['include_dirs']}")
    print(f" > Library directories: {config['library_dirs']}")
    print(f" > SHMEM path: {config['shmem']}")
    print(f" > RCCL path: {config['rccl']}")
    print(f" > Compilation flags: {extra_compile_args}")
    print(f" > Link flags: {config['link_args']}")
    print()

    return CUDAExtension(
        name="ultra_ep._C",
        include_dirs=config["include_dirs"],
        library_dirs=config["library_dirs"],
        sources=sources,
        extra_compile_args=extra_compile_args,
        extra_link_args=config["link_args"],
    )


if __name__ == "__main__":
    setuptools.setup(
        name="ultra_ep",
        version=get_package_version(),
        description="Real-time expert load balancing for large-scale MoE systems",
        long_description=(CURRENT_DIR / "README.md").read_text(encoding="utf-8"),
        long_description_content_type="text/markdown",
        url="https://github.com/HYGON-AI/UltraEP-das",
        license="MIT",
        python_requires=">=3.10",
        packages=setuptools.find_packages(include=["ultra_ep", "ultra_ep.*"]),
        ext_modules=[make_extension()],
        # RDC/device linking is supported only by the Ninja path in PyTorch's
        # extension builder, for both the existing CUDA backend and HIP.
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
        classifiers=[
            "Development Status :: 4 - Beta",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )
