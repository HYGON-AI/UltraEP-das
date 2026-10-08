#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

set -euo pipefail

# Build UltraEP against the bundled rocSHMEM GDA backend for a Dawning/SHCA
# network, then optionally run a command with the same runtime environment.
#
# Examples:
#   ./build_hcu_shca.sh
#   ./build_hcu_shca.sh --force-rocshmem
#   ./build_hcu_shca.sh --wheel
#   ./build_hcu_shca.sh --no-build -- torchrun --nproc_per_node=8 ...

original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
trap 'cd "$original_dir"' EXIT
cd "$script_dir"

build_extension=1
build_wheel=0
force_rocshmem=0
run_command=()

usage() {
    echo "Usage: $0 [--force-rocshmem] [--wheel] [--no-build] [-- command ...]"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --force-rocshmem)
            force_rocshmem=1
            shift
            ;;
        --wheel)
            build_wheel=1
            shift
            ;;
        --no-build)
            build_extension=0
            shift
            ;;
        --)
            shift
            run_command=("$@")
            break
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

prepend_env_path() {
    local variable_name=$1
    local directory=$2
    [ -d "$directory" ] || return 0
    local current_value=${!variable_name:-}
    if [ -n "$current_value" ]; then
        export "$variable_name=$directory:$current_value"
    else
        export "$variable_name=$directory"
    fi
}

find_shca_header_dir() {
    local candidate
    for candidate in \
        "${SHCA_INCLUDE_DIR:-}" \
        "${SHCA_ROOT:-}/include" \
        /usr/include \
        /usr/local/include; do
        [ -n "$candidate" ] || continue
        if [ -f "$candidate/infiniband/shca_dv.h" ]; then
            realpath "$candidate"
            return 0
        fi
    done
    return 1
}

find_shca_library_dir() {
    local candidate
    for candidate in \
        "${SHCA_LIBRARY_DIR:-}" \
        "${SHCA_ROOT:-}/lib64" \
        "${SHCA_ROOT:-}/lib" \
        /usr/lib64 \
        /usr/lib \
        /usr/local/lib64 \
        /usr/local/lib; do
        [ -n "$candidate" ] || continue
        if compgen -G "$candidate/libshca.so*" >/dev/null; then
            realpath "$candidate"
            return 0
        fi
    done

    if command -v ldconfig >/dev/null 2>&1; then
        local library_path
        library_path=$(ldconfig -p 2>/dev/null | awk '/libshca\.so/{print $NF; exit}')
        if [ -n "$library_path" ] && [ -f "$library_path" ]; then
            dirname "$(realpath "$library_path")"
            return 0
        fi
    fi
    return 1
}

detect_gfx_arches() {
    local detected=""
    if command -v rocm_agent_enumerator >/dev/null 2>&1; then
        detected=$(rocm_agent_enumerator 2>/dev/null \
            | sed 's/:.*//' \
            | grep -E '^gfx[0-9a-f]+$' \
            | grep -v '^gfx000$' \
            | sort -u \
            | paste -sd';' - || true)
    fi
    if [ -z "$detected" ] && command -v rocminfo >/dev/null 2>&1; then
        detected=$(rocminfo 2>/dev/null \
            | grep -Eo 'gfx[0-9a-f]+' \
            | grep -v '^gfx000$' \
            | sort -u \
            | paste -sd';' - || true)
    fi
    [ -n "$detected" ] || return 1
    echo "$detected"
}

remove_hipify_outputs() {
    local source_root="${script_dir}/csrc"
    [ -d "$source_root" ] || return 0

    # hipify writes foo.hip next to canonical foo.cu, and may write
    # foo_hip.cuh next to foo.cuh.  Delete only outputs whose source sibling
    # exists, so native HIP sources are never removed.
    while IFS= read -r -d '' generated_file; do
        local source_file="${generated_file%.hip}.cu"
        if [ -f "$source_file" ]; then
            rm -f -- "$generated_file"
            echo "Removed generated hipify file: ${generated_file#$script_dir/}"
        fi
    done < <(find "$source_root" -type f -name '*.hip' -print0)

    while IFS= read -r -d '' generated_file; do
        local source_file="${generated_file%_hip.cuh}.cuh"
        if [ -f "$source_file" ]; then
            rm -f -- "$generated_file"
            echo "Removed generated hipify file: ${generated_file#$script_dir/}"
        fi
    done < <(find "$source_root" -type f -name '*_hip.cuh' -print0)
}

export ROCM_PATH=${ROCM_PATH:-/opt/dtk}
export ROCM_HOME=${ROCM_HOME:-${ROCM_PATH}}
export HIP_HOME=${HIP_HOME:-${ROCM_PATH}/hip}
export ULTRA_EP_BACKEND=hip
export ULTRA_EP_ROCSHMEM_BACKEND=gda_shca
# The target HCU exposes 80 CUs.  Grad Reduce uses persistent CTAs and the
# generic Python default (42) underutilizes this device.  Preserve an explicit
# caller value so other HCU variants can tune it without editing the script.
export ULTRA_EP_GRAD_REDUCE_NUM_SMS=${ULTRA_EP_GRAD_REDUCE_NUM_SMS:-64}

# A CUDA SM list makes torch.utils.cpp_extension generate nvcc -gencode flags.
# HCU/HIP targets are supplied through PYTORCH_ROCM_ARCH below instead.
unset TORCH_CUDA_ARCH_LIST

python_bin=${PYTHON_BIN:-python3}
rocshmem_source=${script_dir}/third-party/rocshmem
export ROCSHMEM_BUILD_DIR=${ROCSHMEM_BUILD_DIR:-${rocshmem_source}/build-ultra-ep-shca}
export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR:-${rocshmem_source}/install-ultra-ep-shca}

if [ "$build_extension" -eq 1 ] \
    && [ -z "${ROCSHMEM_DIR:-${ROCSHMEM_HOME:-}}" ] \
    && [ ! -f "${rocshmem_source}/CMakeLists.txt" ]; then
    echo "The rocSHMEM submodule is not initialized." >&2
    echo "Run 'git submodule update --init --recursive', or set ROCSHMEM_DIR to a compatible installed rocSHMEM." >&2
    exit 1
fi

prepend_env_path PATH "${ROCM_PATH}/bin"
prepend_env_path PATH "${HIP_HOME}/bin"
prepend_env_path LD_LIBRARY_PATH "${ROCM_PATH}/lib"
prepend_env_path LD_LIBRARY_PATH "${ROCM_PATH}/lib64"
prepend_env_path LD_LIBRARY_PATH "${HIP_HOME}/lib"
prepend_env_path LIBRARY_PATH "${ROCM_PATH}/lib"
prepend_env_path LIBRARY_PATH "${ROCM_PATH}/lib64"
prepend_env_path CPATH "${ROCM_PATH}/include"
prepend_env_path CPLUS_INCLUDE_PATH "${ROCM_PATH}/include"

if [ -d /opt/mpi ]; then
    prepend_env_path LD_LIBRARY_PATH /opt/mpi/lib
    prepend_env_path LIBRARY_PATH /opt/mpi/lib
    prepend_env_path CPATH /opt/mpi/include
    prepend_env_path CPLUS_INCLUDE_PATH /opt/mpi/include
fi

if [ -d "${ROCM_PATH}/lib64/cmake" ]; then
    export amd_comgr_DIR=${amd_comgr_DIR:-${ROCM_PATH}/lib64/cmake}
fi
prepend_env_path CMAKE_PREFIX_PATH "${ROCM_PATH}"
prepend_env_path CMAKE_PREFIX_PATH "${ROCM_PATH}/lib/cmake/amd_comgr"
prepend_env_path CMAKE_PREFIX_PATH "${ROCM_PATH}/lib64/cmake/amd_comgr"

if [ -z "${PYTORCH_ROCM_ARCH:-}" ]; then
    if ! PYTORCH_ROCM_ARCH=$(detect_gfx_arches); then
        echo "Unable to detect a gfx architecture; set PYTORCH_ROCM_ARCH explicitly." >&2
        exit 1
    fi
    export PYTORCH_ROCM_ARCH
fi

if [ "$build_extension" -eq 1 ]; then
    command -v "$python_bin" >/dev/null 2>&1 || {
        echo "Python executable was not found: $python_bin" >&2
        exit 1
    }
    command -v cmake >/dev/null 2>&1 || {
        echo "cmake is required to build bundled rocSHMEM." >&2
        exit 1
    }
    command -v hipcc >/dev/null 2>&1 || {
        echo "hipcc was not found below ROCM_PATH/HIP_HOME or PATH." >&2
        exit 1
    }

    if ! shca_include_dir=$(find_shca_header_dir); then
        echo "infiniband/shca_dv.h was not found; set SHCA_INCLUDE_DIR or SHCA_ROOT." >&2
        exit 1
    fi

    # Do not add /usr/include to CPATH/CPLUS_INCLUDE_PATH.  GCC's <cmath>
    # reaches the C header with #include_next <math.h>; putting /usr/include
    # before the normal libstdc++ include directories makes that lookup skip
    # the only math.h and causes CMake's compiler check to fail.  System
    # include directories are searched by hipcc automatically.
    case "$shca_include_dir" in
        /usr/include|/usr/include/|/usr/local/include|/usr/local/include/)
            ;;
        *)
            prepend_env_path CPATH "$shca_include_dir"
            prepend_env_path CPLUS_INCLUDE_PATH "$shca_include_dir"
            ;;
    esac

    if ! printf '#include <cmath>\nint main() { return 0; }\n' \
        | hipcc -x c++ -fsyntax-only - >/dev/null; then
        echo "hipcc cannot compile a host C++ header sanity check." >&2
        echo "Check CPATH/CPLUS_INCLUDE_PATH and hipcc's GCC installation." >&2
        echo "  CPATH=${CPATH:-<unset>}" >&2
        echo "  CPLUS_INCLUDE_PATH=${CPLUS_INCLUDE_PATH:-<unset>}" >&2
        exit 1
    fi
fi

if ! shca_library_dir=$(find_shca_library_dir); then
    echo "libshca.so was not found; set SHCA_LIBRARY_DIR or SHCA_ROOT." >&2
    exit 1
fi
prepend_env_path LD_LIBRARY_PATH "$shca_library_dir"
prepend_env_path LIBRARY_PATH "$shca_library_dir"

# rocSHMEM's multi-backend build must be told to select GDA at runtime, and
# the GDA layer must be told to select the SHCA provider rather than MLX5.
export ROCSHMEM_BACKEND=${ROCSHMEM_BACKEND:-gda}
export ROCSHMEM_GDA_PROVIDER=${ROCSHMEM_GDA_PROVIDER:-shca}
export ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE:-1073741824}

if [ -n "${NETWORK_INTERFACE:-}" ]; then
    export ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=${ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME:-${NETWORK_INTERFACE}}
fi
if [ "$force_rocshmem" -eq 1 ]; then
    export ROCSHMEM_FORCE_REBUILD=1
fi

# Retain any caller-provided CMake flags. The gda_shca preset in setup.py adds
# USE_GDA=ON, USE_IPC=ON and GDA_SHCA=ON itself.
export ROCSHMEM_CMAKE_ARGS=${ROCSHMEM_CMAKE_ARGS:-}

echo "UltraEP HCU/SHCA configuration:"
echo "  ROCM_PATH=${ROCM_PATH}"
echo "  ROCM_HOME=${ROCM_HOME}"
echo "  ULTRA_EP_GRAD_REDUCE_NUM_SMS=${ULTRA_EP_GRAD_REDUCE_NUM_SMS}"
echo "  PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
echo "  ROCSHMEM_SOURCE=${rocshmem_source}"
echo "  ROCSHMEM_BUILD_DIR=${ROCSHMEM_BUILD_DIR}"
echo "  ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR}"
echo "  ROCSHMEM_BACKEND=${ROCSHMEM_BACKEND}"
echo "  ROCSHMEM_GDA_PROVIDER=${ROCSHMEM_GDA_PROVIDER}"
echo "  ROCSHMEM_HEAP_SIZE=${ROCSHMEM_HEAP_SIZE}"
echo "  SHCA_INCLUDE_DIR=${shca_include_dir:-<not checked>}"
echo "  SHCA_LIBRARY_DIR=${shca_library_dir}"
echo "  ROCSHMEM_USE_IB_HCA=${ROCSHMEM_USE_IB_HCA:-<auto>}"
echo "  ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME=${ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME:-<auto>}"

if command -v ibv_devinfo >/dev/null 2>&1; then
    if ! ibv_devinfo >/dev/null 2>&1; then
        echo "Warning: ibv_devinfo did not report an active RDMA device." >&2
    fi
fi

if [ "$build_extension" -eq 1 ]; then
    remove_hipify_outputs

    if [ "$build_wheel" -eq 1 ]; then
        "$python_bin" setup.py bdist_wheel
    else
        "$python_bin" setup.py build_ext --inplace
    fi
fi

if [ "${#run_command[@]}" -gt 0 ]; then
    exec "${run_command[@]}"
fi
