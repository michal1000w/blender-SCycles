#!/usr/bin/env bash
# Build this Blender fork on Linux (x64 or arm64) with the full Cycles feature set (CPU, CUDA,
# OptiX) and install a portable build to ./install (the blender executable with its resources).
#
# It configures Blender's "release" feature set (build_files/cmake/config/blender_release.cmake)
# and detects, without manual setup:
#   * the newest CUDA toolkit (CUDA_PATH, /usr/local/cuda*, nvcc in PATH),
#   * the newest OptiX SDK 8.0+ (OPTIX_ROOT_DIR, ~/NVIDIA-OptiX-SDK-*, /opt/*optix*),
#   * the GPUs in this machine (nvidia-smi), so by default CUDA binaries are compiled only for the
#     architectures that are present (--cuda-arch release for a redistributable build),
#   * Blender's precompiled libraries (lib/linux_<arch>), fetched on first use.
# AMD HIP and Intel oneAPI binaries are built when their compilers are found (--with-hip,
# --with-oneapi). macOS uses ./compile.sh, Windows compileWindows.ps1.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

log() { printf '\n==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

on_error() {
  local status=$?
  printf '\nerror: build script failed near line %s (exit %s)\n' "${BASH_LINENO[0]}" "$status" >&2
  exit "$status"
}
trap on_error ERR

[ "$(uname -s)" = "Linux" ] || die "compileLinux.sh builds on Linux. Use ./compile.sh on macOS and compileWindows.ps1 on Windows."

case "$(uname -m)" in
  x86_64|amd64) ARCH="x64" ;;
  aarch64|arm64) ARCH="arm64" ;;
  *) die "Unsupported architecture: $(uname -m)" ;;
esac

BUILD_TYPE="${BLENDER_BUILD_TYPE:-Release}"
BUILD_DIR=""
RELEASE_DIR="${BLENDER_RELEASE_DIR:-${ROOT_DIR}/install}"
LIBDIR="${BLENDER_LIBDIR:-${ROOT_DIR}/lib/linux_${ARCH}}"
JOBS="${BLENDER_JOBS:-}"
KERNEL_JOBS="${BLENDER_KERNEL_JOBS:-0}"
CUDA_ARCH="${BLENDER_CUDA_ARCH:-auto}"
CUDA_ROOT="${CUDA_PATH:-}"
OPTIX_ROOT="${OPTIX_ROOT_DIR:-}"
NO_CUDA=0
NO_OPTIX=0
WITH_HIP=0
WITH_ONEAPI=0
FETCH_LIBRARIES=auto
CLEAN_RELEASE=0
FRESH=0
PACKAGE=0
SMOKE_TEST=1
RUN_TEST=0
RUN_APP=0
RUN_FILE=""
EXTRA_CMAKE_ARGS=()

usage() {
  cat <<EOF
Usage: ./compileLinux.sh [options] [-- extra-cmake-args...]

Builds Blender with CPU, CUDA and OptiX Cycles and installs it to ${RELEASE_DIR}.

Options:
  -j, --jobs N           Parallel jobs. Default: CPU count.
  --kernel-jobs N        Cycles kernels (CPU kernel files, CUDA and OptiX modules) compiled at
                         once, each needs up to ~16 GB of memory. Default: one per 32 GB of
                         physical memory, at least one (Ninja only).
  --build-type TYPE      Release (default) or RelWithDebInfo.
  --build-dir DIR        CMake build directory. Default: ./build/linux_<arch>_<type>.
  --release-dir DIR      Install directory. Default: ./install.
  --libdir DIR           Precompiled libraries. Default: ./lib/linux_<arch>.
  --cuda-arch LIST       auto (GPUs in this machine, default), release (Blender's list), or a
                         list such as "sm_86;sm_89;compute_75".
  --cuda-root DIR        CUDA toolkit. Default: newest found.
  --optix-root DIR       OptiX SDK (8.0+). Default: OPTIX_ROOT_DIR or newest found.
  --no-cuda              No CUDA kernel binaries and no OptiX.
  --no-optix             No OptiX.
  --with-hip             Build AMD HIP kernel binaries (needs ROCm/HIP).
  --with-oneapi          Build Intel oneAPI kernel binaries.
  --fetch-libraries      Fetch missing libraries and Git LFS data without asking.
  --no-fetch-libraries   Fail if libraries or LFS data are missing.
  --clean-release        Remove the install directory first.
  --fresh                Reconfigure from scratch.
  --package              Also write ./build/blender-jcycles-<version>-linux-<arch>.tar.xz.
  --no-smoke-test        Skip the device probe with the installed blender.
  --test                 Run the feature demo and parity test afterwards.
  --run                  Launch the built Blender.
  --open FILE            Launch the built Blender with FILE.
  -h, --help             Show this help.

Environment: BLENDER_JOBS, BLENDER_KERNEL_JOBS, BLENDER_BUILD_TYPE, BLENDER_RELEASE_DIR,
             BLENDER_LIBDIR, BLENDER_CUDA_ARCH, CUDA_PATH, OPTIX_ROOT_DIR.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    -j[0-9]*) JOBS="${1#-j}"; shift ;;
    -j|--jobs) [ "$#" -ge 2 ] || die "$1 requires a value"; JOBS="$2"; shift 2 ;;
    --kernel-jobs) [ "$#" -ge 2 ] || die "$1 requires a value"; KERNEL_JOBS="$2"; shift 2 ;;
    --build-type) [ "$#" -ge 2 ] || die "$1 requires a value"; BUILD_TYPE="$2"; shift 2 ;;
    --build-dir) [ "$#" -ge 2 ] || die "$1 requires a value"; BUILD_DIR="$2"; shift 2 ;;
    --release-dir) [ "$#" -ge 2 ] || die "$1 requires a value"; RELEASE_DIR="$2"; shift 2 ;;
    --libdir) [ "$#" -ge 2 ] || die "$1 requires a value"; LIBDIR="$2"; shift 2 ;;
    --cuda-arch) [ "$#" -ge 2 ] || die "$1 requires a value"; CUDA_ARCH="$2"; shift 2 ;;
    --cuda-root) [ "$#" -ge 2 ] || die "$1 requires a value"; CUDA_ROOT="$2"; shift 2 ;;
    --optix-root) [ "$#" -ge 2 ] || die "$1 requires a value"; OPTIX_ROOT="$2"; shift 2 ;;
    --no-cuda) NO_CUDA=1; shift ;;
    --no-optix) NO_OPTIX=1; shift ;;
    --with-hip) WITH_HIP=1; shift ;;
    --with-oneapi) WITH_ONEAPI=1; shift ;;
    --fetch-libraries) FETCH_LIBRARIES=1; shift ;;
    --no-fetch-libraries) FETCH_LIBRARIES=0; shift ;;
    --clean-release) CLEAN_RELEASE=1; shift ;;
    --fresh) FRESH=1; shift ;;
    --package) PACKAGE=1; shift ;;
    --no-smoke-test) SMOKE_TEST=0; shift ;;
    --test) RUN_TEST=1; shift ;;
    --run) RUN_APP=1; shift ;;
    --open) [ "$#" -ge 2 ] || die "$1 requires a value"; RUN_APP=1; RUN_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA_CMAKE_ARGS=("$@"); break ;;
    *) die "Unknown option: $1 (see --help)" ;;
  esac
done

case "$BUILD_TYPE" in Release|RelWithDebInfo|Debug) ;; *) die "Invalid build type: ${BUILD_TYPE}" ;; esac
[ -n "$BUILD_DIR" ] || BUILD_DIR="${ROOT_DIR}/build/linux_${ARCH}_${BUILD_TYPE}"
if [ -z "$JOBS" ]; then JOBS="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 1)"; fi
case "$JOBS" in ''|*[!0-9]*) die "Invalid job count: ${JOBS}" ;; esac
case "$KERNEL_JOBS" in ''|*[!0-9]*) die "Invalid kernel job count: ${KERNEL_JOBS}" ;; esac
case "$ROOT_DIR$BUILD_DIR" in *" "*) die "Paths with spaces are not supported by the GPU kernel build." ;; esac

have cmake || die "cmake is required."
have git || die "git is required."
GENERATOR="Ninja"
have ninja || GENERATOR="Unix Makefiles"
if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
  existing="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
  [ -n "$existing" ] && GENERATOR="$existing"
fi

# ------------------------------------------------------------------------------------------------
# Libraries and Git LFS data.

UPSTREAM_LFS="https://projects.blender.org/blender/blender.git/info/lfs"

libraries_ready() {
  [ -e "${LIBDIR}/.git" ] || return 1
  local d
  for d in python ffmpeg openimageio usd embree osl openimagedenoise; do
    [ -d "${LIBDIR}/${d}" ] || return 1
  done
  return 0
}

lfs_ready() {
  local startup="${ROOT_DIR}/release/datafiles/startup.blend"
  [ -f "$startup" ] || return 0
  [ "$(wc -c < "$startup" | tr -d '[:space:]')" -ge 1024 ]
}

if ! libraries_ready; then
  [ "$FETCH_LIBRARIES" != "0" ] || die "Libraries are missing from ${LIBDIR} (run with --fetch-libraries)."
  git lfs version >/dev/null 2>&1 || die "Git LFS is required to fetch Blender's libraries."
  log "Fetching Blender's Linux ${ARCH} libraries into ${LIBDIR}"
  (
    cd "$ROOT_DIR"
    git lfs install --skip-repo >/dev/null
    git config "submodule.lib/linux_${ARCH}.update" checkout
    GIT_LFS_SKIP_SMUDGE=1 git submodule update --init "lib/linux_${ARCH}"
    cd "$LIBDIR"
    git lfs pull
  )
  libraries_ready || die "Libraries are still incomplete after fetching: ${LIBDIR}"
fi

if ! lfs_ready; then
  [ "$FETCH_LIBRARIES" != "0" ] || die "Blender's Git LFS data is missing (run: git lfs pull)."
  log "Fetching Blender's Git LFS data (without tests/files)"
  (cd "$ROOT_DIR" && git -c lfs.fetchexclude="tests/files/**" -c "lfs.url=${UPSTREAM_LFS}" lfs pull)
  lfs_ready || die "Blender's Git LFS data is still incomplete."
fi

# ------------------------------------------------------------------------------------------------
# CUDA, OptiX, GPU architectures.

nvcc_version() {
  "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -n 1
}

version_ge() {
  [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n 1)" = "$2" ]
}

find_cuda() {
  local candidates=() best="" best_version="" c v
  if [ -n "$CUDA_ROOT" ]; then
    candidates+=("$CUDA_ROOT")
  else
    for c in /usr/local/cuda /usr/local/cuda-* /opt/cuda; do [ -d "$c" ] && candidates+=("$c"); done
    if have nvcc; then candidates+=("$(dirname "$(dirname "$(command -v nvcc)")")"); fi
  fi
  for c in "${candidates[@]}"; do
    [ -x "${c}/bin/nvcc" ] || continue
    v="$(nvcc_version "${c}/bin/nvcc")"
    [ -n "$v" ] || continue
    if [ -z "$best" ] || ! version_ge "$best_version" "$v"; then best="$c"; best_version="$v"; fi
  done
  [ -n "$best" ] && printf '%s %s\n' "$best" "$best_version"
  return 0
}

optix_version() {
  local header="$1/include/optix.h" code
  [ -f "$header" ] || return 1
  code="$(sed -n 's/^#define OPTIX_VERSION[[:space:]]*\([0-9]*\).*/\1/p' "$header" | head -n 1)"
  [ -n "$code" ] || return 1
  printf '%d.%d.%d\n' $((code / 10000)) $(((code % 10000) / 100)) $((code % 100))
}

find_optix() {
  local candidates=() best="" best_version="" c v
  if [ -n "$OPTIX_ROOT" ]; then
    candidates+=("$OPTIX_ROOT")
  else
    for c in "$HOME"/NVIDIA-OptiX-SDK-* /opt/NVIDIA-OptiX-SDK-* /opt/optix* /usr/local/optix*; do
      [ -d "$c" ] && candidates+=("$c")
    done
  fi
  for c in "${candidates[@]}"; do
    v="$(optix_version "$c")" || continue
    version_ge "$v" "8.0.0" || continue
    if [ -z "$best" ] || ! version_ge "$best_version" "$v"; then best="$c"; best_version="$v"; fi
  done
  [ -n "$best" ] && printf '%s %s\n' "$best" "$best_version"
  return 0
}

local_cuda_archs() {
  have nvidia-smi || return 0
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null |
    sed -n 's/^[[:space:]]*\([0-9]*\)\.\([0-9]*\)[[:space:]]*$/sm_\1\2/p' | sort -u | paste -sd ';' -
}

CUDA_DIR=""; CUDA_VERSION=""; OPTIX_DIR=""; OPTIX_VERSION=""; CUDA_ARCH_LIST=""
if [ "$NO_CUDA" -eq 0 ]; then
  read -r CUDA_DIR CUDA_VERSION <<<"$(find_cuda)" || true
  if [ -z "$CUDA_DIR" ]; then
    warn "No CUDA toolkit found: building without CUDA kernel binaries and OptiX."
  else
    case "$CUDA_ARCH" in
      auto)
        CUDA_ARCH_LIST="$(local_cuda_archs)"
        [ -n "$CUDA_ARCH_LIST" ] || warn "No NVIDIA GPU detected, compiling Blender's release CUDA architecture list."
        ;;
      release) CUDA_ARCH_LIST="" ;;
      *) CUDA_ARCH_LIST="$CUDA_ARCH" ;;
    esac
    if [ "$NO_OPTIX" -eq 0 ]; then
      read -r OPTIX_DIR OPTIX_VERSION <<<"$(find_optix)" || true
      [ -n "$OPTIX_DIR" ] || warn "No OptiX SDK 8.0+ found (set OPTIX_ROOT_DIR): building without OptiX."
    fi
  fi
fi

if [ "$WITH_HIP" -eq 1 ] && ! have hipcc && [ ! -x /opt/rocm/bin/hipcc ]; then
  die "--with-hip needs ROCm/HIP (hipcc)."
fi

# ------------------------------------------------------------------------------------------------
# Configure, build, install.

mkdir -p "$BUILD_DIR"
INITIAL_CACHE="${BUILD_DIR}/jcycles_linux_config.cmake"
{
  echo "# Generated by compileLinux.sh. Blender's release feature set, adapted to this machine."
  echo "include(\"${ROOT_DIR}/build_files/cmake/config/blender_release.cmake\")"
  echo "set(WITH_CYCLES_DEVICE_CUDA ON CACHE BOOL \"\" FORCE)"
  if [ -n "$CUDA_DIR" ]; then
    echo "set(WITH_CYCLES_CUDA_BINARIES ON CACHE BOOL \"\" FORCE)"
    echo "set(CUDA_TOOLKIT_ROOT_DIR \"${CUDA_DIR}\" CACHE PATH \"\" FORCE)"
    echo "set(CUDA_NVCC_EXECUTABLE \"${CUDA_DIR}/bin/nvcc\" CACHE FILEPATH \"\" FORCE)"
    [ -z "$CUDA_ARCH_LIST" ] || echo "set(CYCLES_CUDA_BINARIES_ARCH \"${CUDA_ARCH_LIST}\" CACHE STRING \"\" FORCE)"
  else
    echo "set(WITH_CYCLES_CUDA_BINARIES OFF CACHE BOOL \"\" FORCE)"
  fi
  if [ -n "$OPTIX_DIR" ]; then
    echo "set(WITH_CYCLES_DEVICE_OPTIX ON CACHE BOOL \"\" FORCE)"
    echo "set(OPTIX_ROOT_DIR \"${OPTIX_DIR}\" CACHE PATH \"\" FORCE)"
  else
    echo "set(WITH_CYCLES_DEVICE_OPTIX OFF CACHE BOOL \"\" FORCE)"
  fi
  if [ "$WITH_HIP" -eq 1 ]; then
    echo "set(WITH_CYCLES_HIP_BINARIES ON CACHE BOOL \"\" FORCE)"
  else
    echo "set(WITH_CYCLES_HIP_BINARIES OFF CACHE BOOL \"\" FORCE)"
    echo "set(WITH_CYCLES_DEVICE_HIPRT OFF CACHE BOOL \"\" FORCE)"
  fi
  if [ "$WITH_ONEAPI" -eq 1 ]; then
    echo "set(WITH_CYCLES_ONEAPI_BINARIES ON CACHE BOOL \"\" FORCE)"
  else
    echo "set(WITH_CYCLES_ONEAPI_BINARIES OFF CACHE BOOL \"\" FORCE)"
  fi
} > "$INITIAL_CACHE"

log "Build configuration"
printf '  Source:       %s\n' "$ROOT_DIR"
printf '  Build dir:    %s (%s, %s, %s jobs)\n' "$BUILD_DIR" "$BUILD_TYPE" "$GENERATOR" "$JOBS"
if [ "$KERNEL_JOBS" -gt 0 ]; then
  printf '  Kernel jobs:  %s\n' "$KERNEL_JOBS"
else
  printf '  Kernel jobs:  automatic, from the physical memory\n'
fi
printf '  Install dir:  %s\n' "$RELEASE_DIR"
printf '  Libraries:    %s\n' "$LIBDIR"
if [ -n "$CUDA_DIR" ]; then
  printf '  CUDA:         %s (%s), architectures: %s\n' "$CUDA_VERSION" "$CUDA_DIR" "${CUDA_ARCH_LIST:-Blender release list}"
else
  printf '  CUDA:         off\n'
fi
if [ -n "$OPTIX_DIR" ]; then printf '  OptiX:        %s (%s)\n' "$OPTIX_VERSION" "$OPTIX_DIR"; else printf '  OptiX:        off\n'; fi

if [ "$CLEAN_RELEASE" -eq 1 ]; then
  case "$RELEASE_DIR" in ""|"/"|"$ROOT_DIR"|"$BUILD_DIR"|"$HOME") die "Refusing to remove ${RELEASE_DIR}" ;; esac
  log "Removing ${RELEASE_DIR}"
  rm -rf "$RELEASE_DIR"
fi

CMAKE_ARGS=(
  -S "$ROOT_DIR"
  -B "$BUILD_DIR"
  -G "$GENERATOR"
  -C "$INITIAL_CACHE"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_INSTALL_PREFIX="$RELEASE_DIR"
  -DWITH_INSTALL_PORTABLE=ON
  -DLIBDIR="$LIBDIR"
  -DWITH_NINJA_POOL_JOBS=ON
  -DCYCLES_KERNEL_MAX_PARALLEL_JOBS="$KERNEL_JOBS"
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)
if have ccache; then
  export CCACHE_BASEDIR="$ROOT_DIR" CCACHE_NOHASHDIR=1
  CMAKE_ARGS+=(-DWITH_COMPILER_CCACHE=ON)
fi
[ "$FRESH" -eq 0 ] || CMAKE_ARGS+=(--fresh)
[ "${#EXTRA_CMAKE_ARGS[@]}" -eq 0 ] || CMAKE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")

log "Configuring"
cmake "${CMAKE_ARGS[@]}"

log "Building and installing"
start_time=$SECONDS
cmake --build "$BUILD_DIR" --target install --parallel "$JOBS"
printf 'Build and install took %d min\n' $(((SECONDS - start_time) / 60))

# ------------------------------------------------------------------------------------------------
# Verify.

BLENDER_BIN="${RELEASE_DIR}/blender"
[ -x "$BLENDER_BIN" ] || die "The blender executable was not installed: ${BLENDER_BIN}"

log "Verifying the installed Cycles add-on and kernels"
ADDON_DIR="$(find "$RELEASE_DIR" -maxdepth 4 -type d -path '*/scripts/addons_core/cycles' -print -quit)"
[ -n "$ADDON_DIR" ] || die "The installed Cycles add-on was not found."
for f in __init__.py engine.py properties.py ui.py; do
  cmp -s "${ROOT_DIR}/intern/cycles/blender/addon/${f}" "${ADDON_DIR}/${f}" || die "Installed Cycles add-on file is missing or stale: ${f}"
done
expected=()
if [ -n "$CUDA_DIR" ] && [ -n "$CUDA_ARCH_LIST" ]; then
  IFS=';' read -r -a archs <<<"$CUDA_ARCH_LIST"
  for a in "${archs[@]}"; do
    case "$a" in
      sm_*) expected+=("kernel_${a}.cubin.zst") ;;
      compute_*) expected+=("kernel_${a}.ptx.zst") ;;
    esac
  done
fi
[ -z "$OPTIX_DIR" ] || expected+=(kernel_optix.ptx.zst kernel_optix_shader_raytrace.ptx.zst)
for k in ${expected[@]+"${expected[@]}"}; do
  [ -f "${ADDON_DIR}/lib/${k}" ] || die "GPU kernel missing from the installation: ${ADDON_DIR}/lib/${k}"
done
[ "${#expected[@]}" -eq 0 ] || printf '  GPU kernels: %s\n' "${expected[*]}"

if [ "$SMOKE_TEST" -eq 1 ]; then
  log "Smoke test"
  "$BLENDER_BIN" --background --factory-startup --python "${ROOT_DIR}/tests/python/jcycles_device_probe.py"
fi

if [ "$PACKAGE" -eq 1 ]; then
  version="$(find "$RELEASE_DIR" -maxdepth 1 -type d -regex '.*/[0-9]+\.[0-9]+' -printf '%f\n' | head -n 1)"
  archive="${ROOT_DIR}/build/blender-jcycles-${version}-linux-${ARCH}.tar.xz"
  log "Packaging ${archive}"
  tar -C "$(dirname "$RELEASE_DIR")" -cJf "$archive" "$(basename "$RELEASE_DIR")"
fi

if [ "$RUN_TEST" -eq 1 ]; then
  log "Running the feature demo and parity test"
  "$BLENDER_BIN" --background --factory-startup --python "${ROOT_DIR}/tests/python/jcycles_feature_demo.py" -- \
    --output "${ROOT_DIR}/build/jcycles_feature_demo"
fi

log "Done"
printf 'Blender: %s\n' "$BLENDER_BIN"
printf 'JCycles add-on for an official Blender: ./compileCycles.sh\n'

if [ "$RUN_APP" -eq 1 ]; then
  if [ -n "$RUN_FILE" ]; then
    [ -f "$RUN_FILE" ] || die "Blend file does not exist: ${RUN_FILE}"
    nohup "$BLENDER_BIN" "$RUN_FILE" >/dev/null 2>&1 &
  else
    nohup "$BLENDER_BIN" >/dev/null 2>&1 &
  fi
fi
