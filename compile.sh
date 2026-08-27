#!/usr/bin/env bash
# Build Blender for Apple Silicon macOS and install the complete app bundle to ./Release.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

BUILD_TYPE="${BLENDER_BUILD_TYPE:-Release}"
ARCH="${BLENDER_ARCH:-arm64}"
BUILD_DIR="${BLENDER_BUILD_DIR:-${ROOT_DIR}/build/macos_${ARCH}_${BUILD_TYPE}}"
RELEASE_DIR="${BLENDER_RELEASE_DIR:-${ROOT_DIR}/Release}"
LIBDIR="${BLENDER_LIBDIR:-${ROOT_DIR}/lib/macos_${ARCH}}"
JOBS="${BLENDER_JOBS:-}"
FETCH_LIBRARIES="${BLENDER_FETCH_LIBRARIES:-auto}"
USE_CCACHE="${BLENDER_USE_CCACHE:-auto}"
REQUESTED_GENERATOR="${BLENDER_CMAKE_GENERATOR:-}"
GENERATOR="$REQUESTED_GENERATOR"
CLEAN_RELEASE=0
PACKAGE_ARCHIVE=0
SMOKE_TEST=1
RUN_APP=0
RUN_FILE=""
EXTRA_CMAKE_ARGS=()

usage() {
  cat <<EOF
Usage: ./compile.sh [options] [-- extra-cmake-args...]

Builds Blender for Apple Silicon macOS and installs the complete app bundle to:
  ${RELEASE_DIR}

Options:
  -j, --jobs N              Parallel build jobs. Default: detected CPU count.
  --build-dir DIR           CMake build directory. Default: ./build/macos_arm64_Release.
  --release-dir DIR         Install/artifact directory. Default: ./Release.
  --libdir DIR              Blender dependency library directory. Default: ./lib/macos_arm64.
  --fetch-libraries         Fetch missing official macOS arm64 dependency libraries and LFS data.
  --no-fetch-libraries      Fail if dependency libraries or LFS data are missing.
  --clean-release           Remove ./Release before installing.
  --package                 Also build Blender's package_archive target and copy zips to ./Release.
  --no-smoke-test           Skip the final background --version run.
  --run                     Launch a new instance of the freshly installed app after building.
  --open FILE               Launch a new instance and open FILE after building.
  -h, --help                Show this help.

Environment:
  BLENDER_JOBS, BLENDER_BUILD_DIR, BLENDER_RELEASE_DIR, BLENDER_LIBDIR,
  BLENDER_FETCH_LIBRARIES, BLENDER_USE_CCACHE, BLENDER_CMAKE_GENERATOR.

Examples:
  ./compile.sh
  ./compile.sh --run
  ./compile.sh --open /path/to/scene.blend
  ./compile.sh -j 8 --no-fetch-libraries
  ./compile.sh -- -DWITH_TRACY=OFF
EOF
}

log() {
  printf '\n==> %s\n' "$*"
}

die() {
  printf '\nerror: %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

require_command() {
  have "$1" || die "Required command not found: $1"
}

on_error() {
  local status=$?
  printf '\nerror: build script failed near line %s (exit %s)\n' "${BASH_LINENO[0]}" "$status" >&2
  exit "$status"
}

trap on_error ERR

while [ "$#" -gt 0 ]; do
  case "$1" in
    -j|--jobs)
      [ "$#" -ge 2 ] || die "$1 requires a value"
      JOBS="$2"
      shift 2
      ;;
    --build-dir)
      [ "$#" -ge 2 ] || die "$1 requires a value"
      BUILD_DIR="$2"
      shift 2
      ;;
    --release-dir)
      [ "$#" -ge 2 ] || die "$1 requires a value"
      RELEASE_DIR="$2"
      shift 2
      ;;
    --libdir)
      [ "$#" -ge 2 ] || die "$1 requires a value"
      LIBDIR="$2"
      shift 2
      ;;
    --fetch-libraries)
      FETCH_LIBRARIES=1
      shift
      ;;
    --no-fetch-libraries)
      FETCH_LIBRARIES=0
      shift
      ;;
    --clean-release)
      CLEAN_RELEASE=1
      shift
      ;;
    --package)
      PACKAGE_ARCHIVE=1
      shift
      ;;
    --no-smoke-test)
      SMOKE_TEST=0
      shift
      ;;
    --run)
      RUN_APP=1
      shift
      ;;
    --open)
      [ "$#" -ge 2 ] || die "$1 requires a value"
      RUN_APP=1
      RUN_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_CMAKE_ARGS=("$@")
      break
      ;;
    *)
      die "Unknown option: $1"
      ;;
  esac
done

detect_jobs() {
  local detected

  detected=""
  if have sysctl; then
    detected="$(sysctl -n hw.ncpu 2>/dev/null || true)"
  fi
  if [ -z "$detected" ] && have getconf; then
    detected="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  fi
  if [ -z "$detected" ]; then
    detected=1
  fi

  printf '%s\n' "$detected"
}

validate_positive_integer() {
  case "$1" in
    ''|*[!0-9]*)
      return 1
      ;;
    *)
      [ "$1" -gt 0 ]
      ;;
  esac
}

safe_rm_release_dir() {
  [ "$CLEAN_RELEASE" -eq 1 ] || return 0

  case "$RELEASE_DIR" in
    ""|"/"|"$ROOT_DIR"|"$BUILD_DIR")
      die "Refusing to remove unsafe release directory: ${RELEASE_DIR}"
      ;;
  esac

  log "Cleaning release directory: ${RELEASE_DIR}"
  rm -rf "$RELEASE_DIR"
}

libraries_ready() {
  [ -e "${LIBDIR}/.git" ] || return 1
  [ -d "${LIBDIR}/python" ] || return 1
  [ -d "${LIBDIR}/ffmpeg" ] || return 1
  [ -d "${LIBDIR}/openimageio" ] || return 1
  [ -d "${LIBDIR}/usd" ] || return 1
  return 0
}

fetch_libraries_if_needed() {
  if libraries_ready; then
    return 0
  fi

  if [ "$FETCH_LIBRARIES" = "0" ]; then
    cat >&2 <<EOF

error: macOS arm64 Blender dependencies are missing from:
  ${LIBDIR}

Fetch them with:
  ./compile.sh --fetch-libraries

or manually run:
  python3 build_files/utils/make_update.py --no-blender --architecture=arm64
EOF
    exit 1
  fi

  require_command git
  require_command python3
  git lfs version >/dev/null 2>&1 || die "Git LFS is required to fetch Blender's precompiled libraries."

  log "Fetching official Blender macOS arm64 dependency libraries"
  (
    cd "$ROOT_DIR"
    python3 build_files/utils/make_update.py --no-blender --architecture=arm64
  )

  libraries_ready || die "Dependency libraries are still incomplete after fetching: ${LIBDIR}"
}

root_lfs_data_ready() {
  local startup_blend
  local startup_size

  startup_blend="${ROOT_DIR}/release/datafiles/startup.blend"
  [ -f "$startup_blend" ] || return 0

  startup_size="$(wc -c < "$startup_blend" | tr -d '[:space:]')"
  [ "$startup_size" -ge 1024 ]
}

fetch_root_lfs_data_if_needed() {
  if root_lfs_data_ready; then
    return 0
  fi

  if [ "$FETCH_LIBRARIES" = "0" ]; then
    cat >&2 <<EOF

error: Blender Git LFS data is incomplete in the source checkout.

Fetch it with:
  git lfs pull

or let this script do it:
  ./compile.sh --fetch-libraries
EOF
    exit 1
  fi

  require_command git
  git lfs version >/dev/null 2>&1 || die "Git LFS is required to fetch Blender's data files."

  log "Fetching Blender root Git LFS data"
  (
    cd "$ROOT_DIR"
    git lfs pull
  )

  root_lfs_data_ready || die "Blender root Git LFS data is still incomplete after git lfs pull."
}

configure_generator() {
  local existing_generator

  existing_generator=""
  if [ -f "${BUILD_DIR}/CMakeCache.txt" ]; then
    existing_generator="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt" | head -n 1)"
  fi

  if [ -n "$existing_generator" ] && [ -z "$REQUESTED_GENERATOR" ]; then
    GENERATOR="$existing_generator"
  fi

  if [ -n "$existing_generator" ] && [ -n "$REQUESTED_GENERATOR" ] && [ "$existing_generator" != "$REQUESTED_GENERATOR" ]; then
    die "Build directory already uses generator '${existing_generator}'. Use a different --build-dir to switch to '${REQUESTED_GENERATOR}'."
  fi

  if [ -z "$GENERATOR" ]; then
    if have ninja; then
      GENERATOR="Ninja"
    else
      GENERATOR="Unix Makefiles"
    fi
  fi

  if [ "$GENERATOR" = "Ninja" ] && ! have ninja; then
    die "Ninja generator requested but ninja was not found in PATH."
  fi
}

configure_ccache() {
  if [ "$USE_CCACHE" = "0" ] || [ "$USE_CCACHE" = "OFF" ]; then
    printf '%s\n' "OFF"
    return 0
  fi

  if have ccache; then
    export CCACHE_BASEDIR="${ROOT_DIR}"
    export CCACHE_NOHASHDIR=1
    printf '%s\n' "ON"
  else
    printf '%s\n' "OFF"
  fi
}

validate_host() {
  [ "$(uname -s)" = "Darwin" ] || die "This script builds the macOS app bundle and must be run on macOS."
  [ "$ARCH" = "arm64" ] || die "This script is intentionally scoped to Apple Silicon arm64 builds."

  if [ "$(uname -m)" != "arm64" ]; then
    if ! sysctl -in hw.optional.arm64 2>/dev/null | grep -q '^1$'; then
      die "This machine does not appear to be Apple Silicon."
    fi
    log "Running from a translated shell, still targeting arm64"
  fi
}

copy_package_archives() {
  [ "$PACKAGE_ARCHIVE" -eq 1 ] || return 0

  log "Building release archive"
  cmake --build "$BUILD_DIR" --target package_archive --parallel "$JOBS"

  if [ -d "${BUILD_DIR}/release" ]; then
    find "${BUILD_DIR}/release" -maxdepth 1 -type f -name '*.zip' -exec cp -p {} "$RELEASE_DIR" \;
  fi
}

verify_installed_cycles_sources() {
  local installed_source
  local installed_addon
  local relative_path
  local source_file
  local installed_file
  local runtime_sources

  installed_source="$(find "${RELEASE_DIR}/Blender.app/Contents/Resources" \
    -type d -path '*/scripts/addons_core/cycles/source' -print -quit)"
  [ -n "$installed_source" ] || die "Installed Cycles source directory was not found in Blender.app."

  runtime_sources=(
    kernel/data_template.h
    kernel/device/metal/bvh.h
    kernel/device/metal/kernel.metal
    kernel/device/gpu/kernel.h
    kernel/geom/pixel_displacement_shader.h
    kernel/integrator/path_state.h
    kernel/integrator/photon_mapping.h
    kernel/integrator/shade_surface.h
    kernel/integrator/state.h
    kernel/integrator/subsurface.h
    kernel/integrator/surface_shader.h
    kernel/types.h
  )

  for relative_path in "${runtime_sources[@]}"; do
    source_file="${ROOT_DIR}/intern/cycles/${relative_path}"
    installed_file="${installed_source}/${relative_path}"
    [ -f "$installed_file" ] || die "Installed Cycles runtime source is missing: ${installed_file}"
    cmp -s "$source_file" "$installed_file" || die \
      "Installed Cycles runtime source is stale: ${relative_path}"
  done

  installed_file="${installed_source}/kernel/integrator/subsurface.h"
  grep -Fq 'defined(__KERNEL_METAL_PIXEL_DISPLACEMENT_SHADE__)' "$installed_file" || die \
    "Installed Cycles runtime is missing pixel-displacement support in Metal shading kernels."

  installed_addon="$(dirname "$installed_source")"
  for relative_path in properties.py ui.py; do
    source_file="${ROOT_DIR}/intern/cycles/blender/addon/${relative_path}"
    installed_file="${installed_addon}/${relative_path}"
    [ -f "$installed_file" ] || die "Installed Cycles add-on file is missing: ${installed_file}"
    cmp -s "$source_file" "$installed_file" || die \
      "Installed Cycles add-on file is stale: ${relative_path}"
  done
}

launch_fresh_app() {
  [ "$RUN_APP" -eq 1 ] || return 0

  require_command open
  log "Launching a fresh Blender instance"
  if [ -n "$RUN_FILE" ]; then
    [ -f "$RUN_FILE" ] || die "Blend file does not exist: ${RUN_FILE}"
    open -n -a "${RELEASE_DIR}/Blender.app" "$RUN_FILE"
  else
    open -n "${RELEASE_DIR}/Blender.app"
  fi
}

if [ -z "$JOBS" ]; then
  JOBS="$(detect_jobs)"
fi
validate_positive_integer "$JOBS" || die "Invalid job count: ${JOBS}"

validate_host
require_command cmake
require_command xcrun
xcrun --find clang >/dev/null
xcrun --sdk macosx --show-sdk-path >/dev/null

fetch_libraries_if_needed
fetch_root_lfs_data_if_needed
configure_generator
WITH_CCACHE="$(configure_ccache)"

log "Build configuration"
printf '  Source:      %s\n' "$ROOT_DIR"
printf '  Build dir:   %s\n' "$BUILD_DIR"
printf '  Release dir: %s\n' "$RELEASE_DIR"
printf '  Libraries:   %s\n' "$LIBDIR"
printf '  Generator:   %s\n' "$GENERATOR"
printf '  Jobs:        %s\n' "$JOBS"
printf '  ccache:      %s\n' "$WITH_CCACHE"

safe_rm_release_dir
mkdir -p "$BUILD_DIR" "$RELEASE_DIR"

CMAKE_ARGS=(
  -S "$ROOT_DIR"
  -B "$BUILD_DIR"
  -G "$GENERATOR"
  -C "$ROOT_DIR/build_files/buildbot/config/blender_macos.cmake"
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
  -DCMAKE_OSX_ARCHITECTURES="$ARCH"
  -DCMAKE_INSTALL_PREFIX="$RELEASE_DIR"
  -DLIBDIR="$LIBDIR"
  -DWITH_COMPILER_CCACHE="$WITH_CCACHE"
  -DWITH_CYCLES_PARALLEL_DEVICE_KERNEL_BUILD=ON
  -DWITH_NINJA_POOL_JOBS=ON
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)

if [ "${#EXTRA_CMAKE_ARGS[@]}" -gt 0 ]; then
  CMAKE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")
fi

log "Configuring Blender"
cmake "${CMAKE_ARGS[@]}"

log "Building and installing Blender"
cmake --build "$BUILD_DIR" --target install --parallel "$JOBS"

copy_package_archives

BLENDER_BIN="${RELEASE_DIR}/Blender.app/Contents/MacOS/Blender"
[ -x "$BLENDER_BIN" ] || die "Expected Blender executable was not created: ${BLENDER_BIN}"

log "Verifying installed Cycles runtime sources"
verify_installed_cycles_sources

if [ "$SMOKE_TEST" -eq 1 ]; then
  log "Running smoke test"
  "$BLENDER_BIN" --background --factory-startup --version >/dev/null
fi

log "Done"
printf 'Blender app: %s\n' "${RELEASE_DIR}/Blender.app"
printf 'Blender bin: %s\n' "$BLENDER_BIN"
if [ "$RUN_APP" -eq 0 ]; then
  printf 'Fresh launch: ./compile.sh --run\n'
  printf 'Note: plain macOS open may reactivate an older running Blender instance.\n'
fi

launch_fresh_app
