#!/usr/bin/env bash
# Package this repository's Blender build as the JCycles add-on (macOS and Linux).
#
# Run ./compile.sh first: it builds and installs Blender to ./install. This script then writes
# ./build/JCycles, a folder you can zip and install in an official Blender as an add-on. It adds a
# "JCycles" render engine that renders with this repository's Cycles (the build is bundled inside).
# On Windows use compileCycles.ps1.

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
COMPILE_FIRST=0
PASS_ARGS=()

usage() {
  cat <<USAGE
Usage: ./compileCycles.sh [options]

Writes ./build/JCycles from the build installed by ./compile.sh.

Options:
  --compile          Run ./compile.sh first (incremental build and install).
  --runtime PATH     Installed build to package. Default: ./install
  --output DIR       Add-on folder to write. Default: ./build/JCycles
  --zip              Also write ./build/JCycles-<version>-<platform>.zip
  --allow-stale      Package even if the installed build is older than the sources.
  -h, --help         Show this help.

Everything after -- is passed to ./compile.sh when --compile is used.
USAGE
}

COMPILE_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --compile) COMPILE_FIRST=1; shift ;;
    --runtime|--output) [ "$#" -ge 2 ] || { echo "error: $1 requires a value" >&2; exit 1; }; PASS_ARGS+=("$1" "$2"); shift 2 ;;
    --zip|--allow-stale|--no-runtime) PASS_ARGS+=("$1"); shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; COMPILE_ARGS=("$@"); break ;;
    *) echo "error: unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

case "$(uname -s)" in
  Darwin|Linux) ;;
  *) echo "error: on Windows run compileCycles.ps1" >&2; exit 1 ;;
esac

if [ "$COMPILE_FIRST" -eq 1 ]; then
  "${ROOT_DIR}/compile.sh" ${COMPILE_ARGS[@]+"${COMPILE_ARGS[@]}"}
fi

PYTHON=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 && \
     "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
    PYTHON="$candidate"
    break
  fi
done
if [ -z "$PYTHON" ]; then
  for candidate in "${ROOT_DIR}"/lib/*/python/bin/python3*; do
    if [ -x "$candidate" ]; then PYTHON="$candidate"; break; fi
  done
fi
[ -n "$PYTHON" ] || { echo "error: Python 3.8 or newer is required" >&2; exit 1; }

exec "$PYTHON" "${ROOT_DIR}/build_files/jcycles/build_jcycles.py" ${PASS_ARGS[@]+"${PASS_ARGS[@]}"}
