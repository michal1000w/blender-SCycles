#!/bin/bash
# Compare two output directories of run_suites.sh, image by image.
#   tools/rebase_validation/compare.sh <Blender.app> <reference directory> <candidate directory>
# With the same directory twice, compares CPU against Metal inside that build instead.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BLENDER="$1/Contents/MacOS/Blender"
REF="$2"
NEW="$3"
compare() {
  # compare <reference image directory> <candidate image directory> <result file>
  pairs=()
  for f in "$1"/*.exr; do
    n=$(basename "$f")
    if [ -f "$2/$n" ]; then pairs+=("$f" "$2/$n"); else echo "missing: $2/$n"; fi
  done
  [ ${#pairs[@]} -eq 0 ] && return
  "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_metal_image_compare.py" -- \
    "${pairs[@]}" 2>&1 | grep '^COMPARE ' | sed 's/^COMPARE //' > "$3"
}
if [ "$REF" = "$NEW" ]; then
  for suite in feature parity nodes; do
    compare "$REF/${suite}_CPU" "$REF/${suite}_METAL" "$REF/cpu_vs_metal_$suite.jsonl"
  done
  python3 "$ROOT/tools/rebase_validation/summarize.py" "$REF" cpu_vs_metal_feature cpu_vs_metal_parity cpu_vs_metal_nodes
else
  for suite in feature_METAL feature_CPU parity_METAL parity_CPU nodes_METAL nodes_CPU; do
    [ -d "$REF/$suite" ] && compare "$REF/$suite" "$NEW/$suite" "$NEW/compare_$suite.jsonl"
  done
  python3 "$ROOT/tools/rebase_validation/summarize.py" "$NEW"
fi
