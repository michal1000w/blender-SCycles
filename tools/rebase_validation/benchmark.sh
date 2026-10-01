#!/bin/bash
# Interleaved render time benchmark of two builds with the fork's feature and parity suites.
#   tools/rebase_validation/benchmark.sh <before Blender.app> <after Blender.app> <output directory> [rounds]
# Round 0 warms the kernel caches and is not reported. Summarize with benchmark_report.py.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BEFORE="$1"; AFTER="$2"; OUT="$3"; ROUNDS="${4:-3}"
mkdir -p "$OUT"
run() {
  # run <label> <app> <round>
  local dir="$OUT/$1_round$3" blender="$2/Contents/MacOS/Blender"
  mkdir -p "$dir"
  "$blender" -b --factory-startup --python "$ROOT/tests/python/cycles_metal_feature_scenes.py" -- \
    --output "$dir/feature_METAL" --device METAL --resolution 192 --samples 64 --timing > "$dir/feature_METAL.log" 2>&1
  "$blender" -b --factory-startup --python "$ROOT/tests/python/cycles_cpu_parity_scenes.py" -- \
    --output "$dir/parity_METAL" --device METAL --resolution 160 --samples 64 > "$dir/parity_METAL.log" 2>&1
  [ "$3" = 0 ] && return
  "$blender" -b --factory-startup --python "$ROOT/tests/python/cycles_metal_feature_scenes.py" -- \
    --output "$dir/feature_CPU" --device CPU --resolution 160 --samples 32 --timing > "$dir/feature_CPU.log" 2>&1
  "$blender" -b --factory-startup --python "$ROOT/tests/python/cycles_cpu_parity_scenes.py" -- \
    --output "$dir/parity_CPU" --device CPU --resolution 128 --samples 32 > "$dir/parity_CPU.log" 2>&1
}
for round in $(seq 0 "$ROUNDS"); do
  run before "$BEFORE" "$round"
  run after "$AFTER" "$round"
  echo "round $round done" >> "$OUT/status.txt"
done
echo done >> "$OUT/status.txt"
