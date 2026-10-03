#!/bin/bash
# Interleaved render time benchmark of two builds on the nested dielectric scenes.
#   tools/nested_dielectrics/benchmark.sh <before Blender.app> <after Blender.app> <output directory> [rounds] [samples] [resolution]
# Both builds render the scenes that do not use the feature (same image expected, the cost of
# having the code). The after build also renders the nested scenes (the cost of using it:
# compare drink_nested with drink_overlap, and each *_nested with its *_reference).
# Round 0 warms the kernel caches and is not reported. Summarize with benchmark_report.py.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BEFORE="$1"; AFTER="$2"; OUT="$3"; ROUNDS="${4:-3}"; SAMPLES="${5:-256}"; RESOLUTION="${6:-256}"
SCRIPT="$ROOT/tests/python/cycles_nested_dielectric_scenes.py"
LEGACY="drink_overlap,drink_gap,shell_reference,rough_shell_reference,bubble_reference,camera_inside_reference,overlap_reference"
NESTED="drink_nested,rod_matched,spheres_123,shell_nested,rough_shell_nested,bubble_nested,camera_inside_nested,overlap_nested"
mkdir -p "$OUT"
run() {
  # run <label> <app> <round> <scenes>
  local blender="$2/Contents/MacOS/Blender"
  for device in METAL CPU; do
    for integrator in pt bdpt; do
      local dir="$OUT/$1_round$3/${device}_${integrator}"
      mkdir -p "$dir"
      "$blender" -b --factory-startup --python "$SCRIPT" -- --output "$dir" --device "$device" \
        --integrator "$integrator" --scenes "$4" --samples "$SAMPLES" --resolution "$RESOLUTION" \
        > "$dir.log" 2>&1
    done
  done
}
for round in $(seq 0 "$ROUNDS"); do
  run before "$BEFORE" "$round" "$LEGACY"
  run after "$AFTER" "$round" "$LEGACY,$NESTED"
  echo "round $round done" >> "$OUT/status.txt"
done
echo done >> "$OUT/status.txt"
