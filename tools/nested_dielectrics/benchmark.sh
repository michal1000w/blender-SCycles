#!/bin/bash
# Interleaved render time benchmark of two builds on the nested dielectric scenes.
#   tools/nested_dielectrics/benchmark.sh <before Blender.app> <after Blender.app> <output directory> [rounds] [samples] [resolution]
# Both builds render the scenes without media (same image expected, the cost of having the
# code) and the ones with priorities, if the before build has them. The after build also renders
# the automatic scenes (the cost of using it: compare each *_auto with its *_nested, and
# separate_auto, whose objects are tracked without being nested, with separate_reference).
# BEFORE_SCENES overrides what the before build renders.
# Round 0 warms the kernel caches and is not reported. Summarize with benchmark_report.py.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BEFORE="$1"; AFTER="$2"; OUT="$3"; ROUNDS="${4:-3}"; SAMPLES="${5:-256}"; RESOLUTION="${6:-256}"
SCRIPT="$ROOT/tests/python/cycles_nested_dielectric_scenes.py"
LEGACY="drink_overlap,drink_gap,shell_reference,rough_shell_reference,bubble_reference,camera_inside_reference,overlap_reference"
NESTED="drink_nested,rod_matched,spheres_123,shell_nested,rough_shell_nested,bubble_nested,camera_inside_nested,overlap_nested"
AUTO="drink_auto,shell_auto,rough_shell_auto,bubble_auto,camera_inside_auto,separate_auto,separate_reference,showcase_auto,showcase_off"
BEFORE_SCENES="${BEFORE_SCENES:-$LEGACY,$NESTED}"
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
  run before "$BEFORE" "$round" "$BEFORE_SCENES"
  run after "$AFTER" "$round" "$LEGACY,$NESTED,$AUTO"
  echo "round $round done" >> "$OUT/status.txt"
done
echo done >> "$OUT/status.txt"
