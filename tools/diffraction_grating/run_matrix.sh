#!/bin/bash
# Render the Grating model scenes in every integrator configuration, one GPU job at a time, and
# compare each configuration with plain path tracing on the CPU:
#   tools/diffraction_grating/run_matrix.sh <Blender.app> <output directory> [scenes] [res] [spp]
# Configurations: CPU and METAL x pt, bdpt, photon, vcm x guiding off/on, plus OSL on the CPU.
# Builds without the fork's integrators (the reference branch) only run what they have.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$(cd "$1" && pwd)"; OUT="$2"; SCENES="${3:-validation}"; RES="${4:-0}"; SPP="${5:-0}"
BLENDER="$APP/Contents/MacOS/Blender"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
# Snapshot of the scenes, so that every configuration renders the same version.
cp "$ROOT/tests/python/cycles_diffraction_grating_scenes.py" "$OUT/scenes.py"
SCRIPT="$OUT/scenes.py"
COMPARE="$ROOT/tests/python/cycles_diffraction_grating_compare.py"
: > "$OUT/status.txt"

run() {
  local name="$1"; shift
  "$BLENDER" -b --factory-startup --python-exit-code 1 --python "$SCRIPT" -- --output "$OUT/$name" --scenes "$SCENES" \
    --resolution "$RES" --samples "$SPP" "$@" > "$OUT/$name.log" 2>&1
  local code=$?
  echo "render $name exit $code" >> "$OUT/status.txt"
  return $code
}

compare() {
  local reference="$1" candidate="$2"; shift 2
  [ -d "$OUT/$reference" ] && [ -d "$OUT/$candidate" ] || return
  "$BLENDER" -b --factory-startup --python-exit-code 1 --python "$COMPARE" -- "$OUT/$reference" "$OUT/$candidate" \
    --report "$OUT/compare_${candidate}_vs_${reference}.json" "$@" \
    > "$OUT/compare_${candidate}_vs_${reference}.log" 2>&1
  echo "compare $candidate vs $reference exit $? ($(grep -E "ALL PASS|FAILED" "$OUT/compare_${candidate}_vs_${reference}.log"))" \
    >> "$OUT/status.txt"
}

for device in CPU METAL; do
  for integrator in pt bdpt photon vcm; do
    for guiding in off on; do
      name="${device}_${integrator}_guiding_${guiding}"
      extra=()
      [ "$guiding" = on ] && extra+=(--guiding)
      run "$name" --device "$device" --integrator "$integrator" ${extra[@]+"${extra[@]}"} || \
        rm -rf "$OUT/$name"
    done
  done
done
run CPU_pt_osl --device CPU --integrator pt --osl || rm -rf "$OUT/CPU_pt_osl"

for name in $(cd "$OUT" && ls -d */ | tr -d /); do
  [ "$name" = CPU_pt_guiding_off ] && continue
  case "$name" in
    *photon*) compare CPU_pt_guiding_off "$name" --mean-tolerance 0.05 --block-tolerance 0.12 ;;
    *) compare CPU_pt_guiding_off "$name" ;;
  esac
done
cat "$OUT/status.txt"
