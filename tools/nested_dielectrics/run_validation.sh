#!/bin/bash
# Render the nested dielectric validation pairs in every integrator configuration and compare
# each nested render with its reference, relative to the noise floor of the reference.
#   tools/nested_dielectrics/run_validation.sh <Blender.app> <output directory> [devices] [configs] [samples] [resolution]
# devices: "CPU METAL" (default). configs: any of pt pt_guiding bdpt bdpt_guiding photon osl.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; OUT="$2"; DEVICES="${3:-CPU METAL}"
CONFIGS="${4:-pt pt_guiding bdpt bdpt_guiding photon}"; SAMPLES="${5:-512}"; RESOLUTION="${6:-160}"
SCRIPT="$ROOT/tests/python/cycles_nested_dielectric_scenes.py"
BLENDER="$APP/Contents/MacOS/Blender"
mkdir -p "$OUT"
status=0
for device in $DEVICES; do
  for config in $CONFIGS; do
    flags=()
    case "$config" in
      pt) flags=(--integrator pt) ;;
      pt_guiding) flags=(--integrator pt --guiding) ;;
      bdpt) flags=(--integrator bdpt) ;;
      bdpt_guiding) flags=(--integrator bdpt --guiding) ;;
      photon) flags=(--integrator photon) ;;
      osl) flags=(--integrator pt --osl) ;;
    esac
    dir="$OUT/${device}_${config}"
    "$BLENDER" -b --factory-startup --python "$SCRIPT" -- --output "$dir" --device "$device" \
      "${flags[@]}" --scenes validation --samples "$SAMPLES" --resolution "$RESOLUTION" \
      > "$dir.log" 2>&1
    "$BLENDER" -b --factory-startup --python "$SCRIPT" -- --output "$dir/noise" --device "$device" \
      "${flags[@]}" --scenes references --samples "$SAMPLES" --resolution "$RESOLUTION" \
      --seed 23 > "$dir.noise.log" 2>&1
    echo "== $device $config" | tee -a "$OUT/summary.txt"
    "$ROOT/tools/nested_dielectrics/compare_pairs.sh" "$APP" "$dir" "$dir/noise" 3.0 |
      tee -a "$OUT/summary.txt"
    [ "${PIPESTATUS[0]}" = 0 ] || status=1
  done
done
exit $status
