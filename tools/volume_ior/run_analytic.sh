#!/bin/bash
# Render the volume index of refraction scenes that check themselves against closed forms, in
# every integrator configuration:
#   tools/volume_ior/run_analytic.sh <Blender.app> <output directory> [devices] [configs] [samples]
# devices: "CPU METAL" (default). configs: any of pt pt_guiding pt_marching bdpt bdpt_guiding
# photon vcm osl. Appends to <output directory>/summary.txt and exits non-zero on a failure.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; OUT="$2"; DEVICES="${3:-CPU METAL}"
CONFIGS="${4:-pt pt_guiding pt_marching bdpt bdpt_guiding photon vcm}"; SAMPLES="${5:-256}"
SCRIPT="$ROOT/tests/python/cycles_volume_ior_scenes.py"
BLENDER="$APP/Contents/MacOS/Blender"
mkdir -p "$OUT"
status=0
for device in $DEVICES; do
  configs="$CONFIGS"
  [ "$device" = CPU ] && [ $# -lt 4 ] && configs="$configs osl"
  for config in $configs; do
    flags=()
    case "$config" in
      pt) flags=(--integrator pt) ;;
      pt_guiding) flags=(--integrator pt --guiding) ;;
      pt_marching) flags=(--integrator pt --ray-marching) ;;
      bdpt) flags=(--integrator bdpt) ;;
      bdpt_guiding) flags=(--integrator bdpt --guiding) ;;
      photon) flags=(--integrator photon) ;;
      vcm) flags=(--integrator vcm) ;;
      osl) flags=(--integrator pt --osl) ;;
    esac
    dir="$OUT/analytic_${device}_${config}"
    "$BLENDER" -b --factory-startup --python "$SCRIPT" -- --output "$dir" --device "$device" \
      "${flags[@]}" --scenes analytic --samples "$SAMPLES" > "$dir.log" 2>&1
    code=$?
    echo "== $device $config" | tee -a "$OUT/summary.txt"
    grep "^VOLUME_IOR" "$dir.log" | tee -a "$OUT/summary.txt"
    if [ $code != 0 ] || [ "$(grep -c "^VOLUME_IOR" "$dir.log")" != 7 ]; then
      echo "FAILED: $device $config (exit code $code), see $dir.log" | tee -a "$OUT/summary.txt"
      status=1
    fi
  done
done
exit $status
