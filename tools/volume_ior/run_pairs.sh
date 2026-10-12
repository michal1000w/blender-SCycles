#!/bin/bash
# Render the volume index of refraction comparison scenes and compare them:
#   tools/volume_ior/run_pairs.sh <Blender.app> <output directory> [devices] [configs] [samples] [resolution]
# 1. A refraction of strength zero must not change the image (`*_zero` against `*_off`).
# 2. Fog with a refraction too weak to see, which splits rays into segments, must match the fog
#    alone up to noise (`segments_fog` against `room_fog_off`).
# 3. Every integrator must render weak refraction as path tracing does on the same device
#    (`room_weak`, `room_weak_fog` against the `pt` configuration), judged against how far it is
#    from path tracing without refraction: bidirectional methods are biased in fog in this fork.
# 4. Devices must agree (every refracting scene of METAL against CPU), if both were rendered.
# Differences are block errors relative to the noise floor: that of two renders of the reference
# with different seeds. Appends to <output directory>/summary.txt, exits non-zero on a failure.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; OUT="$2"; DEVICES="${3:-CPU METAL}"
CONFIGS="${4:-pt pt_guiding bdpt bdpt_guiding photon vcm}"; SAMPLES="${5:-512}"; RESOLUTION="${6:-160}"
SCRIPT="$ROOT/tests/python/cycles_volume_ior_scenes.py"
COMPARE="$ROOT/tests/python/cycles_nested_dielectric_compare.py"
BLENDER="$APP/Contents/MacOS/Blender"
mkdir -p "$OUT"
status=0

compare() { # label threshold reference candidate noise
  local line
  line=$("$BLENDER" -b --factory-startup --python "$COMPARE" -- --threshold "$2" "$3" "$4" "$5" \
    2>/dev/null | grep NESTED_COMPARE)
  "$BLENDER" -b --factory-startup --python-expr "
import json, sys
r = json.loads('''${line#NESTED_COMPARE }''')
print('  %-34s mean ratio %.4f  block error %.4f  noise %.4f  ratio %5.2f  %s' % (
    '$1', r['mean_ratio'], r['block_error'], r['noise_floor'], r['ratio'],
    'FAILED' if r.get('failed') else 'ok'))
" 2>/dev/null | grep "ratio" | tee -a "$OUT/summary.txt"
  echo "$line" | grep -q '"failed": true' && status=1
  [ -n "$line" ] || { echo "  $1 NOT COMPARED FAILED" | tee -a "$OUT/summary.txt"; status=1; }
}

for device in $DEVICES; do
  for config in $CONFIGS; do
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
    dir="$OUT/pairs_${device}_${config}"
    "$BLENDER" -b --factory-startup --python "$SCRIPT" -- --output "$dir" --device "$device" \
      "${flags[@]}" --scenes pairs --samples "$SAMPLES" --resolution "$RESOLUTION" \
      > "$dir.log" 2>&1 || status=1
    "$BLENDER" -b --factory-startup --python "$SCRIPT" -- --output "$dir/noise" --device "$device" \
      "${flags[@]}" --scenes references --samples "$SAMPLES" \
      --resolution "$RESOLUTION" --seed 23 > "$dir.noise.log" 2>&1 || status=1
    echo "== $device $config" | tee -a "$OUT/summary.txt"
    grep "FAILED" "$dir.log" "$dir.noise.log" | tee -a "$OUT/summary.txt"
    compare "zero strength" 1.5 "$dir/room_off.exr" "$dir/room_zero.exr" "$dir/noise/room_off.exr"
    compare "zero strength in fog" 1.5 "$dir/room_fog_off.exr" "$dir/room_fog_zero.exr" \
      "$dir/noise/room_fog_off.exr"
    compare "segments in fog" 1.5 "$dir/room_fog_off.exr" "$dir/segments_fog.exr" \
      "$dir/noise/room_fog_off.exr"
    if [ "$config" != pt ]; then
      ref="$OUT/pairs_${device}_pt"
      compare "weak refraction against pt" 2.0 "$ref/room_weak.exr" "$dir/room_weak.exr" \
        "$ref/noise/room_weak.exr"
      compare "  without refraction" 100 "$ref/room_off.exr" "$dir/room_off.exr" \
        "$ref/noise/room_off.exr"
      compare "weak refraction in fog against pt" 3.0 "$ref/room_weak_fog.exr" \
        "$dir/room_weak_fog.exr" "$ref/noise/room_weak_fog.exr"
      compare "  without refraction" 100 "$ref/room_fog_off.exr" "$dir/room_fog_off.exr" \
        "$ref/noise/room_fog_off.exr"
    fi
  done
done

if [[ " $DEVICES " == *" CPU "* && " $DEVICES " == *" METAL "* ]]; then
  for config in $CONFIGS; do
    cpu="$OUT/pairs_CPU_${config}"; metal="$OUT/pairs_METAL_${config}"
    echo "== METAL against CPU, $config" | tee -a "$OUT/summary.txt"
    for scene in room room_fog room_weak room_weak_fog; do
      compare "$scene" 2.0 "$cpu/$scene.exr" "$metal/$scene.exr" "$cpu/noise/$scene.exr"
    done
  done
fi
exit $status
