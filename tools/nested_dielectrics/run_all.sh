#!/bin/bash
# Complete nested dielectric test run of one build:
#   tools/nested_dielectrics/run_all.sh <Blender.app> <output directory> [devices]
# 1. validation pairs in every integrator configuration (run_validation.sh),
# 2. the deeper pairs (five nested media, camera inside two media, volume ending at a medium),
# 3. stress scenes, which only have to render finite images,
# 4. toggling the priority between renders of a persistent session.
# Writes <output directory>/summary.txt and exits non-zero if anything failed.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; OUT="$2"; DEVICES="${3:-CPU METAL}"
BLENDER="$APP/Contents/MacOS/Blender"
SCENES="$ROOT/tests/python/cycles_nested_dielectric_scenes.py"
mkdir -p "$OUT"
status=0
for device in $DEVICES; do
  configs="pt pt_guiding bdpt bdpt_guiding photon"
  [ "$device" = CPU ] && configs="$configs osl"
  "$ROOT/tools/nested_dielectrics/run_validation.sh" "$APP" "$OUT" "$device" "$configs" || status=1

  for integrator in pt bdpt photon; do
    dir="$OUT/extra_${device}_${integrator}"
    "$BLENDER" -b --factory-startup --python "$SCENES" -- --output "$dir" --device "$device" \
      --integrator "$integrator" --scenes extra --samples 256 --resolution 128 > "$dir.log" 2>&1
    echo "== $device $integrator extra pairs" | tee -a "$OUT/summary.txt"
    # Deterministic sampling: the pairs trace the same paths, the block error is absolute.
    "$ROOT/tools/nested_dielectrics/compare_pairs.sh" "$APP" "$dir" "" 0.01 |
      tee -a "$OUT/summary.txt"
    [ "${PIPESTATUS[0]}" = 0 ] || status=1

    dir="$OUT/stress_${device}_${integrator}"
    "$BLENDER" -b --factory-startup --python "$SCENES" -- --output "$dir" --device "$device" \
      --integrator "$integrator" --scenes stress --samples 64 --resolution 160 > "$dir.log" 2>&1
    echo "== $device $integrator stress scenes" | tee -a "$OUT/summary.txt"
    "$BLENDER" -b --factory-startup --python-expr "
import bpy, sys, glob, numpy as np
bad = 0
files = sorted(glob.glob('$dir/*.exr'))
for path in files:
    image = bpy.data.images.load(path)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    finite = bool(np.isfinite(pixels).all())
    bad += not finite
    print('NESTED_STRESS %-24s mean %.5f %s' % (path.split('/')[-1], pixels.mean(), 'finite' if finite else 'NOT FINITE FAILED'))
if len(files) != 5:
    print('NESTED_STRESS %d of 5 scenes rendered FAILED' % len(files))
    bad += 1
sys.exit(1 if bad else 0)
" 2>&1 | grep NESTED_STRESS | tee -a "$OUT/summary.txt"
    grep -q "NESTED_STRESS.*FAILED" "$OUT/summary.txt" && status=1
  done

  for integrator in pt bdpt; do
    echo "== $device $integrator priority changes in a persistent session" | tee -a "$OUT/summary.txt"
    "$BLENDER" -b --factory-startup \
      --python "$ROOT/tests/python/cycles_nested_dielectric_update.py" -- \
      --device "$device" --integrator "$integrator" 2>&1 | grep NESTED_UPDATE |
      tee -a "$OUT/summary.txt"
    [ "${PIPESTATUS[0]}" = 0 ] || status=1
  done
done
echo "failures: $(grep -c FAILED "$OUT/summary.txt")" | tee -a "$OUT/summary.txt"
exit $status
