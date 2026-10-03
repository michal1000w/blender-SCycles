#!/bin/bash
# Vertex connection and merging validation of one build:
#   tools/vcm/run_validation.sh <Blender.app> <reference directory> <output directory> \
#       [devices] [configurations] [samples] [scenes]
# Renders the validation scenes of tests/python/cycles_vcm_scenes.py in every configuration and
# compares each with the path traced references (render those once, with many samples:
#   blender -b --factory-startup --python tests/python/cycles_vcm_scenes.py -- \
#       --output <reference directory> --samples 32768).
# Configurations: pt, bdpt (merging off), pt_vcm, bdpt_vcm, each also with _guiding.
# Appends to <output directory>/summary.txt and exits non-zero if a comparison failed.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; REF="$2"; OUT="$3"
DEVICES="${4:-CPU METAL}"
CONFIGS="${5:-pt_vcm pt_vcm_guiding bdpt_vcm bdpt_vcm_guiding}"
SAMPLES="${6:-1024}"
SCENES="${7:-validation}"
BLENDER="$APP/Contents/MacOS/Blender"
mkdir -p "$OUT"
status=0
for device in $DEVICES; do
  for config in $CONFIGS; do
    args=()
    case "$config" in bdpt*) args+=(--integrator bdpt) ;; esac
    case "$config" in *vcm*) args+=(--vcm) ;; esac
    case "$config" in *guiding*) args+=(--guiding) ;; esac
    dir="$OUT/${device}_${config}"
    "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_vcm_scenes.py" -- \
      --output "$dir" --device "$device" --samples "$SAMPLES" --scenes "$SCENES" \
      ${args[@]+"${args[@]}"} > "$dir.log" 2>&1
    grep -E "Traceback|Error:|error:" "$dir.log" | head -5
    "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_vcm_compare.py" -- \
      --reference "$REF" --candidate "$dir" --label "${device}_${config}" 2>&1 |
      grep "VCM_COMPARE " | tee -a "$OUT/summary.txt"
    [ "${PIPESTATUS[0]}" = 0 ] || status=1
  done
done
exit $status
