#!/bin/bash
# Vertex connection and merging benchmark:
#   tools/vcm/benchmark.sh <Blender.app> <output directory> [devices] [configurations] \
#       [samples] [scenes] [resolution]
# Renders the scenes with a fixed number of samples in every configuration, one process per
# configuration, and records the render times in <output directory>/<device>_<config>/timing.json.
# Run it for the build before and after a change with the configurations that both have
# (pt pt_guiding bdpt bdpt_guiding), then report with tools/vcm/benchmark_report.py.
# Never run two at once: concurrent GPU renders distort each other.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; OUT="$2"
DEVICES="${3:-CPU METAL}"
CONFIGS="${4:-pt pt_guiding bdpt bdpt_guiding pt_vcm pt_vcm_guiding bdpt_vcm bdpt_vcm_guiding}"
SAMPLES="${5:-256}"
SCENES="${6:-diffuse_box,caustic_glass,mirror_caustic,ring,sun_sky,principled}"
RESOLUTION="${7:-256}"
BLENDER="$APP/Contents/MacOS/Blender"
mkdir -p "$OUT"
for device in $DEVICES; do
  for config in $CONFIGS; do
    args=()
    case "$config" in bdpt*) args+=(--integrator bdpt) ;; esac
    case "$config" in *vcm*) args+=(--vcm) ;; esac
    case "$config" in *guiding*) args+=(--guiding) ;; esac
    dir="$OUT/${device}_${config}"
    rm -f "$dir/timing.json"
    "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_vcm_scenes.py" -- \
      --output "$dir" --device "$device" --samples "$SAMPLES" --scenes "$SCENES" \
      --resolution "$RESOLUTION" ${args[@]+"${args[@]}"} > "$dir.log" 2>&1
    grep "VCM_SCENE" "$dir.log" | sed "s/^/${device}_${config} /"
  done
done
