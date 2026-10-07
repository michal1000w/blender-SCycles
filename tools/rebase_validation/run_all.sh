#!/bin/bash
# Every test suite of the fork for one build, one GPU job at a time:
#   tools/rebase_validation/run_all.sh <Blender.app> <output directory> [stages]
# Stages (default all): suites nested vcm osl metalfx upstream
#   suites   feature, CPU parity and shader node scenes on Metal and CPU, EEVEE nodes
#   nested   nested dielectrics (tools/nested_dielectrics/run_all.sh)
#   vcm      vertex connection and merging against the references in build/vcm/out/ref
#   osl      custom (OSL) cameras, Metal translation against the CPU runtime, and a session
#   metalfx  MetalFX denoiser scenes, noisy and denoised
#   upstream Blender's Cycles render tests on CPU and Metal (needs tests/files/render from LFS)
# Run it for the build before and after a change and compare with compare_all.sh.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; OUT="$2"; STAGES="${3:-suites nested vcm osl metalfx upstream}"
BLENDER="$APP/Contents/MacOS/Blender"
VCM_REF="${VCM_REF:-$ROOT/build/vcm/out/ref}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
APP="$(cd "$APP" && pwd)"
BLENDER="$APP/Contents/MacOS/Blender"
stamp() { echo "$(date '+%H:%M:%S') $*" >> "$OUT/progress.txt"; }

for stage in $STAGES; do
  stamp "start $stage"
  case "$stage" in
    suites)
      "$ROOT/tools/rebase_validation/run_suites.sh" "$APP" "$OUT/suites" > "$OUT/suites.log" 2>&1
      ;;
    nested)
      "$ROOT/tools/nested_dielectrics/run_all.sh" "$APP" "$OUT/nested" > "$OUT/nested.log" 2>&1
      echo "nested exit $?" >> "$OUT/status.txt"
      ;;
    vcm)
      "$ROOT/tools/vcm/run_validation.sh" "$APP" "$VCM_REF" "$OUT/vcm" > "$OUT/vcm.log" 2>&1
      echo "vcm exit $?" >> "$OUT/status.txt"
      ;;
    osl)
      mkdir -p "$OUT/osl"
      for device in CPU METAL; do
        "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_osl_camera_scenes.py" -- \
          --output "$OUT/osl/$device" --device $device > "$OUT/osl/$device.log" 2>&1
        echo "osl scenes $device exit $?" >> "$OUT/status.txt"
      done
      "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_osl_camera_compare.py" -- \
        "$OUT/osl/CPU" "$OUT/osl/METAL" --report "$OUT/osl/cpu_vs_metal.json" \
        > "$OUT/osl/cpu_vs_metal.log" 2>&1
      echo "osl cpu vs metal exit $?" >> "$OUT/status.txt"
      "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_osl_camera_scenes.py" -- \
        --output "$OUT/osl/session" --device METAL --session > "$OUT/osl/session.log" 2>&1
      echo "osl session exit $?" >> "$OUT/status.txt"
      ;;
    metalfx)
      mkdir -p "$OUT/metalfx"
      "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_metalfx_scenes.py" -- \
        --output "$OUT/metalfx" --denoiser NONE --tag noisy > "$OUT/metalfx/noisy.log" 2>&1
      echo "metalfx noisy exit $?" >> "$OUT/status.txt"
      "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_metalfx_scenes.py" -- \
        --output "$OUT/metalfx" --denoiser METALFX --tag metalfx > "$OUT/metalfx/metalfx.log" 2>&1
      echo "metalfx denoise exit $?" >> "$OUT/status.txt"
      "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_metalfx_scenes.py" -- \
        --output "$OUT/metalfx" --denoiser METALFX --upscale QUALITY --tag upscale \
        > "$OUT/metalfx/upscale.log" 2>&1
      echo "metalfx upscale exit $?" >> "$OUT/status.txt"
      ;;
    upstream)
      "$ROOT/tools/rebase_validation/run_upstream_tests.sh" "$APP" CPU "$OUT/upstream_cpu" \
        > "$OUT/upstream_cpu.log" 2>&1
      "$ROOT/tools/rebase_validation/run_upstream_tests.sh" "$APP" METAL "$OUT/upstream_metal" \
        > "$OUT/upstream_metal.log" 2>&1
      ;;
  esac
  stamp "done $stage"
done
stamp "all done"
