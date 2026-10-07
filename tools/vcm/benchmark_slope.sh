#!/bin/bash
# Steady-state cost of vertex connection and merging:
#   tools/vcm/benchmark_slope.sh <Blender.app> <output directory> [devices] [configurations] \
#       [scenes] [resolution] [samples low] [samples high]
# Renders every configuration with two sample counts and prints, per scene, both render times and
# the time per sample between them, which leaves out scene export and kernel loading. Lines of a
# configuration with merging also print the ratio to the same configuration without it.
# Never run two at once: concurrent GPU renders distort each other.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; OUT="$2"
DEVICES="${3:-CPU METAL}"
CONFIGS="${4:-pt pt_vcm pt_guiding pt_vcm_guiding bdpt bdpt_vcm bdpt_guiding bdpt_vcm_guiding}"
SCENES="${5:-diffuse_box,caustic_glass,mirror_caustic,principled,demo_pool,demo_lamp}"
RESOLUTION="${6:-512}"
LOW="${7:-32}"
HIGH="${8:-160}"
BLENDER="$APP/Contents/MacOS/Blender"
mkdir -p "$OUT"
for device in $DEVICES; do
  for config in $CONFIGS; do
    args=()
    case "$config" in bdpt*) args+=(--integrator bdpt) ;; esac
    case "$config" in *vcm*) args+=(--vcm) ;; esac
    case "$config" in *guiding*) args+=(--guiding) ;; esac
    for samples in $LOW $HIGH; do
      dir="$OUT/${device}_${config}_${samples}"
      rm -f "$dir/timing.json"
      "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_vcm_scenes.py" -- \
        --output "$dir" --device "$device" --samples "$samples" --scenes "$SCENES" \
        --resolution "$RESOLUTION" ${args[@]+"${args[@]}"} ${VCM_EXTRA_ARGS:-} > "$dir.log" 2>&1
    done
  done
done
python3 - "$OUT" "$LOW" "$HIGH" "$DEVICES" "$CONFIGS" <<'PY'
import json, sys
from pathlib import Path
out, low, high = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
slopes = {}
for device in sys.argv[4].split():
    for config in sys.argv[5].split():
        try:
            a = json.loads((out / f"{device}_{config}_{low}" / "timing.json").read_text())
            b = json.loads((out / f"{device}_{config}_{high}" / "timing.json").read_text())
        except OSError:
            print(f"VCM_SLOPE {device}_{config} missing")
            continue
        for scene in a:
            slope = (b[scene]["seconds"] - a[scene]["seconds"]) / (high - low)
            slopes[(device, config, scene)] = (slope, a[scene]["seconds"], b[scene]["seconds"])
for (device, config, scene), (slope, t_low, t_high) in slopes.items():
    base = slopes.get((device, config.replace("_vcm", ""), scene)) if "_vcm" in config else None
    ratio = " x%.2f total x%.2f" % (slope / base[0], t_high / base[2]) if base else ""
    print("VCM_SLOPE %-6s %-18s %-16s %6.2f s %6.2f s  %7.2f ms/sample%s" % (
        device, config, scene, t_low, t_high, slope * 1000.0, ratio))
PY
