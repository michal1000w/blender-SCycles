#!/bin/bash
# Interleaved render time benchmark of the Grating model scenes in two builds (for example the
# reference branch and this fork), path tracing on CPU and Metal:
#   tools/diffraction_grating/benchmark.sh <A Blender.app> <B Blender.app> <out> [rounds] [res] [spp]
# Each scene is rendered twice per process and the second render is timed. Round 0 warms the
# kernel caches and is not reported. Summarize with benchmark_report.py.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
A="$(cd "$1" && pwd)"; B="$(cd "$2" && pwd)"; OUT="$3"; ROUNDS="${4:-3}"; RES="${5:-512}"; SPP="${6:-64}"
SCENES="${SCENES:-cd_rectangular,cd_sinusoidal,grid_2d,cloth,sphere_uv,cd_hero}"
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
cp "$ROOT/tests/python/cycles_diffraction_grating_scenes.py" "$OUT/scenes.py"
run() {
  local label="$1" app="$2" round="$3" device
  for device in METAL CPU; do
    [ "$round" = 0 ] && [ "$device" = CPU ] && continue
    "$app/Contents/MacOS/Blender" -b --factory-startup --python-exit-code 1 --python "$OUT/scenes.py" -- \
      --output "$OUT/${label}_round${round}/$device" --device $device --scenes "$SCENES" \
      --resolution "$RES" --samples "$SPP" --timing > "$OUT/${label}_round${round}_$device.log" 2>&1
    echo "$label round $round $device exit $?" >> "$OUT/status.txt"
  done
}
for round in $(seq 0 "$ROUNDS"); do
  run A "$A" "$round"
  run B "$B" "$round"
done
python3 "$ROOT/tools/diffraction_grating/benchmark_report.py" "$OUT" | tee "$OUT/report.txt"
