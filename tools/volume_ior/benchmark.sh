#!/bin/bash
# Render times of the volume index of refraction: the demo scenes and the test room, with the
# refraction and with it switched off in the render settings.
#   tools/volume_ior/benchmark.sh <Blender.app> <output directory> [devices] [rounds]
# The first round of a device warms the kernel caches and is not reported.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; OUT="$2"; DEVICES="${3:-CPU METAL}"; ROUNDS="${4:-2}"
BLENDER="$APP/Contents/MacOS/Blender"
mkdir -p "$OUT"
for device in $DEVICES; do
  for round in $(seq 0 "$ROUNDS"); do
    for mode in on off; do
      flag=""; [ $mode = off ] && flag="--no-refraction"
      "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_volume_ior_demos.py" -- \
        --output "$OUT/scenes" --render --render-dir "$OUT/${device}_${mode}" --device "$device" \
        --samples 64 --resolution 960 $flag 2>&1 | grep "^VOLUME_IOR_DEMO" |
        sed "s/^VOLUME_IOR_DEMO/$device $mode round$round/" >> "$OUT/times.txt"
    done
    for integrator in pt bdpt; do
      "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_volume_ior_scenes.py" -- \
        --output "$OUT/room_${device}_${integrator}" --device "$device" --integrator $integrator \
        --scenes room_off,room,room_weak,room_fog_off,room_fog,room_weak_fog --samples 256 \
        --resolution 256 2>&1 | grep "^VOLUME_IOR" |
        sed "s/^VOLUME_IOR/$device $integrator round$round/" >> "$OUT/times.txt"
    done
  done
done
python3 - "$OUT/times.txt" <<'PY'
import collections, re, statistics, sys
times = collections.defaultdict(list)
for line in open(sys.argv[1]):
    m = re.match(r"(\S+) (\S+) round(\d+)\s+(\S+)\s+rendered in\s+([\d.]+) s", line)
    if m and m.group(3) != "0":
        times[(m.group(1), m.group(2), m.group(4))].append(float(m.group(5)))
for key in sorted(times):
    print("%-6s %-5s %-16s %7.1f s" % (*key, statistics.median(times[key])))
PY
