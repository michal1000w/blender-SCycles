#!/bin/bash
# Render the fork's feature, parity and shader node suites on Metal and CPU.
#   tools/rebase_validation/run_suites.sh <Blender.app> <output directory>
# Run it for the build before and after a change, then compare with compare.sh.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BLENDER="$1/Contents/MacOS/Blender"
OUT="$2"
mkdir -p "$OUT"
for device in METAL CPU; do
  "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_metal_feature_scenes.py" -- \
    --output "$OUT/feature_$device" --device $device --timing > "$OUT/feature_$device.log" 2>&1
  echo "feature $device exit $?" >> "$OUT/status.txt"
  "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_cpu_parity_scenes.py" -- \
    --output "$OUT/parity_$device" --device $device > "$OUT/parity_$device.log" 2>&1
  echo "parity $device exit $?" >> "$OUT/status.txt"
  "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_fork_node_scenes.py" -- \
    --output "$OUT/nodes_$device" --device $device > "$OUT/nodes_$device.log" 2>&1
  echo "nodes $device exit $?" >> "$OUT/status.txt"
done
"$BLENDER" -b --factory-startup --python "$ROOT/tests/python/eevee_fork_node_test.py" -- \
  "$OUT/eevee_nodes" > "$OUT/eevee_nodes.log" 2>&1
echo "eevee nodes exit $?" >> "$OUT/status.txt"
cat "$OUT/status.txt"
