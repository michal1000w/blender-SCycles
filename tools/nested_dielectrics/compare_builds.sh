#!/bin/bash
# Compare the renders of two builds scene by scene: every EXR in <before directory> with the
# file of the same name in <after directory>.
#   tools/nested_dielectrics/compare_builds.sh <Blender.app> <before directory> <after directory>
# Prints the mean, the maximum absolute difference and the relative RMSE per scene, and a count
# of scenes that are identical, close (relative RMSE below 1e-4) and different.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; BEFORE="$2"; AFTER="$3"
args=()
for reference in "$BEFORE"/*.exr; do
  name="$(basename "$reference")"
  [ -f "$AFTER/$name" ] && args+=("$reference" "$AFTER/$name")
done
"$APP/Contents/MacOS/Blender" -b --factory-startup \
  --python "$ROOT/tests/python/cycles_metal_image_compare.py" -- "${args[@]}" 2>&1 |
  grep "^COMPARE" | python3 -c '
import json, sys
identical = close = different = 0
for line in sys.stdin:
    r = json.loads(line.split(" ", 1)[1])
    name = r["candidate"].split("/")[-1][:-4]
    if r["max_abs_diff"] == 0.0:
        identical += 1
        state = "identical"
    elif r["relative_rmse"] < 1e-4:
        close += 1
        state = "close"
    else:
        different += 1
        state = "DIFFERENT"
    print("%-34s mean %9.5f -> %9.5f  max diff %.3e  rel rmse %.3e  %s%s" % (
        name, r["reference_mean"], r["candidate_mean"], r["max_abs_diff"], r["relative_rmse"],
        state, "" if r["finite"] else " NOT FINITE"))
print("identical %d, close %d, different %d" % (identical, close, different))'
