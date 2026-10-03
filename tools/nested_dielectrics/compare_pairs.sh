#!/bin/bash
# Compare the nested and reference render of every validation pair in a directory.
#   tools/nested_dielectrics/compare_pairs.sh <Blender.app> <directory> [noise directory] [threshold]
# With a noise directory (the same scenes rendered with another seed) the error is reported
# relative to the noise floor of the reference.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; DIR="$2"; NOISE="${3:-}"; THRESHOLD="${4:-}"
args=()
[ -n "$THRESHOLD" ] && args+=(--threshold "$THRESHOLD")
# Kinds of a pair: media by priority, automatic, and with equal priorities.
for reference in "$DIR"/*_reference.exr; do
  name="$(basename "$reference" _reference.exr)"
  for kind in nested auto equal; do
    [ -f "$DIR/${name}_${kind}.exr" ] || continue
    args+=(-- "$reference" "$DIR/${name}_${kind}.exr")
    [ -n "$NOISE" ] && [ -f "$NOISE/${name}_reference.exr" ] && args+=("$NOISE/${name}_reference.exr")
  done
done
"$APP/Contents/MacOS/Blender" -b --factory-startup \
  --python "$ROOT/tests/python/cycles_nested_dielectric_compare.py" -- "${args[@]}" 2>&1 |
  grep NESTED_COMPARE | python3 -c '
import json, sys
bad = 0
for line in sys.stdin:
    r = json.loads(line.split(" ", 1)[1])
    name = r["candidate"].split("/")[-1].replace(".exr", "")
    text = "%-24s mean %8.5f -> %8.5f (x%.4f)  block error %.5f" % (
        name, r["reference_mean"], r["candidate_mean"], r["mean_ratio"], r["block_error"])
    if "ratio" in r:
        text += "  noise %.5f  ratio %.2f" % (r["noise_floor"], r["ratio"])
    if r.get("failed") or not r["finite"]:
        text += "  FAILED"
        bad += 1
    print(text)
sys.exit(1 if bad else 0)'
