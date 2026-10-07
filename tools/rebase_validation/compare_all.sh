#!/bin/bash
# Compare two output directories of run_all.sh, stage by stage:
#   tools/rebase_validation/compare_all.sh <Blender.app> <before directory> <after directory>
# Prints image differences of the general suites, the pass/fail lines of the feature suites that
# changed, and the upstream render tests whose result changed.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; REF="$2"; NEW="$3"
BLENDER="$APP/Contents/MacOS/Blender"

echo "######## exit codes (before | after)"
diff <(cat "$REF/suites/status.txt" "$REF/status.txt" 2>/dev/null) \
     <(cat "$NEW/suites/status.txt" "$NEW/status.txt" 2>/dev/null) && echo "same"

echo "######## general suites, after against before"
"$ROOT/tools/rebase_validation/compare.sh" "$APP" "$REF/suites" "$NEW/suites"
echo "######## general suites, CPU against Metal in the after build"
"$ROOT/tools/rebase_validation/compare.sh" "$APP" "$NEW/suites" "$NEW/suites"

for stage in nested vcm; do
  echo "######## $stage: failures before $(grep -c FAILED "$REF/$stage/summary.txt")," \
       "after $(grep -c FAILED "$NEW/$stage/summary.txt")"
  diff <(grep FAILED "$REF/$stage/summary.txt" | awk '{print $1, $2, $3}' | sort) \
       <(grep FAILED "$NEW/$stage/summary.txt" | awk '{print $1, $2, $3}' | sort)
done

echo "######## osl cameras: Metal against CPU in the after build"
grep -c . "$NEW/osl/cpu_vs_metal.log" > /dev/null 2>&1
grep -E "FAIL|fail" "$NEW/osl/cpu_vs_metal.log" | head -20
for device in CPU METAL; do
  echo "######## osl cameras $device, after against before"
  "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_osl_camera_compare.py" -- \
    "$REF/osl/$device" "$NEW/osl/$device" 2>&1 | grep -E "FAIL|fail|passed|images" | head -20
done
echo "######## osl session"
diff <(grep -E "^SESSION|FAIL|Error" "$REF/osl/session.log" | sed 's/[0-9.]*s$//') \
     <(grep -E "^SESSION|FAIL|Error" "$NEW/osl/session.log" | sed 's/[0-9.]*s$//') && echo "same"

echo "######## metalfx, after against before"
for tag in noisy metalfx upscale; do
  "$BLENDER" -b --factory-startup --python "$ROOT/tests/python/cycles_metalfx_compare.py" -- \
    --reference "$REF/metalfx:$tag" --test "$NEW/metalfx" --tags $tag 2>&1 |
    grep -vE "^Blender|^$|^Read|quit"
done

for device in cpu metal; do
  echo "######## upstream render tests $device: result changes (< before, > after)"
  diff <(cd "$REF/upstream_$device" 2>/dev/null && grep -H "FAILED   \]" *.log | grep -v "tests, listed" | sort -u) \
       <(cd "$NEW/upstream_$device" 2>/dev/null && grep -H "FAILED   \]" *.log | grep -v "tests, listed" | sort -u)
  echo "failed before: $(cd "$REF/upstream_$device" 2>/dev/null && grep -h "FAILED   \]" *.log | grep -v "tests, listed" | sort -u | wc -l)," \
       "after: $(cd "$NEW/upstream_$device" 2>/dev/null && grep -h "FAILED   \]" *.log | grep -v "tests, listed" | sort -u | wc -l)"
done
