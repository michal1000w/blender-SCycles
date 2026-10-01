#!/bin/bash
# Run Blender's own Cycles render regression tests against their reference images.
#   tools/rebase_validation/run_upstream_tests.sh <Blender.app> <CPU|METAL|METAL-RT> <output directory> [test directories...]
# Needs the test files: git lfs fetch <upstream remote> <ref> -I 'tests/files/render/**' && git lfs checkout tests/files/render
# Failures listed for both the old and the new build are deviations of the fork from upstream
# (spectral glass, displacement); compare the two lists to find regressions.
R="$(cd "$(dirname "$0")/../.." && pwd)"
APP="$1"; DEV="$2"; OUT="$3"; shift 3
DIRS="$@"
[ -z "$DIRS" ] && DIRS="attributes bake bake_raytrace bsdf camera colorspace denoise denoise_animation displacement gsplat hair image_colorspace image_data_types image_mapping image_mipmap image_texture_limit instancing integrator light light_group light_linking lightprobe mesh motion_blur node_inlining openvdb osl pointcloud principled_bsdf raycast render_layer reports shader shadow shadow_catcher sss texture transparency updates volume"
mkdir -p "$OUT"
DL=$(echo "$DEV" | tr 'A-Z' 'a-z')
for d in $DIRS; do
  OSL=none; [ "$DEV" = CPU ] && OSL=limited
  DYLD_LIBRARY_PATH="$APP/Contents/Resources/lib" "$R/lib/macos_arm64/python/bin/python3.13" "$R/tests/python/cycles_render_tests.py" --blender "$APP/Contents/MacOS/Blender" --testdir "$R/tests/files/render/$d" --outdir "$OUT/cycles" --oiiotool "$R/lib/macos_arm64/openimageio/bin/oiiotool" --device "$DEV" --osl "$OSL" --batch > "$OUT/$d.log" 2>&1
  echo "$d exit $?" >> "$OUT/status.txt"
done
echo done >> "$OUT/status.txt"
