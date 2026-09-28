#!/bin/sh
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
# Launch the complete current installation with its matching runtime resources.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
app_dir="$repo_dir/install/Blender.app"
app_binary="$app_dir/Contents/MacOS/Blender"
runtime_dir="$app_dir/Contents/Resources"
cycles_dir="$runtime_dir/5.3/scripts/addons_core/cycles"
if [ ! -x "$app_binary" ] || [ ! -f "$cycles_dir/shader/node_glass_bsdf.oso" ]; then
    echo 'Install the current build first: ./compile.sh -j4 --no-fetch-libraries' >&2
    exit 1
fi
for addon_file in properties.py ui.py; do
    if ! cmp -s "$repo_dir/intern/cycles/blender/addon/$addon_file" "$cycles_dir/$addon_file"; then
        echo 'The installed Cycles add-on is stale. Run ./compile.sh -j4 --no-fetch-libraries.' >&2
        exit 1
    fi
done
export BLENDER_SYSTEM_RESOURCES="$runtime_dir/5.3"
export DYLD_LIBRARY_PATH="$runtime_dir/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export CYCLES_KERNEL_PATH="$cycles_dir/source"
export CYCLES_SHADER_PATH="$cycles_dir/shader"
exec "$app_binary" "$@"
