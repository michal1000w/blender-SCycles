#!/bin/sh
# Launch the current repository build with matching runtime resources.
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$repo_dir/tools/launch_diffraction_build.sh" "$@"
