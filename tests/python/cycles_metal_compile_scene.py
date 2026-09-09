#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate a matched Metal displacement scene and time its first and warm renders.

Run inside Blender with --python-exit-code 1. Wrap Blender in
cycles_metal_compile_benchmark.py to measure compiler memory and cache conditions.
The defaults reproduce the small cold-compilation reference scene. Increasing
--samples and --resolution also permits measuring steady rendering throughput.
"""

import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import bpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--bdpt", action="store_true")
    parser.add_argument("--guiding", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise RuntimeError("Use an empty output directory")
    root = Path(__file__).resolve().parents[2]
    setup = runpy.run_path(str(root / "tools/pixel_displacement_benchmark.py"))
    setup["create_scene"](2, 64, args.samples, args.resolution)
    scene = bpy.context.scene
    scene.cycles.use_pixel_displacement_resolution_clamp = False
    scene.cycles.use_bidirectional_path_tracing = args.bdpt
    scene.cycles.use_guiding = args.guiding
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    times = []
    for repeat in range(2):
        started = time.perf_counter()
        bpy.ops.render.render()
        times.append(time.perf_counter() - started)
        bpy.data.images["Render Result"].save_render(
            str(args.output / f"render-{repeat}.exr"), scene=scene)
        print("COMPILE_BENCH_RENDER", repeat, times[-1], flush=True)
    (args.output / "render.json").write_text(json.dumps({"seconds": times}, indent=2) + "\n")


if __name__ == "__main__":
    main()
