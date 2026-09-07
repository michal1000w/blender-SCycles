# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Benchmark a loaded scene without saving changes to its .blend file.

Example:
  Blender --background --factory-startup scene.blend --python-exit-code 1 \
    --python tools/pixel_displacement_performance.py -- \
    --mode unlimited --percentage 10 --samples 8 --output /tmp/displacement-test

Run baseline and candidate builds sequentially on the same machine. Compare warm
medians separately from the first render, which includes kernel compilation and
scene preparation. Each run saves an EXR and JSON for fidelity and memory checks.
"""

import argparse
import hashlib
import json
from pathlib import Path
import os
import platform
import resource
import statistics
import sys
import time

import bpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        help="unlimited or a displacement resolution from 64 to 16384")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--percentage", type=int, default=10)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if not 1 <= args.percentage <= 100 or args.samples < 1 or args.repeat < 3:
        parser.error("percentage must be 1..100, samples positive, and repeat at least 3")

    if args.mode != "unlimited" and (not args.mode.isdecimal() or not 64 <= int(args.mode) <= 16384):
        parser.error("mode must be unlimited or an integer from 64 to 16384")
    if ((args.width is None) != (args.height is None) or
            (args.width is not None and min(args.width, args.height) < 1)):
        parser.error("width and height must both be positive integers")

    scene = bpy.context.scene
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.metalrt = "ON"
    prefs.get_devices()
    metal_devices = [device for device in prefs.devices if device.type == "METAL"]
    if not metal_devices:
        raise RuntimeError("This pixel-displacement implementation requires a Metal GPU")
    for device in prefs.devices:
        device.use = device.type == "METAL"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.use_pixel_displacement = True
    scene.cycles.use_pixel_displacement_resolution_clamp = args.mode != "unlimited"
    scene.cycles.pixel_displacement_resolution = 16384 if args.mode == "unlimited" else int(args.mode)
    scene.cycles.samples = args.samples
    scene.cycles.seed = args.seed
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    if args.width is not None:
        scene.render.resolution_x = args.width
        scene.render.resolution_y = args.height
    scene.render.resolution_percentage = args.percentage
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.label}-{args.mode}"
    report = {
        "label": args.label,
        "scene": bpy.data.filepath,
        "scene_sha256": hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        "build_hash": bpy.app.build_hash.decode(),
        "version": bpy.app.version_string,
        "gpu": [device.name for device in metal_devices],
        "mode": args.mode,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y, args.percentage],
        "render_pixels": [scene.render.resolution_x * args.percentage // 100,
                          scene.render.resolution_y * args.percentage // 100],
        "displacement_environment": {key: value for key, value in os.environ.items()
                                     if key.startswith("CYCLES_PIXEL_DISPLACEMENT_")},
        "samples": args.samples,
        "seed": scene.cycles.seed,
        "steps": scene.cycles.pixel_displacement_steps,
        "max_distance": scene.cycles.pixel_displacement_max_distance,
        "scale": scene.cycles.pixel_displacement_scale,
        "renders_seconds": [],
        "completed": False,
    }
    print("DISPLACEMENT_BENCH_START " + json.dumps(report), flush=True)
    for index in range(args.repeat):
        start = time.perf_counter()
        bpy.ops.render.render()
        elapsed = time.perf_counter() - start
        report["renders_seconds"].append(elapsed)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_process_rss_bytes"] = rss if platform.system() == "Darwin" else rss * 1024
        report["renders_completed"] = index + 1
        if index > 0:
            report["warm_median_seconds"] = statistics.median(report["renders_seconds"][1:])
        (args.output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"DISPLACEMENT_BENCH_RENDER {index} {elapsed:.9f}", flush=True)
    bpy.data.images["Render Result"].save_render(str(args.output / f"{stem}.exr"), scene=scene)
    report["completed"] = True
    (args.output / f"{stem}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DISPLACEMENT_BENCH_RESULT " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
