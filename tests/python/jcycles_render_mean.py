# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Render a .blend on one device and print the mean of the image, to check whether a difference
between devices is noise (it shrinks with more samples) or bias (it does not).

  blender --background FILE.blend --python tests/python/jcycles_render_mean.py -- \
      --device CPU|CUDA|OPTIX [--samples N] [--seed N] [--set path=value ...] [--output FILE]

`--set` assigns scene properties, for example `--set cycles.use_guiding=False`.
"""

import argparse
import sys

import bpy
import numpy as np


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", default="", help="Keep the render as this .exr file")
    return parser.parse_args(argv)


def assign(scene, path, text):
    *parents, name = path.split(".")
    target = scene
    for parent in parents:
        target = getattr(target, parent)
    current = getattr(target, name)
    if isinstance(current, bool):
        value = text.lower() in {"1", "true", "yes"}
    else:
        value = type(current)(text)
    setattr(target, name, value)


def main():
    args = parse_args()
    scene = bpy.context.scene
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if args.device == "CPU":
        scene.cycles.device = "CPU"
    else:
        prefs.compute_device_type = args.device
        prefs.refresh_devices()
        for device in prefs.devices:
            device.use = device.type == args.device
        scene.cycles.device = "GPU"
    if args.samples:
        scene.cycles.samples = args.samples
        scene.cycles.use_adaptive_sampling = False
    scene.cycles.seed = args.seed
    for item in args.set:
        path, value = item.split("=", 1)
        assign(scene, path, value)

    scene.render.filepath = args.output or (bpy.app.tempdir + "render_mean.exr")
    scene.render.image_settings.file_format = "OPEN_EXR"
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath)
    pixels = np.array(image.pixels[:], dtype=np.float64).reshape(-1, 4)[:, :3]
    print(f"[render-mean] {args.device} samples={scene.cycles.samples} seed={args.seed} "
          f"{' '.join(args.set)} mean {pixels.mean():.5f}", flush=True)


main()
