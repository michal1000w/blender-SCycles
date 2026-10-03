# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Report of tools/vcm/benchmark.sh runs. Run inside Blender:

  blender -b --factory-startup --python tools/vcm/benchmark_report.py -- \
      --after DIRECTORY [--before DIRECTORY] [--reference DIRECTORY]

Prints, for every configuration and scene, the render time of the build after the change, the
time of the build before it when that directory has the configuration, and with a reference
directory of the same resolution the relative RMSE of the image and the time to reach unit
variance (time times squared error): the lower, the more efficient.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def load(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(image.size[1], image.size[0], image.channels)[:, :, :3]
    bpy.data.images.remove(image)
    return pixels.astype(np.float64)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--after", required=True, type=Path)
    parser.add_argument("--before", type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)

    print("VCM_BENCH %-24s %-16s %9s %9s %7s %8s %10s" % (
        "configuration", "scene", "before s", "after s", "ratio", "rel RMSE", "time*MSE"))
    for directory in sorted(path for path in args.after.iterdir() if path.is_dir()):
        timing_path = directory / "timing.json"
        if not timing_path.exists():
            continue
        after = json.loads(timing_path.read_text())
        before = {}
        if args.before and (args.before / directory.name / "timing.json").exists():
            before = json.loads((args.before / directory.name / "timing.json").read_text())
        for scene, entry in sorted(after.items()):
            seconds = entry["seconds"]
            before_seconds = before.get(scene, {}).get("seconds")
            rmse = None
            if args.reference and (args.reference / (scene + ".exr")).exists():
                reference = load(args.reference / (scene + ".exr"))
                image = load(directory / (scene + ".exr"))
                if reference.shape == image.shape:
                    # Fireflies of either image must not decide the comparison.
                    limit = 10.0 * max(reference.mean(), 1e-12)
                    difference = np.clip(image, 0, limit) - np.clip(reference, 0, limit)
                    rmse = float(np.sqrt((difference ** 2).mean()) / max(reference.mean(), 1e-12))
            print("VCM_BENCH %-24s %-16s %9s %9.2f %7s %8s %10s" % (
                directory.name, scene,
                "%.2f" % before_seconds if before_seconds else "-",
                seconds,
                "%.3f" % (seconds / before_seconds) if before_seconds else "-",
                "%.4f" % rmse if rmse is not None else "-",
                "%.4f" % (seconds * rmse * rmse) if rmse is not None else "-"), flush=True)


main()
