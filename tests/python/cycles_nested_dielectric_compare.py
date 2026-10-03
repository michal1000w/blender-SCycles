# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare nested dielectric renders against their references. Run inside Blender:

  blender -b --factory-startup --python tests/python/cycles_nested_dielectric_compare.py -- \
      [--block N] [--threshold X] reference.exr candidate.exr [noise.exr] [-- next triple ...]

Images are averaged over blocks of pixels before comparing, which removes most Monte Carlo
noise and keeps differences in the transport. With a third image, an independent render of the
reference (another seed), the difference of the two references is the noise floor that the
candidate is judged against: `ratio` is the block error of the candidate over that floor, and
values near 1 mean the candidate is indistinguishable from the reference.

Prints a line per comparison and exits non-zero if any exceeds the threshold.
"""

import json
import sys

import bpy
import numpy as np


def load(path):
    image = bpy.data.images.load(path, check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(image.size[1], image.size[0], image.channels)[:, :, :3]
    bpy.data.images.remove(image)
    return pixels.astype(np.float64)


def blocks(image, size):
    height = image.shape[0] // size * size
    width = image.shape[1] // size * size
    image = image[:height, :width]
    return image.reshape(height // size, size, width // size, size, 3).mean(axis=(1, 3))


def block_error(a, b, size):
    """Root mean square difference of block averages, relative to the mean of the reference."""
    a = blocks(a, size)
    b = blocks(b, size)
    return float(np.sqrt(((a - b) ** 2).mean()) / max(a.mean(), 1e-12))


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    block = 8
    threshold = None
    groups = [[]]
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--block":
            block = int(argv[index + 1])
            index += 2
        elif arg == "--threshold":
            threshold = float(argv[index + 1])
            index += 2
        elif arg == "--":
            groups.append([])
            index += 1
        else:
            groups[-1].append(arg)
            index += 1
    failed = False
    for group in groups:
        if not group:
            continue
        if len(group) not in (2, 3):
            raise SystemExit("Pass reference, candidate and optionally a second reference")
        reference = load(group[0])
        candidate = load(group[1])
        result = {
            "reference": group[0],
            "candidate": group[1],
            "finite": bool(np.isfinite(candidate).all()),
            "reference_mean": float(reference.mean()),
            "candidate_mean": float(candidate.mean()),
            "mean_ratio": float(candidate.mean() / max(reference.mean(), 1e-12)),
            "block_error": block_error(reference, candidate, block),
        }
        value = result["block_error"]
        if len(group) == 3:
            noise = load(group[2])
            result["noise_floor"] = block_error(reference, noise, block)
            result["ratio"] = result["block_error"] / max(result["noise_floor"], 1e-12)
            value = result["ratio"]
        if not result["finite"] or (threshold is not None and value > threshold):
            result["failed"] = True
            failed = True
        print("NESTED_COMPARE " + json.dumps(result), flush=True)
    if failed:
        sys.exit(1)


main()
