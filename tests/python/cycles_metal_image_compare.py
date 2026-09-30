# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare EXR renders pixel-wise. Run inside Blender:

  blender -b --factory-startup --python tests/python/cycles_metal_image_compare.py -- \
      reference.exr candidate.exr [more pairs...]

Prints mean luminance, mean absolute difference, relative RMSE and maximum difference for each
pair, and a JSON line per pair for scripting.
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


def main():
    paths = sys.argv[sys.argv.index("--") + 1:]
    if len(paths) < 2 or len(paths) % 2:
        raise SystemExit("Pass reference/candidate pairs")
    for reference_path, candidate_path in zip(paths[0::2], paths[1::2]):
        reference = load(reference_path)
        candidate = load(candidate_path)
        if reference.shape != candidate.shape:
            raise SystemExit(f"Shape mismatch {reference.shape} {candidate.shape}")
        difference = candidate - reference
        mean = float(reference.mean())
        result = {
            "reference": reference_path,
            "candidate": candidate_path,
            "reference_mean": mean,
            "candidate_mean": float(candidate.mean()),
            "mean_abs_diff": float(np.abs(difference).mean()),
            "relative_rmse": float(np.sqrt((difference ** 2).mean()) / max(mean, 1e-12)),
            "max_abs_diff": float(np.abs(difference).max()),
            "finite": bool(np.isfinite(candidate).all()),
        }
        print("COMPARE " + json.dumps(result))


main()
