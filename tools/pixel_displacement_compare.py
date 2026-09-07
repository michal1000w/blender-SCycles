# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Compare linear render EXRs using Blender's image loader and bundled NumPy.

Blender --background --factory-startup --python tools/pixel_displacement_compare.py -- \
    --baseline baseline.exr --candidate candidate.exr --output comparison --preview
"""

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    arrays, sizes = [], []
    for path, label in ((args.baseline, "baseline"), (args.candidate, "candidate")):
        image = bpy.data.images.load(str(path), check_existing=False)
        try:
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
            arrays.append(pixels.reshape(-1, 4))
            sizes.append(tuple(image.size))
            if args.preview:
                bpy.context.scene.render.image_settings.file_format = "PNG"
                bpy.context.scene.view_settings.view_transform = "AgX"
                image.save_render(str(args.output / (label + ".png")))
        finally:
            bpy.data.images.remove(image)
    if sizes[0] != sizes[1] or arrays[0].shape != arrays[1].shape:
        raise ValueError("Render dimensions do not match")
    a, b = arrays
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("A render contains non-finite values")
    difference = np.abs(a[:, :3].astype(np.float64) - b[:, :3])
    result = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "size": sizes[0],
        "finite": True,
        "max_abs": float(difference.max()),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "mean_abs": float(difference.mean()),
        "relative_l1": float(difference.sum() / max(float(np.abs(a[:, :3]).sum()), 1.0e-30)),
        "pixels_over_0_001": int(np.sum(difference.max(axis=1) > 0.001)),
        "pixels": len(a),
        "p99_abs": float(np.quantile(difference, 0.99)),
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print("IMAGE_COMPARISON", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
