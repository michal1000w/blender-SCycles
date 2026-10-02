# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare the renders of cycles_osl_camera_scenes.py from two devices.

  blender -b --factory-startup --python tests/python/cycles_osl_camera_compare.py -- \\
      REFERENCE_DIRECTORY CANDIDATE_DIRECTORY [--report FILE]

Probe images hold computed values, one sample per pixel: nearly all pixels must agree to
floating point accuracy. A few pixels may differ where a discontinuous function, such as a
comparison or rounding, amplifies the last bit of its argument.

Render images are noisy and the devices do not draw the same lens samples where the shader
hashes values that differ in the last bit, so those are compared after averaging blocks of
pixels.

Prints one line per image and exits with an error if any image fails.
"""

import json
import sys
from pathlib import Path

import bpy
import numpy as np

# Probe values: absolute and relative tolerance, and the fraction of pixels that may exceed it.
PROBE_ABS = 2e-4
PROBE_REL = 2e-4
PROBE_OUTLIERS = 0.004
# Image lookups: graphics hardware interpolates between pixels with a few bits of precision.
TEXTURE_PROBE_ABS = 3e-3
TEXTURE_PROBE_REL = 3e-3
# Render images: difference of the mean, and RMSE of block averages relative to the mean.
RENDER_MEAN = 0.02
RENDER_BLOCK_RMSE = 0.08
RENDER_BLOCK = 8


def load(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(image.size[1], image.size[0], image.channels)[:, :, :3]
    bpy.data.images.remove(image)
    return pixels.astype(np.float64)


def compare_probe(reference, candidate, absolute=PROBE_ABS, relative=PROBE_REL):
    # Undo the encoding of the probe shader. Black pixels have no ray.
    ref = np.where(reference.sum(axis=2, keepdims=True) > 0.0, (reference - 4.0) * 4.0, -99.0)
    cand = np.where(candidate.sum(axis=2, keepdims=True) > 0.0, (candidate - 4.0) * 4.0, -99.0)
    difference = np.abs(cand - ref)
    tolerance = absolute + relative * np.abs(ref)
    outliers = float((difference > tolerance).any(axis=2).mean())
    inside = difference[difference <= tolerance]
    return {
        "kind": "probe",
        "outliers": outliers,
        "max_diff": float(difference.max()),
        "max_diff_inside": float(inside.max()) if inside.size else 0.0,
        "value_range": [float(ref.min()), float(ref.max())],
        "constant": bool(ref.max() - ref.min() < 1e-6),
        "ok": outliers <= PROBE_OUTLIERS,
    }


def block_average(image, block):
    height = image.shape[0] // block * block
    width = image.shape[1] // block * block
    cropped = image[:height, :width]
    return cropped.reshape(height // block, block, width // block, block, 3).mean(axis=(1, 3))


def compare_render(reference, candidate):
    mean = float(reference.mean())
    mean_diff = abs(float(candidate.mean()) - mean) / max(mean, 1e-12)
    ref_blocks = block_average(reference, RENDER_BLOCK)
    cand_blocks = block_average(candidate, RENDER_BLOCK)
    block_rmse = float(np.sqrt(((cand_blocks - ref_blocks) ** 2).mean())) / max(mean, 1e-12)
    pixel_rmse = float(np.sqrt(((candidate - reference) ** 2).mean())) / max(mean, 1e-12)
    return {
        "kind": "render",
        "reference_mean": mean,
        "candidate_mean": float(candidate.mean()),
        "mean_diff": mean_diff,
        "block_rmse": block_rmse,
        "pixel_rmse": pixel_rmse,
        "ok": (mean_diff <= RENDER_MEAN and block_rmse <= RENDER_BLOCK_RMSE) or
              (mean < 1e-9 and float(candidate.mean()) < 1e-9),
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    report_path = None
    if "--report" in argv:
        index = argv.index("--report")
        report_path = Path(argv[index + 1])
        del argv[index:index + 2]
    if len(argv) != 2:
        raise SystemExit("Pass a reference and a candidate directory")
    reference_dir, candidate_dir = Path(argv[0]), Path(argv[1])

    results = {}
    failed = []
    for reference_path in sorted(reference_dir.glob("*.exr")):
        name = reference_path.stem
        candidate_path = candidate_dir / reference_path.name
        if not candidate_path.exists():
            results[name] = {"ok": False, "missing": True}
            failed.append(name)
            print(f"COMPARE {name} MISSING")
            continue
        reference = load(reference_path)
        candidate = load(candidate_path)
        if reference.shape != candidate.shape:
            result = {"ok": False, "shape_mismatch": True}
        elif not np.isfinite(candidate).all():
            result = {"ok": False, "not_finite": True}
        elif name.startswith("probe_texture"):
            result = compare_probe(reference, candidate, TEXTURE_PROBE_ABS, TEXTURE_PROBE_REL)
        elif name.startswith("probe"):
            result = compare_probe(reference, candidate)
        else:
            result = compare_render(reference, candidate)
        results[name] = result
        if not result["ok"]:
            failed.append(name)
        print(f"COMPARE {name} {'ok' if result['ok'] else 'FAILED'} {json.dumps(result)}")

    print(f"SUMMARY {len(results) - len(failed)} of {len(results)} images match")
    if report_path:
        report_path.write_text(json.dumps(results, indent=2) + "\n")
    if failed:
        print("FAILED " + " ".join(failed))
        sys.exit(1)


main()
