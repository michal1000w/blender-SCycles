# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare renders of tests/python/cycles_vcm_scenes.py with a reference directory.

  blender -b --factory-startup --python tests/python/cycles_vcm_compare.py -- \
      --reference DIRECTORY --candidate DIRECTORY [--scenes a,b] [--mean 0.02] [--block 0.10] \
      [--label TEXT]

For every image of the candidate directory that the reference directory has too, prints the
ratio of the image means, the relative error of block means (blocks of 1/8 of the image, which
average out noise but not a wrong amount of light in part of the image) and the relative RMSE of
the pixels. A scene fails when the mean or the block error exceeds its threshold, or when the
image has non-finite or negative pixels. Exits non-zero when any scene failed.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--scenes", default="")
    parser.add_argument("--mean", type=float, default=0.02)
    parser.add_argument("--block", type=float, default=0.10)
    parser.add_argument("--label", default="")
    return parser.parse_args(argv)


def load(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(image.size[1], image.size[0], image.channels)[:, :, :3]
    bpy.data.images.remove(image)
    return pixels.astype(np.float64)


def block_means(image, blocks=8):
    height, width = image.shape[:2]
    by, bx = height // blocks, width // blocks
    cropped = image[:by * blocks, :bx * blocks]
    return cropped.reshape(blocks, by, blocks, bx, 3).mean(axis=(1, 3, 4))


def main():
    args = parse_args()
    wanted = set(filter(None, args.scenes.split(",")))
    failed = 0
    compared = 0
    for candidate_path in sorted(args.candidate.glob("*.exr")):
        name = candidate_path.stem
        reference_path = args.reference / candidate_path.name
        if (wanted and name not in wanted) or not reference_path.exists():
            continue
        reference = load(reference_path)
        candidate = load(candidate_path)
        if reference.shape != candidate.shape:
            print("VCM_COMPARE %-18s shape mismatch FAILED" % name)
            failed += 1
            continue
        compared += 1
        mean = max(reference.mean(), 1.0e-12)
        ratio = candidate.mean() / mean
        reference_blocks = block_means(reference)
        candidate_blocks = block_means(candidate)
        # Relative to the block itself, but not less than a tenth of the image mean: dark blocks
        # hold little light and much relative noise.
        block_error = np.abs(candidate_blocks - reference_blocks) / np.maximum(
            reference_blocks, 0.1 * mean)
        rmse = np.sqrt(((candidate - reference) ** 2).mean()) / mean
        valid = bool(np.isfinite(candidate).all() and (candidate >= 0.0).all())
        ok = valid and abs(ratio - 1.0) <= args.mean and block_error.max() <= args.block
        failed += not ok
        result = {
            "label": args.label,
            "scene": name,
            "mean_ratio": float(ratio),
            "block_error_max": float(block_error.max()),
            "block_error_mean": float(block_error.mean()),
            "relative_rmse": float(rmse),
            "valid": valid,
            "ok": bool(ok),
        }
        print("VCM_COMPARE %-22s %-18s mean %.4f  block max %.4f avg %.4f  rmse %.4f  %s" % (
            args.label, name, ratio, block_error.max(), block_error.mean(), rmse,
            "ok" if ok else ("FAILED" if valid else "INVALID PIXELS FAILED")), flush=True)
        print("VCM_COMPARE_JSON " + json.dumps(result), flush=True)
    if compared == 0:
        print("VCM_COMPARE %s nothing to compare FAILED" % args.label)
        failed += 1
    sys.exit(1 if failed else 0)


main()
