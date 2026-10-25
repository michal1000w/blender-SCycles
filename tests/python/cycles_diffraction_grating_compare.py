# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare renders of cycles_diffraction_grating_scenes.py that differ in noise: two builds,
devices or integrators. Run inside Blender:

  blender -b --factory-startup --python tests/python/cycles_diffraction_grating_compare.py -- \
      REFERENCE_DIR CANDIDATE_DIR [--scenes a,b] [--block 16] [--mean-tolerance 0.03] \
      [--block-tolerance 0.08] [--report report.json]

For every EXR present in both directories it reports the ratio of the channel means (energy) and
the block error: the mean absolute difference of block averaged images relative to the mean of
the reference, which averages most of the noise away while keeping errors of a few pixels' size.
Exits with 1 when a scene exceeds a tolerance.
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


def block_mean(pixels, block):
    h, w = pixels.shape[0] // block * block, pixels.shape[1] // block * block
    p = pixels[:h, :w]
    return p.reshape(h // block, block, w // block, block, 3).mean(axis=(1, 3))


def compare(reference, candidate, block):
    ref_mean = reference.reshape(-1, 3).mean(axis=0)
    cand_mean = candidate.reshape(-1, 3).mean(axis=0)
    ratio = cand_mean / np.maximum(ref_mean, 1e-12)
    rb, cb = block_mean(reference, block), block_mean(candidate, block)
    scale = max(float(reference.mean()), 1e-12)
    block_diff = np.abs(cb - rb).mean(axis=2) / scale
    # Luminance of the brightest structure, to tell a missing highlight from noise.
    return {
        "mean_reference": ref_mean.tolist(),
        "mean_candidate": cand_mean.tolist(),
        "mean_ratio": ratio.tolist(),
        "mean_error": float(np.abs(ratio - 1.0).max()),
        "block_error": float(block_diff.mean()),
        "block_error_p99": float(np.percentile(block_diff, 99)),
        "finite": bool(np.isfinite(candidate).all()),
    }


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--scenes", default="")
    parser.add_argument("--block", type=int, default=16)
    parser.add_argument("--mean-tolerance", type=float, default=0.03)
    parser.add_argument("--block-tolerance", type=float, default=0.08)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    names = sorted(p.stem for p in args.reference.glob("*.exr"))
    if args.scenes:
        wanted = set(args.scenes.split(","))
        names = [n for n in names if n in wanted]
    results = {}
    failed = []
    for name in names:
        candidate_path = args.candidate / f"{name}.exr"
        if not candidate_path.exists():
            results[name] = {"missing": True}
            failed.append(name)
            print(f"{name:24s} MISSING")
            continue
        r = compare(load(args.reference / f"{name}.exr"), load(candidate_path), args.block)
        ok = (r["finite"] and r["mean_error"] <= args.mean_tolerance and
              r["block_error"] <= args.block_tolerance)
        r["pass"] = ok
        results[name] = r
        if not ok:
            failed.append(name)
        ratio = " ".join(f"{v:.4f}" for v in r["mean_ratio"])
        print(f"{name:24s} {'ok  ' if ok else 'FAIL'} mean ratio RGB {ratio}  block error "
              f"{r['block_error']:.4f} (p99 {r['block_error_p99']:.4f})", flush=True)
    if args.report:
        args.report.write_text(json.dumps(results, indent=2))
    print("FAILED: " + ",".join(failed) if failed else "ALL PASS", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
