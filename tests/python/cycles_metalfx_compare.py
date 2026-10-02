# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Compare renders of cycles_metalfx_scenes.py against reference renders.

  blender -b --factory-startup --python tests/python/cycles_metalfx_compare.py -- \\
      --reference DIR[:TAG] --test DIR [--tags a,b] [--png DIR] [--json FILE]

For every <scene>_<tag>.exr in the test directory the image is compared with
<scene>_<reference tag>.exr of the reference directory (tag "reference" by default). Reports

  PSNR    of the display image (sRGB of the clamped linear values), higher is better
  relMSE  mean of (a - b)^2 / (b^2 + 0.01) of the linear values, lower is better
  mean    ratio of the mean linear values, 1.0 when no energy is gained or lost
  edge    ratio of the mean gradient magnitude of the display image, a measure of sharpness:
          below 1.0 the image is blurrier than the reference, above it is noisier

Images of another resolution than the reference are compared after the reference is
resampled to their size with a box filter.

--png writes display images of the test renders, for looking at them.
"""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np


def load(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(height, width, 4)


def display(linear):
    rgb = np.clip(linear[..., :3], 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055)


def box_resize(image, width, height):
    if image.shape[1] == width and image.shape[0] == height:
        return image
    src_h, src_w = image.shape[:2]
    # Area resampling through cumulative sums, exact for any ratio.
    def axis(data, src, dst, ax):
        edges = np.linspace(0.0, src, dst + 1)
        cumulative = np.concatenate(
            [np.zeros_like(np.take(data, [0], axis=ax)), np.cumsum(data, axis=ax, dtype=np.float64)],
            axis=ax)
        lo = np.floor(edges).astype(int)
        frac = edges - lo
        lo = np.clip(lo, 0, src)
        hi = np.clip(lo + 1, 0, src)
        value = np.take(cumulative, lo, axis=ax) * (1 - frac).reshape(
            [-1 if i == ax else 1 for i in range(data.ndim)]) + np.take(
            cumulative, hi, axis=ax) * frac.reshape([-1 if i == ax else 1 for i in range(data.ndim)])
        return np.diff(value, axis=ax) / (src / dst)
    out = axis(image.astype(np.float64), src_h, height, 0)
    out = axis(out, src_w, width, 1)
    return out.astype(np.float32)


def gradient(image):
    gx = np.abs(np.diff(image, axis=1)).mean()
    gy = np.abs(np.diff(image, axis=0)).mean()
    return gx + gy


def metrics(test, reference):
    reference = box_resize(reference, test.shape[1], test.shape[0])
    a = test[..., :3].astype(np.float64)
    b = reference[..., :3].astype(np.float64)
    da = display(a)
    db = display(b)
    mse = np.mean((da - db) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 0 else float("inf")
    relmse = float(np.mean((a - b) ** 2 / (b ** 2 + 0.01)))
    return {
        "psnr": float(psnr),
        "relmse": relmse,
        "mean": float(a.mean() / max(b.mean(), 1e-12)),
        "edge": float(gradient(da) / max(gradient(db), 1e-12)),
        "finite": bool(np.isfinite(test).all()),
        "size": [int(test.shape[1]), int(test.shape[0])],
    }


def over_white(image):
    result = image.copy()
    result[..., :3] += (1.0 - np.clip(image[..., 3:4], 0.0, 1.0))
    return result


def save_png(linear, path):
    height, width = linear.shape[:2]
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[..., :3] = display(linear)
    image = bpy.data.images.new("out", width, height, alpha=False)
    image.pixels.foreach_set(rgba.ravel())
    image.file_format = 'PNG'
    image.filepath_raw = str(path)
    scene = bpy.context.scene
    scene.view_settings.view_transform = 'Standard'
    image.save()
    bpy.data.images.remove(image)


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--tags", default=None)
    parser.add_argument("--png", default=None)
    parser.add_argument("--json", default=None)
    parser.add_argument("--over-white", action="store_true",
                        help="Composite over white before comparing, for transparent films")
    args = parser.parse_args(argv)

    ref_dir, _, ref_tag = args.reference.partition(":")
    ref_dir = Path(ref_dir)
    ref_tag = ref_tag or "reference"
    test_dir = Path(args.test)
    tags = args.tags.split(",") if args.tags else None

    references = {}
    for path in sorted(ref_dir.glob("*_%s.exr" % ref_tag)):
        references[path.name[:-len("_%s.exr" % ref_tag)]] = load(path)

    results = {}
    print("%-12s %-34s %8s %9s %7s %6s" % ("scene", "tag", "PSNR", "relMSE", "mean", "edge"))
    for path in sorted(test_dir.glob("*.exr")):
        stem = path.stem
        scene = next((name for name in references if stem.startswith(name + "_")), None)
        if scene is None:
            continue
        tag = stem[len(scene) + 1:]
        if tags is not None and tag not in tags:
            continue
        test = load(path)
        reference = references[scene]
        if args.over_white:
            test = over_white(test)
            reference = over_white(reference)
        m = metrics(test, reference)
        results.setdefault(scene, {})[tag] = m
        print("%-12s %-34s %8.2f %9.5f %7.4f %6.3f%s" % (
            scene, tag, m["psnr"], m["relmse"], m["mean"], m["edge"],
            "" if m["finite"] else "  NON-FINITE VALUES"))
        if args.png:
            Path(args.png).mkdir(parents=True, exist_ok=True)
            save_png(test, Path(args.png) / (stem + ".png"))

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))


main()
