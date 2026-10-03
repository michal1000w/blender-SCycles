# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Put renders of the same scene side by side. Run inside Blender:

  blender -b --factory-startup --python tools/vcm/contact_sheet.py -- \
      --output sheet.png [--exposure 0.0] image_a.exr image_b.exr ...

EXR inputs are tone mapped with the given exposure and a sRGB curve, with one scale for all.
"""

import argparse
import sys

import bpy
import numpy as np


def load(path):
    image = bpy.data.images.load(path, check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(image.size[1], image.size[0], image.channels)[:, :, :3].copy()
    linear = image.file_format == "OPEN_EXR"
    bpy.data.images.remove(image)
    return pixels, linear


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--exposure", type=float, default=0.0)
    parser.add_argument("images", nargs="+")
    args = parser.parse_args(argv)

    tiles = []
    for path in args.images:
        pixels, linear = load(path)
        if linear:
            pixels = np.clip(pixels * 2.0 ** args.exposure, 0.0, 1.0)
            pixels = np.where(pixels <= 0.0031308, 12.92 * pixels,
                              1.055 * np.power(pixels, 1 / 2.4) - 0.055)
        tiles.append(pixels)
        tiles.append(np.ones((pixels.shape[0], 4, 3), dtype=np.float32))
    sheet = np.concatenate(tiles[:-1], axis=1)
    rgba = np.concatenate([sheet, np.ones((*sheet.shape[:2], 1), dtype=np.float32)], axis=2)
    result = bpy.data.images.new("Sheet", sheet.shape[1], sheet.shape[0], alpha=False)
    # Already display referred: keep the values as they are.
    result.colorspace_settings.name = "Non-Color"
    result.pixels.foreach_set(rgba.ravel().astype(np.float32))
    result.filepath_raw = args.output
    result.file_format = "PNG"
    result.save()


main()
