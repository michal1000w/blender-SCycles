#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare scene-linear EXR renders and emit view-transformed PNG previews."""

import argparse
import json
import math
import os
import sys

import bpy


def load_rgb(path):
    image = bpy.data.images.load(os.path.abspath(path), check_existing=False)
    pixels = list(image.pixels[:])
    rgb = [pixels[index] for index in range(len(pixels)) if index % 4 != 3]
    return image, rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    os.makedirs(options.output_dir, exist_ok=True)

    reference_image, reference = load_rgb(options.reference)
    results = {}
    for path in options.image:
        image, values = load_rgb(path)
        if len(values) != len(reference):
            raise RuntimeError(f"Resolution mismatch: {path}")
        square_error = sum((value - target) ** 2 for value, target in zip(values, reference))
        absolute_error = sum(abs(value - target) for value, target in zip(values, reference))
        name = os.path.splitext(os.path.basename(path))[0]
        results[name] = {
            "rmse": math.sqrt(square_error / len(reference)),
            "mae": absolute_error / len(reference),
            "mean": sum(values) / len(values),
            "reference_mean": sum(reference) / len(reference),
            "max": max(values),
        }
        image.save_render(os.path.join(options.output_dir, name + ".png"), scene=bpy.context.scene)
        bpy.data.images.remove(image)

    reference_image.save_render(
        os.path.join(options.output_dir, "reference.png"), scene=bpy.context.scene
    )
    print("RESTIR_IMAGE_COMPARE " + json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
