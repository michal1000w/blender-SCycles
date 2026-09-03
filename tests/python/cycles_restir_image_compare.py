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
    parser.add_argument(
        "--baseline",
        help="Optional matched-seed image used to report how much of the frame changed",
    )
    parser.add_argument("--output-dir", required=True)
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    os.makedirs(options.output_dir, exist_ok=True)

    reference_image, reference = load_rgb(options.reference)
    baseline_image = None
    baseline = None
    if options.baseline:
        baseline_image, baseline = load_rgb(options.baseline)
        if len(baseline) != len(reference):
            raise RuntimeError(f"Resolution mismatch: {options.baseline}")
    results = {}
    for path in options.image:
        image, values = load_rgb(path)
        if len(values) != len(reference):
            raise RuntimeError(f"Resolution mismatch: {path}")
        square_error = sum((value - target) ** 2 for value, target in zip(values, reference))
        absolute_error = sum(abs(value - target) for value, target in zip(values, reference))
        max_index = max(range(len(values)), key=values.__getitem__)
        max_error_index = max(
            range(len(values)), key=lambda index: abs(values[index] - reference[index])
        )
        sorted_values = sorted(values)
        width = image.size[0]
        pixel_index, channel = divmod(max_index, 3)
        name = os.path.splitext(os.path.basename(path))[0]
        results[name] = {
            "rmse": math.sqrt(square_error / len(reference)),
            "mae": absolute_error / len(reference),
            "mean": sum(values) / len(values),
            "reference_mean": sum(reference) / len(reference),
            "max": max(values),
            "p999": sorted_values[min(int(0.999 * len(sorted_values)), len(sorted_values) - 1)],
            "max_location": [pixel_index % width, pixel_index // width, channel],
            "reference_at_max": reference[max_index],
            "max_abs_error": abs(values[max_error_index] - reference[max_error_index]),
            "max_error_value": values[max_error_index],
            "reference_at_max_error": reference[max_error_index],
            "max_error_location": [
                (max_error_index // 3) % width,
                (max_error_index // 3) // width,
                max_error_index % 3,
            ],
        }
        if baseline is not None:
            pixel_differences = [
                max(abs(values[index + channel] - baseline[index + channel]) for channel in range(3))
                for index in range(0, len(values), 3)
            ]
            results[name]["changed_pixel_fraction"] = {
                "1e-6": sum(value > 1.0e-6 for value in pixel_differences)
                / len(pixel_differences),
                "1e-4": sum(value > 1.0e-4 for value in pixel_differences)
                / len(pixel_differences),
                "1e-3": sum(value > 1.0e-3 for value in pixel_differences)
                / len(pixel_differences),
                "1e-2": sum(value > 1.0e-2 for value in pixel_differences)
                / len(pixel_differences),
            }
            results[name]["mean_matched_seed_pixel_difference"] = sum(pixel_differences) / len(
                pixel_differences
            )
        image.save_render(os.path.join(options.output_dir, name + ".png"), scene=bpy.context.scene)
        bpy.data.images.remove(image)

    reference_image.save_render(
        os.path.join(options.output_dir, "reference.png"), scene=bpy.context.scene
    )
    if baseline_image is not None:
        bpy.data.images.remove(baseline_image)
    print("RESTIR_IMAGE_COMPARE " + json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
