#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare the individual light-transport passes in multilayer OpenEXR renders."""

import argparse
import json
import os
import sys

import numpy as np
import OpenImageIO as oiio


def load_passes(path):
    image = oiio.ImageInput.open(os.path.abspath(path))
    if image is None:
        raise RuntimeError(oiio.geterror())
    spec = image.spec()
    pixels = image.read_image(oiio.FLOAT)
    image.close()
    if pixels is None:
        raise RuntimeError(oiio.geterror())

    groups = {}
    for channel_index, channel_name in enumerate(spec.channelnames):
        pass_name, component = channel_name.rsplit(".", 1)
        if component not in {"R", "G", "B"}:
            continue
        groups.setdefault(pass_name, []).append((component, channel_index))

    result = {}
    component_order = {"R": 0, "G": 1, "B": 2}
    for pass_name, channels in groups.items():
        channels.sort(key=lambda item: component_order[item[0]])
        if len(channels) == 3:
            result[pass_name] = pixels[..., [index for _, index in channels]].astype(
                np.float64
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument(
        "--baseline",
        help="Optional matched-seed multilayer render used to measure changed-pixel coverage",
    )
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    reference = load_passes(options.reference)
    baseline = load_passes(options.baseline) if options.baseline else {}
    results = {}
    for path in options.image:
        passes = load_passes(path)
        image_result = {}
        for pass_name in sorted(reference.keys() & passes.keys()):
            target = reference[pass_name]
            values = passes[pass_name]
            if values.shape != target.shape:
                raise RuntimeError(f"Resolution mismatch for {pass_name}: {path}")
            difference = values - target
            pixel_magnitude = np.max(np.abs(values), axis=2)
            metrics = {
                "rmse": float(np.sqrt(np.mean(np.square(difference)))),
                "mae": float(np.mean(np.abs(difference))),
                "mean": float(np.mean(values)),
                "reference_mean": float(np.mean(target)),
                "active_pixel_fraction": float(np.mean(pixel_magnitude > 1.0e-6)),
                "max_abs_error": float(np.max(np.abs(difference))),
            }
            if pass_name in baseline:
                matched = np.max(np.abs(values - baseline[pass_name]), axis=2)
                metrics["matched_seed_mean_pixel_diff"] = float(np.mean(matched))
                for threshold in (1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2):
                    metrics[f"matched_seed_changed_fraction_{threshold:g}"] = float(
                        np.mean(matched > threshold)
                    )
            image_result[pass_name] = metrics
        results[os.path.splitext(os.path.basename(path))[0]] = image_result

    print("RESTIR_PASS_COMPARE " + json.dumps(results, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
