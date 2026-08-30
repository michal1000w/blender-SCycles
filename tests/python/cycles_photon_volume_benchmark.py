#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Render a reproducible raw Metal photon-map benchmark from the loaded blend file."""

import argparse
import math
import sys
import time

import bpy


def configure_metal():
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "METAL"
    preferences.get_devices()
    found = False
    for device in preferences.devices:
        if device.type == "METAL":
            device.use = True
            found = True
    if not found:
        raise RuntimeError("No Metal Cycles device available")


def main():
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--photons", type=int, default=4194304)
    parser.add_argument("--gather", type=int, default=256)
    parser.add_argument("--camera-samples", type=int, default=2)
    parser.add_argument("--update", type=int, default=8)
    parser.add_argument("--radius", type=float, default=0.04)
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument("--volume-radius-scale", type=float, default=2.26)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--device", choices=("CPU", "METAL"), default="METAL")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--photon-off", action="store_true")
    options = parser.parse_args(arguments)

    if options.device == "METAL":
        configure_metal()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU" if options.device == "METAL" else "CPU"
    scene.cycles.samples = options.samples
    scene.cycles.seed = options.seed
    scene.cycles.use_adaptive_sampling = options.adaptive
    scene.cycles.use_denoising = False
    scene.cycles.use_photon_mapping = not options.photon_off
    scene.cycles.photon_count = options.photons
    scene.cycles.photon_gather_max = options.gather
    scene.cycles.photon_camera_samples = options.camera_samples
    scene.cycles.photon_map_update_samples = options.update
    scene.cycles.photon_radius = options.radius
    scene.cycles.photon_radius_decay = options.decay
    scene.cycles.photon_volume_radius_scale = options.volume_radius_scale
    scene.render.resolution_x = options.resolution
    scene.render.resolution_y = options.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.filepath = options.output

    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - start

    image = bpy.data.images.load(options.output, check_existing=False)
    pixels = list(image.pixels)
    rgb = [pixels[index] for index in range(len(pixels)) if index % 4 != 3]
    if not rgb or not all(math.isfinite(value) for value in rgb):
        raise RuntimeError("Render contains non-finite values")
    print(
        "PHOTON_VOLUME_BENCHMARK "
        f"seconds={elapsed:.9g} device={options.device} samples={options.samples} "
        f"adaptive={int(options.adaptive)} photon={int(not options.photon_off)} "
        f"photons={options.photons} "
        f"gather={options.gather} camera_samples={options.camera_samples} "
        f"update={options.update} seed={options.seed} "
        f"mean={sum(rgb) / len(rgb):.9g} peak={max(rgb):.9g}"
    )


if __name__ == "__main__":
    main()
