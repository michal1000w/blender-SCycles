#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Render the currently loaded scene with controlled ReSTIR settings.

The input .blend is never saved. This is intended for reproducing quality or performance
regressions on real user scenes without rewriting their render configuration.
"""

import argparse
import json
import os
import sys
import time

import bpy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--percentage", type=int, default=25)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--restir-di", action="store_true")
    parser.add_argument("--restir-pt", action="store_true")
    parser.add_argument("--temporal", type=int, default=20)
    parser.add_argument("--spatial", type=int, default=1)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--metalrt", action="store_true")
    parser.add_argument(
        "--passes",
        action="store_true",
        help="Write a multilayer EXR with Cycles direct/indirect component passes",
    )
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])

    scene = bpy.context.scene
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "METAL"
    preferences.metalrt = "ON" if options.metalrt else "OFF"
    preferences.get_devices()
    metal_found = False
    for device in preferences.devices:
        device.use = device.type == "METAL"
        metal_found |= device.use
    if not metal_found:
        raise RuntimeError("No Metal Cycles device available")

    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = options.samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.seed = options.seed
    scene.cycles.use_restir = options.restir_di
    scene.cycles.use_restir_pt = options.restir_pt
    scene.cycles.restir_pt_temporal_history = options.temporal
    scene.cycles.restir_pt_spatial_neighbors = options.spatial
    scene.cycles.restir_light_candidates = options.candidates
    if (options.width is not None):
        scene.render.resolution_x = options.width
    if (options.height is not None):
        scene.render.resolution_y = options.height
    scene.render.resolution_percentage = options.percentage
    if options.passes:
        view_layer = scene.view_layers[0]
        view_layer.use_pass_diffuse_direct = True
        view_layer.use_pass_diffuse_indirect = True
        view_layer.use_pass_glossy_direct = True
        view_layer.use_pass_glossy_indirect = True
        view_layer.use_pass_transmission_direct = True
        view_layer.use_pass_transmission_indirect = True
        view_layer.use_pass_emit = True
        view_layer.use_pass_environment = True
        view_layer.cycles.use_pass_volume_direct = True
        view_layer.cycles.use_pass_volume_indirect = True
    if options.passes:
        scene.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
        scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
    else:
        scene.render.image_settings.media_type = "IMAGE"
        scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = os.path.abspath(options.output)

    started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - started
    result = {
        "elapsed": elapsed,
        "output": scene.render.filepath,
        "resolution": [
            scene.render.resolution_x * options.percentage // 100,
            scene.render.resolution_y * options.percentage // 100,
        ],
        "samples": options.samples,
        "restir_di": options.restir_di,
        "restir_pt": options.restir_pt,
        "metalrt": options.metalrt,
        "temporal": options.temporal,
        "spatial": options.spatial,
        "passes": options.passes,
    }
    print("RESTIR_SCENE_PROBE " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
