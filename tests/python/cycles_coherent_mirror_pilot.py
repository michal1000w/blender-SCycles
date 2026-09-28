# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Small Metal smoke for a prepared coherent mirror scene; no acceptance gates."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import bpy


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--resolution", type=int, default=64)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
args.output.mkdir(parents=True, exist_ok=False)

scene = bpy.context.scene
scene.render.resolution_x = args.resolution
scene.render.resolution_y = args.resolution
scene.cycles.samples = args.samples
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "METAL"
preferences.get_devices()
for device in preferences.devices:
    device.use = device.type == "METAL"
if not any(device.use for device in preferences.devices):
    raise RuntimeError("No Metal device for coherent mirror pilot")
scene.cycles.device = "GPU"
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = str(args.output / "phase_0_pilot.exr")
start = time.monotonic()
bpy.ops.render.render(write_still=True)
seconds = time.monotonic() - start
image = bpy.data.images.load(scene.render.filepath, check_existing=False)
pixels = tuple(image.pixels[:])
finite = bool(pixels) and all(math.isfinite(value) for value in pixels)
channels = image.channels
mean_rgb = [sum(pixels[channel::channels]) / (len(pixels) // channels)
            for channel in range(3)] if finite else None
bpy.data.images.remove(image)
scene.render.image_settings.file_format = "PNG"
bpy.data.images["Render Result"].save_render(str(args.output / "phase_0_pilot.png"), scene=scene)
report = {
    "scope": "small integration smoke, not the mirror acceptance reference",
    "samples": args.samples,
    "resolution": [args.resolution, args.resolution],
    "render_seconds": seconds,
    "all_pixels_finite": finite,
    "mean_rgb": mean_rgb,
    "specular_connections": scene.cycles.use_coherent_specular_connections,
    "max_interface_events": scene.cycles.coherent_max_interface_events,
}
(args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
if not finite:
    raise RuntimeError("Coherent mirror pilot produced nonfinite or missing pixels")
