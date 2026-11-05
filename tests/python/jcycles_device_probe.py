# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test for an installed build: report which Cycles devices are usable and render a
tiny scene on each of them (CPU, then every CUDA, OptiX, HIP, oneAPI and Metal device found).

  blender --background --factory-startup --python tests/python/jcycles_device_probe.py

Exits with a non-zero status when Cycles is missing, a compiled-in backend fails to render or
a render produces non-finite or black pixels.
"""

import math
import sys

import bpy
import _cycles


def log(message):
    print(f"[jcycles-probe] {message}", flush=True)


def device_list(device_type):
    try:
        return [d for d in _cycles.available_devices(device_type) if d[1] == device_type]
    except Exception:
        return []


def render_once(device_type):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device_type == 'CPU':
        scene.cycles.device = 'CPU'
    else:
        prefs.compute_device_type = device_type
        prefs.refresh_devices()
        for d in prefs.devices:
            d.use = (d.type == device_type)
        scene.cycles.device = 'GPU'
    scene.cycles.samples = 4
    scene.cycles.use_denoising = False
    scene.render.resolution_x = 64
    scene.render.resolution_y = 64
    scene.render.resolution_percentage = 100
    bpy.ops.render.render(write_still=False)
    image = bpy.data.images["Render Result"]
    # Render Result pixels are only readable after saving; use a viewer-free check instead.
    path = bpy.app.tempdir + f"jcycles_probe_{device_type}.exr"
    image.save_render(path)
    loaded = bpy.data.images.load(path)
    pixels = list(loaded.pixels)
    bpy.data.images.remove(loaded)
    finite = all(math.isfinite(v) for v in pixels)
    mean = sum(pixels[i] for i in range(len(pixels)) if i % 4 != 3) / (len(pixels) * 0.75)
    return finite, mean


def main():
    log(f"Blender {bpy.app.version_string}, build {bpy.app.build_hash.decode()}")
    if "cycles" not in bpy.context.preferences.addons:
        log("Cycles add-on is not enabled")
        return 1

    bpy.ops.wm.read_factory_settings(use_empty=False)
    # The default cube scene with its light and camera.
    failures = 0
    backends = ['CPU']
    for device_type in ('CUDA', 'OPTIX', 'HIP', 'ONEAPI', 'METAL'):
        devices = device_list(device_type)
        names = ", ".join(d[0] for d in devices) if devices else "none"
        log(f"{device_type:7s}: {names}")
        if devices:
            backends.append(device_type)

    for device_type in backends:
        try:
            finite, mean = render_once(device_type)
        except Exception as ex:
            log(f"{device_type}: render FAILED: {ex}")
            failures += 1
            continue
        status = "ok" if finite and mean > 1e-4 else "BAD"
        if status != "ok":
            failures += 1
        log(f"{device_type}: render {status} (mean {mean:.4f}, finite {finite})")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
