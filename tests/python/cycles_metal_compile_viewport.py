#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Check first visible Metal viewport output with the currently loaded scene.

Run Blender in GUI mode through cycles_metal_compile_benchmark.py with
--require-first-sample, --debug-cycles --log 'cycles.*' --log-level -1.
The wrapper's XDG_CACHE_HOME identifies the log/output directory. This script
preserves scene sampling and transport settings, captures the first displayed
image after a completed sample batch, then quits without saving the blend file.
"""

import os
from pathlib import Path
import re
import time

import bpy


started = time.monotonic()
output = Path(os.environ["XDG_CACHE_HOME"]).parent
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "METAL"
preferences.get_devices()
for device in preferences.devices:
    device.use = device.type == "METAL"
for area in bpy.context.screen.areas:
    if area.type == "VIEW_3D":
        area.spaces.active.shading.type = "RENDERED"
        print("VIEWPORT_RENDER_START", time.monotonic() - started,
              area.width, area.height, flush=True)
first_batch = None
log_offset = 0
pending_log = ""


def check():
    global first_batch, log_offset, pending_log
    with (output / "run.log").open(errors="replace") as log:
        log.seek(log_offset)
        pending_log += log.read()
        log_offset = log.tell()
    lines = pending_log.split("\n")
    pending_log = lines.pop()
    if first_batch is None and any(
            re.search(r"Rendered [1-9][0-9]* samples in .*occupancy:", line) for line in lines):
        first_batch = time.monotonic()
        print("VIEWPORT_FIRST_BATCH_OBSERVED", first_batch - started, flush=True)
    if first_batch is not None and time.monotonic() - first_batch > 3:
        bpy.ops.screen.screenshot(filepath=str(output / "first-image.png"))
        print("VIEWPORT_IMAGE_CAPTURED", time.monotonic() - started, flush=True)
        bpy.ops.wm.quit_blender()
        return None
    if time.monotonic() - started > 160:
        print("VIEWPORT_NO_SAMPLE_TIMEOUT", flush=True)
        bpy.ops.wm.quit_blender()
        return None
    return 0.25


bpy.app.timers.register(check, first_interval=0.25)
