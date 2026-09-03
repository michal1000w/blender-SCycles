#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Interactive Cycles Metal rendered-viewport smoke with a same-session scene update."""

import argparse
import json
import os
import pathlib
import sys
import time
from types import SimpleNamespace

import bpy

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cycles_restir_metal_benchmark import build_scene, set_animation_frame  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metalrt", action="store_true")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--lights", type=int, default=8)
    parser.add_argument("--preview-samples", type=int, default=16)
    parser.add_argument("--settle-seconds", type=float, default=12.0)
    parser.add_argument("--output-dir", default="/tmp/cycles-restir-viewport")
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    os.makedirs(options.output_dir, exist_ok=True)
    # The startup splash is scheduled after --python begins. Disable it for this unsaved test
    # session so screenshots contain the rendered viewport rather than a popup.
    bpy.context.preferences.view.show_splash = False

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    scene = build_scene(
        SimpleNamespace(
            metalrt=options.metalrt,
            resolution=options.resolution,
            lights=options.lights,
            skip_factory_reset=True,
        )
    )
    scene.cycles.preview_samples = options.preview_samples
    scene.cycles.use_preview_denoising = False
    scene.cycles.use_restir = False
    scene.cycles.use_restir_pt = True
    scene.cycles.restir_pt_temporal_history = 20
    scene.cycles.restir_pt_spatial_neighbors = 1
    scene.cycles.restir_pt_spatial_radius = 8
    scene.cycles.restir_pt_footprint_threshold = 0.02
    scene.cycles.restir_pt_min_roughness = 0.2
    scene.cycles.restir_pt_decorrelate = True

    area = next((candidate for candidate in bpy.context.screen.areas if candidate.type == "VIEW_3D"), None)
    if area is None:
        raise RuntimeError("No VIEW_3D area is available for the viewport smoke")
    space = area.spaces.active
    region = next(candidate for candidate in area.regions if candidate.type == "WINDOW")
    with bpy.context.temp_override(area=area, region=region, space_data=space):
        bpy.ops.view3d.view_camera()
    space.shading.type = "RENDERED"
    space.shading.use_scene_world = True
    space.shading.use_scene_lights = True

    started = time.perf_counter()
    paths = [
        os.path.join(options.output_dir, "viewport-initial.png"),
        os.path.join(options.output_dir, "viewport-motion.png"),
    ]
    state = {"stage": 0, "first_seconds": 0.0}

    def capture_viewport(path):
        with bpy.context.temp_override(area=area, region=region, space_data=space):
            bpy.ops.screen.screenshot_area(filepath=path)

    def advance():
        if state["stage"] == 0:
            capture_viewport(paths[0])
            state["first_seconds"] = time.perf_counter() - started
            set_animation_frame(scene, 2)
            state["stage"] = 1
            return options.settle_seconds

        capture_viewport(paths[1])
        result = {
            "backend": "METALRT" if options.metalrt else "METAL",
            "initial_seconds": state["first_seconds"],
            "motion_seconds": time.perf_counter() - started - state["first_seconds"],
            "preview_samples": options.preview_samples,
            "initial_bytes": os.path.getsize(paths[0]),
            "motion_bytes": os.path.getsize(paths[1]),
            "screenshots": paths,
        }
        print("RESTIR_VIEWPORT " + json.dumps(result, sort_keys=True), flush=True)
        bpy.ops.wm.quit_blender()
        return None

    bpy.app.timers.register(advance, first_interval=options.settle_seconds)


if __name__ == "__main__":
    main()
