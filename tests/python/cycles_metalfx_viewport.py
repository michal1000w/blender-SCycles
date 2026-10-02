# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Drive a rendered Cycles viewport through denoiser configurations and capture it.

First build the scene file, then open it in a window (not -b) and run the steps:

  blender -b --factory-startup --python tests/python/cycles_metalfx_viewport.py -- \\
      --build DIRECTORY/materials.blend --scene materials
  blender --factory-startup --window-geometry 0 0 1280 800 DIRECTORY/materials.blend \\
      --python tests/python/cycles_metalfx_viewport.py -- \\
      --output DIRECTORY --steps STEP[,STEP...]

A step is name:denoiser:upscale:seconds[:action], for example

  reference:NONE:NONE:40          plain path tracing for 40 seconds, the reference
  oidn:OPENIMAGEDENOISE:NONE:5
  mfx:METALFX:NONE:5
  mfx_perf:METALFX:PERFORMANCE:5
  mfx_orbit:METALFX:NONE:4:orbit  orbit the camera for 4 seconds, capture while moving
                                  (<name>_moving.png) and 2 seconds after it stopped
  mfx_object:METALFX:NONE:4:object  the same with a moving object
  mfx_resize:METALFX:NONE:4:resize  toggle the sidebar to change the viewport size
  mfx_edit:METALFX:NONE:4:material  change a material while rendering

Every step writes <name>.png with the pixels of the viewport. The sequence runs in one session,
which also exercises switching between the denoisers. Blender quits when the steps are done.
"""

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Matrix


def load_scenes():
    path = Path(__file__).with_name("cycles_metalfx_scenes.py")
    spec = importlib.util.spec_from_file_location("cycles_metalfx_scenes", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Runner:
    def __init__(self, args):
        self.args = args
        self.out_dir = Path(args.output)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.steps = []
        for text in args.steps.split(","):
            parts = text.split(":")
            self.steps.append({
                "name": parts[0],
                "denoiser": parts[1],
                "upscale": parts[2],
                "seconds": float(parts[3]),
                "action": parts[4] if len(parts) > 4 else "",
            })
        self.index = -1
        self.phase = "setup"
        self.phase_start = 0.0
        self.base_view = None
        self.moving_captured = False

    # Context helpers.

    def view3d(self):
        window = bpy.context.window_manager.windows[0]
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                region = next(r for r in area.regions if r.type == 'WINDOW')
                return window, area, region
        raise RuntimeError("no 3D viewport")

    def capture(self, name):
        window, area, region = self.view3d()
        path = str(self.out_dir / (name + ".png"))
        with bpy.context.temp_override(window=window, area=area, region=region):
            bpy.ops.screen.screenshot_area(filepath=path)
        # Crop to the region that shows the render.
        image = bpy.data.images.load(path, check_existing=False)
        width, height = image.size
        info = "%s: area %dx%d region %dx%d at %d,%d" % (
            name, width, height, region.width, region.height,
            region.x - area.x, region.y - area.y)
        bpy.data.images.remove(image)
        print("METALFX_VIEWPORT capture", info)
        sys.stdout.flush()

    def setup_view(self):
        window, area, region = self.view3d()
        scene = bpy.context.scene
        space = area.spaces.active
        space.overlay.show_overlays = False
        space.show_gizmo = False
        space.show_region_header = False
        space.show_region_tool_header = False
        space.show_region_toolbar = False
        space.show_region_ui = False
        space.lens = 70.0
        space.clip_start = 0.1
        space.clip_end = 1000.0
        rv3d = space.region_3d
        rv3d.view_perspective = 'PERSP'
        bpy.context.view_layer.update()
        self.base_view = scene.camera.matrix_world.copy()
        rv3d.view_matrix = self.base_view.inverted()
        scene.view_settings.view_transform = 'Standard'
        scene.render.preview_pixel_size = self.args.pixel_size
        space.shading.type = 'RENDERED'

    def apply_step(self, step):
        cscene = bpy.context.scene.cycles
        cscene.use_preview_denoising = step["denoiser"] != 'NONE'
        if cscene.use_preview_denoising:
            cscene.preview_denoiser = step["denoiser"]
            cscene.preview_denoising_use_gpu = True
            cscene.preview_denoising_start_sample = 1
        cscene.preview_denoising_upscale_quality = step["upscale"]
        cscene.preview_samples = 0 if step["denoiser"] == 'NONE' else self.args.samples
        cscene.use_preview_adaptive_sampling = self.args.adaptive
        # Restart the render so that every step starts from nothing.
        window, area, region = self.view3d()
        rv3d = area.spaces.active.region_3d
        rv3d.view_matrix = self.base_view.inverted()
        bpy.context.scene.update_tag()

    def animate(self, step, t):
        """t runs from 0 to 1 over the duration of the motion."""
        window, area, region = self.view3d()
        space = area.spaces.active
        action = step["action"]
        if action == "orbit":
            angle = math.radians(self.args.orbit_degrees) * t
            rv3d = space.region_3d
            rv3d.view_matrix = (Matrix.Rotation(angle, 4, 'Z') @ self.base_view).inverted()
        elif action == "object":
            obj = bpy.data.objects.get("Suzanne") or bpy.data.objects[0]
            if not hasattr(self, "object_start"):
                self.object_start = obj.location.copy()
            obj.location.x = self.object_start.x + 2.0 * math.sin(t * math.pi)
        elif action == "resize":
            space.show_region_ui = int(t * 6) % 2 == 1
        elif action == "material":
            mat = bpy.data.materials.get("diffuse") or bpy.data.materials[0]
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Base Color"].default_value = (0.1 + 0.8 * t, 0.25, 0.9 - 0.8 * t, 1.0)

    def tick(self):
        now = time.time()
        try:
            if self.phase == "setup":
                self.setup_view()
                self.phase = "next"
                return 1.0

            if self.phase == "next":
                self.index += 1
                if self.index >= len(self.steps):
                    print("METALFX_VIEWPORT done")
                    sys.stdout.flush()
                    bpy.ops.wm.quit_blender()
                    return None
                step = self.steps[self.index]
                self.apply_step(step)
                self.phase = "run"
                self.phase_start = now
                self.moving_captured = False
                print("METALFX_VIEWPORT step", step["name"])
                sys.stdout.flush()
                return 0.05

            step = self.steps[self.index]
            elapsed = now - self.phase_start

            if self.phase == "run":
                if step["action"]:
                    # Let the render settle first, then move.
                    settle = 2.0
                    if elapsed < settle:
                        return 0.05
                    t = min((elapsed - settle) / step["seconds"], 1.0)
                    self.animate(step, t)
                    if t >= 0.6 and not self.moving_captured:
                        self.capture(step["name"] + "_moving")
                        self.moving_captured = True
                    if t >= 1.0:
                        self.phase = "after"
                        self.phase_start = now
                    return 1.0 / 30.0
                if elapsed >= step["seconds"]:
                    self.capture(step["name"])
                    self.phase = "next"
                return 0.05

            if self.phase == "after":
                if elapsed >= 2.0:
                    self.capture(step["name"])
                    # Put the scene back for the next step.
                    if step["action"] in {"object", "material", "resize"}:
                        self.animate(step, 0.0)
                    self.phase = "next"
                return 0.05
        except Exception:
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            bpy.ops.wm.quit_blender()
            return None
        return 0.05


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--build")
    parser.add_argument("--scene", default="materials")
    parser.add_argument("--steps")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--pixel-size", default="1")
    parser.add_argument("--adaptive", action="store_true",
                        help="Use adaptive sampling in the viewport")
    parser.add_argument("--orbit-degrees", type=float, default=25.0)
    args = parser.parse_args(argv)

    scenes = load_scenes()
    if args.build:
        scenes.SCENES[args.scene]()
        scenes.setup_device('METAL')
        Path(args.build).parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args.build).resolve()))
        return

    scenes.setup_device('METAL')

    runner = Runner(args)
    bpy.app.timers.register(runner.tick, first_interval=2.0)


if __name__ == "__main__":
    main()
