# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""GPU regression for direct displacement updates in JAR-test-Cycles.blend.

Run unchanged baseline and candidate sequentially with --factory-startup and the
scene loaded. Pass the output directory after --. No .blend or input image is saved.
"""

import json
from pathlib import Path
import sys

import bpy


def main():
    output = Path(sys.argv[sys.argv.index("--") + 1])
    output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.metalrt = "ON"
    prefs.get_devices()
    if not any(device.type == "METAL" for device in prefs.devices):
        raise RuntimeError("This regression requires a Metal GPU")
    for device in prefs.devices:
        device.use = device.type == "METAL"
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.use_pixel_displacement = True
    scene.cycles.use_pixel_displacement_resolution_clamp = False
    scene.cycles.samples = 4
    scene.cycles.seed = 7
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.resolution_percentage = 5
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    completed = []

    def render(label):
        # Python property changes need an explicit scene tag before obtaining the
        # evaluated scene again with persistent data enabled in background mode.
        scene.update_tag()
        bpy.context.view_layer.update()
        bpy.ops.render.render()
        bpy.data.images["Render Result"].save_render(str(output / (label + ".exr")), scene=scene)
        completed.append(label)
        print("DIRECT_REGRESSION", label, flush=True)

    render("initial")
    # Exercise arbitrary native dimensions while the rigid-image fast path is still
    # eligible, then verify that restoring the source image invalidates temporary inputs.
    nodes = bpy.data.materials["Material.002"].node_tree
    texture = nodes.nodes["Image Texture.003"]
    original_image = texture.image
    odd_image = original_image.copy()
    odd_image.scale(2053, 2063)
    odd_image.update()
    odd_image.pack()
    texture.image = odd_image
    nodes.update_tag()
    render("odd_native")
    texture.image = original_image
    nodes.update_tag()
    render("restored_native")
    bpy.data.images.remove(odd_image)
    scene.cycles.pixel_displacement_scale = 0.4
    render("scale")
    scene.cycles.pixel_displacement_max_distance = 0.015
    render("distance")
    scene.cycles.pixel_displacement_steps = 128
    render("steps")
    obj = bpy.data.objects["SSand"]
    obj.scale = (obj.scale.x * 1.1, obj.scale.y * 0.9, obj.scale.z * 1.3)
    render("transform")
    image = bpy.data.materials["Material.002"].node_tree.nodes["Image Texture.003"].image
    image.scale(2053, 2063)
    image.update()
    # Keep the resized pixels in memory; otherwise Cycles can reload the original
    # file-backed image. The input image and .blend are never written.
    image.pack()
    bpy.data.materials["Material.002"].node_tree.update_tag()
    render("odd_texture")
    scene.cycles.use_pixel_displacement_resolution_clamp = True
    scene.cycles.pixel_displacement_resolution = 3001
    render("odd_clamp")
    scene.cycles.pixel_displacement_resolution = 2048
    render("cached_mode")
    scene.cycles.use_pixel_displacement_resolution_clamp = False
    render("restored_direct")
    (output / "results.json").write_text(json.dumps({"completed": True, "cases": completed}, indent=2))


if __name__ == "__main__":
    main()
