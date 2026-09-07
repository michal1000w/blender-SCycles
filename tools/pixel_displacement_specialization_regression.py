# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Exercise displacement sampler specialization and precision fallbacks on the JAR scene.

Run baseline and candidate sequentially on Metal; pass an output directory after --.
Only test EXRs and a completion report are written; the input scene is never saved.
"""

import json
from pathlib import Path
import sys

import bpy
import numpy as np


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
    nodes = bpy.data.materials["Material.002"].node_tree
    texture = nodes.nodes["Image Texture.003"]
    original_image = texture.image
    displacement = next(node for node in nodes.nodes if node.type == "DISPLACEMENT")
    original_height = displacement.inputs["Height"].links[0].from_socket
    completed = []

    def render(label):
        nodes.update_tag()
        scene.update_tag()
        bpy.context.view_layer.update()
        bpy.ops.render.render()
        bpy.data.images["Render Result"].save_render(str(output / (label + ".exr")), scene=scene)
        completed.append(label)
        print("SPECIALIZATION_REGRESSION", label, flush=True)

    render("initial")
    texture.interpolation = "Cubic"
    render("cubic")
    texture.interpolation = "Closest"
    render("closest")
    texture.interpolation = "Linear"
    texture.projection = "SPHERE"
    render("sphere")
    texture.projection = "FLAT"

    image = bpy.data.images.new("DisplacementSamplerRegression", width=65, height=67, alpha=True)
    y, x = np.mgrid[0:67, 0:65].astype(np.float32)
    pixels = np.stack((0.1 + 0.2 * x / 64, 0.05 + 0.15 * y / 66,
                       0.2 + 0.1 * np.sin(x * 0.2), 0.2 + 0.8 * y / 66), axis=-1)
    image.pixels.foreach_set(pixels.ravel())
    image.update()
    image.pack()
    texture.image = image
    render("rgba_linear")
    nodes.links.new(texture.outputs["Alpha"], displacement.inputs["Height"])
    render("rgba_alpha_height")

    nodes.links.new(original_height, displacement.inputs["Height"])
    texture.image = original_image
    # Translate the whole scene together: view geometry is preserved, but float precision
    # makes moving displacement arithmetic out of (P + D) - P unsafe.
    for obj in scene.objects:
        if obj.parent is None:
            obj.location.x += 100000.0
            obj.update_tag()
    render("large_coordinates")
    (output / "results.json").write_text(json.dumps({"completed": True, "cases": completed}, indent=2))


if __name__ == "__main__":
    main()
