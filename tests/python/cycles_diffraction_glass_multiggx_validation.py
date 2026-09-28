# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Validate bounded local-tint, uncoated Glass MultiGGX diffraction routing.

Run under a Blender build containing the two-sided cache implementation with
``-- OUTPUT_DIR``. The 8x8 CPU renders exercise shader preparation and both
SVM/OSL closure routing; they are not image-quality acceptance tests.
"""

import json
import math
from pathlib import Path
import sys

import bpy


output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output.mkdir(parents=True, exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.shading_system = "--osl" in sys.argv[sys.argv.index("--") + 1:]
scene.cycles.samples = 1
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.render.resolution_x = scene.render.resolution_y = 8
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
bpy.ops.mesh.primitive_cube_add()
material = bpy.data.materials.new("Two-sided diffraction validation")
material.use_nodes = True
bpy.context.object.data.materials.append(material)
tree = material.node_tree
tree.nodes.remove(tree.nodes.get("Principled BSDF"))
glass = tree.nodes.new("ShaderNodeBsdfGlass")
tree.links.new(glass.outputs["BSDF"], tree.nodes["Material Output"].inputs["Surface"])
glass.distribution = "MULTI_GGX"
glass.inputs["Color"].default_value = (1, 1, 1, 1)
glass.inputs["Roughness"].default_value = 0.5
glass.inputs["IOR"].default_value = 1.5
glass.inputs["Diffraction Weight"].default_value = 1.0
glass.inputs["Diffraction Depth"].default_value = 150.0
bpy.ops.object.camera_add(location=(0, 0, 5))
scene.camera = bpy.context.object
noise = tree.nodes.new("ShaderNodeTexNoise")
results = []


def check(name, expected_error=None, socket=None, linked=False, value=None):
    old_value = None
    link = None
    if socket is not None:
        current = glass.inputs[socket].default_value
        old_value = tuple(current) if hasattr(current, "__len__") else current
        if linked:
            source = noise.outputs["Color" if socket == "Color" else "Fac"]
            link = tree.links.new(source, glass.inputs[socket])
        else:
            glass.inputs[socket].default_value = value
    error = None
    finite_pixels = None
    try:
        image_path = output / f"{name}.exr"
        scene.render.filepath = str(image_path)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(image_path), check_existing=False)
        pixels = image.pixels[:]
        finite_pixels = bool(pixels) and all(math.isfinite(channel) for channel in pixels)
        bpy.data.images.remove(image)
        if not finite_pixels:
            raise RuntimeError("Render Result contains no pixels or non-finite channels")
    except RuntimeError as exception:
        error = str(exception)
    finally:
        if link:
            tree.links.remove(link)
        if old_value is not None:
            glass.inputs[socket].default_value = old_value
    passed = error is None if expected_error is None else bool(error and expected_error in error)
    results.append({"case": name, "expected_error": expected_error,
                    "error": error, "finite_pixels": finite_pixels, "passed": passed})
    (output / "report.json").write_text(json.dumps(results, indent=2) + "\n")


check("neutral_uncoated_two_sided")
check("partial_diffraction_uses_same_cache", socket="Diffraction Weight", value=0.5)
check("bounded_colored_return", socket="Color", value=(0.8, 0.35, 0.2, 1.0))
check("bounded_gray_return", socket="Color", value=(0.6, 0.6, 0.6, 1.0))
check("color_above_unit_rejected", "Color components in [0, 1]", "Color",
      value=(1.1, 0.8, 0.8, 1.0))
check("linked_color_return", socket="Color", linked=True)
check("film_rejected", "does not support thin film", "Thin Film Thickness",
      value=300.0)
for socket in ("Roughness", "IOR", "Diffraction Pitch", "Diffraction Depth",
               "Diffraction Duty Cycle"):
    check("linked_" + socket.lower().replace(" ", "_"),
          "requires constant " + socket, socket, linked=True)
check("ior_identity_outside_bounded_mode", "requires finite roughness and IOR > 1",
      "IOR", value=1.0)
check("smooth_positive_relief_single_event", socket="Roughness", value=0.0)
check("flat_relief_native_multiggx", socket="Diffraction Depth", value=0.0)
check("default_off_native_multiggx", socket="Diffraction Weight", value=0.0)

if not all(result["passed"] for result in results):
    raise RuntimeError("Glass MultiGGX diffraction validation failed")
