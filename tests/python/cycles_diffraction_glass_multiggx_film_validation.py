# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Check bounded coated Glass MultiGGX preparation and SVM/OSL routing.

Run with Blender ``--python this_script.py -- OUTPUT_DIR [--osl] [--thick]``.
The 8x8 CPU renders check successful routing and finite EXR pixels, not an
image-quality or spectral furnace target. ``--thick`` additionally exercises
a 2,000 nm / IOR 1.5 film and records its cache-preparation-inclusive time.
"""

import json
import math
from pathlib import Path
import sys
import time

import bpy


args = sys.argv[sys.argv.index("--") + 1:]
output = Path(args[0]).resolve()
output.mkdir(parents=True, exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.shading_system = "--osl" in args
scene.cycles.samples = 1
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.render.resolution_x = scene.render.resolution_y = 8
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
bpy.ops.mesh.primitive_cube_add()
material = bpy.data.materials.new("Bounded coated two-sided Glass")
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
glass.inputs["Diffraction Pitch"].default_value = 1200.0
glass.inputs["Diffraction Depth"].default_value = 125.0
glass.inputs["Diffraction Duty Cycle"].default_value = 0.43
glass.inputs["Thin Film IOR"].default_value = 1.5
glass.inputs["Thin Film Thickness"].default_value = 250.0
bpy.ops.object.camera_add(location=(0, 0, 5))
scene.camera = bpy.context.object
noise = tree.nodes.new("ShaderNodeTexNoise")
results = []


def check(name, expected_error=None, overrides=None, link=None):
    overrides = overrides or {}
    saved = {socket: glass.inputs[socket].default_value for socket in overrides}
    for socket, value in overrides.items():
        glass.inputs[socket].default_value = value
    edge = None
    if link:
        edge = tree.links.new(noise.outputs["Fac"], glass.inputs[link])
    error = None
    finite = None
    start = time.monotonic()
    try:
        exr = output / f"{name}.exr"
        scene.render.filepath = str(exr)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(exr), check_existing=False)
        pixels = image.pixels[:]
        finite = bool(pixels) and all(math.isfinite(channel) for channel in pixels)
        bpy.data.images.remove(image)
        if not finite:
            raise RuntimeError("Coated Glass EXR contains no pixels or nonfinite channels")
    except RuntimeError as exception:
        error = str(exception)
    finally:
        if edge:
            tree.links.remove(edge)
        for socket, value in saved.items():
            glass.inputs[socket].default_value = value
    passed = error is None if expected_error is None else bool(error and expected_error in error)
    results.append({"case": name, "expected_error": expected_error, "error": error,
                    "finite_pixels": finite, "seconds": time.monotonic() - start,
                    "passed": passed})
    (output / "report.json").write_text(json.dumps(results, indent=2) + "\n")


check("coated_neutral_250nm")
check("coated_colored_250nm", overrides={"Color": (0.7, 0.4, 0.2, 1.0)})
check("uncoated_identity", overrides={"Thin Film Thickness": 0.0,
                                      "Thin Film IOR": 2.0})
check("linked_thickness_rejected", "requires constant Thin Film Thickness",
      link="Thin Film Thickness")
check("linked_film_ior_rejected", "requires constant Thin Film IOR", link="Thin Film IOR")
check("film_exceeds_spectral_budget", "exceeds the 256-node spectral cache limit",
      overrides={"Thin Film Thickness": 10000.0})
if "--thick" in args:
    check("coated_2000nm", overrides={"Thin Film Thickness": 2000.0})

if not all(result["passed"] for result in results):
    raise RuntimeError("Coated Glass MultiGGX validation failed")
