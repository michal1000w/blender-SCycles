# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded Principled MultiGGX diffraction host and SVM/OSL routing checks.

Run under Blender with ``-- OUTPUT_DIR [--osl]``. Small CPU renders check
finite EXRs and explicit unsupported combinations; they are not furnace or
image-quality acceptance measurements.
"""

import json
import math
from pathlib import Path
import sys

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
world = bpy.data.worlds.new("Unit white validation environment")
world.use_nodes = True
scene.world = world
world.node_tree.nodes["Background"].inputs["Color"].default_value = (1, 1, 1, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
bpy.ops.mesh.primitive_cube_add()
material = bpy.data.materials.new("Principled MultiGGX validation")
material.use_nodes = True
bpy.context.object.data.materials.append(material)
node = material.node_tree.nodes["Principled BSDF"]
node.distribution = "MULTI_GGX"
noise = material.node_tree.nodes.new("ShaderNodeTexNoise")
tangent = material.node_tree.nodes.new("ShaderNodeCombineXYZ")
tangent.inputs["X"].default_value = 1.0
bpy.ops.object.camera_add(location=(0, 0, 5))
scene.camera = bpy.context.object
defaults = {"Base Color": (0.7, 0.5, 0.3, 1), "Metallic": 0.0,
            "Transmission Weight": 0.0, "Roughness": 0.5, "IOR": 1.5,
            "Specular Tint": (1, 1, 1, 1), "Anisotropic": 0.0,
            "Tangent": (1.0, 0.0, 0.0),
            "Transmission Dispersion Scale": 0.0, "Transmission Dispersion Abbe Number": 20.0,
            "Thin Wall": False,
            "Thin Film Thickness": 0.0, "Thin Film IOR": 1.32,
            "Diffraction Weight": 1.0, "Diffraction Pitch": 1200.0,
            "Diffraction Depth": 125.0, "Diffraction Duty Cycle": 0.43}
results = []


def check(name, overrides=None, links=(), expected_error=None):
    for socket, value in defaults.items():
        for edge in list(node.inputs[socket].links):
            material.node_tree.links.remove(edge)
        node.inputs[socket].default_value = value
    for socket, value in (overrides or {}).items():
        node.inputs[socket].default_value = value
    for socket in links:
        source = tangent.outputs["Vector"] if socket == "Tangent" else noise.outputs[
            "Color" if socket in ("Base Color", "Specular Tint") else "Fac"]
        material.node_tree.links.new(source, node.inputs[socket])
    error = None
    finite = None
    try:
        exr = output / f"{name}.exr"
        scene.render.filepath = str(exr)
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(exr), check_existing=False)
        pixels = image.pixels[:]
        finite = bool(pixels) and all(math.isfinite(value) for value in pixels)
        bpy.data.images.remove(image)
        if not finite:
            raise RuntimeError("Principled EXR has no pixels or nonfinite channels")
    except RuntimeError as exception:
        error = str(exception)
    passed = error is None if expected_error is None else bool(error and expected_error in error)
    results.append({"case": name, "error": error, "expected_error": expected_error,
                    "finite_pixels": finite, "passed": passed})
    (output / "report.json").write_text(json.dumps(results, indent=2) + "\n")


for metallic in (0.0, 0.5, 1.0):
    for transmission in (0.0, 0.5, 1.0):
        check(f"mix_m{metallic:g}_t{transmission:g}",
              {"Metallic": metallic, "Transmission Weight": transmission})
check("linked_base_color", {"Transmission Weight": 0.5}, links=("Base Color",))
check("linked_runtime_weights", links=("Metallic", "Transmission Weight", "Base Color"))
check("reflective_tinted_specular", {"Specular Tint": (0.4, 0.7, 1, 1)})
check("anisotropic_reflection_with_tangent", {"Anisotropic": 0.4}, links=("Tangent",))
check("solid_transmission_film", {"Transmission Weight": 0.5, "Thin Film Thickness": 250.0})
check("solid_tinted_specular", {"Transmission Weight": 0.5,
      "Specular Tint": (0.4, 0.7, 1, 1)})
check("solid_dispersion", {"Transmission Weight": 0.5,
      "Transmission Dispersion Scale": 1.0})
check("solid_linked_specular_tint", {"Transmission Weight": 0.5}, links=("Specular Tint",))
check("solid_generalized_dispersion", {"Transmission Weight": 1.0,
      "Specular Tint": (0.4, 0.7, 1, 1), "Transmission Dispersion Scale": 1.0})
check("solid_linked_dispersion_rejected", {"Transmission Weight": 0.5},
      links=("Transmission Dispersion Scale",), expected_error="requires constant Transmission Dispersion Scale")
check("solid_coated_generalized", {"Transmission Weight": 0.5,
      "Specular Tint": (0.4, 0.7, 1, 1), "Thin Film Thickness": 250.0})
check("solid_coated_linked_specular_tint", {"Transmission Weight": 1.0, "Thin Film Thickness": 250.0}, links=("Specular Tint",))
check("solid_coated_generalized_dispersion", {"Transmission Weight": 1.0,
      "Specular Tint": (0.4, 0.7, 1, 1), "Transmission Dispersion Scale": 1.0,
      "Thin Film Thickness": 250.0})
for thickness in (0.05, 0.1, 0.100001):
    check(f"solid_generalized_film_cutoff_{thickness:g}", {"Transmission Weight": 1.0,
          "Specular Tint": (0.4, 0.7, 1, 1), "Transmission Dispersion Scale": 1.0,
          "Thin Film Thickness": thickness})
for ior in (3.1, 10.0):
    check(f"solid_coated_high_ior_{ior:g}", {"Transmission Weight": 1.0, "IOR": ior,
          "Specular Tint": (0.4, 0.7, 1, 1), "Thin Film Thickness": 300.0,
          "Thin Film IOR": 2.4})
check("solid_thin_wall_rejected", {"Transmission Weight": 0.5, "Thin Wall": True},
      expected_error="requires a solid interface")
check("solid_linked_ior_rejected", {"Transmission Weight": 0.5}, links=("IOR",),
      expected_error="requires constant IOR")
check("solid_linked_film_rejected", {"Transmission Weight": 0.5},
      links=("Thin Film Thickness",), expected_error="requires constant Thin Film Thickness")
check("known_metal_skips_transmission_restrictions", {"Metallic": 1.0,
      "Specular Tint": (0.4, 0.7, 1, 1), "Transmission Dispersion Scale": 1.0},
      links=("Transmission Weight",))
check("known_zero_transmission_skips_physical_restrictions",
      {"Specular Tint": (0.4, 0.7, 1, 1), "Transmission Dispersion Scale": 1.0})

if not all(result["passed"] for result in results):
    raise RuntimeError("Principled MultiGGX validation failed")
