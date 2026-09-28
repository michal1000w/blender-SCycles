# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Validate constant Glossy/Metallic MultiGGX diffraction and linked-input rejection."""

import json
from pathlib import Path
import sys

import bpy


output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output.mkdir(parents=True, exist_ok=False)
metallic = "--metallic" in sys.argv[sys.argv.index("--") + 1:]
principled = "--principled" in sys.argv[sys.argv.index("--") + 1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.shading_system = "--osl" in sys.argv[sys.argv.index("--") + 1:]
scene.cycles.samples = 1
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.render.resolution_x = scene.render.resolution_y = 8
bpy.ops.mesh.primitive_cube_add()
material = bpy.data.materials.new("Glossy validation")
material.use_nodes = True
bpy.context.object.data.materials.append(material)
tree = material.node_tree
tree.nodes.remove(tree.nodes.get("Principled BSDF"))
glossy = tree.nodes.new("ShaderNodeBsdfPrincipled" if principled else
                        "ShaderNodeBsdfMetallic" if metallic else "ShaderNodeBsdfAnisotropic")
if principled:
    glossy.inputs["Metallic"].default_value = 1.0
tree.links.new(glossy.outputs[0], tree.nodes["Material Output"].inputs["Surface"])
tangent = tree.nodes.new("ShaderNodeCombineXYZ")
tangent.inputs["X"].default_value = 1.0
tree.links.new(tangent.outputs["Vector"], glossy.inputs["Tangent"])
noise = tree.nodes.new("ShaderNodeTexNoise")
bpy.ops.object.camera_add(location=(0, 0, 5))
scene.camera = bpy.context.object

results = []


def check(name, coverage, distribution, linked_input=None, expected_error=None):
    glossy.inputs["Diffraction Weight"].default_value = coverage
    glossy.distribution = distribution
    link = None
    if linked_input:
        link = tree.links.new(noise.outputs["Fac"], glossy.inputs[linked_input])
    message = None
    try:
        bpy.ops.render.render()
    except RuntimeError as exception:
        message = str(exception)
    finally:
        if link:
            tree.links.remove(link)
    passed = (message is None) if expected_error is None else bool(
        message and expected_error in message
    )
    results.append({
        "case": name,
        "linked_input": linked_input,
        "distribution": distribution,
        "coverage": coverage,
        "expected_error": expected_error,
        "error": message,
        "passed": passed,
    })
    (output / "report.json").write_text(json.dumps(results, indent=2) + "\n")


check("ordinary_default", 0.0, "MULTI_GGX")
check("supported_multiscatter_constant", 1.0, "MULTI_GGX")
glossy.inputs["Anisotropic" if principled else "Anisotropy"].default_value = 0.4
check("supported_multiscatter_folded_tangent", 1.0, "MULTI_GGX")
check("supported_multiscatter_folded_tangent_partial", 0.5, "MULTI_GGX")
glossy.inputs["Anisotropic" if principled else "Anisotropy"].default_value = 0.0
if not metallic and not principled:
    check("supported_ashikhmin", 1.0, "ASHIKHMIN_SHIRLEY")
if not principled:
    check("supported_beckmann", 1.0, "BECKMANN")
sockets = ["Roughness", "Anisotropic" if principled else "Anisotropy", "Diffraction Pitch", "Diffraction Depth",
           "Diffraction Duty Cycle"]
if not metallic and not principled:
    sockets.append("Diffraction Medium IOR")
for socket in sockets:
    check("linked_" + socket.lower().replace(" ", "_"), 1.0, "MULTI_GGX", socket,
          "Multiscatter GGX diffraction requires constant " + socket)

if principled:
    error = "Principled Multiscatter GGX diffraction currently requires constant Metallic 1"
    for value in (0.0, 0.5):
        glossy.inputs["Metallic"].default_value = value
        check("unsupported_metallic_" + str(value), 1.0, "MULTI_GGX", expected_error=error)
    glossy.inputs["Metallic"].default_value = 1.0
    check("unsupported_linked_metallic", 1.0, "MULTI_GGX", "Metallic", error)


if not all(result["passed"] for result in results):
    raise RuntimeError("Distribution/linked-input validation failed")
