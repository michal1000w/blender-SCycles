# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Render the shader nodes this fork adds or extends (gratings, polarizer, thin film, Fast Volume,
Diffraction BSDF) with SVM and OSL to EXR, for pixel-wise comparison between builds or devices.

Complements cycles_metal_feature_scenes.py, whose scene helpers it reuses. Renders use fixed seeds
and no denoising. Compare outputs with cycles_metal_image_compare.py.

  blender -b --factory-startup --python tests/python/cycles_fork_node_scenes.py -- \
      --output DIRECTORY [--device METAL|CPU] [--scenes a,b,c]
"""

import sys
from pathlib import Path

# Scene helpers and argument parsing of the feature scene suite, without running it.
_source = (Path(__file__).resolve().parent / "cycles_metal_feature_scenes.py").read_text()
exec(compile(_source[:_source.rindex("def main():")], "cycles_metal_feature_scenes.py", "exec"))

def finish(args, name):
    enable_device(args)
    bpy.context.scene.render.filepath = str(args.output / f"{name}.exr")
    bpy.ops.render.render(write_still=True)
    print("SCENE", name, flush=True)

def node_scene(args, idname, values=None, tangent=False, volume=False, props=None):
    new_scene(args)
    add_floor()
    mat, nodes, links, output = material(idname)
    node = nodes.new(idname)
    for k, v in (props or {}).items():
        setattr(node, k, v)
    for k, v in (values or {}).items():
        if k in node.inputs:
            node.inputs[k].default_value = v
    if tangent:
        t = nodes.new("ShaderNodeCombineXYZ"); t.inputs["X"].default_value = 1
        links.new(t.outputs[0], node.inputs["Tangent"])
    links.new(node.outputs[0], output.inputs["Volume" if volume else "Surface"])
    if volume:
        bpy.ops.mesh.primitive_cube_add(size=1.6, location=(0, 0, 0.9))
        bpy.context.object.data.materials.append(mat)
    else:
        add_sphere(mat)
    return node

def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = args.scenes.split(",") if args.scenes != "all" else None
    def want(n): return names is None or n in names
    if want("diffraction_node"):
        node_scene(args, "ShaderNodeBsdfDiffraction", tangent=True); finish(args, "diffraction_node")
    if want("fast_volume"):
        node_scene(args, "ShaderNodeVolumeFast", volume=True); finish(args, "fast_volume")
    if want("glass_polarizer_grating"):
        node_scene(args, "ShaderNodeBsdfGlass", {"Roughness": 0.1, "Polarizer": True, "Polarizer Angle": 0.4,
                   "Diffraction Weight": 0.6, "Diffraction Pitch": 1400, "Diffraction Depth": 200}, tangent=True)
        finish(args, "glass_polarizer_grating")
    if want("glass_grating_default_tangent"):
        node_scene(args, "ShaderNodeBsdfGlass", {"Roughness": 0.15, "Diffraction Weight": 1.0})
        finish(args, "glass_grating_default_tangent")
    if want("refraction_grating"):
        node_scene(args, "ShaderNodeBsdfRefraction", {"Roughness": 0.1, "Diffraction Weight": 0.7}, tangent=True)
        finish(args, "refraction_grating")
    if want("principled_grating_transmission"):
        node_scene(args, "ShaderNodeBsdfPrincipled", {"Transmission Weight": 1.0, "Roughness": 0.15, "Diffraction Weight": 0.8,
                   "Thin Film Thickness": 300.0})
        finish(args, "principled_grating_transmission")
    if want("principled_thinfilm_backface"):
        node_scene(args, "ShaderNodeBsdfPrincipled", {"Transmission Weight": 1.0, "Roughness": 0.05, "Thin Film Thickness": 400.0,
                   "Thin Film IOR": 1.6, "IOR": 1.5})
        finish(args, "principled_thinfilm_backface")
    if want("glass_thinfilm"):
        node_scene(args, "ShaderNodeBsdfGlass", {"Roughness": 0.05, "Thin Film Thickness": 400.0, "Thin Film IOR": 1.6})
        finish(args, "glass_thinfilm")
    if want("glass_anisotropic"):
        node_scene(args, "ShaderNodeBsdfGlass", {"Roughness": 0.3, "Anisotropy": 0.8})
        finish(args, "glass_anisotropic")
    if want("glass_tinted_osl"):
        node_scene(args, "ShaderNodeBsdfGlass", {"Color": (0.9, 0.4, 0.3, 1), "Roughness": 0.1})
        bpy.context.scene.cycles.shading_system = True
        finish(args, "glass_tinted_osl")
    if want("glass_grating_osl"):
        node_scene(args, "ShaderNodeBsdfGlass", {"Roughness": 0.15, "Diffraction Weight": 0.6, "Thin Film Thickness": 300.0}, tangent=True)
        bpy.context.scene.cycles.shading_system = True
        finish(args, "glass_grating_osl")
    if want("principled_osl"):
        node_scene(args, "ShaderNodeBsdfPrincipled", {"Transmission Weight": 0.7, "Roughness": 0.2, "Thin Film Thickness": 350.0, "Diffraction Weight": 0.5})
        bpy.context.scene.cycles.shading_system = True
        finish(args, "principled_osl")
    if want("glossy_grating_osl"):
        node_scene(args, "ShaderNodeBsdfAnisotropic", {"Roughness": 0.22, "Diffraction Weight": 1.0}, tangent=True)
        bpy.context.scene.cycles.shading_system = True
        finish(args, "glossy_grating_osl")
    if want("refraction_osl"):
        node_scene(args, "ShaderNodeBsdfRefraction", {"Roughness": 0.1, "Diffraction Weight": 0.5}, tangent=True)
        bpy.context.scene.cycles.shading_system = True
        finish(args, "refraction_osl")

main()
