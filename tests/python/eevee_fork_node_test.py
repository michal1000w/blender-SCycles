# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Create every shader node this fork adds or extends, check its sockets and render it with EEVEE.

Fails when a node misses an expected socket, has duplicate socket names, renders non-finite
pixels or the magenta of a shader that failed to compile.

  blender -b --factory-startup --python tests/python/eevee_fork_node_test.py -- DIRECTORY
"""
import bpy, sys, json, math
from pathlib import Path
import numpy as np

out = Path(sys.argv[sys.argv.index("--") + 1]); out.mkdir(parents=True, exist_ok=True)
result = {}

def setup():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = scene.render.resolution_y = 96
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.eevee.taa_render_samples = 8
    bpy.ops.object.camera_add(location=(0, -4, 1.5), rotation=(math.radians(75), 0, 0))
    scene.camera = bpy.context.object
    bpy.ops.object.light_add(type="AREA", location=(2, -2, 4)); bpy.context.object.data.energy = 600
    bpy.ops.mesh.primitive_plane_add(size=10)
    world = bpy.data.worlds.new("W"); scene.world = world
    return scene

EXPECT = {
    "ShaderNodeBsdfGlass": ["Tangent", "Anisotropy", "Rotation", "Polarizer", "Polarizer Angle", "Diffraction Weight", "Diffraction Pitch", "Diffraction Depth", "Diffraction Duty Cycle", "Thin Film Thickness"],
    "ShaderNodeBsdfAnisotropic": ["Diffraction Weight", "Diffraction Pitch", "Diffraction Depth", "Diffraction Duty Cycle", "Diffraction Medium IOR", "Tangent"],
    "ShaderNodeBsdfMetallic": ["Diffraction Weight", "Diffraction Pitch", "Diffraction Depth", "Diffraction Duty Cycle"],
    "ShaderNodeBsdfPrincipled": ["Diffraction Weight", "Diffraction Pitch", "Diffraction Depth", "Diffraction Duty Cycle"],
    "ShaderNodeBsdfRefraction": ["Diffraction Weight", "Diffraction Pitch", "Diffraction Depth", "Diffraction Duty Cycle", "Tangent"],
    "ShaderNodeBsdfDiffraction": [],
    "ShaderNodeVolumeFast": [],
}
for idname, expected in EXPECT.items():
    scene = setup()
    mat = bpy.data.materials.new(idname)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    node = nodes.new(idname)
    names = [s.name for s in node.inputs]
    entry = {"inputs": names,
             "missing": [n for n in expected if n not in names],
             "duplicates": sorted({n for n in names if names.count(n) > 1}),
             "identifiers_unique": len({s.identifier for s in node.inputs}) == len(node.inputs)}
    volume = idname == "ShaderNodeVolumeFast"
    for s in node.inputs:
        if s.name == "Diffraction Weight":
            s.default_value = 0.5
    links.new(node.outputs[0], output.inputs["Volume" if volume else "Surface"])
    if volume:
        bpy.ops.mesh.primitive_cube_add(size=1.6, location=(0, 0, 0.9))
    else:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.8, location=(0, 0, 0.8))
    bpy.context.object.data.materials.append(mat)
    scene.render.filepath = str(out / f"{idname}.exr")
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(scene.render.filepath)
    px = np.empty(len(img.pixels), dtype=np.float32); img.pixels.foreach_get(px)
    px = px.reshape(-1, img.channels)[:, :3]
    magenta = float(((px[:, 0] > 0.5) & (px[:, 1] < 0.05) & (px[:, 2] > 0.5)).mean())
    entry.update(finite=bool(np.isfinite(px).all()), mean=float(px.mean()), magenta_fraction=magenta)
    result[idname] = entry
    print("NODE", idname, json.dumps({k: v for k, v in entry.items() if k != "inputs"}), flush=True)
(out / "report.json").write_text(json.dumps(result, indent=1))

failed = [name for name, entry in result.items()
          if entry["missing"] or entry["duplicates"] or not entry["identifiers_unique"] or
          not entry["finite"] or entry["magenta_fraction"] > 0.01]
if failed:
    raise SystemExit(f"FAILED: {failed}")
print("PASSED", len(result), "nodes")
