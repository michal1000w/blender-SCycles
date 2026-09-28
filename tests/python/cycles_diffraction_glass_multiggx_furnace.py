#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Save bounded front/back white-environment Glass diffraction furnaces.

Run under Blender with ``-- OUTPUT_DIR``. This only saves editable scenes; it
does not render. A neutral lossless BSDF in a unit-radiance world must return
unit radiance from either side, independent of its R/T split. The ordinary
MultiGGX scenes are native controls, and single-event GGX scenes expose the
missing-energy baseline. No tinted absorption or film is in this fixture.
"""

import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output.mkdir(parents=True, exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 256
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.use_coherent_specular_connections = False
scene.cycles.max_bounces = 8
scene.cycles.glossy_bounces = 6
scene.cycles.transmission_bounces = 6
scene.render.resolution_x = scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.film_transparent = False
scene.view_settings.view_transform = "Standard"
scene.view_settings.exposure = 0.0

world = bpy.data.worlds.new("Unit-radiance white environment")
scene.world = world
world.use_nodes = True
background = world.node_tree.nodes.get("Background")
background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
background.inputs["Strength"].default_value = 1.0

bpy.ops.mesh.primitive_plane_add(size=20.0)
plane = bpy.context.object
plane.name = "Two-sided neutral rough dielectric diffraction interface"
material = bpy.data.materials.new("Neutral white Glass diffraction")
material.use_nodes = True
plane.data.materials.append(material)
tree = material.node_tree
tree.nodes.remove(tree.nodes.get("Principled BSDF"))
glass = tree.nodes.new("ShaderNodeBsdfGlass")
glass.distribution = "MULTI_GGX"
glass.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
glass.inputs["Roughness"].default_value = 0.5
glass.inputs["IOR"].default_value = 1.5
glass.inputs["Diffraction Weight"].default_value = 1.0
glass.inputs["Diffraction Pitch"].default_value = 1200.0
glass.inputs["Diffraction Depth"].default_value = 125.0
glass.inputs["Diffraction Duty Cycle"].default_value = 0.43
glass.inputs["Thin Film Thickness"].default_value = 0.0
tree.links.new(glass.outputs["BSDF"], tree.nodes["Material Output"].inputs["Surface"])

bpy.ops.object.camera_add()
camera = bpy.context.object
camera.name = "Off-normal furnace camera"
camera.data.type = "ORTHO"
camera.data.ortho_scale = 0.5
camera.data.clip_start = 0.01
camera.data.clip_end = 20.0
scene.camera = camera


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


scenes = {}
for model, distribution, diffraction_weight in (
        ("two_sided", "MULTI_GGX", 1.0),
        ("native", "MULTI_GGX", 0.0),
        ("single_event", "GGX", 1.0)):
    glass.distribution = distribution
    glass.inputs["Diffraction Weight"].default_value = diffraction_weight
    for side, sign in (("front", 1.0), ("back", -1.0)):
        camera.location = (0.6, 0.35, sign * 3.0)
        camera.rotation_euler = (-Vector(camera.location)).to_track_quat("-Z", "Y").to_euler()
        scene["furnace_model"] = model
        scene["furnace_side"] = side
        scene["furnace_expected_unit_radiance"] = 1.0
        scene["furnace_scope"] = "neutral lossless uncoated two-sided Glass; no tinted absorption or film"
        scene.render.filepath = f"//{model}_{side}"
        bpy.context.view_layer.update()
        name = f"{model}_{side}"
        path = output / f"{name}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        scenes[name] = {"path": str(path), "sha256": digest(path)}

manifest = {
    "scope": "unit-radiance white environment, off-normal front/back neutral Glass",
    "expected_linear_rgb_radiance": [1.0, 1.0, 1.0],
    "theory": "uniform unit environment is unchanged by a reciprocal lossless BSDF after summing reflection and transmission",
    "models": {"two_sided": "new approximate MultiGGX diffraction return",
               "native": "ordinary MultiGGX without diffraction",
               "single_event": "GGX diffraction without multiscatter return"},
    "render_policy": "editable fixtures only; fixed 64x64, 256 samples, adaptive and denoising off",
    "binary": str(Path(bpy.app.binary_path).resolve()),
    "binary_sha256": digest(bpy.app.binary_path),
    "script": str(Path(__file__).resolve()),
    "script_sha256": digest(__file__),
    "scenes": scenes,
}
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"DIFFRACTION_GLASS_FURNACE {output}", flush=True)
