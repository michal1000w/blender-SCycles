#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Save editable, neutral coated Glass white-world furnaces without rendering."""

import hashlib
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
scene.view_settings.view_transform = "Standard"

world = bpy.data.worlds.new("Unit white world")
scene.world = world
world.use_nodes = True
background = world.node_tree.nodes.get("Background")
background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
background.inputs["Strength"].default_value = 1.0

bpy.ops.mesh.primitive_plane_add(size=20.0)
plane = bpy.context.object
plane.name = "Coated rough dielectric interface"
material = bpy.data.materials.new("Neutral coated Glass diffraction")
material.use_nodes = True
plane.data.materials.append(material)
tree = material.node_tree
tree.nodes.remove(tree.nodes.get("Principled BSDF"))
glass = tree.nodes.new("ShaderNodeBsdfGlass")
glass.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
glass.inputs["Roughness"].default_value = 0.5
glass.inputs["IOR"].default_value = 1.5
glass.inputs["Diffraction Weight"].default_value = 1.0
glass.inputs["Diffraction Pitch"].default_value = 1200.0
glass.inputs["Diffraction Depth"].default_value = 125.0
glass.inputs["Diffraction Duty Cycle"].default_value = 0.43
glass.inputs["Thin Film IOR"].default_value = 1.32
glass.inputs["Thin Film Thickness"].default_value = 250.0
tree.links.new(glass.outputs["BSDF"], tree.nodes["Material Output"].inputs["Surface"])

bpy.ops.object.camera_add()
camera = bpy.context.object
camera.name = "Off-normal coated furnace camera"
camera.data.type = "ORTHO"
camera.data.ortho_scale = 0.5
camera.data.clip_start = 0.01
camera.data.clip_end = 20.0
scene.camera = camera

scenes = {}
for model, distribution in (("two_sided", "MULTI_GGX"), ("single_event", "GGX")):
    glass.distribution = distribution
    for side, sign in (("front", 1.0), ("back", -1.0)):
        camera.location = (0.6, 0.35, sign * 3.0)
        camera.rotation_euler = (-Vector(camera.location)).to_track_quat("-Z", "Y").to_euler()
        scene["furnace_model"] = model
        scene["furnace_side"] = side
        scene["furnace_scope"] = "neutral lossless real-film Glass interface"
        name = f"{model}_{side}"
        scene.render.filepath = f"//{name}"
        bpy.context.view_layer.update()
        path = output / f"{name}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        scenes[name] = {"path": str(path), "sha256": digest(path)}

(output / "manifest.json").write_text(json.dumps({
    "scope": "64x64 256-sample neutral coated Glass white-world furnace",
    "film_ior": 1.32, "film_thickness_nm": 250.0,
    "film_optical_thickness_nm": 330.0,
    "theory": "unit radiance must survive a lossless reciprocal two-sided BSDF",
    "binary_sha256": digest(bpy.app.binary_path),
    "script_sha256": digest(__file__),
    "scenes": scenes,
}, indent=2) + "\n")
print(f"DIFFRACTION_GLASS_FILM_FURNACE {output}", flush=True)
