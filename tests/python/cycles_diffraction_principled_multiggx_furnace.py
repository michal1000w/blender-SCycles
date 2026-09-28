#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Save bounded front/back white-environment Principled diffraction furnaces.

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


args = sys.argv[sys.argv.index("--") + 1:]
coated = "--film" in args
generalized = "--generalized" in args or coated
output = Path(args[0]).resolve()
output.mkdir(parents=True, exist_ok=False)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = 256
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.use_coherent_specular_connections = False
scene.cycles.blur_glossy = 0.0
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
material = bpy.data.materials.new("Neutral white Principled diffraction")
material.use_nodes = True
plane.data.materials.append(material)
tree = material.node_tree
principled = tree.nodes.get("Principled BSDF")
principled.distribution = "MULTI_GGX"
principled.inputs["Base Color"].default_value = (1.0, 1.0, 1.0, 1.0)
principled.inputs["Roughness"].default_value = 0.5
principled.inputs["IOR"].default_value = 1.5
principled.inputs["Metallic"].default_value = 0.3
principled.inputs["Transmission Weight"].default_value = 0.4
principled.inputs["Specular IOR Level"].default_value = 0.5
principled.inputs["Specular Tint"].default_value = (1.0, 1.0, 1.0, 1.0)
principled.inputs["Coat Weight"].default_value = 0.0
principled.inputs["Sheen Weight"].default_value = 0.0
principled.inputs["Subsurface Weight"].default_value = 0.0
principled.inputs["Diffraction Weight"].default_value = 1.0
principled.inputs["Diffraction Pitch"].default_value = 1200.0
principled.inputs["Diffraction Depth"].default_value = 125.0
principled.inputs["Diffraction Duty Cycle"].default_value = 0.43
principled.inputs["Thin Film Thickness"].default_value = 0.0
if generalized:
    principled.inputs["Metallic"].default_value = 0.0
    principled.inputs["Transmission Weight"].default_value = 1.0
    principled.inputs["Specular Tint"].default_value = (0.3, 0.7, 1.0, 1.0)
    principled.inputs["Transmission Dispersion Scale"].default_value = 1.0
    principled.inputs["Transmission Dispersion Abbe Number"].default_value = 20.0
if coated:
    principled.inputs["Thin Film Thickness"].default_value = 250.0
    principled.inputs["Thin Film IOR"].default_value = 1.32
scope = ("bare generalized Principled transmission1, metallic0, SpecularTintRGB(.3,.7,1), dispersionScale1/Abbe20"
         if generalized else "neutral uncoated Principled metallic0.3/transmission0.4; white reflection, transmission and diffuse components")
if coated:
    scope = scope.replace("bare", "coated film1.32/250nm")
tree.links.new(principled.outputs["BSDF"], tree.nodes["Material Output"].inputs["Surface"])

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
    principled.distribution = distribution
    principled.inputs["Diffraction Weight"].default_value = diffraction_weight
    for side, sign in (("front", 1.0), ("back", -1.0)):
        camera.location = (0.6, 0.35, sign * 3.0)
        camera.rotation_euler = (-Vector(camera.location)).to_track_quat("-Z", "Y").to_euler()
        scene["furnace_model"] = model
        scene["furnace_side"] = side
        scene["furnace_expected_unit_radiance"] = 1.0
        scene["furnace_scope"] = scope
        scene.render.filepath = f"//{model}_{side}"
        bpy.context.view_layer.update()
        name = f"{model}_{side}"
        path = output / f"{name}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(path))
        scenes[name] = {"path": str(path), "sha256": digest(path)}

manifest = {
    "scope": "unit-radiance white environment, front/back " + scope,
    "metallic": 0.0 if generalized else 0.3, "transmission_weight": 1.0 if generalized else 0.4,
    "generalized_dispersion": generalized, "coated_generalized": coated,
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
