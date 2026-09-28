# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-sample Thin Wall diffraction render fixture for native-limit comparisons."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--samples", type=int, default=128)
parser.add_argument("--transport", choices=("pt", "bdpt", "guided"), default="pt")
parser.add_argument("--osl", action="store_true")
parser.add_argument("--device", choices=("cpu", "metal"), default="metal")
parser.add_argument("--resolution", type=int, default=720)
parser.add_argument("--depth", type=float, default=320.0, help="Relief depth in nm")
parser.add_argument("--coverage", type=float, default=1.0)
parser.add_argument("--film", type=float, default=180.0, help="Film thickness in nm")
parser.add_argument("--dispersion", type=float, default=1.0)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
if (
    args.samples < 1
    or args.resolution < 48
    or args.depth < 0
    or not 0 <= args.coverage <= 1
    or not 0 <= args.dispersion <= 1
    or args.film < 0
):
    parser.error("Invalid sample count, resolution, depth, coverage, film, or dispersion")
if args.osl and args.device == "metal":
    args.device = "cpu"

args.output.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = args.samples
scene.cycles.seed = 37
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.use_bidirectional_path_tracing = args.transport == "bdpt"
scene.cycles.use_guiding = args.transport == "guided"
scene.cycles.shading_system = args.osl
scene.cycles.max_bounces = 8
scene.cycles.sample_clamp_direct = 0
scene.cycles.sample_clamp_indirect = 0
scene.cycles.device = "CPU" if args.device == "cpu" else "GPU"
if args.device == "metal":
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "METAL"
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == "METAL"
    assert any(device.use for device in preferences.devices)

scene.render.resolution_x = args.resolution
scene.render.resolution_y = round(args.resolution * 0.56)
scene.render.resolution_percentage = 100
scene.world = bpy.data.worlds.new("Dark neutral world")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.10, 0.12, 0.15, 1)
scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.2
scene.view_settings.view_transform = "Standard"


def thin_sheet(name, roughness):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    principled.distribution = "GGX"
    values = {
        "Thin Wall": 1,
        "Transmission Weight": 1.0,
        "Roughness": roughness,
        "IOR": 1.5,
        "Base Color": (0.55, 0.78, 0.92, 1),
        "Specular Tint": (0.55, 0.8, 1.0, 1),
        "Diffraction Weight": args.coverage,
        "Diffraction Pitch": 1150.0,
        "Diffraction Depth": args.depth,
        "Diffraction Duty Cycle": 0.42,
        "Thin Film Thickness": args.film,
        "Thin Film IOR": 1.35,
        "Transmission Dispersion Scale": args.dispersion,
        "Transmission Dispersion Abbe Number": 12.0,
    }
    for name, value in values.items():
        principled.inputs[name].default_value = value
    tangent = nodes.new("ShaderNodeCombineXYZ")
    tangent.inputs["X"].default_value = 1
    material.node_tree.links.new(tangent.outputs[0], principled.inputs["Tangent"])
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(principled.outputs[0], output.inputs["Surface"])
    return material


def emitting_bar(name, x, z, color):
    bpy.ops.mesh.primitive_plane_add(size=1, location=(x, 1.2, z))
    obj = bpy.context.object
    obj.name = name
    obj.rotation_euler = (math.pi / 2, 0, 0)
    obj.scale = (0.43, 2.6, 1)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1)
    emission.inputs["Strength"].default_value = 3.0
    output = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(emission.outputs[0], output.inputs["Surface"])
    obj.data.materials.append(material)


for x, roughness in zip((-2.1, 0, 2.1), (0.2, 0.6, 1.0)):
    bpy.ops.mesh.primitive_plane_add(size=2.3, location=(x, 0, 1.6))
    window = bpy.context.object
    window.name = f"Thin Wall roughness {roughness:.1f}"
    window.rotation_euler = (math.pi / 2, 0, 0)
    window.data.materials.append(thin_sheet(window.name, roughness))
    for offset, color in ((-0.55, (1, 0.08, 0.02)), (0, (0.03, 0.9, 0.14)),
                          (0.55, (0.02, 0.16, 1))):
        emitting_bar(f"Emitter {x:+.1f} {offset:+.2f}", x + offset, 1.6, color)

bpy.ops.object.light_add(type="AREA", location=(0, -3, 5))
key = bpy.context.object
key.data.energy = 420
key.data.shape = "DISK"
key.data.size = 3
key.rotation_euler = (Vector((0, 0, 1.6)) - key.location).to_track_quat("-Z", "Y").to_euler()
bpy.ops.object.camera_add(location=(0, -9, 2.1))
scene.camera = bpy.context.object
scene.camera.rotation_euler = (Vector((0, 0, 1.6)) - scene.camera.location).to_track_quat(
    "-Z", "Y"
).to_euler()
scene.camera.data.type = "ORTHO"
scene.camera.data.ortho_scale = 8.0

stem = f"thin_wall_d{args.depth:g}_c{args.coverage:g}_f{args.film:g}_s{args.dispersion:g}"
scene.render.image_settings.file_format = "OPEN_EXR"
scene.render.image_settings.color_depth = "32"
scene.render.filepath = str(args.output / f"{stem}.exr")
bpy.ops.wm.save_as_mainfile(filepath=str(args.output / f"{stem}.blend"))
start = time.monotonic()
bpy.ops.render.render(write_still=True)
seconds = time.monotonic() - start
image = bpy.data.images.load(str(args.output / f"{stem}.exr"), check_existing=False)
pixels = tuple(image.pixels[:])
if not pixels or not all(math.isfinite(value) for value in pixels):
    raise RuntimeError("Missing or nonfinite rendered pixels")
mean_rgb = [sum(pixels[channel::image.channels]) / (len(pixels) // image.channels)
            for channel in range(3)]
bpy.data.images.remove(image)
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_depth = "8"
bpy.data.images["Render Result"].save_render(str(args.output / f"{stem}.png"), scene=scene)
(args.output / f"{stem}.json").write_text(json.dumps({
    "depth_nm": args.depth,
    "coverage": args.coverage,
    "film_nm": args.film,
    "dispersion_scale": args.dispersion,
    "roughness": [0.2, 0.6, 1.0],
    "transport": args.transport,
    "osl": args.osl,
    "device": args.device,
    "samples": args.samples,
    "adaptive_sampling": False,
    "denoising": False,
    "render_seconds": seconds,
    "mean_rgb": mean_rgb,
    "all_pixels_finite": True,
    "scope": "Thin Wall appearance and native-limit comparison fixture",
}, indent=2) + "\n")
