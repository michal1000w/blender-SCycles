# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Metal kernel compilation and light-cache start-up benchmark.

Run with the installed build, for example:

  install/Blender.app/Contents/MacOS/Blender --background --factory-startup \
      --python tests/python/cycles_metal_compile_benchmark.py -- \
      --mode bdpt --report /tmp/report.json

It builds a small scene with diffuse, glossy, glass and emissive materials,
renders it once (cold or warm, depending on the caches) and then performs a
series of scene edits (material colour, camera motion, light strength, sample
count, bounce count) rendering after each one. Every render is timed so that a
recompilation caused by a scene edit shows up as a large time for that step.
"""

import argparse
import json
import math
import sys
import time

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="pt",
                        choices=("pt", "bdpt", "pm", "guiding", "bdpt_guiding", "cpu_bdpt"))
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--edits", default="all",
                        help="Comma separated list of edits, 'all' or 'none'")
    parser.add_argument("--volume", action="store_true")
    parser.add_argument("--displacement", action="store_true",
                        help="Displace the floor, which uses pixel level displacement")
    parser.add_argument("--pause", type=float, default=0.0,
                        help="Seconds to wait after the first render, as a user editing the scene")
    parser.add_argument("--report", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args(argv)


def enable_metal():
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    names = []
    for device in prefs.devices:
        device.use = device.type == "METAL"
        if device.use:
            names.append(device.name)
    return names


def material(name, kind, color=(0.8, 0.8, 0.8, 1.0), strength=10.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    if kind == "diffuse":
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.6
    elif kind == "glossy":
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Metallic"].default_value = 1.0
        bsdf.inputs["Roughness"].default_value = 0.15
    elif kind == "glass":
        bsdf = nodes.new("ShaderNodeBsdfGlass")
        bsdf.inputs["Roughness"].default_value = 0.0
        bsdf.inputs["IOR"].default_value = 1.5
    elif kind == "emission":
        bsdf = nodes.new("ShaderNodeEmission")
        bsdf.inputs["Color"].default_value = color
        bsdf.inputs["Strength"].default_value = strength
    else:
        raise ValueError(kind)
    links.new(bsdf.outputs[0], out.inputs["Surface"])
    return mat


def build_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    cycles = scene.cycles
    cycles.device = "CPU" if args.mode == "cpu_bdpt" else "GPU"
    cycles.samples = args.samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.seed = 1
    cycles.max_bounces = 8

    if args.mode in ("bdpt", "bdpt_guiding", "cpu_bdpt"):
        cycles.use_bidirectional_path_tracing = True
    if args.mode == "pm":
        cycles.use_photon_mapping = True
    if args.mode in ("guiding", "bdpt_guiding"):
        cycles.use_guiding = True

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.03, 1.0)
    scene.world = world

    bpy.ops.mesh.primitive_plane_add(size=6.0)
    floor = bpy.context.object
    floor_material = material("Floor", "diffuse", (0.7, 0.7, 0.7, 1.0))
    floor.data.materials.append(floor_material)
    if args.displacement:
        nodes = floor_material.node_tree.nodes
        links = floor_material.node_tree.links
        noise = nodes.new("ShaderNodeTexNoise")
        noise.name = "Displacement Noise"
        noise.inputs["Scale"].default_value = 4.0
        displacement = nodes.new("ShaderNodeDisplacement")
        displacement.name = "Displacement"
        displacement.inputs["Scale"].default_value = 0.2
        links.new(noise.outputs["Fac"], displacement.inputs["Height"])
        links.new(displacement.outputs[0], nodes["Material Output"].inputs["Displacement"])
        floor_material.displacement_method = "DISPLACEMENT"

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-1.1, 0.0, 0.5))
    bpy.context.object.data.materials.append(material("Metal", "glossy", (0.9, 0.6, 0.3, 1.0)))

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.6, location=(0.9, 0.0, 0.6),
                                         segments=48, ring_count=24)
    bpy.ops.object.shade_smooth()
    bpy.context.object.data.materials.append(material("Glass", "glass"))

    bpy.ops.mesh.primitive_plane_add(size=0.8, location=(0.0, 0.0, 3.0))
    lamp = bpy.context.object
    lamp.rotation_euler = (math.pi, 0.0, 0.0)
    lamp.data.materials.append(material("Lamp", "emission", (1.0, 0.95, 0.9, 1.0), 20.0))

    bpy.ops.object.light_add(type="POINT", location=(2.0, -2.0, 2.5))
    light = bpy.context.object
    light.data.energy = 200.0
    light.data.shadow_soft_size = 0.1

    if args.volume:
        bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 1.5, 1.0))
        vol = bpy.data.materials.new("Fog")
        vol.use_nodes = True
        nodes = vol.node_tree.nodes
        nodes.remove(nodes["Principled BSDF"])
        scatter = nodes.new("ShaderNodeVolumeScatter")
        scatter.inputs["Density"].default_value = 0.3
        vol.node_tree.links.new(scatter.outputs[0], nodes["Material Output"].inputs["Volume"])
        bpy.context.object.data.materials.append(vol)

    bpy.ops.object.camera_add(location=(0.0, -5.5, 2.2), rotation=(math.radians(72), 0.0, 0.0))
    scene.camera = bpy.context.object
    return scene


def render(scene, output):
    scene.render.filepath = output
    start = time.perf_counter()
    bpy.ops.render.render(write_still=bool(output))
    return time.perf_counter() - start


def edit_material_color(scene):
    bsdf = bpy.data.materials["Floor"].node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.3, 0.5, 0.8, 1.0)


def edit_camera(scene):
    scene.camera.location.x += 0.35
    scene.camera.rotation_euler.z += math.radians(4.0)


def edit_light(scene):
    bpy.data.lights[0].energy = 350.0
    bpy.data.materials["Lamp"].node_tree.nodes["Emission"].inputs["Strength"].default_value = 12.0


def edit_samples(scene):
    scene.cycles.samples = scene.cycles.samples + 4


def edit_bounces(scene):
    scene.cycles.max_bounces = 5
    scene.cycles.glossy_bounces = 3


def edit_roughness(scene):
    bsdf = bpy.data.materials["Metal"].node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.4


def edit_new_node(scene):
    """Add a node type that was not used before (a noise texture)."""
    mat = bpy.data.materials["Floor"]
    nodes = mat.node_tree.nodes
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 8.0
    mat.node_tree.links.new(noise.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])


def edit_displacement(scene):
    nodes = bpy.data.materials["Floor"].node_tree.nodes
    if "Displacement" in nodes:
        nodes["Displacement"].inputs["Scale"].default_value = 0.35
        nodes["Displacement Noise"].inputs["Scale"].default_value = 6.0


def edit_resolution(scene):
    scene.render.resolution_x = scene.render.resolution_x + 32


def edit_enable_bdpt(scene):
    scene.cycles.use_bidirectional_path_tracing = True


def edit_enable_photons(scene):
    scene.cycles.use_bidirectional_path_tracing = False
    scene.cycles.use_photon_mapping = True


def edit_enable_guiding(scene):
    scene.cycles.use_photon_mapping = False
    scene.cycles.use_guiding = True


def edit_disable_all(scene):
    scene.cycles.use_bidirectional_path_tracing = False
    scene.cycles.use_photon_mapping = False
    scene.cycles.use_guiding = False


EDITS = {
    "material_color": edit_material_color,
    "camera": edit_camera,
    "light": edit_light,
    "samples": edit_samples,
    "bounces": edit_bounces,
    "roughness": edit_roughness,
    "new_node": edit_new_node,
    "resolution": edit_resolution,
    "displacement": edit_displacement,
    "enable_bdpt": edit_enable_bdpt,
    "enable_photons": edit_enable_photons,
    "enable_guiding": edit_enable_guiding,
    "disable_all": edit_disable_all,
}


def main():
    args = parse_args()
    scene = build_scene(args)
    devices = enable_metal() if args.mode != "cpu_bdpt" else []
    report = {
        "mode": args.mode,
        "devices": devices,
        "resolution": args.resolution,
        "samples": args.samples,
        "volume": args.volume,
        "renders": [],
    }

    t = render(scene, args.output)
    report["renders"].append({"step": "initial", "seconds": t})
    print(f"BENCH initial {t:.2f}s", flush=True)
    if args.pause > 0.0:
        time.sleep(args.pause)

    if args.edits == "all":
        edits = [name for name in EDITS if not name.startswith(("enable_", "disable_"))]
    elif args.edits == "transport":
        edits = ["enable_bdpt", "enable_photons", "enable_guiding", "disable_all"]
    elif args.edits == "none":
        edits = []
    else:
        edits = args.edits.split(",")
    for name in edits:
        EDITS[name](scene)
        t = render(scene, "")
        report["renders"].append({"step": name, "seconds": t})
        print(f"BENCH {name} {t:.2f}s", flush=True)

    t = render(scene, "")
    report["renders"].append({"step": "repeat", "seconds": t})
    print(f"BENCH repeat {t:.2f}s", flush=True)

    if args.report:
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)


main()
