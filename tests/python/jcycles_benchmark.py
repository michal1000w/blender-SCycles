# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Render time comparison of standard Cycles scenes, to compare this fork with an official Blender:
run the same command with both and compare the times. The scenes use only standard Cycles
features, so official Blender renders them the same way.

  blender --background --factory-startup --python tests/python/jcycles_benchmark.py -- \
      [--devices CPU,CUDA,OPTIX] [--scenes a,b] [--samples N] [--resolution N] [--repeat N]

Prints one line per scene and device: the time of the first render, which includes loading the
kernels and building the BVH, and the best time of the following renders.
"""

import argparse
import math
import sys
import time

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="CPU,CUDA,OPTIX")
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--repeat", type=int, default=2)
    return parser.parse_args(argv)


def reset_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution * 9 // 16
    scene.render.resolution_percentage = 100
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.image_settings.file_format = "PNG"
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.3
    bpy.ops.object.camera_add(location=(0.0, -9.0, 3.5), rotation=(math.radians(75), 0, 0))
    scene.camera = bpy.context.object
    return scene


def material(name, color, roughness=0.5, metallic=0.0, transmission=0.0):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Transmission Weight"].default_value = transmission
    return mat


def add(op, mat, **kwargs):
    op(**kwargs)
    obj = bpy.context.object
    obj.data.materials.append(mat)
    return obj


def scene_interior(args):
    """Closed room lit by an area light: diffuse interreflection, the common case."""
    scene = reset_scene(args)
    walls = material("Walls", (0.8, 0.8, 0.8, 1))
    for loc, rot in (((0, 0, 0), (0, 0, 0)), ((0, 0, 6), (math.pi, 0, 0)),
                     ((0, 4, 3), (math.radians(90), 0, 0)), ((-4, 0, 3), (0, math.radians(90), 0)),
                     ((4, 0, 3), (0, math.radians(-90), 0))):
        add(bpy.ops.mesh.primitive_plane_add, walls, size=8, location=loc, rotation=rot)
    red = material("Red", (0.8, 0.1, 0.1, 1), 0.3)
    gold = material("Gold", (1.0, 0.7, 0.3, 1), 0.25, 1.0)
    for i, mat in enumerate((red, gold, walls)):
        obj = add(bpy.ops.mesh.primitive_monkey_add, mat, location=(-2.2 + 2.2 * i, 0.5, 1.2))
        obj.modifiers.new("Subdivision", "SUBSURF").levels = 2
        bpy.ops.object.shade_smooth()
    bpy.ops.object.light_add(type="AREA", location=(0, 0, 5.8))
    bpy.context.object.data.energy = 1500
    bpy.context.object.data.size = 2
    return scene


def scene_glass(args):
    """Glass and metal objects outdoors: deep specular paths."""
    scene = reset_scene(args)
    add(bpy.ops.mesh.primitive_plane_add, material("Ground", (0.5, 0.5, 0.5, 1), 0.6), size=40)
    glass = material("Glass", (1, 1, 1, 1), 0.0, 0.0, 1.0)
    chrome = material("Chrome", (0.9, 0.9, 0.9, 1), 0.05, 1.0)
    for i in range(5):
        mat = glass if i % 2 == 0 else chrome
        obj = add(bpy.ops.mesh.primitive_uv_sphere_add, mat, radius=0.8,
                  location=(-4 + 2 * i, 0, 0.8), segments=64, ring_count=32)
        bpy.ops.object.shade_smooth()
    bpy.ops.object.light_add(type="SUN", rotation=(math.radians(40), 0, math.radians(30)))
    bpy.context.object.data.energy = 4
    return scene


def scene_volume(args):
    """Scattering volume lit by a spot light."""
    scene = reset_scene(args)
    add(bpy.ops.mesh.primitive_plane_add, material("Ground", (0.6, 0.6, 0.6, 1)), size=30)
    mat = bpy.data.materials.new("Fog")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    volume = nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = 0.2
    mat.node_tree.links.new(volume.outputs[0], nodes["Material Output"].inputs["Volume"])
    add(bpy.ops.mesh.primitive_cube_add, mat, size=6, location=(0, 0, 3))
    add(bpy.ops.mesh.primitive_monkey_add, material("Red", (0.8, 0.1, 0.1, 1)), location=(0, 0, 1))
    bpy.ops.object.light_add(type="SPOT", location=(0, -2, 6), rotation=(math.radians(20), 0, 0))
    bpy.context.object.data.energy = 3000
    return scene


SCENES = {"interior": scene_interior, "glass": scene_glass, "volume": scene_volume}


def set_device(scene, device):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device == "CPU":
        scene.cycles.device = "CPU"
        return True
    prefs.compute_device_type = device
    prefs.refresh_devices()
    found = False
    for d in prefs.devices:
        d.use = d.type == device
        found |= d.use
    scene.cycles.device = "GPU"
    return found


def main():
    args = parse_args()
    names = list(SCENES) if args.scenes == "all" else args.scenes.split(",")
    print(f"[benchmark] Blender {bpy.app.version_string}, {args.resolution}px, "
          f"{args.samples} samples", flush=True)
    for name in names:
        for device in args.devices.split(","):
            times = []
            for _ in range(args.repeat + 1):
                scene = SCENES[name](args)
                if not set_device(scene, device):
                    break
                scene.render.filepath = bpy.app.tempdir + f"bench_{name}_{device}.png"
                start = time.perf_counter()
                bpy.ops.render.render(write_still=True)
                times.append(time.perf_counter() - start)
            if not times:
                print(f"[benchmark] {name:9s} {device:6s} no device", flush=True)
                continue
            best = min(times[1:]) if len(times) > 1 else times[0]
            print(f"[benchmark] {name:9s} {device:6s} first {times[0]:7.2f} s  "
                  f"best {best:7.2f} s", flush=True)


main()
