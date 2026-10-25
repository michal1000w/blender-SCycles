# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""One light of each kind above a diffuse floor, rendered with path tracing and BDPT, to check
that BDPT converges to the same image for every light type. Run inside Blender:

  blender -b --factory-startup --python-exit-code 1 \
      --python tests/python/cycles_bdpt_light_scenes.py -- --output DIRECTORY \
      [--device CPU|METAL] [--scenes a,b] [--samples N] [--resolution N] [--guiding] \
      [--integrators pt,bdpt,vcm,photon] [--tolerance 0.02]

Renders <scene>_<integrator>.exr and prints the ratio of every integrator's image mean to path
tracing,
exits with 1 when a scene is off by more than the tolerance.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="CPU", choices=("CPU", "METAL"))
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--guiding", action="store_true")
    parser.add_argument("--integrators", default="pt,bdpt")
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--bounces", type=int, default=4)
    parser.add_argument("--clip-start", type=float, default=0.1)
    parser.add_argument("--no-light-tree", action="store_true")
    parser.add_argument("--no-polarization", action="store_true")
    return parser.parse_args(argv)


def look_at(obj, target):
    from mathutils import Vector
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def setup(args, integrator):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.view_settings.view_transform = "Standard"
    cycles = scene.cycles
    cycles.device = "GPU" if args.device == "METAL" else "CPU"
    cycles.samples = args.samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.seed = args.seed
    cycles.max_bounces = args.bounces
    cycles.diffuse_bounces = args.bounces
    cycles.glossy_bounces = args.bounces
    cycles.sample_clamp_direct = 0.0
    cycles.sample_clamp_indirect = 0.0
    # Filter Glossy blurs glass after diffuse bounces of camera paths only: biased by design.
    cycles.blur_glossy = 0.0
    cycles.use_bidirectional_path_tracing = integrator == "bdpt"
    cycles.use_light_tree = not args.no_light_tree
    if args.no_polarization:
        cycles.use_polarization = False
    if integrator == "vcm":
        cycles.use_vertex_merging = True
    if integrator == "photon":
        cycles.use_photon_mapping = True
    if args.guiding:
        cycles.use_guiding = True
        cycles.use_deterministic_guiding = True
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.world = world
    # Floor.
    bpy.ops.mesh.primitive_plane_add(size=4.0)
    floor = bpy.context.object
    mat = bpy.data.materials.new("Floor")
    mat.use_nodes = True
    mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1)
    mat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 1.0
    floor.data.materials.append(mat)
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.clip_start = args.clip_start
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = (0.0, -3.0, 2.0)
    look_at(cam, (0.0, 0.0, 0.0))
    scene.camera = cam
    return scene


def add_lamp(scene, kind, location, energy, **props):
    data = bpy.data.lights.new(kind, kind)
    data.energy = energy
    for key, value in props.items():
        setattr(data, key, value)
    obj = bpy.data.objects.new(kind, data)
    scene.collection.objects.link(obj)
    obj.location = location
    return obj


def scene_point(scene):
    add_lamp(scene, "POINT", (0.0, 0.0, 2.0), 200.0, shadow_soft_size=0.15)


def scene_spot(scene):
    lamp = add_lamp(scene, "SPOT", (0.0, 0.0, 2.0), 300.0, shadow_soft_size=0.1,
                    spot_size=math.radians(70.0))
    look_at(lamp, (0.0, 0.3, 0.0))


def scene_area_square(scene):
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.3)


def scene_area_disk(scene):
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.3, shape="DISK")


def scene_area_rect(scene):
    lamp = add_lamp(scene, "AREA", (0.3, 0.2, 1.5), 200.0, size=0.6, size_y=0.2,
                    shape="RECTANGLE")
    look_at(lamp, (0.0, 0.0, 0.0))


def scene_area_large(scene):
    add_lamp(scene, "AREA", (0.0, 0.0, 1.5), 300.0, size=1.5)


def scene_area_spread(scene):
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.5, spread=math.radians(60.0))


def scene_area_offset(scene):
    lamp = add_lamp(scene, "AREA", (0.0, -1.0, 2.0), 200.0, size=0.3)
    look_at(lamp, (0.0, 0.4, 0.5))


def diffuse_floor(scene):
    floor = bpy.data.objects["Plane"]
    mat = bpy.data.materials.new("Diffuse")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.7, 0.7, 0.7, 1.0)
    mat.node_tree.links.new(diffuse.outputs[0], nodes["Material Output"].inputs["Surface"])
    floor.data.materials[0] = mat


def scene_area_diffuse(scene):
    diffuse_floor(scene)
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.3)


def scene_point_diffuse(scene):
    diffuse_floor(scene)
    add_lamp(scene, "POINT", (0.0, 0.0, 2.0), 200.0, shadow_soft_size=0.15)


def wide_camera(scene, target=(0.0, 0.2, 0.3), lens=35.0):
    scene.camera.data.lens = lens
    look_at(scene.camera, target)


def scene_area_edge(scene):
    """Area light just above the top edge of the frame, seen from behind (the case of an 11%
    too dark BDPT found with the diffraction test scenes)."""
    diffuse_floor(scene)
    wide_camera(scene)
    lamp = add_lamp(scene, "AREA", (0.0, -1.0, 2.0), 200.0, size=0.3)
    look_at(lamp, (0.0, 0.4, 0.5))


def scene_area_edge_narrow(scene):
    """Same, with a narrower lens: the light is outside the frame."""
    diffuse_floor(scene)
    wide_camera(scene, lens=50.0)
    lamp = add_lamp(scene, "AREA", (0.0, -1.0, 2.0), 200.0, size=0.3)
    look_at(lamp, (0.0, 0.4, 0.5))


def scene_point_edge(scene):
    diffuse_floor(scene)
    wide_camera(scene)
    add_lamp(scene, "POINT", (0.0, -1.0, 2.0), 200.0, shadow_soft_size=0.15)


def scene_area_spread_narrow(scene):
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.5, spread=math.radians(10.0))


def scene_area_spread_zero(scene):
    lamp = add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.5, spread=0.0)
    lamp.rotation_euler = (math.radians(10.0), 0.0, 0.0)


def glass_sphere(roughness=0.0, ior=1.5):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(0.0, 0.0, 0.6), segments=48,
                                         ring_count=24)
    sphere = bpy.context.object
    bpy.ops.object.shade_smooth()
    mat = bpy.data.materials.new("Glass")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["IOR"].default_value = ior
    glass.inputs["Roughness"].default_value = roughness
    mat.node_tree.links.new(glass.outputs[0], nodes["Material Output"].inputs["Surface"])
    sphere.data.materials.append(mat)


def scene_caustic_area(scene):
    glass_sphere()
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.6)


def scene_caustic_spread(scene):
    glass_sphere()
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.6, spread=math.radians(40.0))


def scene_caustic_area_rough(scene):
    glass_sphere(0.08)
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.6)


def scene_caustic_ior1(scene):
    glass_sphere(0.08, 1.001)
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.6)


def scene_caustic_ior2(scene):
    glass_sphere(0.08, 2.0)
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.6)


def scene_caustic_area_opaque(scene):
    """The caustic_area scene with a black sphere: its shadow is where only caustics arrive."""
    glass_sphere()
    nodes = bpy.data.materials["Glass"].node_tree.nodes
    nodes.remove(nodes["Glass BSDF"])
    add_lamp(scene, "AREA", (0.0, 0.0, 2.0), 200.0, size=0.6)


def scene_caustic_sphere_light(scene):
    glass_sphere()
    lamp = add_lamp(scene, "POINT", (0.0, 0.0, 2.0), 200.0, shadow_soft_size=0.3)
    lamp.data.use_soft_falloff = False


def scene_caustic_point(scene):
    glass_sphere()
    add_lamp(scene, "POINT", (0.0, 0.0, 2.0), 200.0, shadow_soft_size=0.3)


def scene_caustic_mesh(scene):
    glass_sphere()
    scene_mesh(scene)
    bpy.data.objects["Plane.001"].scale = (2.0, 2.0, 1.0)


def scene_sun(scene):
    lamp = add_lamp(scene, "SUN", (0.0, 0.0, 3.0), 3.0, angle=math.radians(5.0))
    lamp.rotation_euler = (math.radians(20.0), 0.0, 0.0)


def scene_mesh(scene):
    bpy.ops.mesh.primitive_plane_add(size=0.3, location=(0.0, 0.0, 2.0))
    emitter = bpy.context.object
    emitter.rotation_euler = (math.pi, 0.0, 0.0)  # Facing down.
    mat = bpy.data.materials.new("Emitter")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.remove(nodes["Principled BSDF"])
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 100.0
    mat.node_tree.links.new(emission.outputs[0], nodes["Material Output"].inputs["Surface"])
    emitter.data.materials.append(mat)


def scene_world(scene):
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0
    # A wall so that the world is not all the camera sees.
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0.0, 2.0, 2.0),
                                     rotation=(math.pi / 2, 0.0, 0.0))


SCENES = {
    "point": scene_point,
    "spot": scene_spot,
    "area_square": scene_area_square,
    "area_disk": scene_area_disk,
    "area_rect": scene_area_rect,
    "area_large": scene_area_large,
    "area_spread": scene_area_spread,
    "area_spread_narrow": scene_area_spread_narrow,
    "area_spread_zero": scene_area_spread_zero,
    "area_offset": scene_area_offset,
    "area_edge": scene_area_edge,
    "area_edge_narrow": scene_area_edge_narrow,
    "point_edge": scene_point_edge,
    "area_diffuse": scene_area_diffuse,
    "point_diffuse": scene_point_diffuse,
    "caustic_area": scene_caustic_area,
    "caustic_spread": scene_caustic_spread,
    "caustic_point": scene_caustic_point,
    "caustic_area_opaque": scene_caustic_area_opaque,
    "caustic_area_rough": scene_caustic_area_rough,
    "caustic_ior1": scene_caustic_ior1,
    "caustic_ior2": scene_caustic_ior2,
    "caustic_sphere_light": scene_caustic_sphere_light,
    "caustic_mesh": scene_caustic_mesh,
    "sun": scene_sun,
    "mesh": scene_mesh,
    "world": scene_world,
}


def enable_device(args):
    if args.device != "METAL":
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "METAL"


def load(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.reshape(-1, 4)[:, :3].astype(np.float64)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = list(SCENES) if args.scenes == "all" else args.scenes.split(",")
    integrators = args.integrators.split(",")
    report, failed = {}, []
    for name in names:
        means = {}
        for integrator in integrators:
            scene = setup(args, integrator)
            SCENES[name](scene)
            enable_device(args)
            path = args.output / f"{name}_{integrator}.exr"
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            means[integrator] = float(load(path).mean())
        entry = {"means": means}
        for integrator in integrators:
            if integrator == "pt" or "pt" not in means:
                continue
            ratio = means[integrator] / max(means["pt"], 1e-12)
            entry.setdefault("ratios", {})[integrator] = ratio
            if integrator == "bdpt":
                entry["ratio"] = ratio
            ok = abs(ratio - 1.0) <= args.tolerance
            entry["pass"] = entry.get("pass", True) and ok
            if not ok:
                failed.append(f"{name}:{integrator}")
            print(f"RESULT {name:12s} {integrator}/pt {ratio:.4f} {'ok' if ok else 'FAIL'}",
                  flush=True)
        report[name] = entry
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print("FAILED: " + ",".join(failed) if failed else "ALL PASS", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
