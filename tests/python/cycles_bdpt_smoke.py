#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone Metal smoke and caustic-regression scene for Cycles BDPT."""

import argparse
import math
import sys
import time

import bpy
from mathutils import Vector


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def configure_metal():
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "METAL"
    preferences.get_devices()
    found = False
    for device in preferences.devices:
        device.use = device.type == "METAL"
        if device.use:
            found = True
    if not found:
        raise RuntimeError("No Metal Cycles device available")


def principled_material(
    name,
    base_color,
    transmission=0.0,
    roughness=0.4,
    ior=1.5,
    dispersion=0.0,
    subsurface=0.0,
):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    transmission_input = bsdf.inputs.get("Transmission Weight") or bsdf.inputs.get("Transmission")
    transmission_input.default_value = transmission
    if subsurface > 0.0:
        bsdf.inputs["Subsurface Weight"].default_value = subsurface
        bsdf.inputs["Subsurface Radius"].default_value = (0.65, 0.32, 0.16)
        bsdf.inputs["Subsurface Scale"].default_value = 0.45
    if dispersion > 0.0:
        bsdf.inputs["Transmission Dispersion Scale"].default_value = dispersion
        bsdf.inputs["Transmission Dispersion Abbe Number"].default_value = 9.0
    return material


def emission_material(name, strength):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    emission.inputs["Strength"].default_value = strength
    material.node_tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def diffuse_material(name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (*color, 1.0)
    diffuse.inputs["Roughness"].default_value = 0.0
    material.node_tree.links.new(diffuse.outputs["BSDF"], output.inputs["Surface"])
    return material


def hair_material(name, color):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    hair = nodes.new("ShaderNodeBsdfHair")
    hair.inputs["Color"].default_value = (*color, 1.0)
    hair.inputs["RoughnessU"].default_value = 0.35
    hair.inputs["RoughnessV"].default_value = 0.35
    material.node_tree.links.new(hair.outputs["BSDF"], output.inputs["Surface"])
    return material


def ray_portal_material(name, position, direction):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    portal = nodes.new("ShaderNodeBsdfRayPortal")
    portal.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    portal.inputs["Position"].default_value = position
    portal.inputs["Direction"].default_value = direction
    material.node_tree.links.new(portal.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_scene(options):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if options.device == "GPU":
        configure_metal()

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = options.device
    scene.cycles.samples = options.samples
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = options.adaptive
    if options.adaptive:
        scene.cycles.adaptive_threshold = 0.01
        scene.cycles.adaptive_min_samples = 16
    scene.cycles.use_photon_mapping = False
    scene.cycles.use_light_tree = not options.no_light_tree
    scene.cycles.use_bidirectional_path_tracing = options.bdpt
    scene.cycles.bdpt_light_paths = options.light_paths
    scene.cycles.bdpt_max_bounces = options.max_bounces
    scene.cycles.bdpt_update_samples = options.update_samples
    scene.cycles.max_bounces = 10
    scene.cycles.blur_glossy = 0.0
    scene.cycles.pixel_filter_type = "BOX"
    scene.render.resolution_x = options.resolution
    scene.render.resolution_y = options.resolution
    scene.render.resolution_percentage = 100
    if options.crop_border:
        scene.render.use_border = True
        scene.render.use_crop_to_border = True
        scene.render.border_min_x = 0.25
        scene.render.border_max_x = 0.75
        scene.render.border_min_y = 0.25
        scene.render.border_max_y = 0.75
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.filepath = options.output

    world = bpy.data.worlds.new("Black World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = (
        0.8 if options.light_type == "WORLD" else 0.0
    )
    if options.world_absorption > 0.0:
        absorption = world.node_tree.nodes.new("ShaderNodeVolumeAbsorption")
        absorption.inputs["Color"].default_value = (0.72, 0.82, 0.95, 1.0)
        absorption.inputs["Density"].default_value = options.world_absorption
        world.node_tree.links.new(
            absorption.outputs["Volume"], world.node_tree.nodes["World Output"].inputs["Volume"]
        )
    if options.world_scatter > 0.0:
        scatter = world.node_tree.nodes.new("ShaderNodeVolumeScatter")
        scatter.inputs["Color"].default_value = (0.72, 0.82, 0.95, 1.0)
        scatter.inputs["Density"].default_value = options.world_scatter
        scatter.inputs["Anisotropy"].default_value = 0.35
        world.node_tree.links.new(
            scatter.outputs["Volume"], world.node_tree.nodes["World Output"].inputs["Volume"]
        )
    scene.world = world

    floor = None
    if not options.volume_only and not options.no_floor:
        bpy.ops.mesh.primitive_plane_add(size=12.0, location=(0.0, 0.0, 0.0))
        floor = bpy.context.object
        if options.hair_bsdf:
            floor.data.materials.append(hair_material("Hair Closure Receiver", (0.65, 0.68, 0.72)))
        else:
            floor.data.materials.append(
                diffuse_material("Diffuse Receiver", (0.65, 0.68, 0.72))
                if options.diffuse_node
                else principled_material("Diffuse Receiver", (0.65, 0.68, 0.72))
            )

    if options.ray_portal:
        bpy.ops.mesh.primitive_plane_add(size=7.0, location=(0.0, 0.0, 2.7))
        portal_object = bpy.context.object
        portal_object.name = "BDPT Ray Portal"
        portal_object.visible_camera = False
        portal_object.data.materials.append(
            ray_portal_material("BDPT Ray Portal", (0.0, 0.0, 2.55), (0.0, 0.0, -1.0))
        )

    blocker = None
    if options.shadow_link_blocker:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=32, ring_count=16, radius=0.7, location=(-0.85, -0.6, 2.15)
        )
        blocker = bpy.context.object
        blocker.data.materials.append(diffuse_material("Shadow Blocker", (0.3, 0.3, 0.3)))

    if options.backdrop:
        bpy.ops.mesh.primitive_plane_add(
            size=12.0, location=(0.0, 3.0, 3.0), rotation=(math.pi * 0.5, 0.0, 0.0)
        )
        backdrop = bpy.context.object
        backdrop.data.materials.append(
            principled_material("Diffuse Backdrop", (0.45, 0.5, 0.58), roughness=0.55)
        )

    if options.subsurface and not options.volume_only:
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=1.0, location=(0.0, 0.0, 1.25))
        subsurface_object = bpy.context.object
        subsurface_object.data.materials.append(
            principled_material(
                "Subsurface Receiver", (0.72, 0.22, 0.08), roughness=0.42, subsurface=1.0
            )
        )
    elif not options.no_glass and not options.volume_only:
        bpy.ops.mesh.primitive_uv_sphere_add(
            segments=64, ring_count=32, radius=1.0, location=(0.0, 0.0, 1.25)
        )
        glass = bpy.context.object
        glass.data.materials.append(
            principled_material(
                "Glass Caster",
                (1.0, 1.0, 1.0),
                transmission=1.0,
                roughness=0.0,
                ior=1.52,
                dispersion=options.dispersion,
            )
        )

    light = None
    light_type = "MESH" if options.mesh_light else options.light_type
    if light_type == "MESH":
        bpy.ops.mesh.primitive_plane_add(size=0.7, location=(-1.7, -1.2, 4.5))
        light = bpy.context.object
        light.rotation_euler = (
            Vector((0.0, 0.0, 0.5)) - light.location
        ).to_track_quat("Z", "Y").to_euler()
        light.data.materials.append(emission_material("Mesh Emitter", 250.0))
    elif light_type != "WORLD":
        bpy.ops.object.light_add(type=light_type, location=(-1.7, -1.2, 4.5))
        light = bpy.context.object
        if light_type == "AREA":
            light.data.energy = 1100.0
            light.data.shape = "DISK"
            light.data.size = 0.35
            look_at(light, (0.0, 0.0, 0.5))
        elif light_type == "POINT":
            light.data.energy = 350.0
            light.data.shadow_soft_size = options.light_radius
            light.data.use_soft_falloff = not options.physical_sphere
        elif light_type == "SPOT":
            light.data.energy = 900.0
            light.data.shadow_soft_size = options.light_radius
            light.data.use_soft_falloff = not options.physical_sphere
            light.data.spot_size = math.radians(38.0)
            light.data.spot_blend = 0.35
            look_at(light, (0.0, 0.0, 0.5))
        elif light_type == "SUN":
            light.data.energy = 2.0
            light.data.angle = math.radians(options.sun_angle)
            look_at(light, (0.0, 0.0, 0.5))

    if light is not None:
        if options.light_max_bounces >= 0:
            light.data.cycles.max_bounces = options.light_max_bounces
        if options.light_link != "ALL":
            receivers = bpy.data.collections.new("BDPT Test Light Receivers")
            scene.collection.children.link(receivers)
            if options.light_link == "FLOOR" and floor is not None:
                receivers.objects.link(floor)
            else:
                dummy = bpy.data.objects.new("Unlit Receiver", None)
                receivers.objects.link(dummy)
            light.light_linking.receiver_collection = receivers
        if blocker is not None:
            blockers = bpy.data.collections.new("BDPT Test Shadow Blockers")
            scene.collection.children.link(blockers)
            blockers.objects.link(blocker)
            light.light_linking.blocker_collection = blockers

    camera_location = (0.0, 0.0, 6.0) if options.top_camera else (5.8, -6.2, 4.3)
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.data.type = options.camera_type
    if options.camera_type == "ORTHO":
        camera.data.ortho_scale = options.ortho_scale
    camera.data.lens = options.lens
    camera.data.dof.use_dof = options.aperture > 0.0
    camera.data.dof.aperture_fstop = options.aperture if options.aperture > 0.0 else 2.8
    look_at(camera, (0.0, 0.0, 0.0) if options.top_camera else (0.0, 0.2, 0.7))
    if options.view_glass:
        if options.top_camera:
            raise RuntimeError("--view-glass requires the perspective test camera")
        pane_target = (0.0, 0.2, 0.7)
        pane_location = tuple(
            (camera_location[index] + pane_target[index]) * 0.5 for index in range(3)
        )
        half_size = 2.4
        half_thickness = 0.035
        vertices = [
            (-half_size, -half_size, -half_thickness),
            (half_size, -half_size, -half_thickness),
            (half_size, half_size, -half_thickness),
            (-half_size, half_size, -half_thickness),
            (-half_size, -half_size, half_thickness),
            (half_size, -half_size, half_thickness),
            (half_size, half_size, half_thickness),
            (-half_size, half_size, half_thickness),
        ]
        pane_mesh = bpy.data.meshes.new("Camera Glass Pane Mesh")
        # Disconnected parallel interfaces give each triangle a smooth, constant differential
        # normal field without the edge-normal averaging of a smooth-shaded cube.
        pane_mesh.from_pydata(vertices, [], [(0, 3, 2, 1), (4, 5, 6, 7)])
        pane_mesh.update()
        for polygon in pane_mesh.polygons:
            polygon.use_smooth = True
        pane = bpy.data.objects.new("Camera Glass Pane", pane_mesh)
        scene.collection.objects.link(pane)
        pane.location = pane_location
        pane.name = "Camera Glass Pane"
        look_at(pane, pane_target)
        pane.data.materials.append(
            principled_material(
                "Camera Glass Pane",
                (1.0, 1.0, 1.0),
                transmission=1.0,
                roughness=0.0,
                ior=1.45,
            )
        )
    if options.camera_motion:
        scene.render.use_motion_blur = True
        scene.frame_set(0)
        camera.location.x -= 0.45
        camera.data.lens = options.lens * 0.85
        look_at(camera, (0.0, 0.2, 0.7))
        camera.keyframe_insert("location")
        camera.keyframe_insert("rotation_euler")
        camera.data.keyframe_insert("lens")
        scene.frame_set(2)
        camera.location.x += 0.9
        camera.data.lens = options.lens * 1.15
        look_at(camera, (0.0, 0.2, 0.7))
        camera.keyframe_insert("location")
        camera.keyframe_insert("rotation_euler")
        camera.data.keyframe_insert("lens")
        scene.frame_set(1)
    scene.camera = camera

    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - start

    image = bpy.data.images.load(options.output, check_existing=False)
    pixels = list(image.pixels)
    rgb = [pixels[index] for index in range(len(pixels)) if index % 4 != 3]
    if not rgb or not all(math.isfinite(value) for value in rgb):
        raise RuntimeError("Render contains non-finite values")
    if max(rgb) <= 0.0 and not options.allow_empty:
        raise RuntimeError("Render contains no light")
    ordered = sorted(rgb)
    colors = [tuple(pixels[index : index + 3]) for index in range(0, len(pixels), 4)]
    bright_colors = sorted(colors, key=sum)[-max(1, len(colors) // 100) :]
    top_chroma = sum(
        (max(color) - min(color)) / (sum(color) / 3.0 + 1.0e-12) for color in bright_colors
    ) / len(bright_colors)
    percentile_99 = ordered[min(int(0.99 * len(ordered)), len(ordered) - 1)]
    percentile_999 = ordered[min(int(0.999 * len(ordered)), len(ordered) - 1)]
    print(
        "BDPT_SMOKE "
        f"bdpt={int(options.bdpt)} camera={options.camera_type} light={light_type} "
        f"seconds={elapsed:.9g} "
        f"samples={options.samples} "
        f"light_paths={options.light_paths} mean={sum(rgb) / len(rgb):.9g} "
        f"p99={percentile_99:.9g} p999={percentile_999:.9g} peak={max(rgb):.9g} "
        f"top_chroma={top_chroma:.9g}"
    )


def main():
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--bdpt", action="store_true")
    parser.add_argument("--device", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--light-paths", type=int, default=16384)
    parser.add_argument("--update-samples", type=int, default=4)
    parser.add_argument("--max-bounces", type=int, default=8)
    parser.add_argument("--backdrop", action="store_true")
    parser.add_argument("--no-glass", action="store_true")
    parser.add_argument("--view-glass", action="store_true")
    parser.add_argument("--dispersion", type=float, default=0.0)
    parser.add_argument("--no-light-tree", action="store_true")
    parser.add_argument("--aperture", type=float, default=0.0)
    parser.add_argument("--mesh-light", action="store_true")
    parser.add_argument(
        "--light-type", choices=("AREA", "POINT", "SPOT", "SUN", "MESH", "WORLD"), default="AREA"
    )
    parser.add_argument("--light-radius", type=float, default=0.0)
    parser.add_argument("--physical-sphere", action="store_true")
    parser.add_argument("--sun-angle", type=float, default=0.0, help="Sun diameter in degrees")
    parser.add_argument("--light-max-bounces", type=int, default=-1)
    parser.add_argument("--light-link", choices=("ALL", "FLOOR", "NONE"), default="ALL")
    parser.add_argument("--shadow-link-blocker", action="store_true")
    parser.add_argument("--allow-empty", action="store_true")
    parser.add_argument("--crop-border", action="store_true")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--diffuse-node", action="store_true")
    parser.add_argument("--hair-bsdf", action="store_true")
    parser.add_argument("--subsurface", action="store_true")
    parser.add_argument("--ray-portal", action="store_true")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--lens", type=float, default=52.0)
    parser.add_argument("--camera-type", choices=("PERSP", "PANO", "ORTHO"), default="PERSP")
    parser.add_argument("--ortho-scale", type=float, default=7.0)
    parser.add_argument("--world-absorption", type=float, default=0.0)
    parser.add_argument("--world-scatter", type=float, default=0.0)
    parser.add_argument("--volume-only", action="store_true")
    parser.add_argument("--no-floor", action="store_true")
    parser.add_argument("--camera-motion", action="store_true")
    parser.add_argument("--top-camera", action="store_true")
    parser.add_argument("--output", default="/tmp/cycles_bdpt_smoke.exr")
    build_scene(parser.parse_args(arguments))


if __name__ == "__main__":
    main()
