#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone Metal smoke scene for the volumetric caustic photon mapper.

Run with Blender in background mode. Arguments after ``--``:
``--photon`` enables the photon map, ``--heterogeneous`` uses a procedural density,
and ``--output PATH`` writes an OpenEXR result.
The script fails if the render contains non-finite values or no light.
"""

import argparse
import math
import sys

import bpy
from mathutils import Vector


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def material_with_output(name):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    return material, nodes, material.node_tree.links, output


def configure_metal():
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "METAL"
    preferences.get_devices()
    found = False
    for device in preferences.devices:
        if device.type == "METAL":
            device.use = True
            found = True
    if not found:
        raise RuntimeError("No Metal Cycles device available")


def build_scene(
    use_photons, heterogeneous, surface_only, samples, photon_count, volume_radius_scale, output
):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_metal()

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = False
    scene.cycles.use_photon_mapping = use_photons
    scene.cycles.photon_count = photon_count
    scene.cycles.photon_radius = 0.28
    scene.cycles.photon_volume_radius_scale = volume_radius_scale
    scene.cycles.photon_map_update_samples = 4
    scene.cycles.photon_max_bounces = 12
    scene.cycles.pixel_filter_type = "BOX"
    scene.render.resolution_x = 96
    scene.render.resolution_y = 96
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.filepath = output

    world = bpy.data.worlds.new("Black World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    scene.world = world

    if surface_only:
        bpy.ops.mesh.primitive_cube_add(location=(0.0, 2.0, 1.0), scale=(3.8, 0.08, 3.0))
        receiver = bpy.context.object
        receiver.name = "Diffuse Surface Photon Receiver"
        material, nodes, links, output_node = material_with_output("Diffuse Receiver")
        diffuse = nodes.new("ShaderNodeBsdfDiffuse")
        links.new(diffuse.outputs["BSDF"], output_node.inputs["Surface"])
        receiver.data.materials.append(material)
    else:
        bpy.ops.mesh.primitive_cube_add(location=(0.0, 0.0, 1.0), scale=(3.8, 5.0, 3.0))
        volume = bpy.context.object
        volume.name = "Homogeneous Photon Receiver"
        material, nodes, links, output_node = material_with_output("Homogeneous Volume")
        volume_node = nodes.new("ShaderNodeVolumePrincipled")
        volume_node.inputs["Density"].default_value = 0.075
        volume_node.inputs["Color"].default_value = (0.82, 0.9, 1.0, 1.0)
        volume_node.inputs["Anisotropy"].default_value = 0.35
        if heterogeneous:
            texture = nodes.new("ShaderNodeTexNoise")
            texture.noise_dimensions = "3D"
            texture.inputs["Scale"].default_value = 0.7
            texture.inputs["Detail"].default_value = 3.0
            density_scale = nodes.new("ShaderNodeMath")
            density_scale.operation = "MULTIPLY"
            density_scale.inputs[1].default_value = 0.12
            links.new(texture.outputs["Fac"], density_scale.inputs[0])
            links.new(density_scale.outputs[0], volume_node.inputs["Density"])
        links.new(volume_node.outputs["Volume"], output_node.inputs["Volume"])
        volume.data.materials.append(material)

    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=(0.0, -1.0, 1.0), radius=0.8)
    lens = bpy.context.object
    lens.name = "Glass Caustic Caster"
    material, nodes, links, output_node = material_with_output("Glass Lens")
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glass.inputs["Roughness"].default_value = 0.0
    glass.inputs["IOR"].default_value = 1.52
    links.new(glass.outputs["BSDF"], output_node.inputs["Surface"])
    lens.data.materials.append(material)

    bpy.ops.object.light_add(type="POINT", location=(0.0, -3.3, 1.0))
    light = bpy.context.object
    light.name = "Caustic Point Light"
    light.data.energy = 900.0
    light.data.shadow_soft_size = 0.035

    bpy.ops.object.camera_add(location=(6.8, -7.4, 3.6))
    camera = bpy.context.object
    camera.data.lens = 52.0
    look_at(camera, (0.0, 0.4, 1.0))
    scene.camera = camera

    bpy.ops.render.render(write_still=True)

    # Reading Render Result immediately after a GPU background render is not reliable on all
    # platforms. Reloading the written float EXR also validates the actual delivered artifact.
    result = bpy.data.images.load(output, check_existing=False)
    pixels = list(result.pixels)
    rgb = [pixels[index] for index in range(len(pixels)) if index % 4 != 3]
    if not rgb or not all(math.isfinite(value) for value in rgb):
        raise RuntimeError("Render contains non-finite values")
    mean = sum(rgb) / len(rgb)
    peak = max(rgb)
    if peak <= 0.0:
        raise RuntimeError("Render contains no light")
    print(
        "PHOTON_VOLUME_SMOKE "
        f"photon={int(use_photons)} heterogeneous={int(heterogeneous)} "
        f"surface_only={int(surface_only)} "
        f"mean={mean:.9g} peak={peak:.9g}"
    )


def main():
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--photon", action="store_true")
    parser.add_argument("--heterogeneous", action="store_true")
    parser.add_argument("--surface-only", action="store_true")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--photons", type=int, default=65536)
    parser.add_argument("--volume-radius-scale", type=float, default=2.0)
    parser.add_argument("--output", default="/tmp/cycles_photon_volume_smoke.exr")
    options = parser.parse_args(arguments)
    build_scene(
        options.photon,
        options.heterogeneous,
        options.surface_only,
        options.samples,
        options.photons,
        options.volume_radius_scale,
        options.output,
    )


if __name__ == "__main__":
    main()
