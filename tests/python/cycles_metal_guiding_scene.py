#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reproducible indirect-lighting, glossy, mixed-closure, and volume guiding scenes.

Run inside Blender, with arguments after --. Every render writes an EXR, raw linear
RGB NumPy array, JSON settings/metrics, and optionally an editable .blend file.
"""

import argparse
import json
import pathlib
import sys
import time

import bpy
import numpy as np
from mathutils import Vector


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def material(name, color, roughness=0.6, metallic=0.0):
    result = bpy.data.materials.new(name)
    result.use_nodes = True
    shader = result.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return result


def box(name, location, scale, mat):
    bpy.ops.mesh.primitive_cube_add(size=1, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    return obj


def build_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = args.device
    if args.device == "GPU":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        devices = [device for device in prefs.devices if device.type == "METAL"]
        if not devices:
            raise RuntimeError("Metal GPU unavailable")
        for device in prefs.devices:
            device.use = device.type == "METAL"
    else:
        devices = []

    cycles = scene.cycles
    cycles.samples = args.samples
    cycles.time_limit = args.time_limit
    cycles.seed = args.seed
    cycles.use_guiding = args.guiding
    cycles.guiding_training_samples = args.training_samples
    cycles.guiding_gpu_memory_mb = args.memory_mb
    cycles.surface_guiding_probability = args.probability
    cycles.volume_guiding_probability = args.probability
    cycles.use_surface_guiding = not args.volume_only
    cycles.use_volume_guiding = not args.surface_only
    cycles.use_bidirectional_path_tracing = args.bdpt
    cycles.bdpt_light_paths = args.light_paths
    cycles.bdpt_update_samples = 4
    cycles.bdpt_max_bounces = 12
    cycles.use_photon_mapping = False
    cycles.use_denoising = False
    cycles.use_adaptive_sampling = False
    cycles.use_auto_tile = False
    cycles.sample_clamp_direct = 0
    cycles.sample_clamp_indirect = 0
    cycles.blur_glossy = 0
    cycles.max_bounces = 12
    cycles.diffuse_bounces = 12
    cycles.glossy_bounces = 12
    cycles.transmission_bounces = 12
    cycles.volume_bounces = 8
    cycles.pixel_filter_type = "BOX"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(args.output.with_suffix(".exr"))
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0

    world = bpy.data.worlds.new("Black environment")
    world.use_nodes = True
    world.node_tree.nodes.get("Background").inputs["Strength"].default_value = 0
    scene.world = world

    white = material("Diffuse ivory", (0.72, 0.72, 0.72))
    red = material("Diffuse red", (0.65, 0.06, 0.025))
    green = material("Diffuse green", (0.035, 0.42, 0.07))
    # Main room and hidden lighting chamber. The light cannot illuminate the camera-facing
    # objects directly; radiance enters via the narrow opening above the back baffle.
    box("Floor", (0, 0.5, -0.08), (6.2, 7.2, 0.16), white)
    box("Ceiling", (0, 0.5, 4.08), (6.2, 7.2, 0.16), white)
    box("Left wall", (-3.08, 0.5, 2), (0.16, 7.2, 4), red)
    box("Right wall", (3.08, 0.5, 2), (0.16, 7.2, 4), green)
    box("Back wall", (0, 4.08, 2), (6.2, 0.16, 4), white)
    box("Hidden-light baffle", (0, 2, 1.65), (6.2, 0.16, 3.3), white)
    light = bpy.data.lights.new("Hidden area light", "AREA")
    light.energy = 1800
    light.shape = getattr(args, "light_shape", "DISK")
    light.size = 0.75
    lamp = bpy.data.objects.new(light.name, light)
    scene.collection.objects.link(lamp)
    lamp.location = (0, 3.6, 3.7)
    look_at(lamp, (0, 2.4, 1))

    object_mat = white
    if args.scene == "glossy":
        object_mat = material("Rough copper", (0.7, 0.35, 0.1), roughness=0.23, metallic=0.85)
    elif args.scene == "mixed":
        object_mat = bpy.data.materials.new("Diffuse and delta glass mixture")
        object_mat.use_nodes = True
        nodes = object_mat.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        diffuse = nodes.new("ShaderNodeBsdfDiffuse")
        diffuse.inputs["Color"].default_value = (0.55, 0.7, 0.8, 1)
        glass = nodes.new("ShaderNodeBsdfGlass")
        glass.inputs["Roughness"].default_value = 0
        glass.inputs["IOR"].default_value = 1.45
        mix = nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = 0.5
        links = object_mat.node_tree.links
        links.new(diffuse.outputs[0], mix.inputs[1])
        links.new(glass.outputs[0], mix.inputs[2])
        links.new(mix.outputs[0], output.inputs["Surface"])
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.8, location=(-1.1, 0.45, 0.8))
    bpy.context.object.name = "Guiding test sphere"
    bpy.context.object.data.materials.append(object_mat)
    bpy.ops.object.shade_smooth()
    block = box("Tall block", (1.1, 0.7, 1.05), (1.1, 1.2, 2.1), object_mat)
    block.rotation_euler.z = -0.2

    if args.scene == "volume":
        medium = bpy.data.materials.new("Anisotropic participating medium")
        medium.use_nodes = True
        nodes = medium.node_tree.nodes
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        scatter = nodes.new("ShaderNodeVolumeScatter")
        scatter.inputs["Density"].default_value = 0.12
        scatter.inputs["Anisotropy"].default_value = 0.65
        scatter.inputs["Color"].default_value = (0.85, 0.9, 1.0, 1)
        medium.node_tree.links.new(scatter.outputs[0], output.inputs["Volume"])
        box("Medium boundary", (0, 0.5, 2), (5.98, 6.98, 3.98), medium)

    data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", data)
    scene.collection.objects.link(camera)
    camera.location = (0, -7.8, 2.15)
    data.lens = 34
    look_at(camera, (0, 1, 1.7))
    scene.camera = camera
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.save_scene:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix(".blend")))
    if args.build_only:
        return

    warmup_seconds = 0.0
    if args.warmup:
        start = time.perf_counter()
        bpy.ops.render.render(write_still=False)
        warmup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - start
    import OpenImageIO as oiio
    image = oiio.ImageInput.open(scene.render.filepath)
    if image is None:
        raise RuntimeError("No render result")
    rgb = np.asarray(image.read_image(format=oiio.FLOAT))[..., :3]
    image.close()
    if not np.isfinite(rgb).all():
        raise RuntimeError("Nonfinite render values")
    np.save(args.output.with_suffix(".npy"), rgb)
    metrics = {key: str(value) if isinstance(value, pathlib.Path) else value
               for key, value in vars(args).items()}
    metrics.update(seconds=elapsed, warmup_seconds=warmup_seconds,
                   mean=float(rgb.mean()), peak=float(rgb.max()),
                   p999=float(np.quantile(rgb, 0.999)),
                   devices=[device.name for device in devices],
                   blender_version=bpy.app.version_string,
                   build_hash=bpy.app.build_hash.decode())
    args.output.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("METAL_GUIDING_RENDER " + json.dumps(metrics, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("indirect", "glossy", "mixed", "volume"), default="indirect")
    parser.add_argument("--device", choices=("GPU", "CPU"), default="GPU")
    parser.add_argument("--light-shape", choices=("DISK", "SQUARE"), default="DISK",
                        help="Preserve the disk baseline or exercise rectangular solid-angle NEE")
    parser.add_argument("--guiding", action="store_true")
    parser.add_argument("--bdpt", action="store_true")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--time-limit", type=float, default=0.0,
                        help="Cycles render-time budget in seconds; zero renders all samples")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--training-samples", type=int, default=128)
    parser.add_argument("--memory-mb", type=int, default=64)
    parser.add_argument("--probability", type=float, default=0.5)
    parser.add_argument("--light-paths", type=int, default=16384)
    parser.add_argument("--surface-only", action="store_true")
    parser.add_argument("--volume-only", action="store_true")
    parser.add_argument("--save-scene", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--warmup", action="store_true", help="Discard one complete render to warm kernel caches")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    build_scene(args)


if __name__ == "__main__":
    main()
