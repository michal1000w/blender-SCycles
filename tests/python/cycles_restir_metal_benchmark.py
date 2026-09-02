#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Small deterministic quality/performance benchmark for Cycles Metal ReSTIR DI."""

import argparse
import json
import math
import os
import random
import statistics
import sys
import time

import bpy
from mathutils import Vector


def configure_metal(metalrt):
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = "METAL"
    preferences.metalrt = "ON" if metalrt else "OFF"
    preferences.get_devices()
    found = False
    for device in preferences.devices:
        device.use = device.type == "METAL"
        found |= device.use
    if not found:
        raise RuntimeError("No Metal Cycles device available")


def diffuse(name, color, roughness=0.45):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    return material


def look_at(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def build_scene(options):
    if not getattr(options, "skip_factory_reset", False):
        bpy.ops.wm.read_factory_settings(use_empty=True)
    configure_metal(options.metalrt)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_light_tree = True
    scene.cycles.max_bounces = 5
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 3
    scene.cycles.seed = 1977
    scene.cycles.pixel_filter_type = "BOX"
    scene.render.resolution_x = options.resolution
    scene.render.resolution_y = options.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"

    world = bpy.data.worlds.new("Dark World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.015, 0.02, 0.03, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.2
    scene.world = world

    floor_mat = diffuse("Floor", (0.42, 0.45, 0.5), 0.55)
    wall_mat = diffuse("Walls", (0.3, 0.34, 0.4), 0.6)
    glossy_mat = diffuse("Glossy", (0.3, 0.12, 0.04), 0.13)

    bpy.ops.mesh.primitive_plane_add(size=14.0, location=(0.0, 0.0, 0.0))
    bpy.context.object.data.materials.append(floor_mat)
    bpy.ops.mesh.primitive_plane_add(
        size=14.0, location=(0.0, 4.5, 4.5), rotation=(math.pi * 0.5, 0.0, 0.0)
    )
    bpy.context.object.data.materials.append(wall_mat)
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=48, ring_count=24, radius=1.25, location=(-1.2, 0.1, 1.25)
    )
    bpy.context.object.data.materials.append(glossy_mat)
    bpy.ops.mesh.primitive_cube_add(size=1.8, location=(1.45, 0.55, 0.9))
    bpy.context.object.rotation_euler[2] = 0.45
    bpy.context.object.data.materials.append(diffuse("Box", (0.08, 0.24, 0.5), 0.36))

    random.seed(8341)
    for index in range(options.lights):
        angle = 2.0 * math.pi * index / options.lights
        radius = 2.0 + 2.0 * random.random()
        light_data = bpy.data.lights.new(f"Light {index:03d}", type="POINT")
        light_data.energy = 18.0 + 42.0 * random.random()
        light_data.color = (
            0.35 + 0.65 * random.random(),
            0.35 + 0.65 * random.random(),
            0.35 + 0.65 * random.random(),
        )
        light_data.shadow_soft_size = 0.035 + 0.12 * random.random()
        light = bpy.data.objects.new(light_data.name, light_data)
        light.location = (
            math.cos(angle) * radius,
            math.sin(angle) * radius * 0.7,
            1.1 + 3.2 * random.random(),
        )
        light["benchmark_location"] = tuple(light.location)
        bpy.context.collection.objects.link(light)

    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (7.2, -9.4, 6.2)
    camera.data.lens = 48.0
    look_at(camera, (0.0, 0.7, 1.25))
    scene.camera = camera
    return scene


def set_animation_frame(scene, frame):
    """Apply deterministic emitter/object motion without relying on persistent render history."""
    phase = 0.045 * (frame - 1)
    cosine = math.cos(phase)
    sine = math.sin(phase)
    for light in (obj for obj in scene.objects if obj.type == "LIGHT"):
        x, y, z = light["benchmark_location"]
        light.location = (cosine * x - sine * y, sine * x + cosine * y, z)
    box = scene.objects.get("Cube")
    if box is not None:
        box.rotation_euler[2] = 0.45 + 0.035 * (frame - 1)
    scene.frame_set(frame)


def render(scene, samples, restir, output_path, seed, options):
    scene.cycles.samples = samples
    scene.cycles.seed = seed
    scene.cycles.use_restir = restir and not options.enhanced
    scene.cycles.use_restir_pt = restir and options.enhanced
    scene.cycles.restir_pt_temporal_history = options.history
    scene.cycles.restir_pt_spatial_neighbors = options.neighbors
    scene.cycles.restir_pt_spatial_radius = 30
    scene.cycles.restir_pt_footprint_threshold = options.footprint
    scene.cycles.restir_pt_min_roughness = options.min_roughness
    scene.cycles.restir_pt_decorrelate = options.history > 0
    scene.cycles.restir_light_candidates = options.candidates
    scene.cycles.restir_history_length = options.history
    scene.cycles.restir_spatial_neighbors = options.neighbors
    scene.cycles.restir_spatial_radius = 24
    scene.cycles.restir_normal_threshold = 0.8
    scene.cycles.restir_position_threshold = 0.05
    scene.cycles.restir_min_roughness = 0.25
    scene.render.filepath = output_path
    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - start
    image = bpy.data.images.load(output_path, check_existing=False)
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    rgb = [pixels[i] for i in range(len(pixels)) if i % 4 != 3]
    if not rgb or not all(math.isfinite(value) for value in rgb):
        raise RuntimeError("Render produced empty or non-finite pixels")
    return rgb, elapsed


def rmse(image, reference):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(image, reference)) / len(reference))


def image_stats(image):
    ordered = sorted(image)
    return {
        "min": ordered[0],
        "p99": ordered[min(int(0.99 * len(ordered)), len(ordered) - 1)],
        "p999": ordered[min(int(0.999 * len(ordered)), len(ordered) - 1)],
        "max": ordered[-1],
    }


def render_repeated(scene, samples, restir, output_path, seed, options):
    timings = []
    image = None
    for _ in range(options.timing_repeats):
        image, elapsed = render(scene, samples, restir, output_path, seed, options)
        timings.append(elapsed)
    return image, statistics.median(timings)


def residual_flicker(images, references):
    if len(images) < 2:
        return 0.0
    squared_error = 0.0
    count = 0
    previous = [value - reference for value, reference in zip(images[0], references[0])]
    for image, reference in zip(images[1:], references[1:]):
        current = [value - target for value, target in zip(image, reference)]
        squared_error += sum((value - old) ** 2 for value, old in zip(current, previous))
        count += len(current)
        previous = current
    return math.sqrt(squared_error / count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metalrt", action="store_true")
    parser.add_argument("--enhanced", action="store_true")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--lights", type=int, default=48)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--reference-samples", type=int, default=256)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--history", type=int, default=0)
    parser.add_argument("--neighbors", type=int, default=0)
    parser.add_argument("--footprint", type=float, default=0.02)
    parser.add_argument("--min-roughness", type=float, default=0.2)
    parser.add_argument("--animation-frames", type=int, default=1)
    parser.add_argument("--timing-repeats", type=int, default=1)
    parser.add_argument("--output-dir", default="/tmp/cycles-restir-benchmark")
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    os.makedirs(options.output_dir, exist_ok=True)
    scene = build_scene(options)

    references = []
    baselines = []
    restir_images = []
    reference_seconds = 0.0
    baseline_seconds = 0.0
    restir_seconds = 0.0
    for frame in range(1, options.animation_frames + 1):
        set_animation_frame(scene, frame)
        suffix = "" if options.animation_frames == 1 else f"-{frame:04d}"
        reference, elapsed = render(
            scene,
            options.reference_samples,
            False,
            os.path.join(options.output_dir, f"reference{suffix}.exr"),
            9901,
            options,
        )
        references.append(reference)
        reference_seconds += elapsed
        baseline, elapsed = render_repeated(
            scene,
            options.samples,
            False,
            os.path.join(options.output_dir, f"baseline{suffix}.exr"),
            1977,
            options,
        )
        baselines.append(baseline)
        baseline_seconds += elapsed
        restir_image, elapsed = render_repeated(
            scene,
            options.samples,
            True,
            os.path.join(options.output_dir, f"restir{suffix}.exr"),
            1977,
            options,
        )
        restir_images.append(restir_image)
        restir_seconds += elapsed

    reference = [value for image in references for value in image]
    baseline = [value for image in baselines for value in image]
    restir = [value for image in restir_images for value in image]
    result = {
        "backend": "METALRT" if options.metalrt else "METAL",
        "resolution": options.resolution,
        "lights": options.lights,
        "samples": options.samples,
        "reference_samples": options.reference_samples,
        "candidates": options.candidates,
        "history": options.history,
        "footprint": options.footprint,
        "min_roughness": options.min_roughness,
        "neighbors": options.neighbors,
        "animation_frames": options.animation_frames,
        "timing_repeats": options.timing_repeats,
        "reference_seconds": reference_seconds,
        "baseline_seconds": baseline_seconds,
        "restir_seconds": restir_seconds,
        "baseline_rmse": rmse(baseline, reference),
        "restir_rmse": rmse(restir, reference),
        "reference_mean": sum(reference) / len(reference),
        "baseline_mean": sum(baseline) / len(baseline),
        "restir_mean": sum(restir) / len(restir),
        "baseline_stats": image_stats(baseline),
        "restir_stats": image_stats(restir),
        "baseline_residual_flicker": residual_flicker(baselines, references),
        "restir_residual_flicker": residual_flicker(restir_images, references),
    }
    result["rmse_ratio"] = result["restir_rmse"] / result["baseline_rmse"]
    result["time_ratio"] = result["restir_seconds"] / result["baseline_seconds"]
    result["flicker_ratio"] = (
        result["restir_residual_flicker"] / result["baseline_residual_flicker"]
        if result["baseline_residual_flicker"] > 0.0
        else 0.0
    )
    with open(os.path.join(options.output_dir, "metrics.json"), "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print("RESTIR_BENCHMARK " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
