#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Continuous Blender animation-session benchmark for Cycles Metal ReSTIR PT history."""

import argparse
import json
import math
import os
import pathlib
import sys
import time
from types import SimpleNamespace

import bpy

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from cycles_restir_metal_benchmark import build_scene, image_stats, residual_flicker, rmse  # noqa: E402


def key_animation(scene, frames):
    lights = [obj for obj in scene.objects if obj.type == "LIGHT"]
    box = scene.objects.get("Cube")
    for frame in range(1, frames + 1):
        phase = 0.045 * (frame - 1)
        cosine = math.cos(phase)
        sine = math.sin(phase)
        for light in lights:
            x, y, z = light["benchmark_location"]
            light.location = (cosine * x - sine * y, sine * x + cosine * y, z)
            light.keyframe_insert("location", frame=frame)
        if box is not None:
            box.rotation_euler[2] = 0.45 + 0.035 * (frame - 1)
            box.keyframe_insert("rotation_euler", frame=frame, index=2)
    scene.frame_start = 1
    scene.frame_end = frames
    scene.frame_set(1)


def load_rgb(path):
    image = bpy.data.images.load(path, check_existing=False)
    pixels = list(image.pixels[:])
    bpy.data.images.remove(image)
    rgb = [pixels[index] for index in range(len(pixels)) if index % 4 != 3]
    if not rgb or not all(math.isfinite(value) for value in rgb):
        raise RuntimeError(f"Animation frame is empty or non-finite: {path}")
    return rgb


def render_sequence(scene, output_dir, prefix, samples, enhanced, history, neighbors):
    scene.cycles.samples = samples
    scene.cycles.seed = 1977
    scene.cycles.use_restir = False
    scene.cycles.use_restir_pt = enhanced
    scene.cycles.restir_pt_temporal_history = history
    scene.cycles.restir_pt_spatial_neighbors = neighbors
    scene.cycles.restir_pt_spatial_radius = 8
    scene.cycles.restir_pt_footprint_threshold = 0.02
    scene.cycles.restir_pt_min_roughness = 0.2
    scene.cycles.restir_pt_decorrelate = history > 0
    scene.frame_set(1)
    scene.render.filepath = os.path.join(output_dir, prefix + "-")
    start = time.perf_counter()
    bpy.ops.render.render(animation=True)
    elapsed = time.perf_counter() - start
    images = [
        load_rgb(os.path.join(output_dir, f"{prefix}-{frame:04d}.exr"))
        for frame in range(scene.frame_start, scene.frame_end + 1)
    ]
    return images, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metalrt", action="store_true")
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--lights", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--reference-samples", type=int, default=128)
    parser.add_argument("--frames", type=int, default=3)
    parser.add_argument("--history", type=int, default=20)
    parser.add_argument("--neighbors", type=int, default=1)
    parser.add_argument("--output-dir", default="/tmp/cycles-restir-continuous")
    options = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    os.makedirs(options.output_dir, exist_ok=True)

    scene = build_scene(
        SimpleNamespace(
            metalrt=options.metalrt,
            resolution=options.resolution,
            lights=options.lights,
        )
    )
    key_animation(scene, options.frames)

    references, reference_seconds = render_sequence(
        scene, options.output_dir, "reference", options.reference_samples, False, 0, 0
    )
    baselines, baseline_seconds = render_sequence(
        scene, options.output_dir, "baseline", options.samples, False, 0, 0
    )
    enhanced, enhanced_seconds = render_sequence(
        scene,
        options.output_dir,
        "restir",
        options.samples,
        True,
        options.history,
        options.neighbors,
    )

    reference_flat = [value for image in references for value in image]
    baseline_flat = [value for image in baselines for value in image]
    enhanced_flat = [value for image in enhanced for value in image]
    baseline_rmse = rmse(baseline_flat, reference_flat)
    enhanced_rmse = rmse(enhanced_flat, reference_flat)
    baseline_flicker = residual_flicker(baselines, references)
    enhanced_flicker = residual_flicker(enhanced, references)
    result = {
        "backend": "METALRT" if options.metalrt else "METAL",
        "continuous_animation": True,
        "frames": options.frames,
        "resolution": options.resolution,
        "samples": options.samples,
        "reference_samples": options.reference_samples,
        "history": options.history,
        "neighbors": options.neighbors,
        "reference_seconds": reference_seconds,
        "baseline_seconds": baseline_seconds,
        "restir_seconds": enhanced_seconds,
        "baseline_rmse": baseline_rmse,
        "restir_rmse": enhanced_rmse,
        "rmse_ratio": enhanced_rmse / baseline_rmse,
        "baseline_residual_flicker": baseline_flicker,
        "restir_residual_flicker": enhanced_flicker,
        "flicker_ratio": enhanced_flicker / baseline_flicker if baseline_flicker else 0.0,
        "reference_mean": sum(reference_flat) / len(reference_flat),
        "baseline_mean": sum(baseline_flat) / len(baseline_flat),
        "restir_mean": sum(enhanced_flat) / len(enhanced_flat),
        "baseline_stats": image_stats(baseline_flat),
        "restir_stats": image_stats(enhanced_flat),
    }
    result["time_ratio"] = enhanced_seconds / baseline_seconds
    with open(os.path.join(options.output_dir, "metrics.json"), "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True)
        output.write("\n")
    print("RESTIR_CONTINUOUS " + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
