#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Metal compilation feature changes in one persistent render session.

Run in Blender with --python-exit-code 1 and pass --output DIRECTORY after --.
Only a generated scene is used. EXRs allow baseline/candidate comparison.
"""

import argparse
import json
from pathlib import Path
import runpy
import sys
import time

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--warm-up", action="store_true",
                        help="Compatibility flag: every case always measures first and warm renders")
    parser.add_argument("--transitions", action="store_true")
    parser.add_argument("--materials", action="store_true",
                        help="Test node usage, displacement interpolation and shadow-linking edits")
    parser.add_argument("--denoise", action="store_true",
                        help="Exercise denoising and rendering kernel loads on the same Metal device")
    parser.add_argument("--reference-output", type=Path)
    parser.add_argument("--relative-rmse-limit", type=float, default=1e-4)
    parser.add_argument("--guided-relative-rmse-limit", type=float, default=1e-3,
                        help="Allow at most 0.1%% relative RMSE for stochastic guided transport")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.transitions and args.materials:
        parser.error("Select either transport transitions or material transitions")
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise RuntimeError("Use an empty output directory")
    root = Path(__file__).resolve().parents[2]
    setup = runpy.run_path(str(root / "tools/pixel_displacement_benchmark.py"))
    setup["create_scene"](2, 64, 4, 32)
    scene = bpy.context.scene
    scene.cycles.use_denoising = args.denoise
    scene.cycles.denoising_use_gpu = True
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.cycles.seed = 11
    scene.cycles.use_pixel_displacement_resolution_clamp = False
    scene.cycles.photon_count = 2048
    scene.cycles.photon_max_bounces = 3
    scene.cycles.bdpt_light_paths = 2048
    scene.cycles.bdpt_max_bounces = 3
    scene.cycles.guiding_gpu_memory_mb = 16
    scene.cycles.guiding_training_samples = 4
    cases = [("displacement", True, False, False, False)]
    if args.transitions:
        cases += [("disabled", False, False, False, False),
                  ("restored", True, False, False, False),
                  ("photon", True, True, False, False),
                  ("bdpt", True, False, True, False),
                  ("bdpt_guiding", True, False, True, True),
                  ("photon_after_bdpt", True, True, False, False),
                  ("restored_all", True, False, False, False)]
    if args.materials:
        cases += [(label, True, False, False, False) for label in
                  ("noise_added", "noise_removed", "cubic", "linear_restored",
                   "shadow_linked", "shadow_unlinked")]
    material = bpy.data.materials["pixel_displacement_material"]
    noise = None
    reports = []
    images = {}
    for label, displacement, photon, bdpt, guiding in cases:
        if label == "noise_added":
            noise = material.node_tree.nodes.new("ShaderNodeTexNoise")
            material.node_tree.links.new(
                noise.outputs["Color"], material.node_tree.nodes["Principled BSDF"].inputs["Base Color"])
        elif label == "noise_removed":
            material.node_tree.nodes.remove(noise)
        elif label in {"cubic", "linear_restored"}:
            image_node = next(node for node in material.node_tree.nodes if node.type == "TEX_IMAGE")
            image_node.interpolation = "Cubic" if label == "cubic" else "Linear"
        elif label == "shadow_linked":
            blockers = bpy.data.collections.new("Compile Test Blockers")
            scene.collection.children.link(blockers)
            blockers.objects.link(bpy.data.objects["pixel_displacement_grid"])
            bpy.data.objects["key_area"].light_linking.blocker_collection = blockers
        elif label == "shadow_unlinked":
            bpy.data.objects["key_area"].light_linking.blocker_collection = None
        scene.cycles.use_pixel_displacement = displacement
        scene.cycles.use_photon_mapping = photon
        scene.cycles.use_bidirectional_path_tracing = bdpt
        scene.cycles.use_guiding = guiding
        scene.update_tag()
        bpy.context.view_layer.update()
        times = []
        for repeat in range(2):
            started = time.perf_counter()
            bpy.ops.render.render()
            times.append(time.perf_counter() - started)
        path = args.output / f"{label}.exr"
        bpy.data.images["Render Result"].save_render(str(path), scene=scene)
        image = bpy.data.images.load(str(path), check_existing=False)
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgb = pixels.reshape(-1, 4)[:, :3]
        assert np.isfinite(rgb).all() and rgb.mean() > 0, label
        images[label] = rgb
        report = dict(case=label, seconds=times, mean=float(rgb.mean()))
        if args.reference_output:
            reference_image = bpy.data.images.load(
                str(args.reference_output / f"{label}.exr"), check_existing=False)
            reference = np.empty(len(reference_image.pixels), dtype=np.float32)
            reference_image.pixels.foreach_get(reference)
            reference = reference.reshape(-1, 4)[:, :3]
            assert rgb.shape == reference.shape, label
            error = float(np.sqrt(np.mean((rgb - reference) ** 2)) /
                          max(np.sqrt(np.mean(reference ** 2)), 1e-20))
            report.update(exact_equal=bool(np.array_equal(rgb, reference)), relative_rmse=error)
            limit = args.guided_relative_rmse_limit if guiding else args.relative_rmse_limit
            report["relative_rmse_limit"] = limit
            assert error <= limit, (label, error)
        reports.append(report)
        (args.output / "report.json").write_text(json.dumps(reports, indent=2) + "\n")
        print("METAL_FEATURE_RENDER", json.dumps(reports[-1]), flush=True)
    for label in ("restored", "restored_all"):
        if label in images:
            # The baseline also changes slightly after rebuilding its displacement cache.
            # Compare matching states against baseline EXRs, not against the first state.
            print("METAL_RESTORATION_DELTA", label,
                  float(np.max(np.abs(images["displacement"] - images[label]))), flush=True)
    print("METAL_COMPILE_FEATURES_PASSED", flush=True)


if __name__ == "__main__":
    main()
