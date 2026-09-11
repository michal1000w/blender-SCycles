#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise Metal guiding UI, persistent-session resets, budgets, and CPU coexistence.

Run inside Blender with --debug-cycles --log-level debug. Arguments follow --.
The log includes per-render GPU training counts for checking finite training limits.
"""

import argparse
import json
import pathlib
import runpy
import sys

import bpy
import numpy as np
import OpenImageIO as oiio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--scene", choices=("indirect", "volume"), default="indirect")
    parser.add_argument("--bdpt", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    scene_script = pathlib.Path(__file__).with_name("cycles_metal_guiding_scene.py")
    sys.argv = [str(scene_script), "--", "--build-only", "--resolution", "32", "--samples", "32",
                "--scene", args.scene, "--output", str(args.output / "scene")]
    if args.bdpt:
        sys.argv.append("--bdpt")
    runpy.run_path(str(scene_script), run_name="__main__")
    scene = bpy.context.scene
    scene.render.use_persistent_data = True
    prefs = bpy.context.preferences.addons["cycles"].preferences
    from cycles import ui
    assert ui.CYCLES_RENDER_PT_sampling_path_guiding.poll(bpy.context), "Metal guiding UI hidden"
    assert not ui.CYCLES_RENDER_PT_sampling_path_guiding_debug.poll(bpy.context), "CPU controls exposed on Metal"
    results = {}
    images = {}

    def render(name, **settings):
        for key, value in settings.items():
            setattr(scene.cycles, key, value)
        print("GUIDING_REGRESSION_BEGIN " + name, flush=True)
        scene.render.filepath = str(args.output / (name + ".exr"))
        bpy.ops.render.render(write_still=True)
        image = oiio.ImageInput.open(scene.render.filepath)
        assert image is not None, name + ": missing image"
        channels = list(image.spec().channelnames)
        rgb_indices = [channels.index(channel) for channel in ('R', 'G', 'B')]
        pixels = np.asarray(image.read_image(format=oiio.FLOAT))[..., rgb_indices]
        image.close()
        assert np.isfinite(pixels).all(), name + ": nonfinite pixels"
        assert float(pixels.max()) > 0, name + ": empty image"
        images[name] = pixels
        results[name] = dict(mean=float(pixels.mean()), peak=float(pixels.max()), settings=settings)
        print("GUIDING_REGRESSION_END " + name, flush=True)

    render("off_initial", use_guiding=False)
    render("train_one", use_guiding=True, guiding_training_samples=1, guiding_gpu_memory_mb=16)
    render("train_five", guiding_training_samples=5)
    render("resize", guiding_gpu_memory_mb=32)
    render("repeat")
    # At one sample the field is empty: changing group size must preserve every pixel/sample.
    # A nonempty multi-sample render alone would miss skipped training batches.
    render("history_minimum", guiding_gpu_history_memory_mb=16, max_bounces=1024,
           samples=1, guiding_training_samples=0)
    render("history_large", guiding_gpu_history_memory_mb=128)
    assert np.allclose(images["history_minimum"], images["history_large"], rtol=1e-6, atol=1e-6), \
        "History group limit dropped or changed camera samples"
    render("history_resize", guiding_gpu_history_memory_mb=32, max_bounces=12,
           samples=32, guiding_training_samples=5)
    render("shrink", guiding_gpu_memory_mb=16, guiding_gpu_history_memory_mb=16)
    render("surface_only", use_volume_guiding=False)
    render("off_after", use_guiding=False)
    assert np.allclose(images["off_initial"], images["off_after"], rtol=1e-5, atol=1e-5), \
        "Disabling guiding failed to restore the unguided image"
    render("probability_zero", use_guiding=True, surface_guiding_probability=0.0,
           volume_guiding_probability=0.0)
    assert np.allclose(images["off_initial"], images["probability_zero"], rtol=1e-5, atol=1e-5), \
        "Zero guiding probability changed the baseline image"
    render("probability_one", surface_guiding_probability=1.0, volume_guiding_probability=1.0)
    render("product_mis", guiding_directional_sampling_type='MIS')
    render("roughness_weighted", guiding_directional_sampling_type='ROUGHNESS')
    render("product_resampling", guiding_directional_sampling_type='RIS')
    assert not np.allclose(images["product_mis"], images["product_resampling"], rtol=1e-4, atol=1e-5), \
        "Public sampling mode control did not change the render"
    scene.cycles.surface_guiding_probability = 0.5
    scene.cycles.volume_guiding_probability = 0.5
    render("all_training", use_guiding=True, use_volume_guiding=True, guiding_training_samples=0)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.1
    render("lighting_reset", guiding_training_samples=5)
    for device in prefs.devices:
        device.use = device.type in {"METAL", "CPU"}
    render("cpu_metal")
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")
    print("GUIDING_REGRESSION_PASSED " + str(len(results)), flush=True)


if __name__ == "__main__":
    main()
