# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Render a supplied .blend in memory, with PT and BDPT/tree toggles in one session.

blender -b --disable-autoexec scene.blend --python-exit-code 1 --python this.py -- --output DIR
Does not save the blend. Preserves materials, lights and transport settings; changes only render
resolution, seed, sample count, denoising/adaptive sampling and the tested integrator toggles.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np
import OpenImageIO as oiio

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--width", type=int, default=192)
parser.add_argument("--samples", type=int, default=128)
parser.add_argument("--light-paths", type=int)
parser.add_argument("--reference-samples", type=int, default=4096)
options = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
options.output.mkdir(parents=True, exist_ok=True)
s = bpy.context.scene
preferences = bpy.context.preferences.addons["cycles"].preferences
preferences.compute_device_type = "METAL"
preferences.get_devices()
metal = [d for d in preferences.devices if d.type == "METAL"]
if not metal:
    raise RuntimeError("No Metal device")
for d in preferences.devices:
    d.use = d.type == "METAL"
original = {"file": bpy.data.filepath, "resolution": [s.render.resolution_x, s.render.resolution_y],
            "samples": s.cycles.samples, "light_paths": s.cycles.bdpt_light_paths,
            "update_samples": s.cycles.bdpt_update_samples,
            "photon_mapping": s.cycles.use_photon_mapping}
s.render.resolution_y = max(1, round(options.width * s.render.resolution_y / s.render.resolution_x))
s.render.resolution_x = options.width
s.render.resolution_percentage = 100
s.render.engine = "CYCLES"
s.cycles.device = "GPU"
# BDPT takes precedence over photon mapping; disabling BDPT alone can enable a different
# integrator in saved scenes. Explicitly disable both for a genuine path-tracing reference.
s.cycles.use_photon_mapping = False
s.cycles.use_denoising = False
s.cycles.use_adaptive_sampling = False
if options.light_paths is not None:
    s.cycles.bdpt_light_paths = options.light_paths
s.render.image_settings.file_format = "OPEN_EXR"
s.render.image_settings.color_depth = "32"
results = {"original": original, "renders": {}}
images = {}
# This order exercises creating the CDF when enabling BDPT while the tree stays enabled,
# and changing the tree without rebuilding a Blender process or reloading the scene.
for name, bdpt, tree in [("reference", False, True), ("tree", True, True),
                         ("flat", True, False), ("tree_repeat", True, True),
                         ("tree_independent", True, True)]:
    s.cycles.use_bidirectional_path_tracing = bdpt
    s.cycles.use_light_tree = tree
    s.cycles.samples = options.reference_samples if name == "reference" else options.samples
    s.cycles.seed = 7919 if name == "reference" else (1 if name == "tree_independent" else 0)
    s.render.filepath = str(options.output.resolve() / (name + ".exr"))
    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    seconds = time.perf_counter() - start
    inp = oiio.ImageInput.open(s.render.filepath)
    image = inp.read_image(format=oiio.FLOAT)[..., :3].astype(np.float64)
    inp.close()
    if not np.isfinite(image).all():
        raise AssertionError(f"Nonfinite pixels in {name}")
    images[name] = image
    results["renders"][name] = {"seconds": seconds, "mean": float(image.mean()),
                                "mse": float(np.mean((image - images["reference"]) ** 2))}
    print("BDPT_SCENE_RESULT", name, results["renders"][name], flush=True)
    (options.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
results["toggle_max_abs"] = float(np.max(np.abs(images["tree"] - images["tree_repeat"])))
# Cache insertion order is atomic and changes which cached vertices camera paths select.
# A same-seed repeat therefore is not pixel deterministic. Compare its discrepancy with an
# independent-seed discrepancy instead of accepting an arbitrary per-pixel tolerance.
results["toggle_mse"] = float(np.mean((images["tree"] - images["tree_repeat"]) ** 2))
results["independent_mse"] = float(np.mean((images["tree"] - images["tree_independent"]) ** 2))
(options.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
if results["toggle_mse"] > max(1.5 * results["independent_mse"], 1e-10):
    raise AssertionError("Tree toggle discrepancy exceeds independent sampling noise")
