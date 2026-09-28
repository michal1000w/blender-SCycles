#!/usr/bin/env python3
"""Blender worker: same-process Metal coherent ON/OFF/ON routing regression.

Launch with any test binary:
BLENDER --background --factory-startup --python-exit-code 73 --threads 2
  --python THIS_SCRIPT -- --blend SPHERE_PHASE0.blend --output-dir FRESH_DIR
This small render tests routing/cache invalidation, not a physical image oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "same-process ON/OFF/ON Metal routing only; not physical oracle",
              "binary": bpy.app.binary_path, "binary_sha256": sha(bpy.app.binary_path),
              "fixture": str(args.blend.resolve()), "fixture_sha256": sha(args.blend),
              "script_sha256": sha(__file__), "cases": {}, "passed": False}
    try:
        bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
        scene = bpy.context.scene
        if scene.cycles.coherent_polarization_mode != "VECTOR":
            raise RuntimeError("Expected original VECTOR sphere fixture")
        scene.render.engine = "CYCLES"
        scene.render.resolution_x = scene.render.resolution_y = 16
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.cycles.samples = 4
        scene.cycles.seed = 19
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        scene.cycles.device = "GPU"
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "METAL"
        preferences.get_devices()
        metal = [d for d in preferences.devices if d.type == "METAL"]
        if not metal:
            raise RuntimeError("Metal device unavailable")
        for device in preferences.devices:
            device.use = device.type == "METAL"
        report["devices"] = [d.name for d in metal]
        images = {}
        for name, enabled in (("first_on", True), ("off", False), ("third_on", True)):
            scene.cycles.use_coherent_specular_connections = enabled
            path = output / f"{name}.exr"
            scene.render.filepath = str(path)
            start = time.perf_counter()
            status = bpy.ops.render.render(write_still=True)
            if "FINISHED" not in status:
                raise RuntimeError(f"Render failed: {name} {status}")
            image = bpy.data.images.load(str(path), check_existing=False)
            raw = np.asarray(image.pixels[:], dtype=np.float64).reshape(
                (image.size[1], image.size[0], image.channels))[:, :, :3].copy()
            bpy.data.images.remove(image)
            images[name] = raw
            report["cases"][name] = {"enabled": enabled, "exr": str(path),
                                     "sha256": sha(path), "all_finite": bool(np.isfinite(raw).all()),
                                     "rgb_mean": raw.mean(axis=(0, 1)).tolist(),
                                     "minimum": float(raw.min()), "maximum": float(raw.max()),
                                     "render_seconds": time.perf_counter() - start}
        difference = float(np.max(np.abs(images["first_on"] - images["third_on"])))
        report["on_repeat_max_absolute_error"] = difference
        report["on_repeat_absolute_gate"] = 1e-6
        report["first_on_positive"] = bool(np.max(images["first_on"]) > 0)
        report["final_settings"] = {
            "enabled": scene.cycles.use_coherent_specular_connections,
            "polarization": scene.cycles.coherent_polarization_mode,
            "samples": scene.cycles.samples, "seed": scene.cycles.seed,
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
            "adaptive": scene.cycles.use_adaptive_sampling,
            "denoising": scene.cycles.use_denoising,
            "bdpt": scene.cycles.use_bidirectional_path_tracing,
            "guiding": scene.cycles.use_guiding}
        report["passed"] = (all(v["all_finite"] for v in report["cases"].values()) and
                            report["first_on_positive"] and difference <= 1e-6)
        if not report["passed"]:
            raise RuntimeError("Routing gate failed")
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
