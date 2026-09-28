#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Render and gate the saved passive colored Glass furnaces on Metal."""

import argparse
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np


CASES = tuple(f"{model}_{side}" for model in ("two_sided", "single_event")
              for side in ("front", "back"))
ROI = (slice(16, 48), slice(16, 48))
PASSIVE_UPPER = 1.03
FRONT_BACK_TOLERANCE = 0.08
BACK_RETURN_GAIN_MINIMUM = 0.02
CHROMATIC_GAP_MINIMUM = 0.04


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "64x64, 256 fixed samples, passive chromatic uncoated Glass white-world furnace",
        "cases": {}, "central_roi": "pixels [16:48,16:48]", "device": "METAL",
        "passive_upper_each_rgb": PASSIVE_UPPER,
        "front_back_tolerance_each_rgb": FRONT_BACK_TOLERANCE,
        "back_mean_return_gain_minimum": BACK_RETURN_GAIN_MINIMUM,
        "chromatic_adjacent_gap_minimum": CHROMATIC_GAP_MINIMUM,
        "reference_policy": "passivity and return gain; no RGB==input-color assertion",
    }
    report_path = args.output / "report.json"
    try:
        for name in CASES:
            scene_file = (args.fixtures / f"{name}.blend").resolve()
            bpy.ops.wm.open_mainfile(filepath=str(scene_file))
            scene = bpy.context.scene
            assert scene.render.resolution_x == scene.render.resolution_y == 64
            assert scene.cycles.samples == 256
            assert not scene.cycles.use_adaptive_sampling and not scene.cycles.use_denoising
            assert not scene.cycles.use_coherent_specular_connections
            assert scene["furnace_model"] == name.rsplit("_", 1)[0]
            assert scene["furnace_side"] == name.rsplit("_", 1)[1]
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = "METAL"
            preferences.get_devices()
            devices = [device.name for device in preferences.devices if device.type == "METAL"]
            if not devices:
                raise RuntimeError("No Metal device available")
            for device in preferences.devices:
                device.use = device.type == "METAL"
            scene.cycles.device = "GPU"
            scene.render.image_settings.file_format = "OPEN_EXR"
            scene.render.image_settings.color_depth = "32"
            render_path = (args.output / f"{name}.exr").resolve()
            scene.render.filepath = str(render_path)
            start = time.monotonic()
            status = bpy.ops.render.render(write_still=True)
            seconds = time.monotonic() - start
            if "FINISHED" not in status:
                raise RuntimeError(f"Render failed for {name}: {sorted(status)}")
            image = bpy.data.images.load(str(render_path), check_existing=False)
            pixels = np.asarray(image.pixels[:], dtype=np.float64).reshape(64, 64, image.channels)
            bpy.data.images.remove(image)
            rgb = pixels[ROI][:, :, :3]
            report["cases"][name] = {
                "scene": str(scene_file), "exr": str(render_path),
                "render_seconds": seconds, "metal_devices": devices,
                "roi_rgb_mean": [float(value) for value in rgb.mean(axis=(0, 1))],
                "roi_rgb_min": [float(value) for value in rgb.min(axis=(0, 1))],
                "all_finite": bool(np.isfinite(rgb).all()),
                "negative_channels": int(np.count_nonzero(rgb < 0.0)),
            }
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name}: {report['cases'][name]['roi_rgb_mean']}", flush=True)
        two_front = report["cases"]["two_sided_front"]["roi_rgb_mean"]
        two_back = report["cases"]["two_sided_back"]["roi_rgb_mean"]
        single_back = report["cases"]["single_event_back"]["roi_rgb_mean"]
        report["passivity_passed"] = all(
            entry["all_finite"] and entry["negative_channels"] == 0 and
            all(0.0 <= value <= PASSIVE_UPPER for value in entry["roi_rgb_mean"])
            for entry in report["cases"].values())
        report["chromatic_passed"] = all(
            values[1] - values[0] > CHROMATIC_GAP_MINIMUM and
            values[2] - values[1] > CHROMATIC_GAP_MINIMUM
            for values in (two_front, two_back))
        report["front_back_passed"] = all(
            abs(a - b) <= FRONT_BACK_TOLERANCE for a, b in zip(two_front, two_back))
        report["back_return_gain"] = [a - b for a, b in zip(two_back, single_back)]
        report["return_passed"] = (
            float(np.mean(report["back_return_gain"])) > BACK_RETURN_GAIN_MINIMUM and
            all(value > -0.005 for value in report["back_return_gain"]))
        report["passed"] = all(report[key] for key in (
            "passivity_passed", "chromatic_passed", "front_back_passed", "return_passed"))
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}", passed=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passed"]:
        raise RuntimeError(report.get("error", "Colored Glass furnace gate failed"))


if __name__ == "__main__":
    main()
