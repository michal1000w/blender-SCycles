"""Render the declared two-transmission slab and compare its fixed analytic ROI."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--path-tracing", action="store_true")
    parser.add_argument("--guiding", action="store_true")
    parser.add_argument("--warm-measured", type=int, default=0,
                        help="One warmup and this many same-scene measured renders")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    args.output.mkdir(parents=True, exist_ok=False)
    fixture_kind = "mixed R/T" if "mixed" in args.reference.name else "slab"
    result = {"scope": f"phase-0 {fixture_kind} pilot against independent vector reference; overrides are diagnostic only",
              "blend": str(args.blend.resolve()), "reference": str(args.reference.resolve())}
    try:
        bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
        scene = bpy.context.scene
        if args.samples:
            scene.cycles.samples = args.samples
        if args.resolution:
            scene.render.resolution_x = args.resolution
            scene.render.resolution_y = args.resolution
        if args.path_tracing:
            scene.cycles.use_bidirectional_path_tracing = False
        if args.guiding:
            scene.cycles.use_guiding = True
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
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
        scene.render.filepath = str((args.output / "phase_0.exr").resolve())
        if args.warm_measured < 0:
            raise ValueError("--warm-measured must be nonnegative")
        if args.warm_measured:
            scene.render.use_persistent_data = True
        result.update(metal_devices=devices, samples=scene.cycles.samples,
                      resolution=[scene.render.resolution_x, scene.render.resolution_y],
                      bidirectional_path_tracing=scene.cycles.use_bidirectional_path_tracing,
                      guiding=scene.cycles.use_guiding,
                      adaptive_sampling=scene.cycles.use_adaptive_sampling,
                      denoising=scene.cycles.use_denoising,
                      polarization_mode=scene.cycles.coherent_polarization_mode,
                      max_interface_events=scene.cycles.coherent_max_interface_events)
        durations = []
        for index in range(args.warm_measured + 1):
            start = time.monotonic()
            status = bpy.ops.render.render(write_still=(args.warm_measured == 0 or index > 0))
            durations.append(time.monotonic() - start)
            if "FINISHED" not in status:
                raise RuntimeError(f"Render returned {sorted(status)}")
            if args.warm_measured and index == 1 and durations[-1] > 60.0:
                result["warm_early_stop_reason"] = (
                    "First measured render exceeded 60 seconds; remaining repeats skipped")
                break
        result["render_seconds"] = durations[-1]
        if args.warm_measured:
            result["warmup_seconds"] = durations[0]
            result["measured_seconds"] = durations[1:]
            result["median_measured_seconds"] = statistics.median(durations[1:])
        image = bpy.data.images.load(scene.render.filepath, check_existing=False)
        height, width = int(image.size[1]), int(image.size[0])
        channels = int(image.channels)
        pixels = np.asarray(image.pixels[:], dtype=np.float64).reshape(height, width, channels)
        bpy.data.images.remove(image)
        actual = np.mean(pixels[:, :, :3], axis=2)
        with np.load(args.reference.resolve()) as reference_file:
            expected = reference_file["phase_0_radiance"]
            roi = reference_file["roi_mask"].astype(bool)
        if expected.shape != actual.shape and expected.shape[0] % actual.shape[0] == 0:
            factor = expected.shape[0] // actual.shape[0]
            expected = expected.reshape(actual.shape[0], factor, actual.shape[1], factor).mean(axis=(1, 3))
            roi = roi.reshape(actual.shape[0], factor, actual.shape[1], factor).all(axis=(1, 3))
        if actual.shape != expected.shape or roi.shape != expected.shape:
            raise RuntimeError(f"Shape mismatch: render {actual.shape}, reference {expected.shape}")
        error = actual[roi] - expected[roi]
        result["metrics"] = {
            "roi_pixels": int(np.count_nonzero(roi)),
            "all_finite": bool(np.isfinite(actual[roi]).all()),
            "render_mean_radiance": float(np.mean(actual[roi])),
            "reference_mean_radiance": float(np.mean(expected[roi])),
            "absolute_mean_error": float(abs(np.mean(error))),
            "absolute_rmse": float(np.sqrt(np.mean(error * error))),
            "maximum_absolute_error": float(np.max(np.abs(error))),
            "negative_pixel_count_roi": int(np.count_nonzero(actual[roi] < 0.0)),
            "minimum_radiance_roi": float(np.min(actual[roi])),
            "top_half_mean_radiance": float(np.mean(actual[:height // 2][roi[:height // 2]])),
            "bottom_half_mean_radiance": float(np.mean(actual[height // 2:][roi[height // 2:]])),
        }
        result["status"] = "completed"
    except BaseException as error:
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
    (args.output / "report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "completed":
        raise RuntimeError(result["error"])


if __name__ == "__main__":
    main()
