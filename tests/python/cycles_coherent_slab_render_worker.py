"""Render one fixed glass-slab acceptance case in a fresh Blender process."""

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--case", choices=("phase_0", "phase_pi", "incoherent_connector_control",
                                           "distinct_groups", "diagonal_control"), required=True)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--phase0-data", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    report = {"case": args.case, "status": "started"}
    try:
        bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
        scene = bpy.context.scene
        assert scene.render.resolution_x == scene.render.resolution_y == 512
        assert scene.cycles.samples == 512
        if args.samples <= 0:
            raise ValueError("Sample count must be positive")
        scene.cycles.samples = args.samples
        assert scene.cycles.seed == 19
        assert scene.cycles.use_coherent_specular_connections
        assert scene.cycles.coherent_polarization_mode == "VECTOR"
        assert scene.cycles.use_bidirectional_path_tracing
        assert not scene.cycles.use_adaptive_sampling and not scene.cycles.use_denoising
        assert scene.cycles.pixel_filter_type == "BOX" and scene.cycles.filter_width == 1.0
        assert scene.cycles.sample_clamp_direct == scene.cycles.sample_clamp_indirect == 0.0
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "METAL"
        preferences.get_devices()
        devices = [device.name for device in preferences.devices if device.type == "METAL"]
        if not devices:
            raise RuntimeError("No Metal device is available")
        for device in preferences.devices:
            device.use = device.type == "METAL"
        scene.cycles.device = "GPU"
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.filepath = str(args.render.resolve())
        report.update(devices=devices, samples=args.samples, resolution=[512, 512],
                      adaptive_sampling=False, denoising=False, bdpt=True,
                      polarization_mode="VECTOR")
        status = bpy.ops.render.render(write_still=True)
        if "FINISHED" not in status:
            raise RuntimeError(f"Render returned {sorted(status)}")
        image = bpy.data.images.load(scene.render.filepath, check_existing=False)
        pixels = np.asarray(image.pixels[:], dtype=np.float64).reshape(512, 512, image.channels)
        bpy.data.images.remove(image)
        actual = np.mean(pixels[:, :, :3], axis=2)
        with np.load(args.reference.resolve()) as source:
            roi = source["roi_mask"].astype(bool)
            key = {"phase_0": "phase_0_radiance", "phase_pi": "phase_pi_radiance",
                   "incoherent_connector_control": "incoherent_radiance",
                   "distinct_groups": "distinct_groups_radiance",
                   "diagonal_control": "diagonal_control_radiance"}[args.case]
            expected = source[key]
            if actual.shape != expected.shape or roi.shape != expected.shape:
                raise RuntimeError("Render/reference/ROI shape mismatch")
            values = actual[roi]
            errors = values - expected[roi]
            report["metrics"] = {
                "roi_pixels": int(np.count_nonzero(roi)),
                "all_finite": bool(np.isfinite(values).all()),
                "render_mean_radiance": float(np.mean(values)),
                "reference_mean_radiance": float(np.mean(expected[roi])),
                "absolute_mean_error": float(abs(np.mean(errors))),
                "absolute_rmse": float(np.sqrt(np.mean(errors * errors))),
                "negative_pixel_count_roi": int(np.count_nonzero(values < 0.0)),
                "minimum_radiance_roi": float(np.min(values)),
            }
            if args.case == "phase_pi":
                if not args.phase0_data or not args.phase0_data.is_file():
                    raise RuntimeError("Missing phase-0 data for phase difference")
                with np.load(args.phase0_data) as phase0_file:
                    phase0 = phase0_file["actual_roi"]
                expected_delta = source["phase_0_radiance"][roi] - source["phase_pi_radiance"][roi]
                delta_error = phase0 - values - expected_delta
                report["phase_check"] = {
                    "absolute_mean_error": float(abs(np.mean(delta_error))),
                    "absolute_rmse": float(np.sqrt(np.mean(delta_error * delta_error))),
                }
        np.savez_compressed(args.render.with_suffix(".npz"), actual_roi=actual[roi])
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["status"] != "completed":
        raise RuntimeError(report["error"])


if __name__ == "__main__":
    main()
