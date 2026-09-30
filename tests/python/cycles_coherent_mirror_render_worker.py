# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""One scene worker for run_cycles_coherent_mirror_acceptance.py; Blender only."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--render", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--case", choices=("phase_0", "phase_pi", "incoherent_bdpt_control", "incoherent_connector_control"), required=True)
    parser.add_argument("--phase0-data", type=Path)
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--pt-guiding", action="store_true")
    transport.add_argument("--pt", action="store_true")
    transport.add_argument("--photon-mapping", action="store_true",
                           help="Metal path tracing with caustic photon mapping")
    parser.add_argument("--require-specular", action="store_true")
    parser.add_argument("--scene-budgets", action="store_true")
    parser.add_argument("--minimum-glossy-bounces", type=int)
    parser.add_argument("--cpu", action="store_true",
                        help="Render on the CPU device instead of Metal (BDPT is Metal-only)")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    result = {
        "case": args.case,
        "worker_status": "started",
        "blend": str(args.blend.resolve()),
        "render_path": str(args.render.resolve()),
        "samples": 128,
        "device": "METAL",  # Replaced below for --cpu.
        "pixel_filter": "BOX",
        "adaptive_sampling": False,
        "denoising": False,
    }

    try:
        bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
        scene = bpy.context.scene
        result["coherent_specular_enabled"] = scene.cycles.use_coherent_specular_connections
        if args.require_specular and not result["coherent_specular_enabled"]:
            raise RuntimeError("saved scene does not enable coherent specular connections")
        scene.render.engine = "CYCLES"
        scene.cycles.samples = 128
        scene.cycles.seed = 19
        scene.cycles.device = "CPU" if args.cpu else "GPU"
        if args.cpu and not (args.pt_guiding or args.pt):
            raise RuntimeError("BDPT is a Metal feature; CPU cases must request --pt or --pt-guiding")
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        scene.cycles.use_bidirectional_path_tracing = not (args.pt_guiding or args.pt or args.photon_mapping)
        scene.cycles.use_photon_mapping = args.photon_mapping
        if args.photon_mapping:
            # Photon bounces are independent of the scene budget; match them so
            # photons cover exactly the path classes the reference covers.
            scene.cycles.photon_max_bounces = scene.cycles.max_bounces
        result["photon_mapping"] = args.photon_mapping
        scene.cycles.use_guiding = args.pt_guiding
        result["bidirectional_path_tracing"] = scene.cycles.use_bidirectional_path_tracing
        result["guiding"] = scene.cycles.use_guiding
        if not args.scene_budgets:
            scene.cycles.max_bounces = 2
            scene.cycles.diffuse_bounces = 1
            scene.cycles.glossy_bounces = 1
            scene.cycles.transmission_bounces = 0
            scene.cycles.transparent_max_bounces = 0
            scene.cycles.volume_bounces = 0
        result["bounce_budgets"] = {name: getattr(scene.cycles, name) for name in
                                    ("max_bounces", "diffuse_bounces", "glossy_bounces", "transmission_bounces")}
        if args.minimum_glossy_bounces is not None:
            result["required_glossy_bounces"] = args.minimum_glossy_bounces
            if args.minimum_glossy_bounces < 0:
                raise RuntimeError("minimum glossy bounce requirement must be nonnegative")
            if scene.cycles.glossy_bounces < args.minimum_glossy_bounces:
                raise RuntimeError(
                    f"reference requires at least {args.minimum_glossy_bounces} glossy bounces, "
                    f"but the effective scene budget is {scene.cycles.glossy_bounces}; "
                    "render was not attempted")
        scene.cycles.pixel_filter_type = "BOX"
        scene.cycles.filter_width = 1.0
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"

        if args.cpu:
            result["device"] = "CPU"
        else:
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = "METAL"
            preferences.get_devices()
            metal_devices = [device for device in preferences.devices if device.type == "METAL"]
            for device in preferences.devices:
                device.use = device.type == "METAL"
            if not metal_devices:
                raise RuntimeError("No Metal device is available; render was not attempted")
            result["metal_devices"] = [device.name for device in metal_devices]
        args.render.parent.mkdir(parents=True, exist_ok=True)
        scene.render.filepath = str(args.render.resolve())
        render_start = time.perf_counter()
        status = bpy.ops.render.render(write_still=True)
        result["render_seconds_including_initialization"] = time.perf_counter() - render_start
        result["render_status"] = sorted(status)
        if "FINISHED" not in status:
            raise RuntimeError(f"render operator returned {sorted(status)}")

        image = bpy.data.images.load(str(args.render.resolve()), check_existing=False)
        width, height = int(image.size[0]), int(image.size[1])
        result["resolution"] = [width, height]
        channels = int(image.channels)
        values = np.asarray(image.pixels[:], dtype=np.float64)
        bpy.data.images.remove(image)
        pixels = values.reshape((height, width, channels))
        actual = np.mean(pixels[:, :, :3], axis=2)
        reference_file = np.load(str(args.reference.resolve()))
        expected_key = {
            "phase_0": "phase_0_radiance",
            "phase_pi": "phase_pi_radiance",
            "incoherent_bdpt_control": "incoherent_radiance",
            "incoherent_connector_control": "incoherent_radiance",
        }[args.case]
        expected = reference_file[expected_key]
        if expected.ndim == 3:
            actual = pixels[:, :, :3]
        roi = reference_file["roi_mask"].astype(bool)
        if actual.shape != expected.shape or roi.shape != expected.shape[:2]:
            raise RuntimeError(
                f"render/reference shape mismatch: render={actual.shape} reference={expected.shape} ROI={roi.shape}"
            )
        if not np.isfinite(actual[roi]).all():
            raise RuntimeError("render contains non-finite radiance in the predeclared ROI")

        actual_roi = actual[roi]
        expected_roi = expected[roi]
        error = actual_roi - expected_roi
        metrics = {
            "roi_pixels": int(np.count_nonzero(roi)),
            "render_mean_radiance": float(np.mean(actual_roi)),
            "reference_mean_radiance": float(np.mean(expected_roi)),
            "absolute_mean_error": float(abs(np.mean(actual_roi) - np.mean(expected_roi))),
            "absolute_rmse": float(np.sqrt(np.mean(error * error))),
            "maximum_absolute_error": float(np.max(np.abs(error))),
            "all_finite": True,
            "minimum_radiance": float(np.min(actual_roi)),
        }
        if expected.ndim == 3:
            black = roi & np.all(expected == 0.0, axis=2)
            metrics["black_pixels"] = int(np.count_nonzero(black))
            metrics["black_max_absolute_radiance"] = float(np.max(np.abs(actual[black]))) if np.any(black) else 0.0
            metrics["rgb_mean_absolute_error"] = np.abs(np.mean(error, axis=0)).tolist()
            metrics["rgb_rmse"] = np.sqrt(np.mean(error * error, axis=0)).tolist()
        result["metrics"] = metrics
        np.savez_compressed(args.render.with_suffix(".npz"), actual_roi=actual_roi)

        if args.case == "phase_pi":
            if not args.phase0_data or not args.phase0_data.is_file():
                raise RuntimeError("phase-0 radiance array is required for the predeclared phase check")
            phase0 = np.load(str(args.phase0_data.resolve()))["actual_roi"]
            if phase0.shape != actual_roi.shape:
                raise RuntimeError("phase-0 and phase-pi ROI shapes differ")
            expected_difference = reference_file["phase_0_radiance"][roi] - \
                reference_file["phase_pi_radiance"][roi]
            actual_difference = phase0 - actual_roi
            phase_metrics = {
                "actual_mean_phase_difference": float(np.mean(actual_difference)),
                "reference_mean_phase_difference": float(np.mean(expected_difference)),
                "absolute_phase_mean_error": float(abs(np.mean(actual_difference) - np.mean(expected_difference))),
                "phase_difference_rmse": float(np.sqrt(np.mean((actual_difference - expected_difference) ** 2))),
            }
            if expected.ndim == 3:
                phase_error = actual_difference - expected_difference
                phase_metrics["rgb_phase_mean_absolute_error"] = np.abs(np.mean(phase_error, axis=0)).tolist()
                phase_metrics["rgb_phase_rmse"] = np.sqrt(np.mean(phase_error * phase_error, axis=0)).tolist()
            result["phase_check"] = phase_metrics

        result["worker_status"] = "completed"
    except BaseException as error:
        result["worker_status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["worker_status"] != "completed":
        raise RuntimeError(result.get("error", "worker failed"))


if __name__ == "__main__":
    main()
