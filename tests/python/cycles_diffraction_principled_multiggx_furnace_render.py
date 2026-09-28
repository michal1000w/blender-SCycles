"""Render six saved neutral mixed Principled furnaces and record raw linear ROI energy."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np


CASES = tuple(f"{model}_{side}" for model in ("two_sided", "native", "single_event")
              for side in ("front", "back"))
ROI = (slice(16, 48), slice(16, 48))
ENERGY_TOLERANCE = 0.03


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-cases", action="store_true", help="Use explicit independent targets/gates from fixture manifest")
    parser.add_argument("--osl", action="store_true", help="CPU OSL; render only two-sided front/back")
    parser.add_argument("--bdpt-guiding", action="store_true", help="one Metal mixed front case with BDPT and guiding")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.osl and args.bdpt_guiding:
        parser.error("--osl and --bdpt-guiding cannot be combined")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "64x64, 256 fixed samples, Principled front/back furnace; material scope recorded from saved fixture",
              "transport": "BDPT+guiding" if args.bdpt_guiding else "PT",
              "cases": {}, "energy_tolerance_absolute_each_rgb": ENERGY_TOLERANCE,
              "front_back_tolerance_absolute_each_rgb": ENERGY_TOLERANCE,
              "central_roi": "pixels [16:48,16:48]", "device": "CPU OSL" if args.osl else "METAL",
              "binary": str(Path(bpy.app.binary_path).resolve()),
              "binary_sha256": hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
              "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report_path = args.output / "report.json"
    fixture_manifest = json.loads((args.fixtures / "manifest.json").read_text()) if args.manifest_cases else None
    cases = tuple(fixture_manifest["scenes"]) if fixture_manifest else CASES
    if fixture_manifest:
        report["scope"] = fixture_manifest["scope"]
        report["energy_tolerance_absolute_each_rgb"] = fixture_manifest["energy_tolerance_absolute_each_rgb"]
        report["front_back_tolerance_absolute_each_rgb"] = fixture_manifest["energy_tolerance_absolute_each_rgb"]
        report["fixture_manifest"] = str((args.fixtures / "manifest.json").resolve())
    try:
        selected = cases[:1] if args.bdpt_guiding else (cases[:2] if args.osl else cases)
        for name in selected:
            scene_file = (args.fixtures / f"{name}.blend").resolve()
            bpy.ops.wm.open_mainfile(filepath=str(scene_file))
            scene = bpy.context.scene
            report["material_scope"] = scene["furnace_scope"]
            assert scene.render.resolution_x == scene.render.resolution_y == 64
            assert scene.cycles.samples == 256
            assert not scene.cycles.use_adaptive_sampling and not scene.cycles.use_denoising
            assert not scene.cycles.use_coherent_specular_connections
            assert scene["furnace_model"] == name.rsplit("_", 1)[0]
            assert scene["furnace_side"] == name.rsplit("_", 1)[1]
            if args.osl:
                scene.cycles.device = "CPU"
                scene.cycles.shading_system = True
                devices = ["CPU OSL"]
            else:
                preferences = bpy.context.preferences.addons["cycles"].preferences
                preferences.compute_device_type = "METAL"
                preferences.get_devices()
                devices = [device.name for device in preferences.devices if device.type == "METAL"]
                if not devices:
                    raise RuntimeError("No Metal device available")
                for device in preferences.devices:
                    device.use = device.type == "METAL"
                scene.cycles.device = "GPU"
            scene.cycles.use_bidirectional_path_tracing = args.bdpt_guiding
            scene.cycles.use_guiding = args.bdpt_guiding
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
            entry = {"scene": str(scene_file), "scene_sha256": hashlib.sha256(scene_file.read_bytes()).hexdigest(),
                     "exr": str(render_path), "exr_sha256": hashlib.sha256(render_path.read_bytes()).hexdigest(),
                     "render_seconds": seconds, "metal_devices": devices,
                     "samples": scene.cycles.samples, "seed": scene.cycles.seed,
                     "adaptive_sampling": scene.cycles.use_adaptive_sampling,
                     "denoising": scene.cycles.use_denoising,
                     "bdpt": scene.cycles.use_bidirectional_path_tracing, "guiding": scene.cycles.use_guiding,
                     "roi_rgb_mean": [float(value) for value in rgb.mean(axis=(0, 1))],
                     "roi_rgb_min": [float(value) for value in rgb.min(axis=(0, 1))],
                     "all_finite": bool(np.isfinite(pixels).all()),
                     "negative_channels": int(np.count_nonzero(rgb < 0.0))}
            report["cases"][name] = entry
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"{name}: {entry['roi_rgb_mean']}", flush=True)
        if fixture_manifest:
            tolerance = fixture_manifest["energy_tolerance_absolute_each_rgb"]
            report["energy_passed"] = all(
                report["cases"][name]["all_finite"] and
                report["cases"][name]["negative_channels"] == 0 and
                all(abs(value-target) <= tolerance for value,target in zip(
                    report["cases"][name]["roi_rgb_mean"],
                    fixture_manifest["scenes"][name]["expected_linear_rgb_radiance"]))
                for name in selected)
            pairs = [pair for pair in fixture_manifest.get("front_back_pairs",[])
                     if all(name in selected for name in pair)]
            report["front_back_passed"] = all(
                all(abs(a-b) <= tolerance for a,b in zip(report["cases"][first]["roi_rgb_mean"],
                                                       report["cases"][second]["roi_rgb_mean"]))
                for first,second in pairs)
            report["passed"] = report["energy_passed"] and report["front_back_passed"]
        else:
            gated = ("two_sided",) if (args.osl or args.bdpt_guiding) else ("two_sided", "native")
            sides = ("front",) if args.bdpt_guiding else ("front", "back")
            report["energy_passed"] = all(
                report["cases"][f"{model}_{side}"]["all_finite"] and
                report["cases"][f"{model}_{side}"]["negative_channels"] == 0 and
                all(abs(value - 1.0) <= ENERGY_TOLERANCE for value in
                    report["cases"][f"{model}_{side}"]["roi_rgb_mean"])
                for model in gated for side in sides)
            report["front_back_passed"] = None if args.bdpt_guiding else all(
                all(abs(a - b) <= ENERGY_TOLERANCE for a, b in zip(
                    report["cases"][f"{model}_front"]["roi_rgb_mean"],
                    report["cases"][f"{model}_back"]["roi_rgb_mean"]))
                for model in gated)
            report["passed"] = report["energy_passed"] and (args.bdpt_guiding or report["front_back_passed"])
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}", passed=False)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["passed"]:
        raise RuntimeError(report.get("error", "Furnace energy gate failed"))


if __name__ == "__main__":
    main()
