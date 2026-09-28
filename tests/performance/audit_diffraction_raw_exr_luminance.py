#!/usr/bin/env python3
"""Read-only raw-EXR audit; writes only a new standalone JSON report.

Use the Homebrew Python/OpenCV environment with OPENCV_IO_ENABLE_OPENEXR=1.
Blender only loads saved scenes to inspect their EXR color settings; it does
not render or save. Negative scene-linear RGB alone is not negative light:
spectral colors outside the RGB gamut can have negative components and positive Y.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = {"cd_dvd", "glass", "glossy", "metallic", "principled_film", "refraction",
            "covered_disc", "indirect", "thin_wall", "ashikhmin", "two_sided_glass",
            "generalized_dispersion", "coated_generalized_furnace"}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def color_settings(binary, scenes, config):
    # Factory startup avoids loading user add-ons. open_mainfile is read-only
    # here: no render, save, or preferences write is requested.
    probe = """import bpy,json,os
results={}
for name,path in SCENES.items():
    bpy.ops.wm.open_mainfile(filepath=path)
    s=bpy.context.scene
    tree=s.compositing_node_group
    outputs=[{'name':n.name,'file_name':n.file_name,'linear':n.format.linear_colorspace_settings.name,'color_management':n.format.color_management} for n in tree.nodes if n.bl_idname=='CompositorNodeOutputFile'] if tree else []
    results[name]={'blend':bpy.data.filepath,'render_linear':s.render.image_settings.linear_colorspace_settings.name,'render_color_management':s.render.image_settings.color_management,'file_outputs':outputs,'view_transform':s.view_settings.view_transform,'ocio_environment':os.environ.get('OCIO')}
print('AUDIT_COLOR_SETTINGS '+json.dumps(results))
"""
    with tempfile.TemporaryDirectory(prefix="diffraction-exr-audit-") as temporary:
        script = Path(temporary) / "inspect_color.py"
        script.write_text("SCENES=" + repr(scenes) + "\n" + probe)
        env = dict(os.environ, OCIO=str(config))
        result = subprocess.run([str(binary), "--factory-startup", "--disable-autoexec",
                                 "--background", "--python", str(script)],
                                cwd=ROOT, env=env, text=True, capture_output=True)
        marker = "AUDIT_COLOR_SETTINGS "
        lines = [line[len(marker):] for line in result.stdout.splitlines() if line.startswith(marker)]
        if result.returncode or len(lines) != 1:
            raise RuntimeError("Scene color inspection failed: " + result.stdout[-3000:] + result.stderr[-1000:])
        return json.loads(lines[0])


def coefficients(config):
    text = config.read_text()
    scene_role = re.search(r"^\s*scene_linear:\s*(.+)$", text, re.MULTILINE).group(1).strip()
    if scene_role != "Linear Rec.709":
        raise RuntimeError("Audit requires an explicitly supported RGB basis; got " + scene_role)
    block = text.split("    name: Linear Rec.709\n", 1)[1].split("  - !<ColorSpace>", 1)[0]
    match = re.search(r"matrix:\s*\[([^]]+)\]", block)
    matrix = np.array([float(value.strip()) for value in match.group(1).split(",")]).reshape(4, 4)
    # This configured matrix maps XYZ D65 to the EXR RGB basis. Its inverse
    # Y row is more precise than the rounded OCIO luma header constants.
    return scene_role, np.linalg.inv(matrix[:3, :3])[1]


def audit_image(path, y_coefficients):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise RuntimeError("Could not read raw RGB EXR: " + str(path))
    finite_pixels = np.isfinite(image).all(axis=2)
    rgb = image[:, :, [2, 1, 0]].astype(np.float64)  # OpenCV returns BGR(A).
    y = rgb @ y_coefficients
    rgb_negative = (rgb < -1e-6).any(axis=2) & finite_pixels
    y_negative = (y < -1e-6) & finite_pixels
    valid_y = y[finite_pixels]
    scale = np.maximum(1.0, np.abs(rgb).sum(axis=2))
    samples = []
    for flat_index in np.argsort(np.where(finite_pixels, y, np.inf), axis=None)[:8]:
        row, column = np.unravel_index(flat_index, y.shape)
        if not y_negative[row, column]:
            break
        samples.append({"x": int(column), "y": int(row), "rgb": rgb[row, column].tolist(),
                        "luminance": float(y[row, column]),
                        "negative_y_over_rgb_l1": float(-y[row, column] / scale[row, column])})
    return {
        "file": str(path), "sha256": sha(path),
        "width": int(image.shape[1]), "height": int(image.shape[0]), "channels": int(image.shape[2]),
        "all_channels_finite": bool(finite_pixels.all()),
        "nonfinite_pixels": int((~finite_pixels).sum()),
        "rgb_minimum": float(rgb[finite_pixels].min()) if finite_pixels.any() else None,
        "negative_rgb_pixels": int(rgb_negative.sum()),
        "negative_rgb_without_y_below_threshold_pixels": int((rgb_negative & ~y_negative).sum()),
        "minimum_y": float(valid_y.min()) if valid_y.size else None,
        "mean_y": float(valid_y.mean()) if valid_y.size else None,
        "y_below_minus_1e_minus6_pixels": int(y_negative.sum()),
        "y_below_minus_1e_minus6_fraction": float(y_negative.mean()),
        "y_negative_beyond_relative_1e6_pixels": int((y_negative & (y < -1e-6 * scale)).sum()),
        "worst_negative_y_pixels": samples,
    }


def audit_suite(folder):
    manifest_path = folder / "manifest.json"
    manifest_hash = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if set(manifest["runs"]) != REQUIRED:
        raise RuntimeError("Expected exactly 13 completed presentation scenes: " + str(folder))
    binary = Path(manifest["binary"])
    if not binary.is_absolute():
        binary = ROOT / binary
    configs = list((binary.parent.parent / "Resources").glob("*/datafiles/colormanagement/config.ocio"))
    if len(configs) != 1:
        raise RuntimeError("Could not identify packaged OCIO config")
    config = configs[0]
    space, y_coefficients = coefficients(config)
    blends = {name: str(folder / name / "denoising.blend") for name in manifest["runs"]}
    settings = color_settings(binary, blends, config)
    runs = {}
    for name, entry in manifest["runs"].items():
        report_path = folder / name / "report.json"
        report = json.loads(report_path.read_text())
        if entry["exit_code"] != 0 or report["sha256"] != manifest["sha256"]:
            raise RuntimeError("Unsuccessful or mixed-provenance render: " + name)
        noisy_outputs = [output for output in settings[name]["file_outputs"] if output["file_name"] == "noisy"]
        if (len(noisy_outputs) != 1 or noisy_outputs[0]["linear"] != space or
                settings[name]["render_linear"] != space):
            raise RuntimeError("Unsupported raw noisy EXR color space: " + name)
        path = Path(report["passes"]["noisy"]["file"])
        report_hash = sha(report_path)
        runs[name] = audit_image(path, y_coefficients)
        runs[name]["scene_color_settings"] = settings[name]
        runs[name]["source_report_sha256"] = report_hash
        if sha(report_path) != report_hash:
            raise RuntimeError("Source report changed during audit: " + name)
    if sha(manifest_path) != manifest_hash:
        raise RuntimeError("Manifest changed during audit")
    return {"presentation": str(folder), "manifest_sha256": manifest_hash,
            "binary_sha256": manifest["sha256"], "scene_linear_space": space,
            "ocio_config": str(config), "ocio_config_sha256": sha(config),
            "y_coefficients_rgb": y_coefficients.tolist(),
            "coefficient_source": "Inverse Y row of bundled OCIO XYZ-D65-to-Linear-Rec.709 matrix",
            "all_raw_noisy_pixels_finite": all(run["all_channels_finite"] for run in runs.values()),
            "total_negative_y_pixels": sum(run["y_below_minus_1e_minus6_pixels"] for run in runs.values()),
            "runs": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Standalone report already exists; choose a new output path")
    result = {"audited_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "scope": __doc__, "script_sha256": sha(__file__), "opencv_version": cv2.__version__,
              "numpy_version": np.__version__, "negative_threshold": -1e-6,
              "interpretation": "Negative RGB components can represent out-of-gamut spectral color; negative Y is reported separately. Tiny Y residuals relative to RGB magnitude can arise from float EXR rounding or matrix cancellation; this audit does not establish their cause.",
              "current": audit_suite(args.presentation.resolve())}
    if args.compare:
        result["comparison"] = audit_suite(args.compare.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output.resolve()),
                      "all_finite": result["current"]["all_raw_noisy_pixels_finite"],
                      "negative_y_pixels": result["current"]["total_negative_y_pixels"]}))


if __name__ == "__main__":
    main()
