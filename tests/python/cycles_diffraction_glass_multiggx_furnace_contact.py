#!/usr/bin/env python3
"""Present saved Glass furnace EXRs at one fixed display exposure.

The render worker owns the pass/fail report. This reads its raw EXRs without
altering their values and makes a front/back comparison image for inspection.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


MODEL_ORDER = ("two_sided", "native", "single_event")
SIDES = ("front", "back")
EV = -1.0


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(linear_bgr):
    exposed = np.maximum(np.asarray(linear_bgr, dtype=np.float64) * 2.0**EV, 0.0)
    srgb = np.where(exposed <= 0.0031308, 12.92 * exposed,
                    1.055 * exposed ** (1.0 / 2.4) - 0.055)
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.renders / "report.json"
    report = json.loads(report_path.read_text())
    models = tuple(model for model in MODEL_ORDER if all(
        f"{model}_{side}" in report["cases"] for side in SIDES))
    if "two_sided" not in models or len(models) < 2:
        raise RuntimeError("Expected two-sided and at least one comparison model")
    images = []
    inputs = {}
    for side in SIDES:
        row = []
        for model in models:
            name = f"{model}_{side}"
            path = args.renders / f"{name}.exr"
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None or raw.shape != (64, 64, 4) or not np.isfinite(raw).all():
                raise RuntimeError(f"Missing, nonfinite, or unexpected EXR: {path}")
            roi = raw[16:48, 16:48, :3]
            mean_rgb = roi.mean(axis=(0, 1))[::-1]
            worker_mean = np.array(report["cases"][name]["roi_rgb_mean"])
            if np.max(np.abs(mean_rgb - worker_mean)) > 1e-5:
                raise RuntimeError(f"Independent EXR mean disagrees with worker for {name}")
            tile = np.full((316, 256, 3), 30, dtype=np.uint8)
            image = cv2.resize(display(raw[:, :, :3]), (256, 256),
                               interpolation=cv2.INTER_NEAREST)
            tile[42:298] = image
            label = f"{model.replace('_', ' ')} {side}"
            cv2.putText(tile, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (240, 240, 240), 1, cv2.LINE_AA)
            numbers = "RGB " + " / ".join(f"{value:.3f}" for value in mean_rgb)
            cv2.putText(tile, numbers, (8, 311), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, (240, 240, 240), 1, cv2.LINE_AA)
            row.append(tile)
            inputs[name] = {"exr": str(path.resolve()), "sha256": sha256(path),
                            "independent_roi_rgb_mean": mean_rgb.tolist()}
        images.append(np.concatenate(row, axis=1))
    footer = np.full((46, 256 * len(models), 3), 30, dtype=np.uint8)
    cv2.putText(footer, f"Fixed -1 EV + sRGB; same scale for all {2 * len(models)} images",
                (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)
    sheet = np.concatenate((*images, footer), axis=0)
    png = args.output / "glass_multiggx_furnace_fixed_ev-1.png"
    if not cv2.imwrite(str(png), sheet):
        raise RuntimeError(f"Failed to save {png}")
    manifest = {"display": "linear radiance times 2^-1, then IEC sRGB transfer; no per-image scaling",
                "roi": "pixels [16:48,16:48]", "worker_report": str(report_path.resolve()),
                "worker_report_sha256": sha256(report_path), "inputs": inputs,
                "contact_sheet": {"path": str(png.resolve()), "sha256": sha256(png)}}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(png.resolve())


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    main()
