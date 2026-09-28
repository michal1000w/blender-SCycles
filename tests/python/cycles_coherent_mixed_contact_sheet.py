"""Display four mixed R/T renders and independent references at one fixed exposure."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np


CASES = ("phase_0", "phase_pi", "distinct_groups", "diagonal_control")
LABELS = ("Phase 0", "Phase pi", "Distinct groups", "Tiny Lc: diagonals")
EV = -2.0


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(linear):
    exposed = np.maximum(np.asarray(linear, dtype=np.float64) * (2.0**EV), 0.0)
    srgb = np.where(exposed <= 0.0031308, 12.92 * exposed,
                    1.055 * exposed ** (1.0 / 2.4) - 0.055)
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def panel(image_bgr, title):
    tile = np.full((562, 512, 3), 28, np.uint8)
    tile[50:] = image_bgr
    cv2.putText(tile, title, (13, 32), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return tile


def write_sheet(images, labels, output, banner):
    sheet = np.concatenate([panel(image, label) for image, label in zip(images, labels)], axis=1)
    footer = np.full((58, sheet.shape[1], 3), 28, np.uint8)
    cv2.putText(footer, f"{banner} | Fixed {EV:+g} EV, sRGB transfer, no per-image scaling",
                (15, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (235, 235, 235), 2, cv2.LINE_AA)
    sheet = np.concatenate((sheet, footer), axis=0)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Could not write {output}")
    return {"path": str(output), "sha256": digest(output),
            "dimensions": [int(sheet.shape[1]), int(sheet.shape[0])]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with np.load(args.reference.resolve()) as source:
        reference = [source[f"{name}_radiance"].astype(np.float64) for name in CASES]
    actual = []
    inputs = {}
    for name in CASES:
        exr = (args.renders / f"{name}.exr").resolve()
        pixels = cv2.imread(str(exr), cv2.IMREAD_UNCHANGED)
        if pixels is None or pixels.shape[:2] != (512, 512) or pixels.shape[2] < 3:
            raise RuntimeError(f"Missing or unexpected EXR: {exr}")
        rgb = pixels[:, :, :3]
        if not np.isfinite(rgb).all():
            raise RuntimeError(f"Nonfinite EXR: {exr}")
        actual.append(rgb)
        inputs[name] = {"exr": str(exr), "sha256": digest(exr)}
    render_png = args.output / "mixed_v21_actual_128spp_ev-2_srgb.png"
    reference_png = args.output / "mixed_v5_independent_reference_ev-2_srgb.png"
    manifest = {
        "display": "fixed linear radiance * 2^-2, clamp negative to zero, IEC sRGB OETF; same for actual and independent reference",
        "render_report": str((args.renders / "results.json").resolve()),
        "reference_npz": str(args.reference.resolve()),
        "reference_sha256": digest(args.reference.resolve()),
        "cases": inputs,
        "actual": write_sheet([display(image) for image in actual], LABELS, render_png,
                              "Actual Metal BDPT, 128 fixed samples"),
        "reference": write_sheet([display(np.repeat(image[:, :, None], 3, axis=2)) for image in reference],
                                 LABELS, reference_png, "Independent saved-float analytic reference"),
    }
    (args.output / "display_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(render_png)
    print(reference_png)


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    main()
