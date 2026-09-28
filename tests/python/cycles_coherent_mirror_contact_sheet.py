#!/usr/bin/env python3
"""Display existing linear EXRs with one fixed exposure and sRGB transfer."""

import hashlib
import json
import os
from pathlib import Path
import sys
import argparse

import cv2
import numpy as np


CASES = ("phase_0", "phase_pi", "incoherent_connector_control")
LABELS = ("Coherent phase 0", "Coherent phase pi", "Incoherent control")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def display_rgb(linear_bgr, exposure_ev):
    exposed = np.maximum(linear_bgr.astype(np.float64) * 2.0**exposure_ev, 0.0)
    srgb = np.where(exposed <= 0.0031308, 12.92 * exposed,
                    1.055 * np.power(exposed, 1.0 / 2.4) - 0.055)
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255.0).astype(np.uint8)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=root / "build/tests/python/cycles_coherent_mirror_render_v12")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--exposure-ev", type=float, default=2.0)
    parser.add_argument("--name", default="mirror_v12")
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve() if args.output else source / f"display_fixed_ev{args.exposure_ev:g}_srgb"
    output.mkdir(parents=True, exist_ok=True)
    tiles = []
    report = {"source_directory": str(source), "source_render_report_sha256":
              sha256(source / "results.json"),
              "script_sha256": sha256(__file__),
              "display_transform": f"linear scene radiance * 2^{args.exposure_ev:g}, clip negative to zero, IEC sRGB OETF, clip above one; identical transform for all three cases",
              "exposure_ev": args.exposure_ev, "cases": {}}
    for name, label in zip(CASES, LABELS):
        exr = source / f"{name}.exr"
        pixels = cv2.imread(str(exr), cv2.IMREAD_UNCHANGED)
        if pixels is None or pixels.shape != (512, 512, 4):
            raise RuntimeError(f"Missing or unexpected EXR: {exr}")
        linear = pixels[:, :, :3]
        if not np.isfinite(linear).all():
            raise RuntimeError(f"Nonfinite EXR values: {exr}")
        displayed = display_rgb(linear, args.exposure_ev)
        png = output / f"{name}_ev{args.exposure_ev:g}_srgb.png"
        if not cv2.imwrite(str(png), displayed):
            raise RuntimeError(f"Could not write {png}")
        tile = np.full((562, 512, 3), 28, np.uint8)
        tile[50:562] = displayed
        cv2.putText(tile, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (230, 230, 230), 2, cv2.LINE_AA)
        tiles.append(tile)
        report["cases"][name] = {"exr": str(exr), "exr_sha256": sha256(exr),
                                 "png": str(png), "png_sha256": sha256(png),
                                 "linear_min": float(linear.min()),
                                 "linear_max": float(linear.max()),
                                 "negative_channels_clipped": int((linear < 0).sum())}
    sheet = np.concatenate(tiles, axis=1)
    footer = np.full((62, sheet.shape[1], 3), 28, np.uint8)
    cv2.putText(footer, f"Fixed display: {args.exposure_ev:+g} EV, sRGB transfer, no per-image scaling",
                (16, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 2, cv2.LINE_AA)
    cv2.putText(footer, f"Linear radiance 0 -> black; {2.0**-args.exposure_ev:g} -> display white",
                (16, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 190, 190), 1, cv2.LINE_AA)
    sheet = np.concatenate((sheet, footer), axis=0)
    contact = output / f"{args.name}_contact_sheet_ev{args.exposure_ev:g}_srgb.png"
    if not cv2.imwrite(str(contact), sheet):
        raise RuntimeError(f"Could not write {contact}")
    report["contact_sheet"] = {"path": str(contact), "sha256": sha256(contact),
                               "dimensions": [int(sheet.shape[1]), int(sheet.shape[0])]}
    (output / "display_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(contact)


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    sys.exit(main())
