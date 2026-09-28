#!/usr/bin/env python3
"""Inspect saved raw Thin Wall furnace cases at fixed -1EV/sRGB; never rerender."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def display(raw):
    x = np.maximum(np.asarray(raw, dtype=np.float64) * 0.5, 0)
    x = np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--renders", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    report_path = a.renders / "report.json"
    report = json.loads(report_path.read_text())
    tiles, inputs = [], {}
    for name, case in report["cases"].items():
        path = Path(case["exr"])
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.shape != (64, 64, 4) or not np.isfinite(raw).all():
            raise RuntimeError(f"Invalid raw EXR {path}")
        mean = raw[16:48, 16:48, :3].astype(np.float64).mean(axis=(0, 1))[::-1]
        error = float(np.max(np.abs(mean - case["roi_rgb_mean"])))
        if error > 1e-8:
            raise RuntimeError(f"Raw/report ROI disagreement {name}: {error}")
        tile = np.full((314, 256, 3), 28, np.uint8)
        tile[38:294] = cv2.resize(display(raw[:, :, :3]), (256, 256),
                                  interpolation=cv2.INTER_NEAREST)
        cv2.putText(tile, name, (7, 25), cv2.FONT_HERSHEY_SIMPLEX, .46,
                    (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(tile, "RGB " + "/".join(f"{v:.4f}" for v in mean),
                    (7, 308), cv2.FONT_HERSHEY_SIMPLEX, .39,
                    (240, 240, 240), 1, cv2.LINE_AA)
        tiles.append(tile)
        inputs[name] = {"exr": str(path.resolve()), "sha256": sha(path),
                        "raw_roi_rgb_mean": mean.tolist(), "report_mean_max_error": error}
    columns = 4
    rows = []
    for i in range(0, len(tiles), columns):
        row = tiles[i:i + columns]
        row += [np.full_like(tiles[0], 28)] * (columns - len(row))
        rows.append(np.concatenate(row, axis=1))
    footer = np.full((70, 256 * columns, 3), 28, np.uint8)
    cv2.putText(footer, "Fixed -1 EV + sRGB; unit white = 0.5 linear; raw 64px nearest enlargement",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(footer, "Finite physical unit-world validation, no denoising; these are not beauty renders",
                (10, 54), cv2.FONT_HERSHEY_SIMPLEX, .5, (240, 240, 240), 1, cv2.LINE_AA)
    png = a.output / "thin_wall_raw_fixed_ev-1.png"
    if not cv2.imwrite(str(png), np.concatenate(rows + [footer], axis=0)):
        raise RuntimeError("PNG save failed")
    summary = {"scope": report["scope"], "presentation": "fixed -1EV then IEC sRGB, no individual scaling",
               "roi": "pixels [16:48,16:48] in raw EXR orientation",
               "case_count": len(inputs), "passed": report["passed"],
               "all_finite": all(c["all_finite"] for c in report["cases"].values()),
               "max_rgb_unit_energy_error": max(abs(v - 1) for c in inputs.values() for v in c["raw_roi_rgb_mean"]),
               "negative_channels": sum(c["negative_channels"] for c in report["cases"].values()),
               "worker_report": str(report_path.resolve()), "worker_report_sha256": sha(report_path),
               "script_sha256": sha(Path(__file__)), "inputs": inputs,
               "png": str(png.resolve()), "png_sha256": sha(png)}
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(png.resolve())


if __name__ == "__main__":
    main()
