#!/usr/bin/env python3
"""Read saved final_benchmark.py outputs; compare pixels/timing without rendering."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(directory):
    report = json.loads((directory / "report.json").read_text())
    pixels = np.load(directory / "pixels.npy", allow_pickle=False)
    measured = [r["seconds"] for r in report["runs"] if not r["warmup"]]
    if len(measured) != 3 or sum(r["warmup"] for r in report["runs"]) != 1:
        raise ValueError(f"Expected exactly one warmup plus three trials: {directory}")
    if not np.isfinite(pixels).all() or not np.isfinite(measured).all():
        raise ValueError(f"Nonfinite pixels/times: {directory}")
    return report, pixels, statistics.median(measured)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current", type=Path, required=True)
    p.add_argument("--v32", type=Path, default=ROOT / "build/tests/python/cycles_diffraction_benchmark_default_off_v32")
    p.add_argument("--v3", type=Path, default=ROOT / "tests/output/diffraction/benchmark_final_v3")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    current, pixels, median = read(a.current)
    settings = ("samples", "adaptive_sampling", "denoising", "transport", "persistent_data", "resolution")
    comparisons = {}
    for name, directory in (("v32", a.v32), ("historical_v3", a.v3)):
        old, old_pixels, old_median = read(directory)
        scene_match = current["scene_sha256"] == old["scene_sha256"]
        script_match = current["script_sha256"] == old["script_sha256"]
        settings_match = all(current.get(k) == old.get(k) for k in settings)
        shape_match = pixels.shape == old_pixels.shape
        if not (scene_match and script_match and settings_match and shape_match):
            raise ValueError(f"Unmatched scene/script/settings/pixel shape against {name}")
        difference = pixels.astype(np.float64) - old_pixels.astype(np.float64)
        comparisons[name] = {
            "scene_sha256_match": scene_match, "script_sha256_match": script_match,
            "reported_settings_match": settings_match, "raw_shape": list(pixels.shape),
            "all_finite": True, "raw_max_absolute_difference": float(np.abs(difference).max()),
            "raw_rmse": float(np.sqrt(np.mean(difference * difference))),
            "baseline_median_seconds": old_median, "current_median_seconds": median,
            "current_over_baseline_ratio": median / old_median,
            "observed_percent_slower": 100 * (median / old_median - 1),
            "baseline_report": str((directory / "report.json").resolve()),
            "baseline_report_sha256": sha(directory / "report.json"),
            "baseline_pixels_sha256": sha(directory / "pixels.npy")}
    result = {"scope": "Read-only saved-output comparison; unchanged final_benchmark.py, one warmup plus three measured runs",
              "timing_limit": "One matched fixture, EXR writing included; no universal fastest, zero overhead, equal-error or isolated causal claim",
              "current_report": str((a.current / "report.json").resolve()),
              "current_report_sha256": sha(a.current / "report.json"),
              "current_pixels_sha256": sha(a.current / "pixels.npy"),
              "comparison_script_sha256": sha(Path(__file__)),
              "scene_sha256": current["scene_sha256"], "benchmark_script_sha256": current["script_sha256"],
              "reported_settings": {k: current.get(k) for k in settings}, "comparisons": comparisons}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(a.output.resolve())


if __name__ == "__main__":
    main()
