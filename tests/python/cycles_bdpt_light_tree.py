#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Metal BDPT light-tree regression and linear-image convergence measurements.

Run with a Python containing numpy and OpenImageIO. Each render has a fresh process;
logs, commands, EXRs and incremental JSON results are retained. First-use Metal compilation
is included in elapsed time, so use a second run for timing comparisons.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import OpenImageIO as oiio


def pixels(path):
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read {path}")
    data = image.read_image(format=oiio.FLOAT)[..., :3].astype(np.float64)
    image.close()
    if not np.isfinite(data).all():
        raise AssertionError(f"Nonfinite pixels: {path}")
    return data


def validate_results(results):
    # Diffuse controls have low-variance references. Do not apply the same test to sharp
    # caustics: even a high-sample PT image can miss their transport class entirely.
    if results["settings"]["samples"] >= 128 and results["settings"]["reference_samples"] >= 4096 and results["settings"]["seeds"] >= 3:
        for case, summary in results["comparisons"].items():
            if case != "glass" and summary["tree"]["relative_mean_error"] >= 0.02:
                raise AssertionError(f"{case}: tree BDPT mean differs from PT by at least 2%")
            if case in {"point", "area", "mesh", "mixed", "linked", "volume"} and summary["mse_gain"] <= 1.0:
                raise AssertionError(f"{case}: tree did not reduce error in the many-light fixture")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", default=["area", "point", "mesh", "sun", "world", "mixed", "linked", "glass", "volume", "bounded_volume"])
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--reference-samples", type=int, default=4096)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=48)
    parser.add_argument("--verify-only", action="store_true", help="Validate existing results.json without rendering")
    options = parser.parse_args()
    if options.verify_only:
        validate_results(json.loads((options.output / "results.json").read_text()))
        print("Recorded light-tree regression checks passed")
        return
    options.output.mkdir(parents=True, exist_ok=True)
    smoke = Path(__file__).with_name("cycles_bdpt_smoke.py").resolve()
    results = {"settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()}, "renders": {}, "comparisons": {}}
    env = os.environ.copy()
    env.setdefault("BLENDER_USER_RESOURCES", str(options.output.resolve() / "user"))
    Path(env["BLENDER_USER_RESOURCES"]).mkdir(parents=True, exist_ok=True)

    def save():
        (options.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    def render(name, args):
        output = options.output.resolve() / (name + ".exr")
        command = [str(options.blender.resolve()), "--background", "--factory-startup", "--debug-cycles",
                   "--python-exit-code", "1", "--python", str(smoke), "--",
                   "--resolution", str(options.resolution), "--light-paths", "32768",
                   "--output", str(output), *args]
        print(f"Rendering {name}", flush=True)
        log = options.output / (name + ".log")
        with log.open("w") as transcript:
            completed = subprocess.run(command, stdout=transcript, stderr=subprocess.STDOUT, env=env)
        text = log.read_text()
        line = next((line for line in text.splitlines() if line.startswith("BDPT_SMOKE ")), None)
        if completed.returncode or line is None:
            raise RuntimeError(f"{name} failed: {log}\n{text[-4000:]}")
        if "--no-light-tree" not in args and "Use light tree with" not in text:
            raise AssertionError(f"{name}: requested tree was not built")
        if "--bdpt" in args and "--device" not in args and "Use light distribution with" not in text:
            raise AssertionError(f"{name}: missing emission CDF alongside tree")
        if "--bdpt" in args and "--device" not in args and "BDPT light cache:" not in text:
            raise AssertionError(f"{name}: BDPT cache was not scheduled")
        metrics = {key: float(value) for key, value in re.findall(r"([a-z0-9_]+)=([-+0-9.eE]+)", line)}
        timing = re.search(r"Render time \(without synchronization\): ([0-9.eE+-]+)", text)
        if timing is None:
            raise AssertionError(f"{name}: missing render-only timing")
        metrics["render_seconds"] = float(timing.group(1))
        results["renders"][name] = {"command": command, **metrics}
        save()
        return pixels(output), metrics

    scenes = {
        "bounded_volume": ["--volume-only", "--bounded-volume", "--world-scatter", "0.08", "--volume-bounces", "2", "--extra-lights", "31"],
        "mixed": ["--no-glass", "--diffuse-node", "--extra-lights", "15", "--world-strength", "0.8", "--sun-fill", "0.2"],
        "linked": ["--no-glass", "--diffuse-node", "--extra-lights", "15", "--light-link", "FLOOR"],
        "area": ["--no-glass", "--diffuse-node", "--extra-lights", "31"],
        "point": ["--no-glass", "--diffuse-node", "--light-type", "POINT", "--extra-lights", "31"],
        "mesh": ["--no-glass", "--diffuse-node", "--mesh-light", "--extra-lights", "15"],
        "sun": ["--no-glass", "--diffuse-node", "--light-type", "SUN", "--sun-angle", "3"],
        "world": ["--no-glass", "--diffuse-node", "--light-type", "WORLD"],
        "glass": ["--extra-lights", "15"],
        "volume": ["--volume-only", "--world-scatter", "0.08", "--volume-bounces", "2", "--extra-lights", "31"],
    }
    for case in options.cases:
        scene = scenes[case]
        ref, _ = render(case + "_reference", scene + ["--samples", str(options.reference_samples), "--seed", "7919"])
        summary = {}
        for tree in (False, True):
            errors, seconds, images = [], [], []
            for seed in range(options.seeds):
                args = scene + ["--bdpt", "--samples", str(options.samples), "--seed", str(seed)]
                if not tree:
                    args += ["--no-light-tree"]
                image, metrics = render(f"{case}_{'tree' if tree else 'flat'}_{seed}", args)
                errors.append(float(np.mean((image - ref) ** 2)))
                seconds.append(metrics["render_seconds"])
                images.append(image)
            ensemble = np.mean(images, axis=0)
            summary["tree" if tree else "flat"] = {
                "mse": float(np.mean(errors)), "mse_per_seed": errors,
                "render_seconds": float(np.mean(seconds)), "mean": float(ensemble.mean()),
                "relative_mean_error": float(abs(ensemble.mean() - ref.mean()) / max(ref.mean(), 1e-12)),
            }
        summary["reference_mean"] = float(ref.mean())
        summary["mse_gain"] = summary["flat"]["mse"] / max(summary["tree"]["mse"], 1e-30)
        summary["time_normalized_gain"] = (summary["flat"]["mse"] * summary["flat"]["render_seconds"] /
                                          max(summary["tree"]["mse"] * summary["tree"]["render_seconds"], 1e-30))
        results["comparisons"][case] = summary
        save()
        print(case, json.dumps(summary), flush=True)
    validate_results(results)


if __name__ == "__main__":
    main()
