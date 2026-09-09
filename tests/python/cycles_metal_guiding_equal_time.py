#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Calibrate sample counts using time alone, then compare independent seeds at matched wall time.

Cycles' time-limit predictor can overshoot a short budget before it has a steady-state timing
estimate. This experiment instead calibrates fixed sample counts for each variant, including
fresh field training and output time. Calibration never examines image quality. All calibration
runs and achieved timing differences are retained; a failed timing match is reported explicitly.
"""

import argparse
import json
import pathlib

import numpy as np

from cycles_metal_guiding_benchmark import (
    digest, reference_regions, render_reference, run_render, scene_digest, summarize,
)


def calibrate(args, scene, integrator, guiding, device="GPU"):
    variant = "cpu_guided" if device == "CPU" else "guided" if guiding else "baseline"
    observations = []
    samples = 128
    for iteration in range(8):
        name = f"{scene}_{integrator}_{variant}_calibration_{iteration}"
        metrics, _ = run_render(args, scene, "pt" if device == "CPU" else integrator,
                               guiding, samples, 1729, name, device=device, warmup=True)
        seconds = metrics["seconds"]
        previous = next((item for item in reversed(observations) if item["samples"] == samples), None)
        stable = previous is not None and abs(seconds - previous["seconds"]) <= 0.03 * seconds
        observations.append(dict(samples=samples, seconds=seconds, repeated_timing_stable=stable))
        on_target = abs(seconds - args.wall_seconds) <= 0.04 * args.wall_seconds
        if on_target and stable:
            break
        if previous is not None and not stable:
            # Asynchronous kernel specialization can finish between the first warm render
            # and its repeat. Re-estimate from the new speed, retaining the drift evidence.
            samples = max(16, min(1048576, int(round(samples * args.wall_seconds / seconds))))
            continue
        if on_target:
            # Confirm the same sample count after specialization has had time to settle.
            continue
        if len(observations) == 1:
            samples = max(256, min(8192, int(samples * args.wall_seconds / seconds * 1.5)))
            continue
        recent = observations[-3:]
        x = np.array([item["samples"] for item in recent], dtype=float)
        y = np.array([item["seconds"] for item in recent], dtype=float)
        if np.ptp(x) == 0:
            samples = max(16, min(1048576, int(round(samples * args.wall_seconds / seconds))))
            continue
        slope, intercept = np.polyfit(x, y, 1)
        if slope <= 0:
            samples *= 2
        else:
            samples = int(round((args.wall_seconds - intercept) / slope))
        samples = max(16, min(samples, 1048576))
    confirmed = [item for item in observations if item["repeated_timing_stable"]]
    best = min(confirmed or observations[-3:], key=lambda item: abs(item["seconds"] - args.wall_seconds))
    return best["samples"], observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--scenes", nargs="+", default=["indirect", "volume"])
    parser.add_argument("--scene-script", type=pathlib.Path,
                        default=pathlib.Path(__file__).with_name("cycles_metal_guiding_scene.py"))
    parser.add_argument("--integrators", nargs="+", choices=("pt", "bdpt"), default=["pt", "bdpt"])
    parser.add_argument("--wall-seconds", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--reference-samples", type=int, default=32768)
    parser.add_argument("--reference-training-samples", type=int)
    parser.add_argument("--reference-device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--reference-guiding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-seeds", nargs="+", type=int, default=[991, 1993])
    parser.add_argument("--reference-directory", type=pathlib.Path)
    parser.add_argument("--training-samples", type=int, default=128)
    parser.add_argument("--memory-mb", type=int, default=64)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare-cpu", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.wall_seconds <= 0:
        parser.error("--wall-seconds must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.blender = args.blender.resolve()
    args.scene_script = args.scene_script.resolve()
    args.binary_digest = digest(args.blender)
    args.script_digest = scene_digest(args.scene_script)
    report = dict(target_wall_seconds=args.wall_seconds, cases={})
    for scene in args.scenes:
        ref_metrics, reference = render_reference(args, scene)
        for integrator in args.integrators:
            case = dict(reference=ref_metrics, variants={})
            selected = {}
            configurations = [("baseline", "GPU", False), ("guided", "GPU", True)]
            if args.compare_cpu:
                configurations.append(("cpu_guided", "CPU", True))
            for name, device, guiding in configurations:
                count, observations = calibrate(args, scene, integrator, guiding, device)
                selected[name] = count
                case["variants"][name] = dict(samples=count, calibration=observations)
            # Alternate variants within each independent seed to reduce timing drift.
            pixels_by_variant = {name: [] for name, _, _ in configurations}
            times_by_variant = {name: [] for name, _, _ in configurations}
            for seed_index, seed in enumerate(args.seeds):
                offset = seed_index % len(configurations)
                order = configurations[offset:] + configurations[:offset]
                for name, device, guiding in order:
                    metrics, pixels = run_render(args, scene, "pt" if device == "CPU" else integrator,
                                                 guiding, selected[name], seed,
                                                 f"{scene}_{integrator}_{name}_{seed}",
                                                 device=device, warmup=True)
                    pixels_by_variant[name].append(pixels)
                    times_by_variant[name].append(metrics["seconds"])
            for name, _, _ in configurations:
                case["variants"][name].update(summarize(pixels_by_variant[name],
                                                       times_by_variant[name], reference,
                                                       reference_regions(ref_metrics)))
            baseline, guided = case["variants"]["baseline"], case["variants"]["guided"]
            ratio = guided["mean_seconds"] / baseline["mean_seconds"]
            case.update(measured_wall_time_ratio=ratio, matched_within_five_percent=abs(ratio - 1) <= 0.05,
                        mse_gain=baseline["mean_mse"] / max(guided["mean_mse"], 1e-30))
            if args.compare_cpu:
                cpu = case["variants"]["cpu_guided"]
                cpu_ratio = guided["mean_seconds"] / cpu["mean_seconds"]
                case.update(cpu_guided_to_metal_mse_ratio=cpu["mean_mse"] / max(guided["mean_mse"], 1e-30),
                            cpu_guided_to_metal_log1p_mse_ratio=cpu["mean_log1p_mse"] /
                            max(guided["mean_log1p_mse"], 1e-30),
                            metal_to_cpu_wall_time_ratio=cpu_ratio,
                            cpu_matched_within_five_percent=abs(cpu_ratio - 1) <= 0.05)
            report["cases"][f"{scene}_{integrator}"] = case
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("EQUAL_TIME_RESULT " + json.dumps({f"{scene}_{integrator}": case}), flush=True)


if __name__ == "__main__":
    main()
