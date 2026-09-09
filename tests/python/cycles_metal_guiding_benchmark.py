#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Measure Metal guiding image error and end-to-end render time with reproducible seeds.

Runs a complete discarded warmup for each configuration, so the measured render includes
fresh field training but excludes cold shader compilation. Both times are retained. Uses
CPU OpenPGL guided references by default, checks independent reference seeds, saves raw linear
data, and reports regressions unchanged.
"""

import argparse
import copy
import hashlib
import json
import os
import pathlib
import re
import subprocess

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def scene_digest(path):
    base = pathlib.Path(__file__).with_name("cycles_metal_guiding_scene.py").resolve()
    if path.resolve() == base:
        return digest(path)
    # Composed fixtures depend on the shared room factory as well as their own source.
    return hashlib.sha256((digest(path) + digest(base)).encode()).hexdigest()


def kernel_source_digest(blender):
    # Metal kernels are compiled from installed source, independently of the executable.
    # Preserve that provenance as well as the binary hash when testing an installed macOS app.
    roots = sorted((blender.parents[1] / 'Resources').glob('*/scripts/addons_core/cycles/source'))
    if not roots:
        return None
    result = hashlib.sha256()
    for root in roots:
        for path in sorted(root.rglob('*')):
            if path.is_file():
                result.update(str(path.relative_to(root)).encode() + b'\0')
                result.update(path.read_bytes())
    return result.hexdigest()


def reference_regions(metadata):
    return metadata.get("object_regions", metadata.get("runs", [{}])[0].get("object_regions", {}))


def reference_uncertainty(metadata, stack):
    if len(stack) < 2:
        return
    signals = stack.mean(axis=(1, 2, 3))
    metadata["signal_standard_error"] = float(signals.std(ddof=1) / np.sqrt(len(stack)))
    regions = reference_regions(metadata)
    if regions:
        metadata["object_regions"] = regions
        metadata["region_reference_mse"] = {}
        metadata["region_signal_standard_error"] = {}
        for name, (x0, y0, x1, y1) in regions.items():
            values = stack[:, y0:y1, x0:x1]
            metadata["region_reference_mse"][name] = float(values.var(axis=0, ddof=1).mean() / len(stack))
            metadata["region_signal_standard_error"][name] = float(
                values.mean(axis=(1, 2, 3)).std(ddof=1) / np.sqrt(len(stack)))


def run_render(args, scene, integrator, guiding, samples, seed, name, time_limit=0.0, device="GPU",
               warmup=None):
    prefix = args.output / name
    warmup = device == "GPU" and not getattr(args, "quality_only", False) if warmup is None else warmup
    if not hasattr(args, 'kernel_digest'):
        args.kernel_digest = kernel_source_digest(args.blender)
    config = dict(scene=scene, integrator=integrator, guiding=guiding, samples=samples, device=device,
                  seed=seed, resolution=args.resolution, training_samples=args.training_samples,
                  memory_mb=args.memory_mb, time_limit=time_limit, binary_sha256=args.binary_digest,
                  script_sha256=args.script_digest, kernel_sources_sha256=args.kernel_digest, warmup=warmup)
    stamp = prefix.with_suffix(".config.json")
    if args.resume and stamp.exists() and prefix.with_suffix(".npy").exists():
        if json.loads(stamp.read_text()) == config:
            cached = json.loads(prefix.with_suffix(".json").read_text())
            counts_verified = time_limit > 0 or cached.get("rendered_samples") == samples
            budget_verified = integrator != "bdpt" or bool(cached.get("bdpt_light_batches"))
            if counts_verified and budget_verified:
                return cached, np.load(prefix.with_suffix(".npy"))
    command = [args.blender, "--background", "--factory-startup", "--debug-cycles", "--python-exit-code", "1",
               "--python", args.scene_script, "--", "--scene", scene, "--samples", str(samples),
               "--device", device,
               "--seed", str(seed), "--resolution", str(args.resolution),
               "--training-samples", str(args.training_samples), "--memory-mb", str(args.memory_mb),
               "--time-limit", str(time_limit), "--output", prefix]
    if warmup:
        command.append("--warmup")
    if integrator == "bdpt":
        command.append("--bdpt")
    if guiding:
        command.append("--guiding")
    if seed == args.seeds[0]:
        command.append("--save-scene")
    env = dict(os.environ)
    env["BLENDER_USER_RESOURCES"] = str(args.output / "blender-user")
    if integrator == "bdpt":
        command[1:1] = ["--log-level", "debug"]
        env["CYCLES_BDPT_DIAGNOSTICS"] = "1"
    print(f"RENDER {name}", flush=True)
    with prefix.with_suffix(".log").open("w") as log:
        result = subprocess.run([str(value) for value in command], env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"{name} failed, exit {result.returncode}; see {prefix.with_suffix('.log')}")
    if not prefix.with_suffix(".json").exists() or not prefix.with_suffix(".npy").exists():
        raise RuntimeError(f"{name} produced no metrics or linear image")
    metrics = json.loads(prefix.with_suffix(".json").read_text())
    render_log = prefix.with_suffix(".log").read_text()
    counts = re.findall(r"Rendered (\d+) samples in", render_log)
    metrics["rendered_samples"] = int(counts[-1]) if counts else None
    if integrator == "bdpt":
        metrics["bdpt_cache_capacities"] = [int(value) for value in
            re.findall(r"BDPT light cache: (\d+) vertices", render_log)]
        metrics["bdpt_light_batches"] = [dict(sample=int(sample), emitted=int(emitted),
            cached=int(cached), sensor_shadows=int(shadows)) for sample, emitted, cached, shadows in
            re.findall(r"BDPT diagnostics: sample=(\d+) emitted=(\d+) cached=(\d+) sensor_shadows=(\d+)",
                       render_log)]
    prefix.with_suffix(".json").write_text(json.dumps(metrics, indent=2) + "\n")
    if time_limit == 0 and metrics["rendered_samples"] != samples:
        raise RuntimeError(f"{name}: requested {samples} samples, observed {metrics['rendered_samples']}")
    if integrator == "bdpt" and not metrics["bdpt_light_batches"]:
        raise RuntimeError(f"{name}: missing actual BDPT light-budget diagnostics")
    stamp.write_text(json.dumps(config, indent=2) + "\n")
    return json.loads(prefix.with_suffix(".json").read_text()), np.load(prefix.with_suffix(".npy"))


def render_reference(args, scene):
    reference_training = getattr(args, 'reference_training_samples', None)
    if reference_training is None:
        reference_training = args.training_samples
    source = getattr(args, "reference_directory", None)
    if source is not None:
        source = source.resolve()
        metadata = json.loads((source / (scene + "_reference.json")).read_text())
        expected = dict(device=args.reference_device, guiding=args.reference_guiding,
                        samples_per_seed=args.reference_samples, seeds=args.reference_seeds)
        if any(metadata[key] != value for key, value in expected.items()):
            raise ValueError("Imported reference settings do not match the requested reference")
        source_configs = []
        for seed in args.reference_seeds:
            config = json.loads((source / f"{scene}_reference_{seed}.config.json").read_text())
            if (config["resolution"] != args.resolution or
                    config["training_samples"] != reference_training or
                    config["script_sha256"] != args.script_digest):
                raise ValueError("Imported reference scene or training configuration differs")
            source_configs.append(config)
        reference_path = source / (scene + "_reference.npy")
        reference = np.load(reference_path)
        if reference.shape != (args.resolution, args.resolution, 3) or not np.isfinite(reference).all():
            raise ValueError("Invalid imported reference pixels")
        metadata = dict(metadata, imported_from=str(source), reference_sha256=digest(reference_path),
                        source_configs=source_configs)
        seed_paths = [source / f"{scene}_reference_{seed}.npy" for seed in args.reference_seeds]
        if all(path.exists() for path in seed_paths):
            reference_uncertainty(metadata, np.stack([np.load(path).astype(np.float64) for path in seed_paths]))
        np.save(args.output / (scene + "_reference.npy"), reference)
        (args.output / (scene + "_reference.json")).write_text(json.dumps(metadata, indent=2) + "\n")
        return metadata, reference
    images, runs = [], []
    reference_args = copy.copy(args)
    reference_args.training_samples = reference_training
    for seed in args.reference_seeds:
        metrics, pixels = run_render(reference_args, scene, "pt", args.reference_guiding,
                                     args.reference_samples, seed, f"{scene}_reference_{seed}",
                                     device=args.reference_device)
        runs.append(metrics)
        images.append(pixels.astype(np.float64))
    stack = np.stack(images)
    reference = stack.mean(axis=0).astype(np.float32)
    np.save(args.output / (scene + "_reference.npy"), reference)
    metrics = dict(device=args.reference_device, guiding=args.reference_guiding,
                   samples_per_seed=args.reference_samples, seeds=args.reference_seeds,
                   mean=float(reference.mean()), runs=runs,
                   estimated_reference_mse=float(stack.var(axis=0, ddof=1).mean() / len(images))
                   if len(images) > 1 else None)
    reference_uncertainty(metrics, stack)
    (args.output / (scene + "_reference.json")).write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics, reference


def summarize(images, times, reference, regions=None):
    image_stack = np.stack(images).astype(np.float64)
    reference = reference.astype(np.float64)
    mse = np.mean((image_stack - reference) ** 2, axis=(1, 2, 3))
    # A secondary HDR metric gives dim details more influence without altering the saved
    # linear images or replacing the primary radiometric MSE. Apply the same transform to
    # every image and the CPU reference; never normalize images individually.
    log_error = np.log1p(np.maximum(image_stack, 0)) - np.log1p(np.maximum(reference, 0))
    log_mse = np.mean(log_error ** 2, axis=(1, 2, 3))
    means = np.mean(image_stack, axis=(1, 2, 3))
    # Error of the ensemble mean reveals systematic differences that individual noisy images hide.
    ensemble_mse = float(np.mean((image_stack.mean(axis=0) - reference) ** 2))
    result = dict(mse_per_seed=mse.tolist(), mean_mse=float(mse.mean()),
                mse_standard_error=float(mse.std(ddof=1) / np.sqrt(len(mse))) if len(mse) > 1 else None,
                log1p_mse_per_seed=log_mse.tolist(), mean_log1p_mse=float(log_mse.mean()),
                log1p_mse_standard_error=float(log_mse.std(ddof=1) / np.sqrt(len(log_mse)))
                if len(log_mse) > 1 else None,
                seconds_per_seed=times, mean_seconds=float(np.mean(times)),
                mean_signal=float(means.mean()),
                signal_standard_error=float(means.std(ddof=1) / np.sqrt(len(means))) if len(means) > 1 else None,
                relative_mean_difference=float((means.mean() - reference.mean()) / max(reference.mean(), 1e-20)),
                ensemble_mse=ensemble_mse,
                mean_time_times_mse=float(np.mean(np.asarray(times) * mse)))
    if regions:
        result["regions"] = {}
        for name, (x0, y0, x1, y1) in regions.items():
            if not (0 <= x0 < x1 <= reference.shape[1] and 0 <= y0 < y1 <= reference.shape[0]):
                raise ValueError("Invalid reference region " + name)
            region = reference[y0:y1, x0:x1]
            values = image_stack[:, y0:y1, x0:x1]
            signals = values.mean(axis=(1, 2, 3))
            result["regions"][name] = dict(
                bounds=[x0, y0, x1, y1], mean_mse=float(((values - region) ** 2).mean()),
                mean_log1p_mse=float((log_error[:, y0:y1, x0:x1] ** 2).mean()),
                ensemble_mse=float(((values.mean(axis=0) - region) ** 2).mean()),
                mean_signal=float(signals.mean()), reference_mean=float(region.mean()),
                signal_standard_error=float(signals.std(ddof=1) / np.sqrt(len(signals)))
                if len(signals) > 1 else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--scenes", nargs="+", choices=("indirect", "glossy", "mixed", "volume", "aperture",
                                                               "rough_glass", "coated", "transmission", "subsurface"),
                        default=["indirect", "glossy", "mixed", "volume"])
    parser.add_argument("--scene-script", type=pathlib.Path,
                        default=pathlib.Path(__file__).with_name("cycles_metal_guiding_scene.py"))
    parser.add_argument("--integrators", nargs="+", choices=("pt", "bdpt"), default=["pt", "bdpt"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 47])
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--reference-samples", type=int, default=8192)
    parser.add_argument("--reference-training-samples", type=int,
                        help="Training setting for the reference; defaults to the tested setting")
    parser.add_argument("--reference-device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--reference-guiding", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-seeds", nargs="+", type=int, default=[991, 1993])
    parser.add_argument("--reference-directory", type=pathlib.Path,
                        help="Explicitly reuse matching references and retain their original provenance")
    parser.add_argument("--references-only", action="store_true")
    parser.add_argument("--compare-cpu", action=argparse.BooleanOptionalAction, default=True,
                        help="Also measure CPU OpenPGL guiding at the same sample count")
    parser.add_argument("--quality-only", action="store_true",
                        help="Validate fixed-sample images without warmups or performance claims")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--training-samples", type=int, default=128)
    parser.add_argument("--memory-mb", type=int, default=64)
    parser.add_argument("--time-budget", type=float, default=0.0,
                        help="Also compare images at this equal Cycles render-time budget")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.quality_only and args.time_budget:
        parser.error("--quality-only cannot be combined with a render-time budget")
    args.blender = args.blender.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.scene_script = args.scene_script.resolve()
    args.binary_digest = digest(args.blender)
    args.script_digest = scene_digest(args.scene_script)
    report = dict(binary_sha256=args.binary_digest, script_sha256=args.script_digest,
                  comparison_protocol="equal_camera_samples",
                  samples_per_pixel=args.samples, seeds=args.seeds,
                  training_samples=args.training_samples,
                  timing_comparisons_valid=not args.quality_only,
                  work_accounting="BDPT also traces light subpaths; equal SPP is not equal ray count.",
                  cases={})
    for scene in args.scenes:
        ref_metrics, reference = render_reference(args, scene)
        if args.references_only:
            continue
        for integrator in args.integrators:
            variants = {}
            for guiding in (False, True):
                images, times = [], []
                for seed in args.seeds:
                    name = f"{scene}_{integrator}_{'guided' if guiding else 'baseline'}_{seed}"
                    metrics, pixels = run_render(args, scene, integrator, guiding, args.samples, seed, name)
                    images.append(pixels)
                    times.append(metrics["seconds"])
                variants["guided" if guiding else "baseline"] = summarize(
                    images, times, reference, reference_regions(ref_metrics))
            baseline, guided = variants["baseline"], variants["guided"]
            if args.compare_cpu:
                images, times = [], []
                for seed in args.seeds:
                    # CPU has no BDPT backend. Compare both Metal integrators to CPU guided PT.
                    name = f"{scene}_cpu_guided_{seed}"
                    metrics, pixels = run_render(args, scene, "pt", True, args.samples, seed, name,
                                                 device="CPU", warmup=not args.quality_only)
                    images.append(pixels)
                    times.append(metrics["seconds"])
                variants["cpu_guided"] = summarize(images, times, reference, reference_regions(ref_metrics))
                variants["cpu_guided_to_metal_mse_ratio"] = (
                    variants["cpu_guided"]["mean_mse"] / max(guided["mean_mse"], 1e-30))
                variants["cpu_guided_to_metal_log1p_mse_ratio"] = (
                    variants["cpu_guided"]["mean_log1p_mse"] / max(guided["mean_log1p_mse"], 1e-30))
                # Report observed equal-sample quality independently of runtime. A speedup
                # must not turn a measured quality regression into a passing result.
                variants["observed_equal_sample_regressions"] = {
                    name: [metric for metric in ("mean_mse", "mean_log1p_mse")
                           if guided[metric] > other[metric]]
                    for name, other in (("cpu_guided", variants["cpu_guided"]),
                                        ("metal_unguided", baseline))}
            variants.update(reference=ref_metrics,
                            equal_sample_mse_gain=baseline["mean_mse"] / max(guided["mean_mse"], 1e-30),
                            time_adjusted_efficiency_gain=None if args.quality_only else
                            baseline["mean_time_times_mse"] / max(guided["mean_time_times_mse"], 1e-30))
            if args.time_budget > 0:
                timed = {}
                for guiding in (False, True):
                    images, times, counts = [], [], []
                    for seed in args.seeds:
                        name = f"{scene}_{integrator}_{'guided' if guiding else 'baseline'}_timed_{seed}"
                        metrics, pixels = run_render(args, scene, integrator, guiding, 1048576,
                                                     seed, name, args.time_budget)
                        images.append(pixels)
                        times.append(metrics["seconds"])
                        counts.append(metrics["rendered_samples"])
                    result = summarize(images, times, reference, reference_regions(ref_metrics))
                    result["rendered_samples_per_seed"] = counts
                    timed["guided" if guiding else "baseline"] = result
                variants["equal_render_budget"] = dict(
                    seconds=args.time_budget, variants=timed,
                    mse_gain=timed["baseline"]["mean_mse"] / max(timed["guided"]["mean_mse"], 1e-30),
                    measured_wall_time_ratio=timed["guided"]["mean_seconds"] /
                    timed["baseline"]["mean_seconds"])
            key = f"{scene}_{integrator}"
            report["cases"][key] = variants
            (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print("RESULT " + json.dumps({key: variants}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
