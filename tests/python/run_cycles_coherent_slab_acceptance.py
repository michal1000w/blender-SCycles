#!/usr/bin/env python3
"""Fixed absolute-reference acceptance for the bounded two-transmission Glass slab."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SLAB_CASES = ("phase_0", "phase_pi", "incoherent_connector_control")
MIXED_CASES = ("phase_0", "phase_pi", "distinct_groups", "diagonal_control")
# Frozen before the 512-sample acceptance run; raw radiance, no fitted scale.
GATES = {
    "absolute_mean_error_max_radiance": 0.005,
    "absolute_rmse_max_radiance": 0.012,
    "phase_difference_mean_error_max_radiance": 0.005,
    "phase_difference_rmse_max_radiance": 0.012,
    "minimum_radiance_min": -1e-6,
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def case_passes(name, worker):
    metrics = worker.get("metrics", {})
    if worker.get("status") != "completed" or not metrics.get("all_finite"):
        return False
    if metrics.get("absolute_mean_error", float("inf")) > GATES["absolute_mean_error_max_radiance"]:
        return False
    if metrics.get("absolute_rmse", float("inf")) > GATES["absolute_rmse_max_radiance"]:
        return False
    if metrics.get("minimum_radiance_roi", float("-inf")) < GATES["minimum_radiance_min"]:
        return False
    if name == "phase_pi":
        phase = worker.get("phase_check", {})
        if phase.get("absolute_mean_error", float("inf")) > GATES["phase_difference_mean_error_max_radiance"]:
            return False
        if phase.get("absolute_rmse", float("inf")) > GATES["phase_difference_rmse_max_radiance"]:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("slab", "mixed"), default="slab")
    parser.add_argument("--samples", type=int, default=512)
    args = parser.parse_args()
    blender, fixtures, output = (p.resolve() for p in (args.blender, args.fixtures, args.output_dir))
    if not blender.is_file():
        parser.error(f"Blender binary missing: {blender}")
    if args.samples <= 0 or args.samples > 512:
        parser.error("Samples must be in 1..512 for this fixed fixture")
    cases = MIXED_CASES if args.suite == "mixed" else SLAB_CASES
    reference = fixtures / ("mixed_rt_reference.npz" if args.suite == "mixed" else
                            "snell_slab_reference.npz")
    if not reference.is_file():
        parser.error(f"Independent reference missing: {reference}")
    for name in cases:
        if not (fixtures / f"{name}.blend").is_file():
            parser.error(f"Scene fixture missing: {name}.blend")
    output.mkdir(parents=True, exist_ok=False)
    worker = Path(__file__).with_name("cycles_coherent_slab_render_worker.py").resolve()
    result = {
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "binary": str(blender), "binary_sha256": sha256(blender),
        "fixtures": str(fixtures), "reference": str(reference),
        "reference_sha256": sha256(reference),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "worker_sha256": sha256(worker),
        "settings": {"resolution": [512, 512], "samples": args.samples, "seed": 19,
                     "device": "METAL", "bdpt": True, "adaptive": False,
                     "denoising": False, "pixel_filter": "BOX"},
        "scope": ("two-transmission Glass plus interior Mirror three-event T/R/T and direct arms"
                  if args.suite == "mixed" else
                  "two-transmission ideal Glass slab; branches beyond two events excluded by fixture"),
        "suite": args.suite,
        "gates": GATES, "cases": {},
    }
    result_path = output / "results.json"
    environment = os.environ.copy()
    for key in ("BLENDER_SYSTEM_RESOURCES", "DYLD_LIBRARY_PATH", "CYCLES_KERNEL_PATH", "CYCLES_SHADER_PATH"):
        environment.pop(key, None)
    for name in cases:
        scene = fixtures / f"{name}.blend"
        render = output / f"{name}.exr"
        report = output / f"{name}_worker.json"
        command = [str(blender), "-t", "3", "--background", "--factory-startup",
                   "--python-exit-code", "73", "--python", str(worker), "--",
                   "--blend", str(scene), "--reference", str(reference),
                   "--render", str(render), "--report", str(report), "--case", name,
                   "--samples", str(args.samples)]
        if name == "phase_pi":
            command.extend(("--phase0-data", str(output / "phase_0.npz")))
        entry = {"scene_sha256": sha256(scene), "command": command, "status": "running"}
        result["cases"][name] = entry
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        try:
            process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                     check=False, env=environment, timeout=240)
        except subprocess.TimeoutExpired:
            entry.update(status="timeout", timeout_seconds=240, passed=False)
            result["passed"] = False
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            return 1
        entry.update(exit_code=process.returncode, stdout=process.stdout, stderr=process.stderr)
        worker_report = json.loads(report.read_text()) if report.is_file() else {}
        entry["worker_result"] = worker_report
        if render.is_file():
            entry.update(render_sha256=sha256(render), render_size_bytes=render.stat().st_size)
        entry["passed"] = process.returncode == 0 and case_passes(name, worker_report)
        entry["status"] = "passed" if entry["passed"] else "failed"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: {entry['status']}", flush=True)
        if not entry["passed"]:
            break
    result["passed"] = len(result["cases"]) == len(cases) and all(
        case["passed"] for case in result["cases"].values())
    result["completed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"results_json={result_path}", flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
