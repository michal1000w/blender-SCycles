#!/usr/bin/env python3
"""Sequential Metal render runner for the reflected-path coherence fixture.

This runner has fixed absolute radiance and phase gates. A finite image alone
never passes; the phase-0 and phase-pi renders must match the independent
virtual-source references inside the predeclared ROI.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
BASE_CASES = ("phase_0", "phase_pi", "incoherent_bdpt_control")
GATES = {
    "absolute_mean_error_max_radiance": 0.005,
    "absolute_rmse_max_radiance": 0.012,
    "phase_difference_mean_error_max_radiance": 0.005,
    "phase_difference_rmse_max_radiance": 0.012,
    "scope": "fixed absolute radiance and phase errors; no fitted scale or zero-relative guard",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_environment():
    environment = os.environ.copy()
    for key in ("BLENDER_SYSTEM_RESOURCES", "DYLD_LIBRARY_PATH", "CYCLES_KERNEL_PATH", "CYCLES_SHADER_PATH"):
        environment.pop(key, None)
    return environment


def case_passed(case, worker):
    metrics = worker.get("metrics", {})
    if worker.get("worker_status") != "completed" or not metrics.get("all_finite"):
        return False
    if metrics.get("minimum_radiance", 0.0) < -1e-6:
        return False
    if metrics.get("absolute_mean_error", float("inf")) > GATES["absolute_mean_error_max_radiance"]:
        return False
    if metrics.get("absolute_rmse", float("inf")) > GATES["absolute_rmse_max_radiance"]:
        return False
    if metrics.get("black_max_absolute_radiance", 0.0) > 1e-6:
        return False
    if any(error > GATES["absolute_rmse_max_radiance"] for error in metrics.get("rgb_rmse", [])):
        return False
    if any(error > GATES["absolute_mean_error_max_radiance"] for error in metrics.get("rgb_mean_absolute_error", [])):
        return False
    if case == "phase_pi":
        phase = worker.get("phase_check", {})
        if phase.get("absolute_phase_mean_error", float("inf")) > GATES["phase_difference_mean_error_max_radiance"]:
            return False
        if phase.get("phase_difference_rmse", float("inf")) > GATES["phase_difference_rmse_max_radiance"]:
            return False
        if any(error > GATES["phase_difference_rmse_max_radiance"] for error in phase.get("rgb_phase_rmse", [])):
            return False
        if any(error > GATES["phase_difference_mean_error_max_radiance"] for error in phase.get("rgb_phase_mean_absolute_error", [])):
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blender", type=Path,
        default=ROOT / "build/diffraction_delivery_20260927_coherence_specialization/Blender.app/Contents/MacOS/Blender",
    )
    parser.add_argument(
        "--fixtures", type=Path,
        default=ROOT / "build/tests/python/cycles_coherent_mirror_acceptance_specialization",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "build/tests/python/cycles_coherent_mirror_render_specialization_20260927",
    )
    parser.add_argument("--cases", nargs="+", choices=("phase_0", "phase_pi", "incoherent_connector_control"))
    transport = parser.add_mutually_exclusive_group()
    transport.add_argument("--pt-guiding", action="store_true")
    transport.add_argument("--pt", action="store_true")
    parser.add_argument("--scene-budgets", action="store_true")
    parser.add_argument("--persistent-process", action="store_true",
                        help="Run unchanged workers sequentially in one Blender process")
    args = parser.parse_args()
    blender, fixtures, output = args.blender.resolve(), args.fixtures.resolve(), args.output_dir.resolve()
    if not blender.is_file():
        parser.error(f"Blender executable not found: {blender}")
    manifest = json.loads((fixtures / "manifest.json").read_text())
    specular_connections = bool(manifest.get("specular_connections", False))
    cases = ("phase_0", "phase_pi", "incoherent_connector_control") if specular_connections else BASE_CASES
    if args.cases:
        cases = tuple(args.cases)
    if "phase_pi" in cases and "phase_0" not in cases:
        parser.error("phase_pi requires phase_0 in the same run")
    for name in cases:
        if not (fixtures / f"{name}.blend").is_file():
            parser.error(f"scene fixture not found: {fixtures / (name + '.blend')}")
    reference = fixtures / "virtual_source_reference.npz"
    if not reference.is_file():
        parser.error(f"reference data not found: {reference}")
    output.mkdir(parents=True, exist_ok=False)

    worker_script = Path(__file__).with_name("cycles_coherent_mirror_render_worker.py").resolve()
    result = {
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "binary": str(blender),
        "binary_sha256": sha256(blender),
        "fixture_directory": str(fixtures),
        "worker_script": str(worker_script),
        "worker_script_sha256": sha256(worker_script),
        "runner_script_sha256": sha256(Path(__file__).resolve()),
        "reference_npz": str(reference),
        "reference_npz_sha256": sha256(reference),
        "execution": "sequential fresh background process per scene; Metal only",
        "fixed_settings": {
            "device": "METAL",
            "samples": 128,
            "seed": 19,
            "adaptive_sampling": False,
            "denoising": False,
            "pixel_filter": "BOX",
            "filter_width": 1.0,
            "bidirectional_path_tracing": not (args.pt_guiding or args.pt),
            "guiding": args.pt_guiding,
            "preserve_saved_bounce_budgets": args.scene_budgets,
        },
        "gates": GATES,
        "expected_renderer_state": (
            "specular connector tested against independently prepared physical reference: " + manifest.get("scope", "mirror")
            if specular_connections else
            "reflected-path coherent transport is unsupported; expected phase-profile checks to fail"
        ),
        "cases": {},
    }
    result_path = output / "results.json"
    phase0_npz = output / "phase_0.npz"

    def worker_arguments(case):
        arguments = ["--blend", str(fixtures / f"{case}.blend"),
                     "--reference", str(reference), "--render", str(output / f"{case}.exr"),
                     "--report", str(output / f"{case}_worker.json"), "--case", case]
        if case == "phase_pi":
            arguments.extend(("--phase0-data", str(phase0_npz)))
        for enabled, option in ((args.pt_guiding, "--pt-guiding"), (args.pt, "--pt"),
                                (specular_connections, "--require-specular"),
                                (args.scene_budgets, "--scene-budgets")):
            if enabled:
                arguments.append(option)
        return arguments

    shared = None
    batch_state = {}
    if args.persistent_process:
        batch_script = worker_script.with_name("cycles_coherent_mirror_batch_worker.py")
        jobs_path, state_path = output / "batch_jobs.json", output / "batch_state.json"
        jobs_path.write_text(json.dumps({"worker": str(worker_script), "state": str(state_path),
            "jobs": [{"case": case, "arguments": worker_arguments(case),
                      "stdout": str(output / f"{case}_stdout.log"),
                      "stderr": str(output / f"{case}_stderr.log")} for case in cases]}, indent=2) + "\n")
        batch_command = [str(blender), "--background", "--factory-startup", "--python-exit-code",
                         "73", "--threads", "2", "--python", str(batch_script), "--", str(jobs_path)]
        result["execution"] = "sequential unchanged workers in one persistent background process; Metal only"
        result["shared_process"] = {"command": batch_command, "batch_worker_sha256": sha256(batch_script),
                                    "jobs_sha256": sha256(jobs_path), "timeout_seconds": 360 * len(cases),
                                    "status": "running"}
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        try:
            shared = subprocess.run(batch_command, cwd=ROOT, text=True, capture_output=True,
                                    check=False, env=render_environment(), timeout=360 * len(cases))
            result["shared_process"].update(status="completed", process_exit_code=shared.returncode,
                                            stdout=shared.stdout, stderr=shared.stderr)
        except subprocess.TimeoutExpired as error:
            shared = SimpleNamespace(returncode=None)
            result["shared_process"].update(status="timeout", process_exit_code=None,
                                            stdout=str(error.stdout or ""), stderr=str(error.stderr or ""))
        if state_path.is_file():
            batch_state = json.loads(state_path.read_text()).get("jobs", {})
        result["shared_process"]["case_execution"] = batch_state

    for case in cases:
        render_path = output / f"{case}.exr"
        report_path = output / f"{case}_worker.json"
        command = [
            str(blender), "--background", "--factory-startup",
            "--python-exit-code", "73", "--threads", "2", "--python", str(worker_script), "--",
            "--blend", str(fixtures / f"{case}.blend"),
            "--reference", str(reference),
            "--render", str(render_path),
            "--report", str(report_path),
            "--case", case,
        ]
        if case == "phase_pi":
            command.extend(("--phase0-data", str(phase0_npz)))
        if args.pt_guiding:
            command.append("--pt-guiding")
        if args.pt:
            command.append("--pt")
        if specular_connections:
            command.append("--require-specular")
        if args.scene_budgets:
            command.append("--scene-budgets")
        entry = {
            "command": command,
            "scene_sha256": sha256(fixtures / f"{case}.blend"),
            "cwd": str(ROOT),
            "status": "running",
            "render_path": str(render_path),
            "worker_report": str(report_path),
        }
        result["cases"][case] = entry
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        try:
            if args.persistent_process:
                job = batch_state.get(case, {})
                process = SimpleNamespace(returncode=job.get("exit_code", 73),
                    stdout=(output / f"{case}_stdout.log").read_text() if (output / f"{case}_stdout.log").is_file() else "",
                    stderr=(output / f"{case}_stderr.log").read_text() if (output / f"{case}_stderr.log").is_file() else "")
                entry["execution"] = "runpy worker in shared process; command documents equivalent fresh invocation"
                entry["shared_case_status"] = job.get("status", "not_completed")
            else:
                process = subprocess.run(
                    command,
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                    env=render_environment(),
                    timeout=360,
                )
        except subprocess.TimeoutExpired:
            entry.update(status="timeout", timeout_seconds=360, passed=False)
            result["passed"] = False
            result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            return 1
        entry["process_exit_code"] = process.returncode
        entry["stdout"] = process.stdout
        entry["stderr"] = process.stderr
        worker = None
        if report_path.is_file():
            try:
                worker = json.loads(report_path.read_text())
            except json.JSONDecodeError as error:
                entry["worker_report_error"] = str(error)
        entry["worker_result"] = worker
        if render_path.is_file():
            entry["render_sha256"] = sha256(render_path)
            entry["render_size_bytes"] = render_path.stat().st_size
        entry["passed"] = process.returncode == 0 and case_passed(case, worker or {})
        entry["status"] = "passed" if entry["passed"] else (
            "render_failed" if process.returncode != 0 or worker is None else "acceptance_failed"
        )
        print(f"{case}: {entry['status']}", flush=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    result["phase_response_passed"] = result["cases"].get("phase_pi", {}).get("passed", False)
    result["passed"] = all(entry["passed"] for entry in result["cases"].values()) and (shared is None or shared.returncode == 0)
    result["completed_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"results_json={result_path}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
