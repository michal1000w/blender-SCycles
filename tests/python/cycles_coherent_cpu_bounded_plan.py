#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU plan for the retained bounded sphere fixtures (v39 mixed R/TT, v40 TRT/TRRT).

Uses the corrected-Jones v39 references and the v40 references unchanged, with
the fixed absolute radiance/phase gates recorded in the v39 reassessment
(mean 0.005, RMSE 0.012). The stricter v40 TRT presence gate (phase-pi mean
within 4e-4 of the independent reference) is added for TRT/TRRT phase-pi.
PT and CPU path guiding are run; BDPT is Metal-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = {
    "R": ROOT / "tests/output/diffraction/coherent_sphere_planar_v39_1mw_jones/R",
    "TT": ROOT / "tests/output/diffraction/coherent_sphere_planar_v39_1mw_jones/TT",
    "TRT": ROOT / "tests/output/diffraction/coherent_sphere_internal_v40_1mw/TRT",
    "TRRT": ROOT / "tests/output/diffraction/coherent_sphere_internal_v40_1mw/TRRT",
}
GATE = {"absolute_mean_error": 0.005, "absolute_rmse": 0.012}
PHASE_GATE = {"absolute_phase_mean_error": 0.005, "phase_difference_rmse": 0.012}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metal", action="store_true",
                        help="Metal PT, BDPT and guiding instead of CPU PT and guiding")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    jobs = []
    transports = ("pt", "bdpt", "pt-guiding") if args.metal else ("pt", "pt-guiding")
    prefix = "metal" if args.metal else "cpu"
    for transport in transports:
        for folder, fixtures in FIXTURES.items():
            reference = fixtures / "virtual_source_reference.npz"
            for case in ("phase_0", "phase_pi", "incoherent_connector_control"):
                blend = fixtures / f"{case}.blend"
                render = out / f"{transport}_{folder}" / f"{case}.exr"
                render.parent.mkdir(parents=True, exist_ok=True)
                arguments = ["--blend", str(blend), "--reference", str(reference),
                             "--render", str(render),
                             "--report", str(render.with_name(case + "_worker.json")),
                             "--case", case, "--require-specular", "--scene-budgets"]
                if transport != "bdpt":
                    arguments.append("--" + transport)
                if not args.metal:
                    arguments.append("--cpu")
                if case == "phase_pi":
                    arguments += ["--phase0-data", str(render.with_name("phase_0.npz"))]
                gate = dict(GATE)
                label = f"{prefix}_{transport}_{folder}_{case}"
                jobs.append({
                    "case": label,
                    "arguments": arguments,
                    "stdout": str(out / (label + "_stdout.log")),
                    "stderr": str(out / (label + "_stderr.log")),
                    "gate": gate,
                    "phase_gate": dict(PHASE_GATE) if case == "phase_pi" else None,
                    "scene_sha256": digest(blend),
                    "reference_sha256": digest(reference),
                })
                if folder in ("TRT", "TRRT") and case == "phase_pi":
                    # v40 declared TRT presence tolerance on the phase-pi mean.
                    jobs[-1]["gate"]["absolute_mean_error"] = 0.0004
    plan = {
        "scope": prefix.upper() + " bounded sphere regression: v39 mixed R/TT (corrected Jones) and v40 TRT/TRRT",
        "worker": str(ROOT / "tests/python/cycles_coherent_mirror_render_worker.py"),
        "state": str(out / "batch_state.json"),
        "jobs": jobs,
    }
    (out / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
    print(len(jobs), "jobs ->", out / "plan.json")


if __name__ == "__main__":
    main()
