#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end feature fallback regression suite for Cycles Metal reservoir lighting."""

import argparse
import math
import pathlib
import re
import subprocess
import tempfile


METRIC = re.compile(r"([a-z0-9_]+)=([-+0-9.eE]+)")


def run_render(blender, smoke, output_dir, name, arguments):
    output = output_dir / f"{name}.exr"
    command = [
        str(blender),
        "--background",
        "--factory-startup",
        "--python",
        str(smoke),
        "--",
        *arguments,
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    transcript = completed.stdout + completed.stderr
    line = next((line for line in transcript.splitlines() if line.startswith("BDPT_SMOKE ")), None)
    if completed.returncode or line is None:
        raise RuntimeError(f"{name} failed (exit {completed.returncode})\n{transcript[-5000:]}")
    metrics = {key: float(value) for key, value in METRIC.findall(line)}
    if not metrics or not all(math.isfinite(value) for value in metrics.values()):
        raise AssertionError(f"{name} reported invalid metrics: {metrics}")
    print(f"{name:28s} {line}")
    return metrics


def require_equal(reference, candidate, name, relative_tolerance=0.0):
    for metric in ("mean", "p99", "p999", "peak"):
        scale = max(abs(reference[metric]), abs(candidate[metric]), 1.0e-12)
        if abs(reference[metric] - candidate[metric]) > relative_tolerance * scale:
            raise AssertionError(
                f"{name}: fallback changed {metric}: {reference[metric]} != {candidate[metric]}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", required=True, type=pathlib.Path)
    parser.add_argument(
        "--smoke", type=pathlib.Path, default=pathlib.Path(__file__).with_name("cycles_bdpt_smoke.py")
    )
    parser.add_argument("--keep-output", type=pathlib.Path)
    parser.add_argument(
        "--enhanced-only",
        action="store_true",
        help="Run only ReSTIR PT Enhanced, BDPT interoperability, and shadow-catcher cases",
    )
    options = parser.parse_args()

    if options.keep_output:
        options.keep_output.mkdir(parents=True, exist_ok=True)
        output_context = None
        output_dir = options.keep_output
    else:
        output_context = tempfile.TemporaryDirectory(prefix="cycles-restir-metal-")
        output_dir = pathlib.Path(output_context.name)

    gpu = ["--device", "GPU", "--restir", "--samples", "4", "--resolution", "24"]
    cases = {
        "point_light_tree": gpu + ["--light-type", "POINT", "--no-glass"],
        "flat_distribution": gpu + ["--light-type", "POINT", "--no-light-tree", "--no-glass"],
        "mesh_emitter": gpu + ["--light-type", "MESH", "--no-glass"],
        "world_emitter": gpu + ["--light-type", "WORLD", "--no-glass"],
        "sharp_glass_fallback": gpu + ["--light-type", "AREA"],
        "hair_fallback": gpu + ["--hair-bsdf", "--no-glass"],
        "bssrdf_fallback": gpu + ["--subsurface", "--no-glass"],
        "volume_transition": gpu
        + ["--world-scatter", "0.08", "--volume-bounces", "2", "--no-glass"],
        "light_linking": gpu + ["--light-link", "FLOOR", "--no-glass"],
        "panorama": gpu + ["--camera-type", "PANO", "--no-glass"],
        "border_adaptive": gpu + ["--crop-border", "--adaptive", "--no-glass"],
        "photon_only": [
            "--device",
            "GPU",
            "--photon",
            "--photon-count",
            "1024",
            "--photon-camera-samples",
            "1",
            "--samples",
            "2",
            "--resolution",
            "24",
        ],
        "photon_combination": gpu
        + ["--photon", "--photon-count", "1024", "--samples", "2"],
        "metalrt_motion": gpu
        + ["--metalrt", "--camera-motion", "--aperture", "0.05", "--no-glass"],
        "reuse_opt_in": gpu
        + ["--restir-history", "4", "--restir-neighbors", "1", "--no-glass"],
    }
    results = {}
    if not options.enhanced_only:
        cpu_common = ["--device", "CPU", "--no-glass", "--samples", "8", "--resolution", "16"]
        cpu_pt = run_render(options.blender, options.smoke, output_dir, "cpu_pt", cpu_common)
        cpu_restir = run_render(
            options.blender,
            options.smoke,
            output_dir,
            "cpu_restir_fallback",
            cpu_common + ["--restir"],
        )
        require_equal(cpu_pt, cpu_restir, "CPU Metal-only fallback")
        cpu_restir_pt = run_render(
            options.blender,
            options.smoke,
            output_dir,
            "cpu_restir_pt_fallback",
            cpu_common + ["--restir-pt"],
        )
        require_equal(cpu_pt, cpu_restir_pt, "CPU ReSTIR PT Metal-only fallback")
        for name, arguments in cases.items():
            results[name] = run_render(options.blender, options.smoke, output_dir, name, arguments)

    enhanced = [
        "--device",
        "GPU",
        "--restir-pt",
        "--restir-pt-history",
        "1",
        "--samples",
        "4",
        "--resolution",
        "16",
    ]
    enhanced_cases = {
        "pt_point": enhanced + ["--light-type", "POINT", "--no-glass"],
        "pt_mesh": enhanced + ["--light-type", "MESH", "--no-glass"],
        "pt_world": enhanced + ["--light-type", "WORLD", "--no-glass"],
        "pt_glass": enhanced + ["--light-type", "AREA"],
        "pt_hair": enhanced + ["--hair-bsdf", "--no-glass"],
        "pt_bssrdf": enhanced + ["--subsurface", "--no-glass"],
        "pt_volume": enhanced
        + ["--world-scatter", "0.08", "--volume-bounces", "2", "--no-glass"],
        "pt_photon_reference": [
            "--device",
            "GPU",
            "--photon",
            "--photon-count",
            "1024",
            "--samples",
            "2",
            "--resolution",
            "16",
        ],
        "pt_photon": enhanced + ["--photon", "--photon-count", "1024", "--samples", "2"],
        "pt_metalrt_motion": enhanced
        + ["--metalrt", "--camera-motion", "--aperture", "0.05", "--no-glass"],
        "pt_spatial": enhanced
        + ["--restir-pt-neighbors", "1", "--light-type", "POINT", "--no-glass"],
    }
    for name, arguments in enhanced_cases.items():
        results[name] = run_render(options.blender, options.smoke, output_dir, name, arguments)
    enhanced_photon_ratio = results["pt_photon"]["mean"] / results["pt_photon_reference"]["mean"]
    if not 0.75 < enhanced_photon_ratio < 1.25:
        raise AssertionError(
            f"Photon/ReSTIR PT disjoint-estimator energy ratio is invalid: {enhanced_photon_ratio}"
        )
    if not options.enhanced_only:
        photon_mean_ratio = results["photon_combination"]["mean"] / results["photon_only"]["mean"]
        if not 0.75 < photon_mean_ratio < 1.25:
            raise AssertionError(f"Photon/ReSTIR energy ratio is invalid: {photon_mean_ratio}")

    # BDPT-compatible vertices select its complete recursive estimator and must be bit exact.
    bdpt_common = [
        "--device",
        "GPU",
        "--bdpt",
        "--no-glass",
        "--samples",
        "2",
        "--light-paths",
        "1024",
        "--update-samples",
        "1",
        "--resolution",
        "16",
    ]
    bdpt = run_render(options.blender, options.smoke, output_dir, "bdpt", bdpt_common)
    bdpt_restir = run_render(
        options.blender, options.smoke, output_dir, "bdpt_restir_fallback", bdpt_common + ["--restir"]
    )
    require_equal(bdpt, bdpt_restir, "BDPT technique-aware fallback", 1.0e-6)
    bdpt_restir_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "bdpt_restir_pt_fallback",
        bdpt_common + ["--restir-pt"],
    )
    require_equal(bdpt, bdpt_restir_pt, "BDPT/ReSTIR PT technique-aware fallback", 1.0e-6)

    shadow_common = [
        "--device",
        "GPU",
        "--shadow-catcher",
        "--allow-empty",
        "--no-glass",
        "--samples",
        "4",
        "--resolution",
        "16",
    ]
    run_render(options.blender, options.smoke, output_dir, "shadow_catcher", shadow_common)
    run_render(
        options.blender,
        options.smoke,
        output_dir,
        "shadow_catcher_fallback",
        shadow_common + ["--restir"],
    )
    run_render(
        options.blender,
        options.smoke,
        output_dir,
        "shadow_catcher_restir_pt_fallback",
        shadow_common + ["--restir-pt"],
    )

    base_count = 0 if options.enhanced_only else len(cases) + 3
    print(f"RESTIR_REGRESSION passed={base_count + len(enhanced_cases) + 6} output={output_dir}")
    del output_context


if __name__ == "__main__":
    main()
