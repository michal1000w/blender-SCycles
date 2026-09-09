#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Metal regression and benchmark suite for Cycles BDPT.

Run this with the system Python. It launches the supplied Blender build for each independent
render so cache refresh, device selection, and startup state are exercised as users encounter them.
"""

import argparse
import pathlib
import re
import subprocess
import tempfile
import time


METRIC = re.compile(r"([a-z0-9_]+)=([-+0-9.eE]+)")


def relative_error(a, b):
    return abs(a - b) / max(abs(b), 1.0e-12)


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
    print(f"{name:28s} {line}")
    return metrics


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender", required=True, type=pathlib.Path)
    parser.add_argument("--smoke", type=pathlib.Path, default=pathlib.Path(__file__).with_name("cycles_bdpt_smoke.py"))
    parser.add_argument("--keep-output", type=pathlib.Path)
    options = parser.parse_args()

    if options.keep_output:
        options.keep_output.mkdir(parents=True, exist_ok=True)
        output_context = None
        output_dir = options.keep_output
    else:
        output_context = tempfile.TemporaryDirectory(prefix="cycles-bdpt-metal-")
        output_dir = pathlib.Path(output_context.name)

    started = time.perf_counter()
    common = ["--no-glass", "--resolution", "64"]

    cpu_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "cpu_fallback_pt",
        ["--device", "CPU", "--no-glass", "--samples", "256", "--resolution", "32"],
    )
    cpu_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "cpu_fallback_bdpt",
        [
            "--device",
            "CPU",
            "--bdpt",
            "--no-glass",
            "--samples",
            "256",
            "--resolution",
            "32",
        ],
    )
    require(
        cpu_bdpt["mean"] == cpu_pt["mean"] and cpu_bdpt["peak"] == cpu_pt["peak"],
        "Enabling Metal-only BDPT changed the CPU fallback render",
    )

    diffuse_pt = run_render(
        options.blender, options.smoke, output_dir, "diffuse_pt", common + ["--diffuse-node", "--samples", "4096"]
    )
    diffuse_lt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "diffuse_pure_light_trace",
        common
        + [
            "--diffuse-node",
            "--bdpt",
            "--samples",
            "1",
            "--update-samples",
            "1",
            "--max-bounces",
            "1",
            "--light-paths",
            "1048576",
        ],
    )
    require(
        relative_error(diffuse_lt["mean"], diffuse_pt["mean"]) < 0.01,
        "Lambertian pure light tracing differs from PT by at least 1%",
    )

    # A path stopped after its first connectible hit has exactly one reservoir candidate,
    # regardless of the configured BDPT maximum. This catches the old fixed-bounce selection,
    # which retained only 1/max_bounces of these paths and amplified the resulting sparse splats.
    short_paths_1 = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "short_path_support_1",
        common
        + [
            "--diffuse-node",
            "--bdpt",
            "--samples",
            "1",
            "--update-samples",
            "1",
            "--max-bounces",
            "1",
            "--light-max-bounces",
            "0",
            "--light-paths",
            "262144",
        ],
    )
    short_paths_32 = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "short_path_support_32",
        common
        + [
            "--diffuse-node",
            "--bdpt",
            "--samples",
            "1",
            "--update-samples",
            "1",
            "--max-bounces",
            "32",
            "--light-max-bounces",
            "0",
            "--light-paths",
            "262144",
        ],
    )
    require(
        relative_error(short_paths_32["mean"], short_paths_1["mean"]) < 1.0e-6
        and relative_error(short_paths_32["p999"], short_paths_1["p999"]) < 1.0e-6
        and relative_error(short_paths_32["peak"], short_paths_1["peak"]) < 1.0e-6,
        "Configured BDPT maximum amplified one-vertex light paths",
    )

    diffuse_passes = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "diffuse_light_trace_passes",
        common
        + [
            "--diffuse-node",
            "--bdpt",
            "--samples",
            "1",
            "--update-samples",
            "1",
            "--max-bounces",
            "1",
            "--light-paths",
            "1048576",
            "--report-passes",
        ],
    )
    require(
        diffuse_passes["diffuse_direct_mean"] > 0.1
        and diffuse_passes["diffuse_indirect_mean"] == 0.0,
        "A first-bounce sensor splat was not routed exclusively to the direct pass",
    )

    principled_pt = run_render(
        options.blender, options.smoke, output_dir, "principled_pt", common + ["--samples", "4096"]
    )
    principled_lt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "principled_pure_light_trace",
        common
        + [
            "--bdpt",
            "--samples",
            "1",
            "--update-samples",
            "1",
            "--max-bounces",
            "1",
            "--light-paths",
            "1048576",
        ],
    )
    require(
        relative_error(principled_lt["mean"], principled_pt["mean"]) < 0.01,
        "Principled adjoint light tracing differs from PT by at least 1%",
    )

    mixed = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "principled_full_mis",
        common
        + [
            "--bdpt",
            "--samples",
            "4096",
            "--update-samples",
            "4",
            "--max-bounces",
            "1",
            "--light-paths",
            "65536",
        ],
    )
    require(
        relative_error(mixed["mean"], principled_pt["mean"]) < 0.01,
        "Full BDPT strategy mixture differs from PT by at least 1%",
    )

    hair_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "hair_bsdf_pt",
        common + ["--hair-bsdf", "--samples", "4096"],
    )
    hair_lt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "hair_bsdf_light_trace",
        common
        + [
            "--hair-bsdf",
            "--bdpt",
            "--samples",
            "1",
            "--update-samples",
            "1",
            "--max-bounces",
            "1",
            "--light-paths",
            "1048576",
        ],
    )
    require(
        relative_error(hair_lt["mean"], hair_pt["mean"]) < 0.01,
        "Hair BSDF adjoint light tracing differs from PT by at least 1%",
    )

    ray_portal = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "ray_portal_light_trace",
        common
        + [
            "--ray-portal",
            "--bdpt",
            "--samples",
            "1",
            "--update-samples",
            "1",
            "--max-bounces",
            "2",
            "--light-paths",
            "1048576",
        ],
    )
    require(
        ray_portal["peak"] > 10.0,
        "Ray Portal BSDF did not remap a BDPT light subpath",
    )

    caustic_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "glass_caustic_pt",
        ["--samples", "256", "--resolution", "64"],
    )
    caustic_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "glass_caustic_bdpt",
        [
            "--bdpt",
            "--samples",
            "256",
            "--resolution",
            "64",
            "--light-paths",
            "65536",
            "--update-samples",
            "8",
            "--max-bounces",
            "8",
        ],
    )
    require(
        caustic_bdpt["p999"] > 4.0 * caustic_pt["p999"] and caustic_bdpt["peak"] > 20.0,
        "BDPT did not resolve the focused glass caustic",
    )

    opaque_reflective = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "opaque_reflective_caustic",
        [
            "--bdpt",
            "--opaque-caster",
            "--samples",
            "8",
            "--resolution",
            "64",
            "--light-paths",
            "262144",
            "--update-samples",
            "8",
            "--max-bounces",
            "8",
            "--report-passes",
        ],
    )
    opaque_no_reflective = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "opaque_reflective_caustic_disabled",
        [
            "--bdpt",
            "--opaque-caster",
            "--no-reflective-caustics",
            "--samples",
            "8",
            "--resolution",
            "64",
            "--light-paths",
            "262144",
            "--update-samples",
            "8",
            "--max-bounces",
            "8",
            "--report-passes",
        ],
    )
    require(
        opaque_no_reflective["diffuse_indirect_mean"]
        < 0.15 * opaque_reflective["diffuse_indirect_mean"],
        "Disabling reflective caustics did not remove opaque BDPT caustic wings",
    )

    view_glass_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "glass_caustic_through_glass_pt",
        ["--samples", "256", "--resolution", "64", "--view-glass", "--flat-view-glass"],
    )
    view_glass_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "glass_caustic_through_glass_bdpt",
        [
            "--bdpt",
            "--samples",
            "256",
            "--resolution",
            "64",
            "--light-paths",
            "65536",
            "--update-samples",
            "8",
            "--max-bounces",
            "8",
            "--view-glass",
            "--flat-view-glass",
        ],
    )
    require(
        view_glass_bdpt["peak"] > 5.0 * view_glass_pt["peak"]
        and view_glass_bdpt["peak"] > 20.0,
        "BDPT manifold connection did not retain the caustic through refractive glass",
    )

    spectral_caustic = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "glass_caustic_spectral_bdpt",
        [
            "--bdpt",
            "--samples",
            "64",
            "--resolution",
            "64",
            "--light-paths",
            "65536",
            "--update-samples",
            "8",
            "--max-bounces",
            "8",
            "--dispersion",
            "1.0",
        ],
    )
    require(
        spectral_caustic["top_chroma"] > 2.0 * caustic_bdpt["top_chroma"],
        "BDPT dispersive caustic did not retain spectral wavelength separation",
    )

    caustic_dof = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "glass_caustic_dof",
        [
            "--bdpt",
            "--samples",
            "256",
            "--resolution",
            "64",
            "--light-paths",
            "65536",
            "--update-samples",
            "8",
            "--max-bounces",
            "8",
            "--aperture",
            "0.1",
        ],
    )
    require(caustic_dof["p999"] > 20.0, "Thin-lens BDPT did not retain the focused caustic")
    require(
        relative_error(caustic_dof["mean"], caustic_bdpt["mean"]) < 0.03,
        "Thin-lens sensor changed mean energy by at least 3%",
    )

    caustic_motion = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "glass_caustic_camera_motion",
        [
            "--bdpt",
            "--samples",
            "128",
            "--resolution",
            "64",
            "--light-paths",
            "65536",
            "--update-samples",
            "8",
            "--max-bounces",
            "8",
            "--camera-motion",
        ],
    )
    require(caustic_motion["p999"] > 20.0, "Motion-blurred BDPT lost the focused caustic")

    caustic_adaptive = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "glass_caustic_adaptive",
        [
            "--bdpt",
            "--adaptive",
            "--samples",
            "256",
            "--resolution",
            "64",
            "--light-paths",
            "65536",
            "--update-samples",
            "8",
            "--max-bounces",
            "8",
        ],
    )
    require(caustic_adaptive["p999"] > 20.0, "Adaptive sampling lost the focused BDPT caustic")
    require(
        relative_error(caustic_adaptive["mean"], caustic_bdpt["mean"]) < 0.02,
        "Adaptive sampling changed BDPT mean energy by at least 2%",
    )

    refresh_a = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "refresh_16k_4",
        ["--bdpt", "--samples", "64", "--resolution", "64", "--light-paths", "16384", "--update-samples", "4"],
    )
    refresh_b = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "refresh_64k_16",
        ["--bdpt", "--samples", "64", "--resolution", "64", "--light-paths", "65536", "--update-samples", "16"],
    )
    require(
        relative_error(refresh_a["mean"], refresh_b["mean"]) < 0.03,
        "Equal-work cache refresh configurations differ by at least 3%",
    )

    scale_32 = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "resolution_32",
        ["--bdpt", "--no-glass", "--samples", "1", "--resolution", "32", "--max-bounces", "1", "--light-paths", "262144"],
    )
    scale_64 = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "resolution_64",
        ["--bdpt", "--no-glass", "--samples", "1", "--resolution", "64", "--max-bounces", "1", "--light-paths", "1048576"],
    )
    require(
        relative_error(scale_32["mean"], scale_64["mean"]) < 0.02,
        "Sensor normalization changes by at least 2% with resolution",
    )

    crop_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "crop_border_pt",
        common + ["--crop-border", "--samples", "4096"],
    )
    crop_lt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "crop_border_light_trace",
        common
        + [
            "--crop-border",
            "--bdpt",
            "--samples",
            "1",
            "--update-samples",
            "1",
            "--max-bounces",
            "1",
            "--light-paths",
            "1048576",
        ],
    )
    require(
        relative_error(crop_lt["mean"], crop_pt["mean"]) < 0.01,
        "Cropped-border light tracing differs from PT by at least 1%",
    )

    for camera_type in ("ORTHO", "PANO"):
        camera_pt = run_render(
            options.blender,
            options.smoke,
            output_dir,
            f"{camera_type.lower()}_pt",
            common + ["--camera-type", camera_type, "--samples", "4096"],
        )
        camera_lt = run_render(
            options.blender,
            options.smoke,
            output_dir,
            f"{camera_type.lower()}_pure_light_trace",
            common
            + [
                "--camera-type",
                camera_type,
                "--bdpt",
                "--samples",
                "1",
                "--update-samples",
                "1",
                "--max-bounces",
                "1",
                "--light-paths",
                "1048576",
            ],
        )
        require(
            relative_error(camera_lt["mean"], camera_pt["mean"]) < 0.01,
            f"{camera_type} pure light tracing differs from PT by at least 1%",
        )

    unsupported_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "ortho_dof_fallback_pt",
        common + ["--camera-type", "ORTHO", "--aperture", "0.1", "--samples", "4096"],
    )
    unsupported_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "ortho_dof_fallback_bdpt",
        common
        + [
            "--camera-type",
            "ORTHO",
            "--aperture",
            "0.1",
            "--bdpt",
            "--samples",
            "4096",
            "--light-paths",
            "65536",
        ],
    )
    require(
        relative_error(unsupported_bdpt["mean"], unsupported_pt["mean"]) < 1.0e-6,
        "Unsupported camera did not fall back exactly to regular path tracing",
    )
    require(
        unsupported_bdpt["seconds"] < 2.0 * unsupported_pt["seconds"],
        "Unsupported camera fallback still schedules substantial BDPT work",
    )

    shadow_catcher_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "shadow_catcher_pt",
        common
        + ["--shadow-catcher", "--allow-empty", "--samples", "512", "--report-passes"],
    )
    shadow_catcher_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "shadow_catcher_bdpt_fallback",
        common
        + [
            "--shadow-catcher",
            "--allow-empty",
            "--bdpt",
            "--samples",
            "512",
            "--light-paths",
            "65536",
            "--report-passes",
        ],
    )
    require(
        shadow_catcher_bdpt["mean"] == shadow_catcher_pt["mean"]
        and shadow_catcher_bdpt["peak"] == shadow_catcher_pt["peak"]
        and shadow_catcher_bdpt["shadow_catcher_mean"]
        == shadow_catcher_pt["shadow_catcher_mean"]
        and shadow_catcher_bdpt["shadow_catcher_mean"] > 0.0,
        "Shadow catcher mode did not fall back exactly to regular path tracing",
    )
    require(
        shadow_catcher_bdpt["seconds"] < 2.0 * shadow_catcher_pt["seconds"],
        "Shadow catcher fallback still schedules substantial BDPT work",
    )

    for light_type in ("POINT", "SPOT", "SUN", "MESH", "WORLD"):
        emitter_pt = run_render(
            options.blender,
            options.smoke,
            output_dir,
            f"{light_type.lower()}_emitter_pt",
            common + ["--light-type", light_type, "--samples", "4096"],
        )
        emitter_lt = run_render(
            options.blender,
            options.smoke,
            output_dir,
            f"{light_type.lower()}_emitter_light_trace",
            common
            + [
                "--light-type",
                light_type,
                "--bdpt",
                "--samples",
                "1",
                "--update-samples",
                "1",
                "--max-bounces",
                "1",
                "--light-paths",
                "1048576",
            ],
        )
        require(
            emitter_lt["mean"] > 0.0 and emitter_lt["peak"] > emitter_lt["mean"],
            f"{light_type} light tracing is empty",
        )
        require(
            relative_error(emitter_lt["mean"], emitter_pt["mean"]) < 0.02,
            f"{light_type} light tracing differs from PT by at least 2%",
        )

    sun_soft_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "finite_angle_sun_pt",
        common + ["--light-type", "SUN", "--sun-angle", "5", "--samples", "4096"],
    )
    sun_soft_lt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "finite_angle_sun_light_trace",
        common
        + [
            "--light-type",
            "SUN",
            "--sun-angle",
            "5",
            "--bdpt",
            "--samples",
            "1",
            "--update-samples",
            "1",
            "--max-bounces",
            "1",
            "--light-paths",
            "1048576",
        ],
    )
    require(
        relative_error(sun_soft_lt["mean"], sun_soft_pt["mean"]) < 0.02,
        "Finite-angle sun light tracing differs from PT by at least 2%",
    )

    for light_type in ("POINT", "SPOT"):
        impostor_common = common + ["--light-type", light_type, "--light-radius", "0.35"]
        impostor_pt = run_render(
            options.blender,
            options.smoke,
            output_dir,
            f"{light_type.lower()}_soft_impostor_pt",
            impostor_common + ["--samples", "4096"],
        )
        impostor_bdpt = run_render(
            options.blender,
            options.smoke,
            output_dir,
            f"{light_type.lower()}_soft_impostor_bdpt",
            impostor_common + ["--bdpt", "--samples", "4096", "--light-paths", "65536"],
        )
        require(
            relative_error(impostor_bdpt["mean"], impostor_pt["mean"]) < 0.005,
            f"Receiver-facing finite-radius {light_type} fallback differs from PT by at least 0.5%",
        )

        sphere_common = impostor_common + ["--physical-sphere"]
        sphere_pt = run_render(
            options.blender,
            options.smoke,
            output_dir,
            f"{light_type.lower()}_physical_sphere_pt",
            sphere_common + ["--samples", "4096"],
        )
        sphere_lt = run_render(
            options.blender,
            options.smoke,
            output_dir,
            f"{light_type.lower()}_physical_sphere_light_trace",
            sphere_common
            + [
                "--bdpt",
                "--samples",
                "1",
                "--update-samples",
                "1",
                "--max-bounces",
                "1",
                "--light-paths",
                "1048576",
            ],
        )
        require(
            relative_error(sphere_lt["mean"], sphere_pt["mean"]) < 0.02,
            f"Physical sphere {light_type} light tracing differs from PT by at least 2%",
        )

    light_link_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "light_link_excluded_pt",
        common + ["--light-link", "NONE", "--allow-empty", "--samples", "64"],
    )
    light_link_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "light_link_excluded_bdpt",
        common
        + [
            "--light-link",
            "NONE",
            "--allow-empty",
            "--bdpt",
            "--samples",
            "64",
            "--light-paths",
            "65536",
        ],
    )
    require(
        light_link_pt["peak"] == 0.0 and light_link_bdpt["peak"] == 0.0,
        "Excluded light leaked through a PT or BDPT strategy",
    )

    shadow_link_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "shadow_link_pt",
        common
        + ["--light-link", "FLOOR", "--shadow-link-blocker", "--samples", "1024"],
    )
    shadow_link_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "shadow_link_bdpt",
        common
        + [
            "--light-link",
            "FLOOR",
            "--shadow-link-blocker",
            "--bdpt",
            "--samples",
            "1024",
            "--light-paths",
            "65536",
            "--update-samples",
            "4",
            "--max-bounces",
            "2",
        ],
    )
    require(
        relative_error(shadow_link_bdpt["mean"], shadow_link_pt["mean"]) < 0.015,
        "Linked-shadow BDPT differs from PT by at least 1.5%",
    )

    light_max_0 = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "light_max_bounces_0",
        common
        + [
            "--backdrop",
            "--light-max-bounces",
            "0",
            "--bdpt",
            "--samples",
            "1",
            "--light-paths",
            "1048576",
            "--update-samples",
            "1",
            "--max-bounces",
            "2",
        ],
    )
    light_max_1 = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "light_max_bounces_1",
        common
        + [
            "--backdrop",
            "--light-max-bounces",
            "1",
            "--bdpt",
            "--samples",
            "1",
            "--light-paths",
            "1048576",
            "--update-samples",
            "1",
            "--max-bounces",
            "2",
        ],
    )
    require(
        light_max_1["mean"] > 1.08 * light_max_0["mean"],
        "Per-emitter max_bounces did not enable the expected second-bounce transport",
    )

    absorption_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "world_absorption_pt",
        ["--no-glass", "--samples", "4096", "--resolution", "64", "--world-absorption", "0.08"],
    )
    absorption_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "world_absorption_bdpt",
        [
            "--bdpt",
            "--no-glass",
            "--samples",
            "1",
            "--resolution",
            "64",
            "--max-bounces",
            "1",
            "--light-paths",
            "1048576",
            "--world-absorption",
            "0.08",
        ],
    )
    require(
        relative_error(absorption_bdpt["mean"], absorption_pt["mean"]) < 0.01,
        "BDPT absorption transmittance differs from PT by at least 1%",
    )

    volume_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "volume_scatter_pt",
        [
            "--no-glass",
            "--volume-only",
            "--samples",
            "16384",
            "--resolution",
            "64",
            "--world-scatter",
            "0.18",
        ],
    )
    volume_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "volume_scatter_bdpt",
        [
            "--bdpt",
            "--no-glass",
            "--volume-only",
            "--samples",
            "512",
            "--resolution",
            "64",
            "--max-bounces",
            "8",
            "--light-paths",
            "1048576",
            "--update-samples",
            "512",
            "--world-scatter",
            "0.18",
        ],
    )
    require(
        relative_error(volume_bdpt["mean"], volume_pt["mean"]) < 0.015,
        "BDPT volume scattering differs from PT by at least 1.5%",
    )

    volume_light_tracing = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "volume_scatter_light_tracing",
        [
            "--bdpt",
            "--no-glass",
            "--volume-only",
            "--samples",
            "1",
            "--resolution",
            "64",
            "--max-bounces",
            "8",
            "--light-paths",
            "1048576",
            "--update-samples",
            "1",
            "--world-scatter",
            "0.18",
        ],
    )
    require(
        relative_error(volume_light_tracing["mean"], volume_pt["mean"]) < 0.08,
        "Dense volume light tracing did not reconstruct the PT reference within 8%",
    )

    volume_direct_only = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "glass_volume_direct_only",
        [
            "--bdpt",
            "--no-floor",
            "--samples",
            "1",
            "--resolution",
            "64",
            "--max-bounces",
            "1",
            "--light-paths",
            "1048576",
            "--update-samples",
            "1",
            "--volume-bounces",
            "0",
            "--world-scatter",
            "0.12",
            "--report-passes",
        ],
    )
    volume_caustic = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "glass_volume_caustic",
        [
            "--bdpt",
            "--no-floor",
            "--samples",
            "1",
            "--resolution",
            "64",
            "--max-bounces",
            "8",
            "--light-paths",
            "1048576",
            "--update-samples",
            "1",
            "--volume-bounces",
            "0",
            "--world-scatter",
            "0.12",
            "--report-passes",
        ],
    )
    require(
        volume_direct_only["volume_indirect_mean"] == 0.0
        and volume_caustic["volume_indirect_mean"] > 0.005,
        "Specular light subpaths did not produce a volume-indirect caustic contribution",
    )
    require(
        volume_caustic["volume_indirect_p999"]
        > 4.0 * volume_caustic["volume_indirect_mean"],
        "Post-glass volume connection was not spatially focused",
    )

    multiscatter_volume_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "volume_multiscatter_pt",
        [
            "--no-glass",
            "--volume-only",
            "--samples",
            "16384",
            "--resolution",
            "32",
            "--volume-bounces",
            "8",
            "--world-scatter",
            "0.18",
        ],
    )
    multiscatter_volume_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "volume_multiscatter_bdpt",
        [
            "--bdpt",
            "--no-glass",
            "--volume-only",
            "--samples",
            "2048",
            "--resolution",
            "32",
            "--max-bounces",
            "8",
            "--volume-bounces",
            "8",
            "--light-paths",
            "1048576",
            "--update-samples",
            "2048",
            "--world-scatter",
            "0.18",
            "--report-passes",
        ],
    )
    require(
        relative_error(multiscatter_volume_bdpt["mean"], multiscatter_volume_pt["mean"]) < 0.015,
        "BDPT mode changed multiple-scattering volume energy by at least 1.5%",
    )
    require(
        multiscatter_volume_bdpt["volume_direct_mean"] > 0.0
        and multiscatter_volume_bdpt["volume_indirect_mean"] > 0.0,
        "Multiple-scattering volume transport was not split into direct and indirect passes",
    )

    subsurface_pt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "subsurface_pt",
        [
            "--no-glass",
            "--no-floor",
            "--subsurface",
            "--samples",
            "8192",
            "--resolution",
            "64",
        ],
    )
    subsurface_bdpt = run_render(
        options.blender,
        options.smoke,
        output_dir,
        "subsurface_bdpt",
        [
            "--bdpt",
            "--no-glass",
            "--no-floor",
            "--subsurface",
            "--samples",
            "8192",
            "--resolution",
            "64",
            "--max-bounces",
            "8",
            "--light-paths",
            "1048576",
            "--update-samples",
            "8192",
        ],
    )
    require(
        relative_error(subsurface_bdpt["mean"], subsurface_pt["mean"]) < 0.015,
        "BDPT mode changed BSSRDF transport by at least 1.5%",
    )
    require(subsurface_bdpt["peak"] > 5.0, "BDPT mode lost the subsurface highlight")

    print(f"BDPT_METAL_REGRESSION PASS seconds={time.perf_counter() - started:.3f} output={output_dir}")
    if output_context:
        output_context.cleanup()


if __name__ == "__main__":
    main()
