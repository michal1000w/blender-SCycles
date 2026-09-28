# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded white-furnace acceptance for constant-input Metallic MultiGGX diffraction.

Run with a diffraction-enabled Blender build:
  blender -b -t 4 --python tests/python/cycles_diffraction_multiggx_metallic_acceptance.py -- \
    --output /absolute/output/directory --device CPU [--osl] [--case conductor_full_film0]

Unit-reflector conductor and F82 cases target one. Colored cases check only
finite passive image means, not an analytic color reference. Each case saves
its scene, raw EXR, PNG, and file hashes. This image check does not replace
directional BSDF/PDF and reciprocity tests.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy


CASES = {}
for mode in ("conductor", "f82"):
    for coverage_name, coverage in (("full", 1.0), ("half", 0.5)):
        for film in (0.0, 150.0):
            CASES[f"{mode}_{coverage_name}_film{int(film)}"] = {
                "mode": mode, "roughness": 0.6, "anisotropy": 0.0,
                "pitch": 1600.0, "depth": 150.0, "coverage": coverage,
                "film": film, "colored": False,
            }
    CASES[f"{mode}_anisotropic_folded_tangent"] = {
        "mode": mode, "roughness": 0.55, "anisotropy": 0.45,
        "pitch": 1600.0, "depth": 150.0, "coverage": 1.0,
        "film": 0.0, "colored": False,
    }
    CASES[f"{mode}_flat"] = {
        "mode": mode, "roughness": 0.6, "anisotropy": 0.0,
        "pitch": 1600.0, "depth": 0.0, "coverage": 1.0,
        "film": 0.0, "colored": False,
    }
    CASES[f"{mode}_disabled"] = {
        "mode": mode, "roughness": 0.6, "anisotropy": 0.0,
        "pitch": 1600.0, "depth": 150.0, "coverage": 0.0,
        "film": 0.0, "colored": False,
    }
    CASES[f"{mode}_colored"] = {
        "mode": mode, "roughness": 0.6, "anisotropy": 0.0,
        "pitch": 1600.0, "depth": 150.0, "coverage": 1.0,
        "film": 0.0, "colored": True,
    }


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("CPU", "METAL"), default="CPU")
    parser.add_argument("--principled", action="store_true", help="Test fully metallic Principled instead of Metallic")
    parser.add_argument("--osl", action="store_true", help="CPU OSL shading")
    parser.add_argument("--case", action="append", choices=tuple(CASES),
                        help="select one or more cases; default is all sixteen")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=32)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.osl and args.device != "CPU":
        parser.error("OSL requires CPU")
    if not 1 <= args.samples <= 16384 or not 16 <= args.resolution <= 128:
        parser.error("samples must be 1..16384 and resolution 16..128")
    if args.principled and args.case and any(not name.startswith("f82_") for name in args.case):
        parser.error("Principled supports only the F82 cases")
    default_cases = [name for name in CASES if not args.principled or name.startswith("f82_")]
    args.case = list(dict.fromkeys(args.case or default_cases))
    return args


def configure_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = args.samples
    scene.cycles.seed = 11
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.time_limit = 0
    scene.cycles.device = "CPU"
    scene.cycles.shading_system = args.osl
    scene.cycles.use_bidirectional_path_tracing = False
    scene.cycles.use_guiding = False
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    world = bpy.data.worlds.new("Uniform unit furnace")
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = 1.0

    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
    sphere = bpy.context.object
    sphere.name = "Metallic MultiGGX furnace sphere"
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True
    bpy.ops.object.camera_add(location=(0.0, -4.0, 0.0))
    camera = bpy.context.object
    camera.rotation_euler = (sphere.location - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2.3
    scene.camera = camera

    material = bpy.data.materials.new("Metallic MultiGGX grating")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    metallic = nodes.new("ShaderNodeBsdfPrincipled" if args.principled else "ShaderNodeBsdfMetallic")
    if args.principled:
        metallic.inputs["Metallic"].default_value = 1.0
    metallic.distribution = "MULTI_GGX"
    tangent = nodes.new("ShaderNodeCombineXYZ")
    tangent.inputs["X"].default_value = 1.0
    material.node_tree.links.new(tangent.outputs["Vector"], metallic.inputs["Tangent"])
    material.node_tree.links.new(metallic.outputs["BSDF"], output.inputs["Surface"])
    sphere.data.materials.append(material)

    if args.device == "METAL":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "METAL"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == "METAL"
        if not any(device.use for device in preferences.devices):
            raise RuntimeError("No enabled Metal device")
        scene.cycles.device = "GPU"
    return scene, metallic


def object_pixel_means(path, resolution):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = tuple(image.pixels[:])
        if len(pixels) != resolution * resolution * 4 or not all(math.isfinite(x) for x in pixels):
            raise RuntimeError("EXR has invalid dimensions or nonfinite pixels")
        # Entire interior disk lies on the sphere, clear of the antialiased silhouette.
        radius = resolution / 2.3 * 0.82
        channels = [[], [], []]
        for y in range(resolution):
            for x in range(resolution):
                if (x + 0.5 - resolution / 2) ** 2 + (y + 0.5 - resolution / 2) ** 2 > radius**2:
                    continue
                offset = 4 * (y * resolution + x)
                for channel in range(3):
                    channels[channel].append(pixels[offset + channel])
        if len(channels[0]) < 20:
            raise RuntimeError("Sphere measurement region is too small")
        return [sum(values) / len(values) for values in channels], len(channels[0])
    finally:
        bpy.data.images.remove(image)


def set_case(metallic, spec):
    principled = metallic.bl_idname == "ShaderNodeBsdfPrincipled"
    if not principled:
        metallic.fresnel_type = "PHYSICAL_CONDUCTOR" if spec["mode"] == "conductor" else "F82"
    if spec["mode"] == "conductor":
        metallic.inputs["IOR"].default_value = (0.3, 0.8, 1.3) if spec["colored"] else (0.0, 0.0, 0.0)
        metallic.inputs["Extinction"].default_value = (3.0, 3.0, 3.0) if spec["colored"] else (1.0, 1.0, 1.0)
    else:
        metallic.inputs["Base Color"].default_value = (0.35, 0.55, 0.8, 1.0) if spec["colored"] else (1.0, 1.0, 1.0, 1.0)
        metallic.inputs["Specular Tint" if principled else "Edge Tint"].default_value = (0.6, 0.8, 0.95, 1.0) if spec["colored"] else (1.0, 1.0, 1.0, 1.0)
    for socket, value in (
        ("Roughness", spec["roughness"]),
        ("Anisotropic" if principled else "Anisotropy", spec["anisotropy"]),
        ("Thin Film Thickness", spec["film"]),
        ("Thin Film IOR", 1.33),
        ("Diffraction Weight", spec["coverage"]),
        ("Diffraction Pitch", spec["pitch"]),
        ("Diffraction Depth", spec["depth"]),
        ("Diffraction Duty Cycle", 0.41),
    ):
        metallic.inputs[socket].default_value = value


def main():
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene, metallic = configure_scene(args)
    report = {
        "status": "running",
        "scope": __doc__,
        "device": args.device,
        "principled": args.principled,
        "osl": args.osl,
        "samples": args.samples,
        "resolution": args.resolution,
        "seed": 11,
        "adaptive_sampling": False,
        "denoising": False,
        "energy_tolerance": 0.02,
        "flat_disabled_tolerance": 0.01,
        "colored_passivity_limit": 1.02,
        "selected_cases": args.case,
        "script_sha256": sha256(__file__),
        "binary_sha256": sha256(bpy.app.binary_path),
        "runs": [],
    }

    def save_report():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save_report()
    for name in args.case:
        spec = CASES[name]
        set_case(metallic, spec)
        scene.cycles.seed = 11
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        exr = output / f"{name}.exr"
        blend = output / f"{name}.blend"
        png = output / f"{name}.png"
        scene.render.filepath = str(exr)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend))
        bpy.ops.render.render(write_still=True)
        mean, pixel_count = object_pixel_means(exr, args.resolution)
        energy_error = max(abs(x - 1.0) for x in mean)
        passed = (all(0.0 <= x <= report["colored_passivity_limit"] for x in mean)
                  if spec["colored"] else energy_error <= report["energy_tolerance"])
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"
        bpy.data.images["Render Result"].save_render(str(png), scene=scene)
        case_report = {
            "case": name,
            **spec,
            "duty": 0.41,
            "mean_rgb": mean,
            "object_pixels": pixel_count,
            "maximum_energy_error": energy_error,
            "criterion": "colored_passivity" if spec["colored"] else "unit_furnace",
            "passed": passed,
            "blend_sha256": sha256(blend),
            "exr_sha256": sha256(exr),
            "png_sha256": sha256(png),
        }
        report["runs"].append(case_report)
        save_report()
        print(json.dumps(case_report), flush=True)

    by_name = {run["case"]: run for run in report["runs"]}
    for mode in ("conductor", "f82"):
        flat, disabled = by_name.get(f"{mode}_flat"), by_name.get(f"{mode}_disabled")
        if flat and disabled:
            difference = max(abs(a - b) for a, b in zip(flat["mean_rgb"], disabled["mean_rgb"]))
            report[f"{mode}_flat_disabled_difference"] = difference
            report[f"{mode}_flat_disabled_passed"] = difference <= report["flat_disabled_tolerance"]
    report["status"] = "passed" if all(run["passed"] for run in report["runs"]) and all(
        report.get(f"{mode}_flat_disabled_passed", True) for mode in ("conductor", "f82")
    ) else "failed"
    save_report()
    if report["status"] != "passed":
        raise RuntimeError("Metallic MultiGGX furnace acceptance failed; see report.json")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
