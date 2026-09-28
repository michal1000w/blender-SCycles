# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded white-furnace render check for constant-input reflective Glossy MultiGGX.

Run with a diffraction-enabled Blender build:
  blender -b -t 4 --python tests/python/cycles_diffraction_multiggx_glossy_acceptance.py -- \
    --output /absolute/output/directory --device CPU [--osl] [--case relief_r06_p1600]

The white unit-reflector target is one. This checks image energy at several
view directions, not directional cache accuracy, PDF normalization, or a
general passivity bound. Each selected case saves its scene and raw image.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy


CASES = {
    **{
        f"relief_r{int(roughness * 10):02d}_p{pitch}": (roughness, pitch, 150.0, 1.0)
        for roughness in (0.3, 0.6, 0.9)
        for pitch in (740.0, 1600.0)
    },
    "flat_r06_p1600": (0.6, 1600.0, 0.0, 1.0),
    "disabled_r06_p1600": (0.6, 1600.0, 150.0, 0.0),
    "partial_r06_p1600": (0.6, 1600.0, 150.0, 0.5),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("CPU", "METAL"), default="CPU")
    parser.add_argument("--osl", action="store_true", help="CPU OSL shading")
    parser.add_argument("--case", action="append", choices=tuple(CASES),
                        help="select one or more cases; default is all nine")
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--resolution", type=int, default=32)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if args.osl and args.device != "CPU":
        parser.error("OSL requires CPU")
    if not 1 <= args.samples <= 16384 or not 16 <= args.resolution <= 128:
        parser.error("samples must be 1..16384 and resolution 16..128")
    args.case = list(dict.fromkeys(args.case or CASES))
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
    sphere.name = "White glossy test sphere"
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True
    bpy.ops.object.camera_add(location=(0.0, -4.0, 0.0))
    camera = bpy.context.object
    camera.rotation_euler = (sphere.location - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    camera.data.ortho_scale = 2.3
    scene.camera = camera

    material = bpy.data.materials.new("White reflective Glossy MultiGGX grating")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    glossy = nodes.new("ShaderNodeBsdfAnisotropic")
    glossy.distribution = "MULTI_GGX"
    glossy.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    glossy.inputs["Anisotropy"].default_value = 0.0
    tangent = nodes.new("ShaderNodeCombineXYZ")
    tangent.inputs["X"].default_value = 1.0
    material.node_tree.links.new(tangent.outputs["Vector"], glossy.inputs["Tangent"])
    material.node_tree.links.new(glossy.outputs["BSDF"], output.inputs["Surface"])
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
    return scene, glossy


def object_pixel_means(path, resolution):
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        pixels = tuple(image.pixels[:])
        if len(pixels) != resolution * resolution * 4 or not all(math.isfinite(x) for x in pixels):
            raise RuntimeError("EXR has invalid dimensions or nonfinite pixels")
        # Ortho sphere radius is 1, camera width 2.3. This interior disk is
        # wholly on the object, avoiding antialiased silhouette/background.
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


def main():
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scene, glossy = configure_scene(args)
    report = {
        "status": "running",
        "scope": __doc__,
        "device": args.device,
        "osl": args.osl,
        "samples": args.samples,
        "resolution": args.resolution,
        "seed": 11,
        "adaptive_sampling": False,
        "denoising": False,
        "energy_tolerance": 0.02,
        "flat_disabled_tolerance": 0.01,
        "selected_cases": args.case,
        "script_sha256": sha256(__file__),
        "binary_sha256": sha256(bpy.app.binary_path),
        "runs": [],
    }

    def save_report():
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save_report()
    for name in args.case:
        roughness, pitch, depth, weight = CASES[name]
        glossy.inputs["Roughness"].default_value = roughness
        glossy.inputs["Diffraction Pitch"].default_value = pitch
        glossy.inputs["Diffraction Depth"].default_value = depth
        glossy.inputs["Diffraction Duty Cycle"].default_value = 0.41
        glossy.inputs["Diffraction Weight"].default_value = weight
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
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"
        bpy.data.images["Render Result"].save_render(str(png), scene=scene)
        case_report = {
            "case": name,
            "roughness": roughness,
            "pitch_nm": pitch,
            "depth_nm": depth,
            "duty": 0.41,
            "coverage": weight,
            "mean_rgb": mean,
            "object_pixels": pixel_count,
            "maximum_energy_error": energy_error,
            "passes_energy": energy_error <= report["energy_tolerance"],
            "blend_sha256": sha256(blend),
            "exr_sha256": sha256(exr),
            "png_sha256": sha256(png),
        }
        report["runs"].append(case_report)
        save_report()
        print(json.dumps(case_report), flush=True)

    by_name = {run["case"]: run for run in report["runs"]}
    if "flat_r06_p1600" in by_name and "disabled_r06_p1600" in by_name:
        report["flat_disabled_difference"] = max(
            abs(a - b) for a, b in zip(by_name["flat_r06_p1600"]["mean_rgb"],
                                       by_name["disabled_r06_p1600"]["mean_rgb"])
        )
        report["flat_disabled_passed"] = (
            report["flat_disabled_difference"] <= report["flat_disabled_tolerance"]
        )
    report["status"] = "passed" if all(run["passes_energy"] for run in report["runs"]) and \
        report.get("flat_disabled_passed", True) else "failed"
    save_report()
    if report["status"] != "passed":
        raise RuntimeError("Glossy MultiGGX white-furnace acceptance failed; see report.json")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
