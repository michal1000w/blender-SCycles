#!/usr/bin/env python3
"""CPU host acceptance for coherent direct-light validation in a Blender build.

Run from the repository root with:
  python3 tests/python/cycles_coherent_validation.py [--blender PATH] [--output FILE]

Every case runs in a fresh background Blender process with a minimal clean
environment. This validates scene/light acceptance only; it does not test
coherent indirect or multipath transport.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


EXPECTED_ERRORS = {
    "radius": "Coherent direct light must be a zero-radius, shadow-casting, non-caustic point light",
    "wavelength_mismatch": "Coherent direct lights in one group must share wavelength, length, visibility, and light group",
    "length_mismatch": "Coherent direct lights in one group must share wavelength, length, visibility, and light group",
    "source_limit": "Coherent direct light group exceeds 16 point sources",
    "textured_emission": "Coherent direct light requires constant emission",
    "transparent_shadow": "Coherent direct point sources do not support volume or transparent-shadow shaders",
    "volume_shader": "Coherent direct point sources do not support volume or transparent-shadow shaders",
    "incremental_shader_change": "Coherent direct point sources do not support volume or transparent-shadow shaders",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_environment(temp_dir):
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(temp_dir),
        "TMPDIR": str(temp_dir),
        "LANG": "C.UTF-8",
    }


def run_case(blender, script, case_name, result_path, temp_dir):
    command = [
        str(blender), "--background", "--factory-startup",
        "--python-exit-code", "73", "--python", str(script), "--",
        "--worker", case_name, "--worker-result", str(result_path),
    ]
    process = subprocess.run(command, text=True, capture_output=True, check=False,
                              env=clean_environment(temp_dir))
    entry = {
        "case": case_name,
        "command": command,
        "process_exit_code": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr,
        "worker_result": None,
    }
    if result_path.exists():
        try:
            entry["worker_result"] = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            entry["worker_result_error"] = str(error)
    expected = EXPECTED_ERRORS.get(case_name)
    combined_log = process.stdout + "\n" + process.stderr
    if expected:
        entry["expected_error"] = expected
        entry["error_observed"] = expected in combined_log
        worker_error = (entry["worker_result"] or {}).get("render_exception", "")
        entry["passed"] = (
            entry["error_observed"] and expected in worker_error and
            entry["worker_result"] is not None and
            entry["worker_result"].get("worker_status") == "completed"
        )
        if case_name == "incremental_shader_change":
            worker = entry["worker_result"] or {}
            entry["passed"] = (
                entry["error_observed"] and expected in worker.get("second_render_exception", "") and
                worker.get("first_render_status") == ["FINISHED"] and
                worker.get("second_render_exception") is not None
            )
    else:
        worker = entry["worker_result"] or {}
        entry["passed"] = (
            process.returncode == 0 and worker.get("worker_status") == "completed" and
            worker.get("render_status") == ["FINISHED"] and worker.get("all_pixels_finite") is True
        )
    return entry


def worker(case_name, result_path):
    import bpy
    import math
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 1
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.use_bidirectional_path_tracing = False
    scene.cycles.use_guiding = False
    scene.cycles.use_photon_mapping = case_name == "photon_mapping"
    scene.render.resolution_x = 4
    scene.render.resolution_y = 4
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.world = bpy.data.worlds.new("Black world")
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0

    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0.0, 0.0, 0.0))
    receiver = bpy.context.object
    receiver.name = "CPU validation receiver"
    material = bpy.data.materials.new("Diffuse receiver")
    material.diffuse_color = (0.8, 0.8, 0.8, 1.0)
    receiver.data.materials.append(material)

    if case_name == "transparent_shadow":
        material.use_nodes = True
        material.use_transparent_shadow = True
        nodes = material.node_tree.nodes
        nodes.clear()
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        output = nodes.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
    elif case_name == "volume_shader":
        material.use_nodes = True
        nodes = material.node_tree.nodes
        nodes.clear()
        volume = nodes.new("ShaderNodeVolumePrincipled")
        output = nodes.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(volume.outputs["Volume"], output.inputs["Volume"])

    bpy.ops.object.camera_add(location=(0.0, -4.0, 3.0))
    scene.camera = bpy.context.object
    scene.camera.rotation_euler = (Vector((0.0, 0.0, 0.0)) - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera.data.lens = 50

    count = 17 if case_name == "source_limit" else 2
    lights = []
    for index in range(count):
        bpy.ops.object.light_add(type="POINT", location=((index - (count - 1) / 2) * 0.1, 0.0, 2.0))
        light = bpy.context.object
        light.name = f"Coherent point {index:02d}"
        light.data.energy = 2.0
        light.data.shadow_soft_size = 0.0
        settings = light.data.cycles
        settings.coherence_group = 1
        settings.coherence_wavelength_nm = 550.0
        settings.coherence_length_m = 0.002
        lights.append(light)

    if case_name == "radius":
        lights[0].data.shadow_soft_size = 0.1
    elif case_name == "wavelength_mismatch":
        lights[1].data.cycles.coherence_wavelength_nm = 610.0
    elif case_name == "length_mismatch":
        lights[1].data.cycles.coherence_length_m = 0.003
    elif case_name == "textured_emission":
        light_data = lights[0].data
        light_data.use_nodes = True
        tree = light_data.node_tree
        tree.nodes.clear()
        texture = tree.nodes.new("ShaderNodeTexNoise")
        emission = tree.nodes.new("ShaderNodeEmission")
        output = tree.nodes.new("ShaderNodeOutputLight")
        tree.links.new(texture.outputs["Color"], emission.inputs["Color"])
        tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])

    result = {
        "worker_status": "completed",
        "case": case_name,
        "binary": str(Path(bpy.app.binary_path).resolve()),
        "blender_version": bpy.app.version_string,
    }
    if case_name == "incremental_shader_change":
        first_path = result_path.with_name(result_path.stem + "_first.png")
        scene.render.filepath = str(first_path)
        try:
            result["first_render_status"] = sorted(bpy.ops.render.render(write_still=True))
            first_image = bpy.data.images.load(str(first_path), check_existing=False)
            first_pixels = list(first_image.pixels[:])
            result["first_render_size"] = list(first_image.size)
            result["first_render_all_pixels_finite"] = (
                bool(first_pixels) and all(math.isfinite(value) for value in first_pixels)
            )
            bpy.data.images.remove(first_image)
        except BaseException as error:
            result["first_render_exception"] = f"{type(error).__name__}: {error}"
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            return

        # Mutate the already-synced, nonemissive receiver after the first
        # successful scene update; rebuilding the shader must re-run validation.
        material.use_nodes = True
        material.use_transparent_shadow = True
        nodes = material.node_tree.nodes
        nodes.clear()
        transparent = nodes.new("ShaderNodeBsdfTransparent")
        output = nodes.new("ShaderNodeOutputMaterial")
        material.node_tree.links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
        scene.render.filepath = str(result_path.with_name(result_path.stem + "_second.png"))
        try:
            result["second_render_status"] = sorted(bpy.ops.render.render(write_still=True))
        except BaseException as error:
            result["second_render_exception"] = f"{type(error).__name__}: {error}"
        result_path.write_text(json.dumps(result, indent=2) + "\n")
        return

    try:
        render_path = result_path.with_suffix(".png")
        scene.render.filepath = str(render_path)
        status = bpy.ops.render.render(write_still=True)
        result["render_status"] = sorted(status)
        image = bpy.data.images.load(str(render_path), check_existing=False) if render_path.exists() else None
        pixels = list(image.pixels[:]) if image else []
        result["render_result_size"] = list(image.size) if image else []
        result["render_result_pixel_count"] = len(pixels)
        result["all_pixels_finite"] = bool(pixels) and all(math.isfinite(value) for value in pixels)
        if image:
            bpy.data.images.remove(image)
    except BaseException as error:
        result["render_exception"] = f"{type(error).__name__}: {error}"
        result["all_pixels_finite"] = False
    result_path.write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blender",
        type=Path,
        default=Path("build/diffraction_delivery_20260927_coherent_direct_v1/Blender.app/Contents/MacOS/Blender"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/tests/python/cycles_coherent_validation.json"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    blender = args.blender.resolve()
    output = args.output.resolve()
    if not blender.is_file():
        parser.error(f"Blender executable not found: {blender}")
    output.parent.mkdir(parents=True, exist_ok=True)

    cases = ["valid", *EXPECTED_ERRORS]
    # The renderer's photon-map predicate is explicitly Metal-only, so this
    # validation branch cannot be reached in the required CPU-only run.
    results = {
        "binary": str(blender),
        "binary_sha256": sha256(blender),
        "scope": "CPU host validation and tiny direct render; no GPU or multipath transport",
        "cases": {},
        "photon_mapping": {
            "status": "skipped",
            "reason": "use_photon_mapping_on_device is false for CPU devices; testing it requires Metal",
        },
    }
    with tempfile.TemporaryDirectory(prefix="cycles-coherent-validation-") as temporary:
        temp_dir = Path(temporary)
        for case_name in cases:
            worker_result = temp_dir / f"{case_name}.json"
            entry = run_case(blender, Path(__file__).resolve(), case_name, worker_result, temp_dir)
            results["cases"][case_name] = entry
            print(f"{case_name}: {'PASS' if entry['passed'] else 'FAIL'}", flush=True)
            if not entry["passed"]:
                print(entry.get("expected_error", "valid case did not render"), file=sys.stderr)
                print(entry["stdout"], file=sys.stderr)
                print(entry["stderr"], file=sys.stderr)
    results["passed"] = all(case["passed"] for case in results["cases"].values())
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"results_json={output}")
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    if "--worker" in sys.argv:
        separator = sys.argv.index("--") if "--" in sys.argv else len(sys.argv)
        worker_args = argparse.ArgumentParser()
        worker_args.add_argument("--worker", required=True)
        worker_args.add_argument("--worker-result", type=Path, required=True)
        worker_options = worker_args.parse_args(sys.argv[separator + 1:])
        worker(worker_options.worker, worker_options.worker_result)
    else:
        raise SystemExit(main())
