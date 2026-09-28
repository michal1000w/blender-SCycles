#!/usr/bin/env python3
"""Author/run bounded primary and camera-transmitted coherent pass accounting.

Blender --background --python THIS -- --prepare-only --output FIXTURES
Blender --background --python THIS -- --fixtures FIXTURES --output RENDERS
Recomposition is an independent pass-accounting identity, not an image fit.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pass_flags(layer, enabled):
    for prop in layer.bl_rna.properties:
        if prop.identifier.startswith("use_pass_") and prop.type == "BOOLEAN" and not prop.is_readonly:
            setattr(layer, prop.identifier, False)
    layer.use_pass_combined = True
    layer.cycles.denoising_store_passes = False
    if enabled:
        for lobe in ("diffuse", "glossy", "transmission"):
            for part in ("direct", "indirect", "color"):
                setattr(layer, f"use_pass_{lobe}_{part}", True)
        layer.use_pass_emit = layer.use_pass_environment = True


def read(path):
    import OpenImageIO as oiio
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise RuntimeError(oiio.geterror())
    channels = {}
    sub = 0
    while image.seek_subimage(sub, 0):
        spec = image.spec()
        pixels = np.asarray(image.read_image(format=oiio.FLOAT))
        channels.update({name: pixels[..., i] for i, name in enumerate(spec.channelnames)})
        sub += 1
    image.close()
    return channels


def rgb(channels, name):
    values = []
    for component in "RGB":
        matches = [v for k, v in channels.items() if k.endswith(f".{name}.{component}")]
        if len(matches) != 1:
            raise RuntimeError(f"Missing/ambiguous {name}/{component}: {list(channels)}")
        values.append(matches[0])
    return np.stack(values, axis=-1).astype(np.float64)


def prepare(output):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cycles_coherent_mirror_acceptance import configure_scene, add_horizontal_rectangle
    scene, detector, mirror, masks, sources, camera = configure_scene(True)
    for mask in masks:
        bpy.data.objects.remove(mask, do_unlink=True)
    detector.cycles.coherent_interface = "DETECTOR"
    mirror.cycles.coherent_interface = "MIRROR"
    detector.data.materials[0].node_tree.nodes.get("Diffuse BSDF").inputs["Color"].default_value = (.25, .5, .75, 1)
    scene.cycles.coherent_max_interface_events = 1
    scene.cycles.use_coherent_specular_connections = True
    scene.cycles.coherent_polarization_mode = "SCALAR"
    scene.cycles.samples = 32
    scene.cycles.diffuse_bounces = 0
    scene.cycles.transmission_bounces = 1
    scene.cycles.blur_glossy = 0
    scene.render.resolution_x = scene.render.resolution_y = 32
    camera.data.ortho_scale = .04
    pass_flags(scene.view_layers[0], True)
    output.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.save_as_mainfile(filepath=str((output / "primary.blend").resolve()))
    material = bpy.data.materials.new("Unmarked ideal camera glass")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    glass = nodes.new("ShaderNodeBsdfGlass")
    glass.distribution = "GGX"
    glass.inputs["Roughness"].default_value = 0
    glass.inputs["IOR"].default_value = 1.5
    out = nodes.new("ShaderNodeOutputMaterial")
    material.node_tree.links.new(glass.outputs["BSDF"], out.inputs["Surface"])
    # Camera aperture x~1; direct illumination crosses this z plane at x~.5,
    # reflected illumination at x~.833. Neither hits this finite window.
    add_horizontal_rectangle("Camera-only glass window", (1, 0, 1.5), .06, .06, material)
    bpy.ops.wm.save_as_mainfile(filepath=str((output / "secondary.blend").resolve()))
    (output / "manifest.json").write_text(json.dumps({
        "scope": "Primary colored Lambertian, and same detector through unmarked delta camera glass; both direct and mirror source routes",
        "radiometry": "Existing RGB-envelope coherent model, signed half-pair allocation",
        "prepared_only": True, "resolution": [32, 32], "samples": 32,
        "coherent_specular_enabled": scene.cycles.use_coherent_specular_connections,
        "polarization": scene.cycles.coherent_polarization_mode,
        "scenes": {name: {"path": str((output / f"{name}.blend").resolve()), "sha256": sha(output / f"{name}.blend")}
                   for name in ("primary", "secondary")}}, indent=2) + "\n")


def run(fixtures, output):
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Signed coherent light-pass accounting; primary+secondary same-seed Combined invariance and native Color recomposition",
              "binary_sha256": sha(bpy.app.binary_path), "script_sha256": sha(__file__), "cases": {}, "passed": False}
    try:
        for name in ("primary", "secondary"):
            path = fixtures / f"{name}.blend"
            bpy.ops.wm.open_mainfile(filepath=str(path.resolve()))
            scene = bpy.context.scene
            if not scene.cycles.use_coherent_specular_connections:
                raise RuntimeError("Fixture must enable coherent specular connections")
            scene.cycles.device = "GPU"
            scene.cycles.seed = 19
            scene.cycles.use_adaptive_sampling = scene.cycles.use_denoising = False
            scene.render.image_settings.media_type = "MULTI_LAYER_IMAGE"
            scene.render.image_settings.file_format = "OPEN_EXR_MULTILAYER"
            scene.render.image_settings.color_depth = "32"
            prefs = bpy.context.preferences.addons["cycles"].preferences
            prefs.compute_device_type = "METAL"
            prefs.get_devices()
            for d in prefs.devices:
                d.use = d.type == "METAL"
            if not any(d.use for d in prefs.devices):
                raise RuntimeError("No Metal device")
            saved = {}
            for enabled in (False, True):
                pass_flags(scene.view_layers[0], enabled)
                exr = output / f"{name}_{'passes' if enabled else 'combined_only'}.exr"
                scene.render.filepath = str(exr.resolve())
                start = time.perf_counter()
                status = bpy.ops.render.render(write_still=True)
                if "FINISHED" not in status:
                    raise RuntimeError(str(status))
                channels = read(exr)
                if not all(np.isfinite(v).all() for v in channels.values()):
                    raise RuntimeError("Nonfinite pass")
                saved[enabled] = channels
                report["cases"].setdefault(name, {"scene_sha256": sha(path), "renders": {}})["renders"][str(enabled)] = {
                    "exr": str(exr.resolve()), "sha256": sha(exr), "seconds": time.perf_counter() - start}
            passes = saved[True]
            combined = rgb(passes, "Combined")
            reconstructed = rgb(passes, "Emission") + rgb(passes, "Environment")
            components = {}
            for lobe in ("Diffuse", "Glossy", "Transmission"):
                direct, indirect, color = (rgb(passes, f"{lobe} {part}") for part in ("Direct", "Indirect", "Color"))
                reconstructed += color * (direct + indirect)
                components[lobe] = {"direct_mean": direct.mean(axis=(0, 1)).tolist(),
                                    "indirect_mean": indirect.mean(axis=(0, 1)).tolist(),
                                    "direct_abs_max": float(np.abs(direct).max())}
            invariant = float(np.max(np.abs(combined - rgb(saved[False], "Combined"))))
            error = float(np.max(np.abs(combined - reconstructed)))
            case = report["cases"][name]
            case.update({"all_finite": True, "combined_mean": combined.mean(axis=(0, 1)).tolist(),
                         "combined_invariance_max_error": invariant, "recomposition_max_error": error,
                         "components": components})
            if invariant > 1e-6 or error > 2e-6 or not np.any(combined > 0):
                raise RuntimeError(f"{name} invariance/recomposition gate")
            if name == "primary":
                if not (max(components["Diffuse"]["direct_mean"]) > 0 and max(components["Diffuse"]["indirect_mean"]) > 0):
                    raise RuntimeError("Missing primary direct/indirect classes")
            else:
                if max(v["direct_abs_max"] for v in components.values()) > 1e-6 or max(components["Transmission"]["indirect_mean"]) <= 0:
                    raise RuntimeError("Secondary first-camera transmission attribution gate")
        report["passed"] = True
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--fixtures", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if a.prepare_only:
        prepare(a.output)
    else:
        if not a.fixtures:
            p.error("--fixtures required to render")
        run(a.fixtures, a.output)


if __name__ == "__main__":
    main()
