# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Vertex connection and merging together with other Cycles features.

  blender -b --factory-startup --python tests/python/cycles_vcm_stress.py -- \
      --output DIRECTORY [--device CPU|METAL] [--integrator pt|bdpt] [--cases a,b,c]

Every case renders a scene of tests/python/cycles_vcm_scenes.py with a feature that merging has
to cooperate with, once without merging as reference and once with. The image with merging must
be finite, without negative pixels, and its mean must match the reference. Cases with render
passes also check that the light passes and the light groups add up to the combined pass.
Prints one VCM_STRESS line per case and exits non-zero if any case failed.
"""

import argparse
import importlib.util
import math
import sys
from pathlib import Path

import bpy
import numpy as np

spec = importlib.util.spec_from_file_location(
    "cycles_vcm_scenes", Path(__file__).with_name("cycles_vcm_scenes.py"))
scenes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scenes)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="CPU", choices=("METAL", "CPU"))
    parser.add_argument("--integrator", default="pt", choices=("pt", "bdpt"))
    parser.add_argument("--cases", default="all")
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--reference-samples", type=int, default=2048)
    parser.add_argument("--resolution", type=int, default=96)
    return parser.parse_args(argv)


def scene_args(args, vcm, samples, **overrides):
    """Arguments of the scenes module for one render."""
    values = dict(
        output=args.output, device=args.device, integrator=args.integrator if vcm else "pt",
        vcm=vcm, guiding=False, scenes="", resolution=args.resolution, samples=samples, seed=7,
        threads=0, bounces=8, radius=0.0, alpha=0.75, merge_max=16, light_paths=65536,
        update_samples=8, time_limit=0.0, suffix="", save_blend=False, no_render=False, png=False)
    values.update(overrides)
    return argparse.Namespace(**values)


def setup_device(args, scene):
    if args.device == "METAL":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "METAL"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == "METAL"
        scene.cycles.device = "GPU"


def render(args, scene, name, multilayer=False):
    setup_device(args, scene)
    path = args.output / (name + ".exr")
    settings = scene.render.image_settings
    if multilayer:
        if hasattr(settings, "media_type"):
            settings.media_type = "MULTI_LAYER_IMAGE"
        settings.file_format = "OPEN_EXR_MULTILAYER"
    else:
        settings.file_format = "OPEN_EXR"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    return path


def load(path):
    """Channels of an EXR by name."""
    import OpenImageIO as oiio
    image = oiio.ImageInput.open(str(path))
    channels = {}
    subimage = 0
    # Every pass of a multilayer file is a part of its own.
    while image.seek_subimage(subimage, 0):
        spec = image.spec()
        pixels = image.read_image(subimage, 0, 0, spec.nchannels, "float")
        channels.update({name: pixels[:, :, index].astype(np.float64)
                         for index, name in enumerate(spec.channelnames)})
        subimage += 1
    image.close()
    return channels


def combined(channels):
    names = [name for name in channels if name.split(".")[-1] in "RGB" and
             ("Combined" in name or "." not in name)]
    if not names:
        names = [name for name in channels if name in ("R", "G", "B")]
    return np.stack([channels[name] for name in sorted(names)], axis=-1)


def layer(channels, name):
    found = sorted(key for key in channels if key.split(".")[-2:-1] == [name] and
                   key.split(".")[-1] in "RGB")
    if len(found) != 3:
        return None
    # Sorted is B, G, R.
    return np.stack([channels[found[2]], channels[found[1]], channels[found[0]]], axis=-1)


def passes_error(channels):
    """Relative difference between the combined pass and the sum of its light passes."""
    total = None
    for kind in ("Diffuse", "Glossy", "Transmission"):
        color = layer(channels, kind + " Color")
        direct = layer(channels, kind + " Direct")
        indirect = layer(channels, kind + " Indirect")
        if color is None or direct is None or indirect is None:
            return None
        part = (direct + indirect) * color
        total = part if total is None else total + part
    for name in ("Emission", "Environment"):
        part = layer(channels, name)
        if part is not None:
            total = total + part
    image = layer(channels, "Combined")
    return abs(total.mean() - image.mean()) / max(image.mean(), 1e-12)


def lightgroups_error(channels, groups):
    total = None
    for group in groups:
        part = layer(channels, "Combined_" + group)
        if part is None:
            return None
        total = part if total is None else total + part
    image = layer(channels, "Combined")
    return abs(total.mean() - image.mean()) / max(image.mean(), 1e-12)


# ------------------------------------------------------------------------------------------------
# Cases: a function that builds the scene from scene arguments, and the allowed mean difference.


def case_passes(sargs):
    scene = scenes.scene_principled(sargs)
    layer_settings = scene.view_layers[0]
    for name in ("diffuse_direct", "diffuse_indirect", "diffuse_color", "glossy_direct",
                 "glossy_indirect", "glossy_color", "transmission_direct",
                 "transmission_indirect", "transmission_color", "emit", "environment"):
        setattr(layer_settings, "use_pass_" + name, True)
    return scene


def case_lightgroups(sargs):
    scene = scenes.scene_emissive_mesh(sargs)
    view_layer = scene.view_layers[0]
    for name, objects in (("warm", ("Panel",)), ("cool", ("Bulb", "Card"))):
        view_layer.lightgroups.add(name=name)
        for object_name in objects:
            bpy.data.objects[object_name].lightgroup = name
    return scene


def case_adaptive(sargs):
    scene = scenes.scene_caustic_glass(sargs)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.adaptive_min_samples = 64
    return scene


def case_denoise(sargs):
    scene = scenes.scene_caustic_glass(sargs)
    scene.cycles.use_denoising = True
    return scene


def case_clamp(sargs):
    scene = scenes.scene_mirror_caustic(sargs)
    scene.cycles.sample_clamp_indirect = 4.0
    scene.cycles.sample_clamp_direct = 8.0
    return scene


def case_shadow_catcher(sargs):
    scene = scenes.scene_caustic_glass(sargs)
    scene.render.film_transparent = True
    bpy.data.objects["Floor"].is_shadow_catcher = True
    return scene


def case_panorama(sargs):
    scene = scenes.scene_caustic_glass(sargs)
    scene.camera.location = (0.0, -0.6, 1.0)
    scene.camera.data.type = "PANO"
    scene.camera.data.panorama_type = "EQUIRECTANGULAR"
    return scene


def case_fisheye_dof(sargs):
    scene = scenes.scene_caustic_glass(sargs)
    camera = scene.camera.data
    camera.type = "PANO"
    camera.panorama_type = "FISHEYE_EQUISOLID"
    camera.fisheye_lens = 12.0
    camera.fisheye_fov = math.radians(140)
    camera.dof.use_dof = True
    camera.dof.focus_distance = 3.8
    camera.dof.aperture_fstop = 1.0
    return scene


def case_texture(sargs):
    scene = scenes.scene_caustic_glass(sargs)
    image = bpy.data.images.new("Tiles", 256, 256)
    pixels = np.ones((256, 256, 4), dtype=np.float32)
    yy, xx = np.mgrid[0:256, 0:256]
    pixels[..., 0] = 0.3 + 0.5 * ((xx // 32 + yy // 32) % 2)
    pixels[..., 1] = 0.4 + 0.4 * ((xx // 16) % 2)
    pixels[..., 2] = 0.5
    image.pixels.foreach_set(pixels.ravel())
    mat = bpy.data.objects["Floor"].data.materials[0]
    nodes = mat.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    bsdf = next(node for node in nodes if node.type == "BSDF_DIFFUSE")
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Color"])
    return scene


def case_dispersion(sargs):
    scene = scenes.scene_caustic_glass(sargs)
    mat = scenes.principled("Dispersive", (1, 1, 1), 0.0, transmission=1.0)
    bsdf = next(node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    if "Dispersion" in bsdf.inputs:
        bsdf.inputs["Dispersion"].default_value = 0.6
    bpy.data.objects["Glass"].data.materials[0] = mat
    return scene


def case_subsurface(sargs):
    scene = scenes.scene_diffuse_box(sargs)
    mat = scenes.principled("Skin", (0.8, 0.5, 0.4), 0.4)
    bsdf = next(node for node in mat.node_tree.nodes if node.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Subsurface Weight"].default_value = 1.0
    bsdf.inputs["Subsurface Radius"].default_value = (0.3, 0.15, 0.1)
    bpy.data.objects["Sphere"].data.materials[0] = mat
    return scene


def case_transparent(sargs):
    """Surfaces with a transparent closure, which a light subpath passes and stores a vertex at."""
    scene = scenes.scene_diffuse_box(sargs)
    mat, nodes, links, output = scenes.new_material("Veil")
    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.5
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.8, 0.6, 0.2, 1.0)
    links.new(transparent.outputs[0], mix.inputs[1])
    links.new(diffuse.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    for index in range(4):
        scenes.add_plane("Veil%d" % index, (0, 0, 0.9 + 0.2 * index), (0, 0, 0), (1.6, 1.6), mat)
    return scene


def case_translucent(sargs):
    """A sheet lit from behind: the light vertex and the camera vertex are on opposite sides."""
    scene = scenes.scene_diffuse_box(sargs)
    mat, nodes, links, output = scenes.new_material("Paper")
    bsdf = nodes.new("ShaderNodeBsdfTranslucent")
    bsdf.inputs["Color"].default_value = (0.8, 0.8, 0.7, 1.0)
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    scenes.add_plane("Sheet", (0, -0.2, 1.0), (math.pi / 2, 0, 0), (1.8, 1.8), mat)
    return scene


def case_many_lights(sargs):
    scene = scenes.scene_diffuse_box(sargs)
    colors = ((1, 0.3, 0.2), (0.2, 1, 0.3), (0.3, 0.4, 1), (1, 1, 0.3), (1, 0.3, 1), (0.3, 1, 1))
    for index, color in enumerate(colors):
        angle = index * math.pi / 3
        scenes.add_area_light(
            "Extra%d" % index, (0.7 * math.cos(angle), 0.7 * math.sin(angle), 1.2 + 0.1 * index),
            0.05 + 0.05 * index, 4.0 + 3.0 * index, target=(0, 0, 0.3), color=color)
    return scene


def case_light_linking(sargs):
    scene = scenes.scene_diffuse_box(sargs)
    light = scenes.add_area_light("Linked", (0.5, -0.6, 1.5), 0.3, 40.0, target=(0, 0, 0.3),
                                  color=(0.3, 0.5, 1.0))
    collection = bpy.data.collections.new("Receivers")
    scene.collection.children.link(collection)
    collection.objects.link(bpy.data.objects["Sphere"])
    light.light_linking.receiver_collection = collection
    return scene


def case_one_sample_maps(sargs):
    """A new light pass for every sample: several caches per batch on Metal."""
    sargs.update_samples = 1
    return scenes.scene_caustic_glass(sargs)


def case_few_light_paths(sargs):
    sargs.light_paths = 1024
    sargs.merge_max = 1
    return scenes.scene_caustic_glass(sargs)


def case_many_light_paths(sargs):
    sargs.light_paths = 1048576
    sargs.radius = 0.05
    return scenes.scene_caustic_glass(sargs)


def case_large_radius(sargs):
    """Merge radius of ten times the default: blurred early, but the amount of light holds."""
    sargs.radius = 0.06
    return scenes.scene_diffuse_box(sargs)


CASES = {
    "passes": (case_passes, 0.02),
    "lightgroups": (case_lightgroups, 0.02),
    "adaptive": (case_adaptive, 0.03),
    "denoise": (case_denoise, 0.03),
    # Clamping removes light from the rare bright samples of the path tracer, which merging
    # replaces by many dim ones: the image with merging is brighter and closer to the truth.
    "clamp": (case_clamp, 0.50),
    "shadow_catcher": (case_shadow_catcher, 0.02),
    "panorama": (case_panorama, 0.02),
    "fisheye_dof": (case_fisheye_dof, 0.02),
    "texture": (case_texture, 0.02),
    "dispersion": (case_dispersion, 0.03),
    "subsurface": (case_subsurface, 0.02),
    "transparent": (case_transparent, 0.02),
    "translucent": (case_translucent, 0.02),
    "many_lights": (case_many_lights, 0.02),
    "light_linking": (case_light_linking, 0.02),
    "one_sample_maps": (case_one_sample_maps, 0.02),
    "few_light_paths": (case_few_light_paths, 0.03),
    "many_light_paths": (case_many_light_paths, 0.03),
    "large_radius": (case_large_radius, 0.05),
}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = list(CASES) if args.cases == "all" else args.cases.split(",")
    failed = 0
    for name in names:
        build, tolerance = CASES[name]
        multilayer = name in ("passes", "lightgroups")
        results = {}
        for kind, vcm, samples in (("reference", False, args.reference_samples),
                                   ("vcm", True, args.samples)):
            scene = build(scene_args(args, vcm, samples))
            path = render(args, scene, "%s_%s" % (name, kind), multilayer)
            results[kind] = load(path)
        reference = combined(results["reference"])
        image = combined(results["vcm"])
        valid = bool(np.isfinite(image).all() and (image >= 0.0).all())
        ratio = image.mean() / max(reference.mean(), 1e-12)
        notes = []
        ok = valid and abs(ratio - 1.0) <= tolerance
        if name == "passes":
            for kind in ("reference", "vcm"):
                error = passes_error(results[kind])
                notes.append("%s passes sum error %s" % (
                    kind, "missing" if error is None else "%.4f" % error))
                ok = ok and error is not None and error < 0.01
        if name == "lightgroups":
            for kind in ("reference", "vcm"):
                error = lightgroups_error(results[kind], ("warm", "cool"))
                notes.append("%s groups sum error %s" % (
                    kind, "missing" if error is None else "%.4f" % error))
                ok = ok and error is not None and error < 0.01
        failed += not ok
        print("VCM_STRESS %-6s %-5s %-18s mean %.4f %s %s" % (
            args.device, args.integrator, name, ratio, "; ".join(notes),
            "ok" if ok else ("FAILED" if valid else "INVALID PIXELS FAILED")), flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
