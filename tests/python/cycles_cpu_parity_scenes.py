# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Render light-cache transport and pixel displacement scenes on the CPU or Metal to EXR.

Complements cycles_metal_feature_scenes.py with the configurations that exercise bidirectional
path tracing, photon mapping and pixel-level displacement: every emitter type, light tree on and
off, adaptive sampling, tiled rendering, light passes and light groups, motion blur, spectral
dispersion, OSL shading, volumes and path guiding. Renders use fixed seeds and no denoising.
Compare CPU and Metal renders of the same build with cycles_metal_image_compare.py.

  blender -b --factory-startup --python tests/python/cycles_cpu_parity_scenes.py -- \
      --output DIRECTORY [--device CPU|METAL] [--scenes a,b,c] [--samples N] [--resolution N]
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="CPU", choices=("METAL", "CPU"))
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--threads", type=int, default=0)
    return parser.parse_args(argv)


def new_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    if args.threads:
        scene.render.threads_mode = "FIXED"
        scene.render.threads = args.threads
    cycles = scene.cycles
    cycles.device = "GPU" if args.device == "METAL" else "CPU"
    cycles.samples = args.samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.seed = 11
    cycles.max_bounces = 8
    cycles.caustics_reflective = True
    cycles.caustics_refractive = True
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.03, 0.035, 0.05, 1)
    scene.world = world
    bpy.ops.object.camera_add(location=(0.0, -5.0, 2.0), rotation=(math.radians(72), 0.0, 0.0))
    scene.camera = bpy.context.object
    return scene


def material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    return mat, nodes, mat.node_tree.links, output


def principled(name, color=(0.8, 0.8, 0.8, 1), roughness=0.5, metallic=0.0):
    mat, nodes, links, output = material(name)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat, bsdf


def glass(name="Glass", roughness=0.0, ior=1.5, dispersion=0.0):
    mat, nodes, links, output = material(name)
    bsdf = nodes.new("ShaderNodeBsdfGlass")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    if dispersion and "Dispersion" in bsdf.inputs:
        bsdf.inputs["Dispersion"].default_value = dispersion
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat, bsdf


def add_floor(mat=None, size=8.0):
    bpy.ops.mesh.primitive_plane_add(size=size)
    floor = bpy.context.object
    floor.data.materials.append(mat or principled("Floor", (0.6, 0.6, 0.6, 1))[0])
    return floor


def add_sphere(mat, location=(0.0, 0.0, 0.8), radius=0.8):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=48,
                                         ring_count=24)
    bpy.ops.object.shade_smooth()
    obj = bpy.context.object
    obj.data.materials.append(mat)
    return obj


def add_light(kind, location, energy, **settings):
    bpy.ops.object.light_add(type=kind, location=location)
    light = bpy.context.object
    light.data.energy = energy
    for key, value in settings.items():
        setattr(light.data, key, value)
    return light


def caustic_setup(scene, light_kind="AREA"):
    add_floor()
    add_sphere(glass()[0], (0.6, 0.0, 0.8), 0.7)
    add_sphere(principled("Chrome", (0.9, 0.9, 0.9, 1), 0.05, 1.0)[0], (-1.1, 0.3, 0.6), 0.5)
    if light_kind == "AREA":
        light = add_light("AREA", (1.0, -1.0, 3.5), 400.0, size=0.8)
        light.rotation_euler = (math.radians(20), math.radians(15), 0.0)
    elif light_kind == "POINT":
        add_light("POINT", (0.5, -0.5, 3.0), 400.0, shadow_soft_size=0.0)
    elif light_kind == "SPHERE":
        light = add_light("POINT", (0.5, -0.5, 3.0), 400.0, shadow_soft_size=0.15)
        if hasattr(light.data, "use_soft_falloff"):
            light.data.use_soft_falloff = False
    elif light_kind == "SPOT":
        light = add_light("SPOT", (0.5, -1.0, 3.5), 800.0, spot_size=math.radians(40),
                          shadow_soft_size=0.0)
        light.rotation_euler = (math.radians(15), 0.0, 0.0)
    elif light_kind == "SUN":
        light = add_light("SUN", (0.0, 0.0, 5.0), 3.0, angle=math.radians(1.0))
        light.rotation_euler = (math.radians(25), math.radians(10), 0.0)
    elif light_kind == "WORLD":
        nodes = scene.world.node_tree.nodes
        links = scene.world.node_tree.links
        sky = nodes.new("ShaderNodeTexSky")
        links.new(sky.outputs["Color"], nodes["Background"].inputs["Color"])
        nodes["Background"].inputs["Strength"].default_value = 0.4
    elif light_kind == "MESH":
        mat, nodes, links, output = material("Emitter")
        emission = nodes.new("ShaderNodeEmission")
        emission.inputs["Strength"].default_value = 60.0
        links.new(emission.outputs[0], output.inputs["Surface"])
        bpy.ops.mesh.primitive_plane_add(size=0.6, location=(0.8, -0.6, 3.0))
        bpy.context.object.data.materials.append(mat)


def bdpt(scene, light_paths=16384):
    scene.cycles.use_bidirectional_path_tracing = True
    scene.cycles.bdpt_light_paths = light_paths


def photons(scene, count=16384):
    scene.cycles.use_photon_mapping = True
    scene.cycles.photon_count = count


def make_bdpt_scene(kind, light_tree=True):
    def build(args):
        scene = new_scene(args)
        bdpt(scene)
        scene.cycles.use_light_tree = light_tree
        caustic_setup(scene, kind)
    return build


def make_photon_scene(kind):
    def build(args):
        scene = new_scene(args)
        photons(scene)
        caustic_setup(scene, kind)
    return build


def scene_bdpt_adaptive(args):
    scene = new_scene(args)
    bdpt(scene)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.adaptive_min_samples = 8
    caustic_setup(scene)


def scene_bdpt_tiles(args):
    scene = new_scene(args)
    bdpt(scene)
    scene.cycles.use_auto_tile = True
    scene.cycles.tile_size = 32
    caustic_setup(scene)


def scene_bdpt_update1(args):
    scene = new_scene(args)
    bdpt(scene)
    scene.cycles.bdpt_update_samples = 1
    caustic_setup(scene)


def scene_bdpt_passes(args):
    scene = new_scene(args)
    bdpt(scene)
    view_layer = scene.view_layers[0]
    view_layer.use_pass_diffuse_direct = True
    view_layer.use_pass_diffuse_indirect = True
    view_layer.use_pass_glossy_direct = True
    view_layer.use_pass_glossy_indirect = True
    view_layer.use_pass_transmission_direct = True
    view_layer.use_pass_transmission_indirect = True
    lightgroup = view_layer.lightgroups.add(name="key")
    caustic_setup(scene)
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.lightgroup = lightgroup.name


def scene_bdpt_motion(args):
    scene = new_scene(args)
    bdpt(scene)
    scene.render.use_motion_blur = True
    caustic_setup(scene)
    sphere = [o for o in scene.objects if o.type == "MESH" and o.name.startswith("Sphere")][0]
    scene.frame_set(1)
    sphere.keyframe_insert("location", frame=1)
    sphere.location.x += 0.4
    sphere.keyframe_insert("location", frame=2)
    scene.frame_set(1)


def scene_bdpt_dispersion(args):
    scene = new_scene(args)
    bdpt(scene)
    add_floor()
    add_sphere(glass("Prism", 0.0, 1.5, 0.08)[0], (0.0, 0.0, 0.8), 0.7)
    add_light("POINT", (0.3, -0.4, 3.0), 500.0, shadow_soft_size=0.0)


def scene_bdpt_volume_spot(args):
    scene = new_scene(args)
    bdpt(scene)
    caustic_setup(scene, "SPOT")
    bpy.ops.mesh.primitive_cube_add(size=3.0, location=(0, 0.5, 1.52))
    mat, nodes, links, output = material("Haze")
    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Density"].default_value = 0.15
    links.new(scatter.outputs[0], output.inputs["Volume"])
    bpy.context.object.data.materials.append(mat)


def scene_bdpt_guiding(args):
    scene = new_scene(args)
    bdpt(scene)
    scene.cycles.use_guiding = True
    caustic_setup(scene)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-1.6, 1.2, 0.5))
    bpy.context.object.data.materials.append(principled("Box", (0.3, 0.6, 0.3, 1))[0])


def scene_bdpt_osl(args):
    scene = new_scene(args)
    bdpt(scene)
    scene.cycles.shading_system = True
    caustic_setup(scene)


def scene_bdpt_polarizer(args):
    scene = new_scene(args)
    bdpt(scene)
    caustic_setup(scene)
    mat, bsdf = glass("Polarizer")
    bsdf.inputs["Polarizer"].default_value = True
    bsdf.inputs["Polarizer Angle"].default_value = 0.6
    bpy.ops.mesh.primitive_plane_add(size=1.2, location=(0, -1.8, 1.2),
                                     rotation=(math.radians(80), 0, 0))
    bpy.context.object.data.materials.append(mat)


def scene_photons_volume(args):
    scene = new_scene(args)
    photons(scene)
    caustic_setup(scene, "SPOT")
    bpy.ops.mesh.primitive_cube_add(size=3.0, location=(0, 0.5, 1.52))
    mat, nodes, links, output = material("Haze")
    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Density"].default_value = 0.15
    links.new(scatter.outputs[0], output.inputs["Volume"])
    bpy.context.object.data.materials.append(mat)


def scene_photons_dispersion(args):
    scene = new_scene(args)
    photons(scene)
    add_floor()
    add_sphere(glass("Prism", 0.0, 1.5, 0.08)[0], (0.0, 0.0, 0.8), 0.7)
    add_light("POINT", (0.3, -0.4, 3.0), 500.0, shadow_soft_size=0.0)


def scene_photons_adaptive(args):
    scene = new_scene(args)
    photons(scene)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.adaptive_min_samples = 8
    caustic_setup(scene)


def scene_photons_motion(args):
    scene = new_scene(args)
    photons(scene)
    scene.render.use_motion_blur = True
    caustic_setup(scene)
    sphere = [o for o in scene.objects if o.type == "MESH" and o.name.startswith("Sphere")][0]
    sphere.keyframe_insert("location", frame=1)
    sphere.location.x += 0.4
    sphere.keyframe_insert("location", frame=2)
    scene.frame_set(1)


def scene_photons_lightgroup(args):
    scene = new_scene(args)
    photons(scene)
    view_layer = scene.view_layers[0]
    lightgroup = view_layer.lightgroups.add(name="key")
    view_layer.use_pass_diffuse_indirect = True
    caustic_setup(scene)
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.lightgroup = lightgroup.name


def displaced_plane(name, height_fn, scale=0.4, subdivision=5, size=4.0):
    bpy.ops.mesh.primitive_plane_add(size=size)
    plane = bpy.context.object
    bpy.ops.object.modifier_add(type="SUBSURF")
    plane.modifiers[0].levels = subdivision
    plane.modifiers[0].render_levels = subdivision
    plane.modifiers[0].subdivision_type = "SIMPLE"
    mat, bsdf = principled(name, (0.7, 0.7, 0.75, 1))
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    displacement = nodes.new("ShaderNodeDisplacement")
    displacement.inputs["Scale"].default_value = scale
    links.new(height_fn(nodes), displacement.inputs["Height"])
    links.new(displacement.outputs[0], nodes["Material Output"].inputs["Displacement"])
    mat.displacement_method = "DISPLACEMENT"
    plane.data.materials.append(mat)
    return plane, mat, bsdf


def noise_height(nodes):
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    return noise.outputs["Fac"]


def displacement_lighting():
    light = add_light("AREA", (1.5, -1.5, 3.5), 400.0, size=1.0)
    light.rotation_euler = (math.radians(20), math.radians(20), 0.0)


def scene_pixdisp_basic(args):
    new_scene(args)
    displacement_lighting()
    displaced_plane("Displaced", noise_height)


def scene_pixdisp_lowres(args):
    """Coarse base mesh: displacement detail comes entirely from the per-hit solve."""
    new_scene(args)
    displacement_lighting()
    displaced_plane("Displaced coarse", noise_height, scale=0.3, subdivision=1)


def scene_pixdisp_clamped(args):
    scene = new_scene(args)
    scene.cycles.use_pixel_displacement_resolution_clamp = True
    scene.cycles.pixel_displacement_resolution = 256
    displacement_lighting()
    displaced_plane("Displaced cached", noise_height)


def scene_pixdisp_image(args):
    new_scene(args)
    displacement_lighting()
    image = bpy.data.images.new("Height", 128, 128, float_buffer=True)
    image.generated_type = "COLOR_GRID"

    def image_height(nodes):
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = image
        return tex.outputs["Color"]

    displaced_plane("Displaced image", image_height, scale=0.2)


def scene_pixdisp_instances(args):
    """Instanced displaced objects, which need per-geometry BVHs."""
    new_scene(args)
    displacement_lighting()
    plane, _, _ = displaced_plane("Displaced instanced", noise_height, size=1.5, subdivision=3)
    plane.location = (-1.0, 0.0, 0.0)
    for index, x in enumerate((1.0, 0.0)):
        copy = plane.copy()
        copy.location = (x, 0.6 * index, 0.2 * index)
        copy.rotation_euler = (0.0, 0.0, 0.5 * index)
        bpy.context.collection.objects.link(copy)


def scene_pixdisp_shadows_glass(args):
    """Displaced receiver under glass, shadow rays and caustics hitting displaced surfaces."""
    new_scene(args)
    displacement_lighting()
    displaced_plane("Displaced receiver", noise_height, scale=0.25)
    add_sphere(glass()[0], (0.0, 0.0, 0.9), 0.5)


def scene_pixdisp_osl(args):
    scene = new_scene(args)
    scene.cycles.shading_system = True
    displacement_lighting()
    _, mat, bsdf = displaced_plane("Displaced OSL", noise_height)
    ao = mat.node_tree.nodes.new("ShaderNodeAmbientOcclusion")
    mat.node_tree.links.new(ao.outputs["Color"], bsdf.inputs["Base Color"])


def scene_pixdisp_bdpt(args):
    scene = new_scene(args)
    bdpt(scene)
    displacement_lighting()
    displaced_plane("Displaced BDPT", noise_height, scale=0.25)
    add_sphere(glass()[0], (0.0, 0.0, 0.9), 0.5)


def scene_pixdisp_disabled(args):
    """Pixel displacement off: regular true displacement, the Embree BVH is kept on the CPU."""
    scene = new_scene(args)
    scene.cycles.use_pixel_displacement = False
    displacement_lighting()
    displaced_plane("Displaced mesh", noise_height)


SCENES = {
    "bdpt_area": make_bdpt_scene("AREA"),
    "bdpt_area_notree": make_bdpt_scene("AREA", light_tree=False),
    "bdpt_point": make_bdpt_scene("POINT"),
    "bdpt_sphere": make_bdpt_scene("SPHERE"),
    "bdpt_spot": make_bdpt_scene("SPOT"),
    "bdpt_sun": make_bdpt_scene("SUN"),
    "bdpt_world": make_bdpt_scene("WORLD"),
    "bdpt_mesh": make_bdpt_scene("MESH"),
    "bdpt_mesh_notree": make_bdpt_scene("MESH", light_tree=False),
    "bdpt_adaptive": scene_bdpt_adaptive,
    "bdpt_tiles": scene_bdpt_tiles,
    "bdpt_update1": scene_bdpt_update1,
    "bdpt_passes": scene_bdpt_passes,
    "bdpt_motion": scene_bdpt_motion,
    "bdpt_dispersion": scene_bdpt_dispersion,
    "bdpt_volume_spot": scene_bdpt_volume_spot,
    "bdpt_guiding": scene_bdpt_guiding,
    "bdpt_osl": scene_bdpt_osl,
    "bdpt_polarizer": scene_bdpt_polarizer,
    "photons_area": make_photon_scene("AREA"),
    "photons_point": make_photon_scene("POINT"),
    "photons_spot": make_photon_scene("SPOT"),
    "photons_sun": make_photon_scene("SUN"),
    "photons_world": make_photon_scene("WORLD"),
    "photons_mesh": make_photon_scene("MESH"),
    "photons_volume": scene_photons_volume,
    "photons_dispersion": scene_photons_dispersion,
    "photons_adaptive": scene_photons_adaptive,
    "photons_motion": scene_photons_motion,
    "photons_lightgroup": scene_photons_lightgroup,
    "pixdisp_basic": scene_pixdisp_basic,
    "pixdisp_lowres": scene_pixdisp_lowres,
    "pixdisp_clamped": scene_pixdisp_clamped,
    "pixdisp_image": scene_pixdisp_image,
    "pixdisp_instances": scene_pixdisp_instances,
    "pixdisp_shadows_glass": scene_pixdisp_shadows_glass,
    "pixdisp_osl": scene_pixdisp_osl,
    "pixdisp_bdpt": scene_pixdisp_bdpt,
    "pixdisp_disabled": scene_pixdisp_disabled,
}


def enable_device(args):
    if args.device != "METAL":
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "METAL"


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = list(SCENES) if args.scenes == "all" else args.scenes.split(",")
    report = {}
    for name in names:
        SCENES[name](args)
        enable_device(args)
        scene = bpy.context.scene
        if scene.cycles.shading_system and args.device == "METAL":
            # OSL is CPU (and OptiX) only; keep SVM on Metal for a comparable reference.
            scene.cycles.shading_system = False
        scene.render.filepath = str(args.output / f"{name}.exr")
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        seconds = time.perf_counter() - start
        report[name] = {"render_seconds": seconds}
        print(f"SCENE {name} {json.dumps(report[name])}", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


main()
