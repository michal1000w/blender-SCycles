# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Render a suite of small Cycles feature scenes to EXR for pixel-wise comparison.

Every scene uses fixed seeds, fixed sample counts and no adaptive sampling or denoising, so the
same build renders it identically. Comparing the output of two kernel configurations (for
example CYCLES_METAL_VISIBLE_SHADING=0 against the default) with
cycles_metal_image_compare.py validates that both render the same result.

  blender -b --factory-startup --python tests/python/cycles_metal_feature_scenes.py -- \
      --output DIRECTORY [--device METAL|CPU] [--scenes a,b,c] [--timing]
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
    parser.add_argument("--device", default="METAL", choices=("METAL", "CPU"))
    parser.add_argument("--scenes", default="all",
                        help="Comma separated scene names. 'all' skips diffraction_realistic, whose "
                        "Realistic grating cache takes very long to build")
    parser.add_argument("--resolution", type=int, default=96)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--timing", action="store_true",
                        help="Render every scene twice and report the second render time")
    return parser.parse_args(argv)


def new_scene(args, samples=None):
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
    cycles = scene.cycles
    cycles.device = "GPU" if args.device == "METAL" else "CPU"
    cycles.samples = samples or args.samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.seed = 7
    cycles.max_bounces = 8
    cycles.caustics_reflective = True
    cycles.caustics_refractive = True
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.06, 0.08, 1)
    scene.world = world
    bpy.ops.object.camera_add(location=(0.0, -5.0, 2.0), rotation=(math.radians(72), 0.0, 0.0))
    scene.camera = bpy.context.object
    bpy.ops.object.light_add(type="AREA", location=(1.5, -1.5, 3.5))
    light = bpy.context.object
    light.data.energy = 400.0
    light.data.size = 1.0
    light.rotation_euler = (math.radians(20), math.radians(20), 0.0)
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


def add_floor(mat=None):
    bpy.ops.mesh.primitive_plane_add(size=8.0)
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


def glass(name="Glass", roughness=0.0):
    mat, nodes, links, output = material(name)
    bsdf = nodes.new("ShaderNodeBsdfGlass")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = 1.5
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat, bsdf


def scene_basic(args):
    new_scene(args)
    add_floor()
    add_sphere(principled("Metal", (0.9, 0.6, 0.3, 1), 0.2, 1.0)[0], (-1.0, 0.0, 0.8))
    add_sphere(glass()[0], (1.0, 0.0, 0.8))


def scene_textures(args):
    new_scene(args)
    mat, bsdf = principled("Textured")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    coord = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (3, 3, 3)
    noise = nodes.new("ShaderNodeTexNoise")
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    ramp = nodes.new("ShaderNodeValToRGB")
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    wave = nodes.new("ShaderNodeTexWave")
    math_node = nodes.new("ShaderNodeMath")
    math_node.operation = "MULTIPLY"
    math_node.inputs[1].default_value = 0.8
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.5
    checker = nodes.new("ShaderNodeTexChecker")
    links.new(coord.outputs["Object"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
    links.new(mapping.outputs["Vector"], voronoi.inputs["Vector"])
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], mix.inputs["A"])
    links.new(voronoi.outputs["Color"], mix.inputs["B"])
    links.new(wave.outputs["Fac"], math_node.inputs[0])
    links.new(math_node.outputs[0], mix.inputs["Factor"])
    links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
    links.new(checker.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    add_floor(mat)
    add_sphere(mat)


def scene_image_normal_map(args):
    new_scene(args)
    image = bpy.data.images.new("Generated", 64, 64, float_buffer=True)
    image.generated_type = "COLOR_GRID"
    mat, bsdf = principled("Image")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    normal_map = nodes.new("ShaderNodeNormalMap")
    normal_map.inputs["Strength"].default_value = 0.3
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(tex.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    add_floor(mat)
    bpy.ops.mesh.primitive_cube_add(size=1.2, location=(0, 0, 0.6))
    bpy.context.object.data.materials.append(mat)


def scene_volume(args):
    new_scene(args)
    add_floor()
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0, 0, 1.0))
    mat, nodes, links, output = material("Fog")
    scatter = nodes.new("ShaderNodeVolumePrincipled")
    scatter.inputs["Density"].default_value = 0.6
    scatter.inputs["Color"].default_value = (0.8, 0.5, 0.3, 1)
    links.new(scatter.outputs[0], output.inputs["Volume"])
    bpy.context.object.data.materials.append(mat)


def scene_subsurface(args):
    new_scene(args)
    add_floor()
    mat, bsdf = principled("Skin", (0.8, 0.4, 0.3, 1), 0.4)
    bsdf.inputs["Subsurface Weight"].default_value = 1.0
    bsdf.inputs["Subsurface Scale"].default_value = 0.2
    add_sphere(mat)


def scene_hair(args):
    new_scene(args)
    add_floor()
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.6, location=(0, 0, 0.8))
    emitter = bpy.context.object
    emitter.data.materials.append(principled("Scalp", (0.3, 0.2, 0.1, 1))[0])
    mat, nodes, links, output = material("Hair")
    hair = nodes.new("ShaderNodeBsdfHairPrincipled")
    links.new(hair.outputs[0], output.inputs["Surface"])
    emitter.data.materials.append(mat)
    modifier = emitter.modifiers.new("Hair", "PARTICLE_SYSTEM")
    settings = modifier.particle_system.settings
    settings.type = "HAIR"
    settings.count = 300
    settings.hair_length = 0.4
    settings.material = 2
    settings.use_advanced_hair = False


def scene_raytrace_nodes(args):
    new_scene(args)
    mat, bsdf = principled("AO Bevel")
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    bevel = nodes.new("ShaderNodeBevel")
    bevel.inputs["Radius"].default_value = 0.1
    links.new(ao.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bevel.outputs["Normal"], bsdf.inputs["Normal"])
    add_floor()
    bpy.ops.mesh.primitive_cube_add(size=1.2, location=(0, 0, 0.6))
    bpy.context.object.data.materials.append(mat)


def scene_passes(args):
    scene = new_scene(args)
    view_layer = scene.view_layers[0]
    view_layer.use_pass_normal = True
    view_layer.use_pass_diffuse_color = True
    view_layer.use_pass_glossy_direct = True
    view_layer.use_pass_emit = True
    aov = view_layer.aovs.add()
    aov.name = "custom"
    aov.type = "COLOR"
    mat, bsdf = principled("AOV", (0.3, 0.7, 0.2, 1), 0.3)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    aov_node = nodes.new("ShaderNodeOutputAOV")
    aov_node.aov_name = "custom"
    noise = nodes.new("ShaderNodeTexNoise")
    links.new(noise.outputs["Color"], aov_node.inputs["Color"])
    add_floor()
    add_sphere(mat)
    scene.cycles.use_denoising = False
    scene.view_layers[0].cycles.denoising_store_passes = True


def scene_displacement(args):
    new_scene(args)
    bpy.ops.mesh.primitive_plane_add(size=4.0)
    plane = bpy.context.object
    bpy.ops.object.modifier_add(type="SUBSURF")
    plane.modifiers[0].levels = 5
    plane.modifiers[0].render_levels = 5
    plane.modifiers[0].subdivision_type = "SIMPLE"
    mat, bsdf = principled("Displaced", (0.7, 0.7, 0.75, 1))
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 3.0
    displacement = nodes.new("ShaderNodeDisplacement")
    displacement.inputs["Scale"].default_value = 0.4
    links.new(noise.outputs["Fac"], displacement.inputs["Height"])
    links.new(displacement.outputs[0], nodes["Material Output"].inputs["Displacement"])
    mat.displacement_method = "DISPLACEMENT"
    plane.data.materials.append(mat)


def scene_world_sky(args):
    scene = new_scene(args)
    nodes = scene.world.node_tree.nodes
    links = scene.world.node_tree.links
    sky = nodes.new("ShaderNodeTexSky")
    links.new(sky.outputs["Color"], nodes["Background"].inputs["Color"])
    nodes["Background"].inputs["Strength"].default_value = 0.3
    scene.world.cycles.sampling_method = "MANUAL"
    scene.world.cycles.sample_map_resolution = 256
    add_floor()
    add_sphere(principled("Chrome", (0.9, 0.9, 0.9, 1), 0.05, 1.0)[0])


def scene_mnee(args):
    new_scene(args)
    floor = add_floor()
    floor.cycles.is_caustics_receiver = True
    sphere = add_sphere(glass("Caustic Glass")[0])
    sphere.cycles.is_caustics_caster = True
    bpy.ops.object.light_add(type="POINT", location=(0, 0, 3.0))
    light = bpy.context.object
    light.data.energy = 300.0
    light.data.cycles.is_caustics_light = True


def scene_motion_blur(args):
    scene = new_scene(args)
    scene.render.use_motion_blur = True
    add_floor()
    sphere = add_sphere(principled("Moving", (0.2, 0.4, 0.9, 1))[0])
    scene.frame_set(1)
    sphere.keyframe_insert("location", frame=1)
    sphere.location.x += 1.0
    sphere.keyframe_insert("location", frame=2)
    scene.frame_set(1)


def scene_shadow_catcher(args):
    scene = new_scene(args)
    scene.render.film_transparent = True
    floor = add_floor()
    floor.is_shadow_catcher = True
    add_sphere(principled("Object", (0.8, 0.2, 0.2, 1))[0])


def scene_bdpt(args):
    scene = new_scene(args)
    scene.cycles.use_bidirectional_path_tracing = True
    scene.cycles.bdpt_light_paths = 16384
    add_floor()
    add_sphere(glass()[0], (0.8, 0.0, 0.8))
    add_sphere(principled("Metal", (0.9, 0.9, 0.9, 1), 0.1, 1.0)[0], (-1.0, 0.0, 0.8), 0.6)


def scene_bdpt_volume(args):
    scene_bdpt(args)
    bpy.ops.mesh.primitive_cube_add(size=1.5, location=(0, 1.2, 0.75))
    mat, nodes, links, output = material("Smoke")
    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Density"].default_value = 0.4
    links.new(scatter.outputs[0], output.inputs["Volume"])
    bpy.context.object.data.materials.append(mat)


def scene_photons(args):
    scene = new_scene(args)
    scene.cycles.use_photon_mapping = True
    scene.cycles.photon_count = 16384
    add_floor()
    add_sphere(glass()[0])


def scene_guiding(args):
    scene = new_scene(args)
    scene.cycles.use_guiding = True
    add_floor()
    add_sphere(principled("Guided", (0.8, 0.8, 0.8, 1), 0.3)[0])
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(-1.5, 0.5, 0.5))
    bpy.context.object.data.materials.append(principled("Box", (0.3, 0.6, 0.3, 1))[0])


def scene_diffraction(args):
    new_scene(args)
    add_floor()
    mat, nodes, links, output = material("Grating")
    bsdf = nodes.new("ShaderNodeBsdfAnisotropic")
    bsdf.inputs["Color"].default_value = (0.8, 0.8, 0.8, 1)
    for key, value in {"Roughness": 0.22, "Diffraction Weight": 1.0,
                       "Diffraction Pitch": 1200, "Diffraction Depth": 320,
                       "Diffraction Duty Cycle": 0.41}.items():
        bsdf.inputs[key].default_value = value
    tangent = nodes.new("ShaderNodeCombineXYZ")
    tangent.inputs["X"].default_value = 1
    links.new(tangent.outputs[0], bsdf.inputs["Tangent"])
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    add_sphere(mat)


def scene_diffraction_realistic(args):
    new_scene(args)
    add_floor()
    mat, nodes, links, output = material("Realistic grating")
    bsdf = nodes.new("ShaderNodeBsdfDiffraction")
    bsdf.quality = "REALISTIC"
    for name, value in dict(pitch=740, depth=150, duty_cycle=.41, incident_ior=1,
                            ridge_ior=.9, ridge_extinction=6, groove_ior=1, substrate_ior=.9,
                            substrate_extinction=6).items():
        setattr(bsdf, name, value)
    tangent = nodes.new("ShaderNodeCombineXYZ")
    tangent.inputs["X"].default_value = 1
    links.new(tangent.outputs[0], bsdf.inputs["Tangent"])
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=1.5, location=(0, 0, 0.5),
                                     rotation=(math.radians(60), 0, 0))
    bpy.context.object.data.materials.append(mat)


def scene_polarizer(args):
    new_scene(args)
    add_floor()
    mat, bsdf = glass("Polarizer")
    bsdf.inputs["Polarizer"].default_value = True
    bsdf.inputs["Polarizer Angle"].default_value = 0.6
    bpy.ops.mesh.primitive_plane_add(size=1.5, location=(0, -1.0, 1.0),
                                     rotation=(math.radians(80), 0, 0))
    bpy.context.object.data.materials.append(mat)
    add_sphere(principled("Behind", (0.8, 0.8, 0.2, 1), 0.1)[0], (0, 1.0, 0.8))


SCENES = {
    "basic": scene_basic,
    "textures": scene_textures,
    "image_normal_map": scene_image_normal_map,
    "volume": scene_volume,
    "subsurface": scene_subsurface,
    "hair": scene_hair,
    "raytrace_nodes": scene_raytrace_nodes,
    "passes": scene_passes,
    "displacement": scene_displacement,
    "world_sky": scene_world_sky,
    "mnee": scene_mnee,
    "motion_blur": scene_motion_blur,
    "shadow_catcher": scene_shadow_catcher,
    "bdpt": scene_bdpt,
    "bdpt_volume": scene_bdpt_volume,
    "photons": scene_photons,
    "guiding": scene_guiding,
    "diffraction": scene_diffraction,
    "polarizer": scene_polarizer,
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
    scenes = dict(SCENES, diffraction_realistic=scene_diffraction_realistic)
    names = list(SCENES) if args.scenes == "all" else args.scenes.split(",")
    report = {}
    for name in names:
        scenes[name](args)
        enable_device(args)
        scene = bpy.context.scene
        scene.render.filepath = str(args.output / f"{name}.exr")
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        seconds = time.perf_counter() - start
        entry = {"first_render_seconds": seconds}
        if args.timing:
            start = time.perf_counter()
            bpy.ops.render.render(write_still=False)
            entry["render_seconds"] = time.perf_counter() - start
        report[name] = entry
        print(f"SCENE {name} {json.dumps(entry)}", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


main()
