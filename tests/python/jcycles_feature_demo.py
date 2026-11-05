# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""
Demo scenes for the custom Cycles features of this fork, rendered on every device, with a
device parity check and a report.

  blender --background --factory-startup --python tests/python/jcycles_feature_demo.py -- \
      --output DIRECTORY [--devices CPU,CUDA,OPTIX] [--scenes a,b] [--samples N] \
      [--resolution N] [--no-blend] [--denoise]

`--denoise` renders with denoising and adaptive sampling on, as production scenes do (the
denoiser runs on the rendering device, which exercises its kernel loading).

For every scene it:
  * saves the scene as DIRECTORY/blend/<scene>.blend, to open and inspect in Blender,
  * renders it on the CPU twice with different seeds, and on every requested GPU backend that
    has a device, to DIRECTORY/<device>/<scene>.exr and .png,
  * compares each GPU render with the CPU render of the same seed. The second CPU seed measures
    the Monte Carlo noise of the scene, so the tolerance follows the noise: a GPU render passes
    when it differs from the CPU no more than another CPU render does (with a margin), and its
    mean brightness matches within the noise of the mean.

Results go to DIRECTORY/report.json and DIRECTORY/report.html. The exit status is non-zero
when a render fails or a GPU render does not match the CPU.
"""

import argparse
import html
import json
import math
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np


# --------------------------------------------------------------------------------------------
# Arguments


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--devices", default="CPU,CUDA,OPTIX",
                        help="Comma separated: CPU, CUDA, OPTIX, HIP, ONEAPI, METAL")
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=160,
                        help="Image width; height is 3/4 of it")
    parser.add_argument("--no-blend", action="store_true", help="Do not save .blend files")
    parser.add_argument("--list", action="store_true", help="List the scenes and exit")
    parser.add_argument("--denoise", action="store_true",
                        help="Render with denoising and adaptive sampling, as production scenes do")
    return parser.parse_args(argv)


# --------------------------------------------------------------------------------------------
# Scene helpers


def new_scene(args, name):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.name = name
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution * 3 // 4
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"
    cycles = scene.cycles
    cycles.samples = args.samples
    cycles.use_adaptive_sampling = args.denoise
    cycles.use_denoising = args.denoise
    cycles.seed = 1
    cycles.max_bounces = 12
    cycles.caustics_reflective = True
    cycles.caustics_refractive = True
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.06, 0.08, 1)
    scene.world = world
    bpy.ops.object.camera_add(location=(0.0, -5.5, 2.4), rotation=(math.radians(70), 0.0, 0.0))
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


def glass(name="Glass", roughness=0.0, ior=1.5, color=(1, 1, 1, 1)):
    mat, nodes, links, output = material(name)
    bsdf = nodes.new("ShaderNodeBsdfGlass")
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    bsdf.inputs["Color"].default_value = color
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat, bsdf


def add_floor(mat=None, size=10.0):
    bpy.ops.mesh.primitive_plane_add(size=size)
    floor = bpy.context.object
    floor.name = "Floor"
    floor.data.materials.append(mat or principled("Floor", (0.6, 0.6, 0.6, 1))[0])
    return floor


def add_sphere(mat, location=(0.0, 0.0, 0.8), radius=0.8, name="Sphere"):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=location, segments=64,
                                         ring_count=32)
    bpy.ops.object.shade_smooth()
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def add_cube(mat, location, size, name="Cube"):
    bpy.ops.mesh.primitive_cube_add(size=size, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


def add_light(kind, location, energy, rotation=(0.0, 0.0, 0.0), **settings):
    bpy.ops.object.light_add(type=kind, location=location, rotation=rotation)
    light = bpy.context.object
    light.data.energy = energy
    for key, value in settings.items():
        setattr(light.data, key, value)
    return light


def area_light(energy=400.0, size=1.0):
    return add_light("AREA", (1.5, -1.5, 3.5), energy,
                     rotation=(math.radians(20), math.radians(20), 0.0), size=size)


def caustic_spot(energy=1500.0):
    return add_light("SPOT", (0.0, 0.0, 4.0), energy, spot_size=math.radians(35),
                     shadow_soft_size=0.05)


# --------------------------------------------------------------------------------------------
# Scenes. Each returns the scene and a short description.


def scene_path_tracing(args):
    """Reference: standard path tracing, nothing from this fork."""
    scene = new_scene(args, "path_tracing")
    area_light()
    add_floor()
    add_sphere(principled("Plastic", (0.8, 0.2, 0.15, 1), 0.3)[0], (-1.0, 0.0, 0.6), 0.6)
    add_sphere(glass()[0], (0.8, 0.3, 0.7), 0.7)
    add_cube(principled("Metal", (0.9, 0.8, 0.6, 1), 0.2, 1.0)[0], (0.0, 1.5, 0.5), 1.0)
    return scene, "Standard path tracing (no fork features), the baseline"


def caustic_scene(args, name):
    scene = new_scene(args, name)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.2
    caustic_spot()
    add_floor()
    add_sphere(glass("Caustic glass")[0], (0.0, 0.0, 0.9), 0.8)
    add_sphere(principled("Mirror", (0.95, 0.95, 0.95, 1), 0.02, 1.0)[0], (-1.6, 0.6, 0.5), 0.5)
    return scene


def scene_bdpt(args):
    scene = caustic_scene(args, "bdpt")
    scene.cycles.use_bidirectional_path_tracing = True
    scene.cycles.bdpt_light_paths = 65536
    scene.cycles.bdpt_max_bounces = 10
    return scene, "Bidirectional path tracing: glass sphere caustic from a spot light"


def scene_vcm(args):
    scene = caustic_scene(args, "vcm")
    scene.cycles.use_bidirectional_path_tracing = True
    scene.cycles.use_vertex_merging = True
    scene.cycles.bdpt_light_paths = 65536
    return scene, "Vertex connection and merging (BDPT + vertex merging) caustics"


def scene_photon_mapping(args):
    scene = caustic_scene(args, "photon_mapping")
    scene.cycles.use_photon_mapping = True
    scene.cycles.photon_count = 131072
    return scene, "Photon mapping caustics"


def scene_photon_volume(args):
    scene = new_scene(args, "photon_volume")
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.1
    caustic_spot(2500.0)
    add_floor()
    mat, nodes, links, output = material("Haze")
    volume = nodes.new("ShaderNodeVolumePrincipled")
    volume.inputs["Density"].default_value = 0.15
    links.new(volume.outputs[0], output.inputs["Volume"])
    # 2 mm above the floor, see scene_nested_dielectrics().
    add_cube(mat, (0.0, 0.0, 1.502), 3.0, "Haze")
    add_sphere(glass()[0], (0.0, 0.0, 2.2), 0.5)
    scene.cycles.use_photon_mapping = True
    scene.cycles.photon_count = 131072
    return scene, "Photon mapping through a scattering volume (volumetric caustic)"


def scene_bdpt_volume(args):
    scene, _ = scene_bdpt(args)
    scene.name = "bdpt_volume"
    mat, nodes, links, output = material("Smoke")
    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Density"].default_value = 0.25
    links.new(scatter.outputs[0], output.inputs["Volume"])
    # 2 mm above the floor, see scene_nested_dielectrics().
    add_cube(mat, (1.2, 1.2, 0.752), 1.5, "Smoke")
    return scene, "BDPT with a scattering volume"


def scene_mnee(args):
    scene = new_scene(args, "mnee")
    floor = add_floor()
    floor.cycles.is_caustics_receiver = True
    sphere = add_sphere(glass("MNEE glass")[0], (0.0, 0.0, 0.9), 0.8)
    sphere.cycles.is_caustics_caster = True
    light = add_light("POINT", (0.0, 0.0, 3.5), 400.0)
    light.data.cycles.is_caustics_light = True
    return scene, "Manifold next event estimation (shadow caustics)"


def displaced_plane(subdivision=5, scale=0.35):
    bpy.ops.mesh.primitive_plane_add(size=4.0)
    plane = bpy.context.object
    plane.name = "Displaced"
    modifier = plane.modifiers.new("Subdivision", "SUBSURF")
    modifier.levels = subdivision
    modifier.render_levels = subdivision
    modifier.subdivision_type = "SIMPLE"
    mat, bsdf = principled("Displaced", (0.7, 0.7, 0.75, 1), 0.4)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 4.0
    noise.inputs["Detail"].default_value = 6.0
    displacement = nodes.new("ShaderNodeDisplacement")
    displacement.inputs["Scale"].default_value = scale
    links.new(noise.outputs["Fac"], displacement.inputs["Height"])
    links.new(displacement.outputs[0], mat.node_tree.nodes["Material Output"].inputs["Displacement"])
    mat.displacement_method = "DISPLACEMENT"
    plane.data.materials.append(mat)
    return plane


def scene_pixel_displacement(args):
    scene = new_scene(args, "pixel_displacement")
    area_light()
    displaced_plane(subdivision=3)
    scene.cycles.use_pixel_displacement = True
    scene.cycles.pixel_displacement_max_distance = 0.4
    add_sphere(principled("Ball", (0.2, 0.4, 0.9, 1), 0.2)[0], (0.6, 0.3, 0.7), 0.5)
    return scene, "Pixel-level displacement of a coarse mesh (detail solved per ray hit)"


def scene_pixel_displacement_glass(args):
    scene, _ = scene_pixel_displacement(args)
    scene.name = "pixel_displacement_glass"
    add_sphere(glass("Glass over displacement")[0], (-0.9, -0.4, 0.7), 0.5, "Glass ball")
    scene.cycles.use_bidirectional_path_tracing = True
    scene.cycles.bdpt_light_paths = 32768
    return scene, "Pixel displacement with glass and BDPT (shadows, refraction, light paths)"


def grating_material(name="Grating"):
    mat, nodes, links, output = material(name)
    bsdf = nodes.new("ShaderNodeBsdfAnisotropic")
    bsdf.inputs["Color"].default_value = (0.85, 0.85, 0.85, 1)
    for key, value in {"Roughness": 0.15, "Diffraction Weight": 1.0,
                       "Diffraction Pitch": 1200.0, "Diffraction Depth": 320.0,
                       "Diffraction Duty Cycle": 0.41}.items():
        bsdf.inputs[key].default_value = value
    tangent = nodes.new("ShaderNodeCombineXYZ")
    tangent.inputs["X"].default_value = 1.0
    links.new(tangent.outputs[0], bsdf.inputs["Tangent"])
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat


def scene_diffraction_grating(args):
    scene = new_scene(args, "diffraction_grating")
    add_light("POINT", (1.5, -2.0, 3.0), 1500.0, shadow_soft_size=0.02)
    add_floor(principled("Dark floor", (0.1, 0.1, 0.1, 1))[0])
    add_sphere(grating_material(), (0.0, 0.0, 0.9), 0.85)
    return scene, "Diffraction grating on a glossy BSDF (spectral colors)"


def scene_diffraction_node(args):
    scene = new_scene(args, "diffraction_node")
    add_light("POINT", (1.5, -2.0, 3.0), 1500.0, shadow_soft_size=0.02)
    add_floor(principled("Dark floor", (0.1, 0.1, 0.1, 1))[0])
    mat, nodes, links, output = material("Diffraction BSDF")
    bsdf = nodes.new("ShaderNodeBsdfDiffraction")
    tangent = nodes.new("ShaderNodeCombineXYZ")
    tangent.inputs["X"].default_value = 1.0
    links.new(tangent.outputs[0], bsdf.inputs["Tangent"])
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0.9),
                                     rotation=(math.radians(60), 0, 0))
    bpy.context.object.data.materials.append(mat)
    return scene, "Diffraction BSDF node (default Fast quality)"


def scene_glass_diffraction(args):
    scene = new_scene(args, "glass_diffraction")
    area_light(600.0)
    add_floor()
    mat, bsdf = glass("Diffracting glass", 0.05)
    bsdf.inputs["Diffraction Weight"].default_value = 1.0
    bsdf.inputs["Diffraction Pitch"].default_value = 1600.0
    add_sphere(mat, (0.0, 0.0, 0.9), 0.85)
    return scene, "Glass BSDF with a transmission diffraction grating"


def scene_polarizer(args):
    scene = new_scene(args, "polarizer")
    area_light(600.0)
    add_floor()
    for i, angle in enumerate((0.0, math.radians(45), math.radians(90))):
        mat, bsdf = glass(f"Polarizer {i}")
        bsdf.inputs["Polarizer"].default_value = True
        bsdf.inputs["Polarizer Angle"].default_value = angle
        bpy.ops.mesh.primitive_plane_add(size=1.2, location=((i - 1) * 1.3, -1.0, 1.0),
                                         rotation=(math.radians(80), 0, 0))
        bpy.context.object.data.materials.append(mat)
    add_sphere(principled("Behind", (0.9, 0.85, 0.2, 1), 0.1)[0], (0, 1.0, 0.8))
    scene.cycles.use_polarization = True
    return scene, "Glass polarizers (0, 45 and 90 degree analyzers)"


def scene_coherent_light(args):
    scene = new_scene(args, "coherent_light")
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    add_floor(principled("Screen", (0.9, 0.9, 0.9, 1), 1.0)[0])
    for i, x in enumerate((-0.002, 0.002)):
        light = add_light("POINT", (x, 0.0, 2.0), 40.0, shadow_soft_size=0.0)
        light.name = f"Coherent source {i}"
        light.data.cycles.coherence_group = 1
        light.data.cycles.coherence_wavelength_nm = 550.0
        light.data.cycles.coherence_length_m = 0.01
    scene.camera.location = (0.0, -2.5, 3.0)
    scene.camera.rotation_euler = (math.radians(40), 0.0, 0.0)
    return scene, "Coherent point sources: two-source interference fringes on a screen"


def scene_nested_dielectrics(args):
    scene = new_scene(args, "nested_dielectrics")
    area_light(600.0)
    add_floor()
    glass_mat, _ = glass("Glass", 0.0, 1.5)
    glass_mat.cycles.nested_priority = 2
    water_mat, _ = glass("Water", 0.0, 1.33, (0.85, 0.92, 1.0, 1))
    water_mat.cycles.nested_priority = 1
    # The glass stands 2 mm above the floor. Standing on it, its bottom face and the floor are
    # one plane: which of them a ray meets first is decided by rounding and differs between
    # devices, and a path that meets the floor first is still inside the glass.
    bpy.ops.mesh.primitive_cylinder_add(radius=0.6, depth=1.6, location=(0, 0, 0.802))
    bpy.ops.object.shade_smooth()
    bpy.context.object.name = "Glass"
    bpy.context.object.data.materials.append(glass_mat)
    # The liquid lies inside the glass; the priorities resolve the overlap of their volumes.
    bpy.ops.mesh.primitive_cylinder_add(radius=0.57, depth=1.0, location=(0, 0, 0.552))
    bpy.ops.object.shade_smooth()
    bpy.context.object.name = "Water"
    bpy.context.object.data.materials.append(water_mat)
    add_sphere(principled("Behind", (0.2, 0.7, 0.3, 1))[0], (0.3, 1.5, 0.5), 0.5)
    return scene, "Nested dielectrics: water inside a glass with overlapping geometry"


def scene_volume_ior(args):
    scene = new_scene(args, "volume_ior")
    area_light(500.0)
    add_floor()
    mat, nodes, links, output = material("Gradient IOR")
    gradient = nodes.new("ShaderNodeTexGradient")
    mapping = nodes.new("ShaderNodeMath")
    mapping.operation = "MULTIPLY_ADD"
    mapping.inputs[1].default_value = 0.15
    mapping.inputs[2].default_value = 1.0
    ior = nodes.new("ShaderNodeVolumeIOR")
    # The node defaults to the temperature model, whose sockets hide the IOR input.
    ior.mode = "IOR"
    links.new(gradient.outputs["Fac"], mapping.inputs[0])
    links.new(mapping.outputs[0], ior.inputs["IOR"])
    links.new(ior.outputs[0], output.inputs["Volume"])
    add_cube(mat, (0.0, -0.5, 1.0), 1.6, "IOR volume")
    checker_mat, nodes, links, output = material("Checker")
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 12.0
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    links.new(checker.outputs["Color"], bsdf.inputs["Color"])
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 1.5, 1.5), rotation=(math.radians(90), 0, 0))
    bpy.context.object.data.materials.append(checker_mat)
    scene.cycles.use_volume_ior = True
    return scene, "Volume IOR: light bending through a graded-index volume"


def scene_volume_fast(args):
    scene = new_scene(args, "volume_fast")
    area_light(500.0)
    add_floor()
    mat, nodes, links, output = material("Fast volume")
    volume = nodes.new("ShaderNodeVolumeFast")
    volume.inputs["Density"].default_value = 1.5
    volume.inputs["Scatter Color"].default_value = (0.8, 0.5, 0.3, 1)
    links.new(volume.outputs[0], output.inputs["Volume"])
    add_sphere(mat, (0.0, 0.0, 1.0), 0.9, "Fast volume")
    return scene, "Volume Fast node"


def scene_path_guiding(args):
    scene = new_scene(args, "path_guiding")
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    add_floor()
    wall_mat = principled("Walls", (0.8, 0.8, 0.8, 1), 0.8)[0]
    # The light hides behind a panel and only lights the back wall: all visible light is
    # indirect, the case guiding is for.
    bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0, 3.0, 3.0), rotation=(math.radians(90), 0, 0))
    bpy.context.object.data.materials.append(wall_mat)
    bpy.ops.mesh.primitive_plane_add(size=1.6, location=(0, 1.6, 0.8), rotation=(math.radians(90), 0, 0))
    bpy.context.object.data.materials.append(wall_mat)
    # Area lights emit along local -Z, rotated here to face the back wall (+Y).
    add_light("AREA", (0.0, 1.8, 0.8), 600.0, rotation=(math.radians(90), 0, 0), size=0.5)
    add_sphere(principled("Guided", (0.8, 0.8, 0.8, 1), 0.3)[0], (-1.0, 0.0, 0.6), 0.6)
    add_sphere(principled("Glossy", (0.9, 0.9, 0.9, 1), 0.15, 1.0)[0], (1.0, 0.3, 0.6), 0.6)
    scene.cycles.use_guiding = True
    # The CPU guides with OpenPGL, GPUs with their own guiding: both converge to the same image,
    # but with only a few samples each is at a different stage of learning the light.
    scene.cycles.samples = max(args.samples, 1024)
    return scene, "Path guiding with indirect light only (OpenPGL on the CPU)"


# --------------------------------------------------------------------------------------------
# Combinations: several features in one scene, as production scenes use them.


def enable_guiding(scene):
    """Guiding with the indirect sample clamp off. Clamping removes a different amount of energy
    from each sampling strategy, and the CPU (OpenPGL) and the GPUs (GPU guiding field) guide
    differently, so clamped guided renders do not converge to the same image."""
    scene.cycles.use_guiding = True
    scene.cycles.sample_clamp_indirect = 0.0


def scene_bdpt_guiding(args):
    scene, _ = scene_bdpt(args)
    scene.name = "bdpt_guiding"
    enable_guiding(scene)
    # GPU guiding and OpenPGL learn differently; give both enough samples to converge.
    scene.cycles.samples = max(args.samples, 256)
    return scene, "BDPT with path guiding (GPU guiding field on CUDA/OptiX, OpenPGL on the CPU)"


def scene_bdpt_volume_guiding(args):
    scene, _ = scene_bdpt_volume(args)
    scene.name = "bdpt_volume_guiding"
    enable_guiding(scene)
    scene.cycles.use_volume_guiding = True
    scene.cycles.samples = max(args.samples, 256)
    return scene, "BDPT, scattering volume and surface + volume guiding"


def scene_photon_guiding(args):
    scene, _ = scene_photon_mapping(args)
    scene.name = "photon_guiding"
    enable_guiding(scene)
    scene.cycles.samples = max(args.samples, 256)
    return scene, "Photon mapping with path guiding"


def scene_vcm_volume(args):
    scene, _ = scene_bdpt_volume(args)
    scene.name = "vcm_volume"
    scene.cycles.use_vertex_merging = True
    return scene, "VCM (BDPT + vertex merging) with a scattering volume"


def scene_prism(args):
    """Close to the attached prism scenes: a spot light through a glass prism onto a floor
    inside a scattering volume, with BDPT, guiding, polarization and diffraction enabled."""
    scene = new_scene(args, "prism")
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
    add_floor(principled("Screen", (0.8, 0.8, 0.8, 1), 0.9)[0])
    mat, bsdf = principled("Prism glass", (1, 1, 1, 1), 0.0)
    bsdf.inputs["Transmission Weight"].default_value = 1.0
    bsdf.inputs["IOR"].default_value = 1.6
    bpy.ops.mesh.primitive_cylinder_add(vertices=3, radius=0.6, depth=1.2, location=(0, 0, 1.1),
                                        rotation=(math.radians(90), 0, 0))
    bpy.context.object.name = "Prism"
    bpy.context.object.data.materials.append(mat)
    haze, nodes, links, output = material("Haze")
    scatter = nodes.new("ShaderNodeVolumeScatter")
    scatter.inputs["Density"].default_value = 0.05
    links.new(scatter.outputs[0], output.inputs["Volume"])
    add_cube(haze, (0.0, 0.0, 2.002), 4.0, "Haze")
    add_light("SPOT", (-1.5, 0.0, 2.6), 3000.0, rotation=(0.0, math.radians(-60), 0.0),
              spot_size=math.radians(8), shadow_soft_size=0.01)
    cycles = scene.cycles
    cycles.use_bidirectional_path_tracing = True
    cycles.bdpt_light_paths = 65536
    enable_guiding(scene)
    cycles.use_polarization = True
    cycles.use_diffraction_effects = True
    cycles.use_coherent_interference = True
    cycles.samples = max(args.samples, 256)
    return scene, "Prism caustic in haze: BDPT, guiding, polarization, diffraction flags"


def scene_nested_volume_ior_bdpt(args):
    scene, _ = scene_nested_dielectrics(args)
    scene.name = "nested_volume_ior_bdpt"
    mat, nodes, links, output = material("Graded volume")
    gradient = nodes.new("ShaderNodeTexGradient")
    mapping = nodes.new("ShaderNodeMath")
    mapping.operation = "MULTIPLY_ADD"
    mapping.inputs[1].default_value = 0.1
    mapping.inputs[2].default_value = 1.0
    ior = nodes.new("ShaderNodeVolumeIOR")
    ior.mode = "IOR"
    links.new(gradient.outputs["Fac"], mapping.inputs[0])
    links.new(mapping.outputs[0], ior.inputs["IOR"])
    links.new(ior.outputs[0], output.inputs["Volume"])
    add_cube(mat, (-1.4, -0.4, 0.802), 1.0, "IOR volume")
    scene.cycles.use_volume_ior = True
    scene.cycles.use_bidirectional_path_tracing = True
    scene.cycles.bdpt_light_paths = 32768
    return scene, "Nested dielectrics, a graded-index volume and BDPT together"


def scene_displacement_guiding(args):
    scene, _ = scene_pixel_displacement(args)
    scene.name = "displacement_guiding"
    enable_guiding(scene)
    scene.cycles.samples = max(args.samples, 256)
    return scene, "Pixel displacement with path guiding (CUDA kernels on OptiX)"


def scene_grating_polarizer_bdpt(args):
    scene, _ = scene_diffraction_grating(args)
    scene.name = "grating_polarizer_bdpt"
    mat, bsdf = glass("Analyzer")
    bsdf.inputs["Polarizer"].default_value = True
    bsdf.inputs["Polarizer Angle"].default_value = math.radians(30)
    bpy.ops.mesh.primitive_plane_add(size=1.5, location=(0.0, -1.6, 1.0),
                                     rotation=(math.radians(80), 0, 0))
    bpy.context.object.data.materials.append(mat)
    scene.cycles.use_polarization = True
    scene.cycles.use_bidirectional_path_tracing = True
    scene.cycles.bdpt_light_paths = 32768
    return scene, "Diffraction grating seen through a polarizer, with BDPT"


def scene_mnee_guiding(args):
    scene, _ = scene_mnee(args)
    scene.name = "mnee_guiding"
    enable_guiding(scene)
    scene.cycles.samples = max(args.samples, 256)
    return scene, "MNEE shadow caustics with path guiding"


SCENES = {
    "path_tracing": scene_path_tracing,
    "bdpt": scene_bdpt,
    "bdpt_volume": scene_bdpt_volume,
    "vcm": scene_vcm,
    "photon_mapping": scene_photon_mapping,
    "photon_volume": scene_photon_volume,
    "mnee": scene_mnee,
    "pixel_displacement": scene_pixel_displacement,
    "pixel_displacement_glass": scene_pixel_displacement_glass,
    "diffraction_grating": scene_diffraction_grating,
    "diffraction_node": scene_diffraction_node,
    "glass_diffraction": scene_glass_diffraction,
    "polarizer": scene_polarizer,
    "coherent_light": scene_coherent_light,
    "nested_dielectrics": scene_nested_dielectrics,
    "volume_ior": scene_volume_ior,
    "volume_fast": scene_volume_fast,
    "path_guiding": scene_path_guiding,
    "bdpt_guiding": scene_bdpt_guiding,
    "bdpt_volume_guiding": scene_bdpt_volume_guiding,
    "photon_guiding": scene_photon_guiding,
    "vcm_volume": scene_vcm_volume,
    "prism": scene_prism,
    "nested_volume_ior_bdpt": scene_nested_volume_ior_bdpt,
    "displacement_guiding": scene_displacement_guiding,
    "grating_polarizer_bdpt": scene_grating_polarizer_bdpt,
    "mnee_guiding": scene_mnee_guiding,
}


# --------------------------------------------------------------------------------------------
# Devices


def available_backends(requested):
    import _cycles
    found = []
    for backend in requested:
        if backend == "CPU":
            found.append(backend)
            continue
        try:
            devices = [d for d in _cycles.available_devices(backend) if d[1] == backend]
        except Exception:
            devices = []
        if devices:
            found.append(backend)
        else:
            print(f"[jcycles-demo] no {backend} device, skipping it", flush=True)
    return found


def use_backend(scene, backend):
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if backend == "CPU":
        scene.cycles.device = "CPU"
        return
    prefs.compute_device_type = backend
    prefs.refresh_devices()
    for device in prefs.devices:
        device.use = device.type == backend
    scene.cycles.device = "GPU"


# --------------------------------------------------------------------------------------------
# Rendering and comparison


def render(scene, path_exr, path_png):
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.filepath = str(path_exr)
    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    seconds = time.perf_counter() - start
    image = bpy.data.images.load(str(path_exr))
    width, height = image.size
    pixels = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    image.filepath_raw = str(path_png)
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    return pixels.reshape(height, width, 4)[:, :, :3], seconds


def blur(rgb, size=4):
    height = rgb.shape[0] // size * size
    width = rgb.shape[1] // size * size
    block = rgb[:height, :width].reshape(height // size, size, width // size, size, 3)
    return block.mean(axis=(1, 3))


def difference(a, b):
    """Relative RMSE of the blurred images and relative difference of the means."""
    mean = max(float(b.mean()), 1e-6)
    blurred = float(np.sqrt(np.mean((blur(a) - blur(b)) ** 2))) / mean
    return blurred, (float(a.mean()) - float(b.mean())) / mean


def compare(gpu, cpu, cpu_other, denoise=False):
    noise_rmse, noise_mean = difference(cpu_other, cpu)
    rmse, mean = difference(gpu, cpu)
    # Same seed GPU and CPU renders follow the same sample sequence until floating point
    # differences let a path diverge, so they normally differ less than two CPU seeds. Allow a
    # margin over that noise, plus a floor for nearly noise free scenes.
    # Denoisers on different devices differ slightly, and denoised seeds hardly differ.
    rmse_limit = 1.5 * noise_rmse + (0.05 if denoise else 0.01)
    mean_limit = 3.0 * abs(noise_mean) + (0.03 if denoise else 0.015)
    finite = bool(np.isfinite(gpu).all())
    passed = finite and rmse <= rmse_limit and abs(mean) <= mean_limit
    return {
        "passed": passed,
        "finite": finite,
        "relative_rmse": rmse,
        "relative_rmse_limit": rmse_limit,
        "mean_difference": mean,
        "mean_difference_limit": mean_limit,
        "cpu_seed_noise_rmse": noise_rmse,
    }


def write_html(path, report, backends):
    rows = []
    for name, entry in report["scenes"].items():
        cells = [f"<td><b>{html.escape(name)}</b><br><small>{html.escape(entry['description'])}</small></td>"]
        for backend in backends:
            result = entry["renders"].get(backend)
            if result is None:
                cells.append("<td>-</td>")
                continue
            if "error" in result:
                cells.append(f"<td class='fail'>ERROR<br><small>{html.escape(result['error'][:300])}</small></td>")
                continue
            check = result.get("comparison")
            status = ""
            klass = ""
            if check is not None:
                klass = "pass" if check["passed"] else "fail"
                status = (f"<br>{'PASS' if check['passed'] else 'FAIL'} "
                          f"rmse {check['relative_rmse']:.3f}/{check['relative_rmse_limit']:.3f} "
                          f"mean {check['mean_difference']:+.3f}")
            cells.append(f"<td class='{klass}'><img src='{backend}/{name}.png'><br>"
                         f"{result['seconds']:.1f} s{status}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    header = "".join(f"<th>{b}</th>" for b in backends)
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>JCycles feature demo</title>"
        "<style>body{font-family:sans-serif;background:#1d1d1d;color:#ddd}"
        "td,th{padding:6px;vertical-align:top;border-bottom:1px solid #333}"
        "img{width:240px;image-rendering:pixelated}.pass{color:#8c8}.fail{color:#f77}</style>"
        f"<h1>JCycles feature demo</h1><p>{html.escape(report['blender'])}, "
        f"{report['samples']} samples</p>"
        f"<table><tr><th>Scene</th>{header}</tr>{''.join(rows)}</table>",
        encoding="utf-8")


def main():
    args = parse_args()
    if args.list:
        for name, builder in SCENES.items():
            print(name)
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    backends = available_backends([b.strip().upper() for b in args.devices.split(",") if b.strip()])
    if "CPU" not in backends:
        backends.insert(0, "CPU")
    names = list(SCENES) if args.scenes == "all" else [s.strip() for s in args.scenes.split(",")]
    for directory in backends + ["CPU_seed2", "blend"]:
        (output / directory).mkdir(exist_ok=True)

    report = {"blender": f"Blender {bpy.app.version_string} ({bpy.app.build_hash.decode()})",
              "samples": args.samples, "denoise": args.denoise, "backends": backends,
              "scenes": {}}
    failures = 0
    for name in names:
        print(f"[jcycles-demo] scene {name}", flush=True)
        scene, description = SCENES[name](args)
        entry = {"description": description, "renders": {}}
        report["scenes"][name] = entry
        if not args.no_blend:
            bpy.ops.wm.save_as_mainfile(filepath=str(output / "blend" / f"{name}.blend"))
        images = {}
        for backend in backends + ["CPU_seed2"]:
            device = "CPU" if backend == "CPU_seed2" else backend
            try:
                use_backend(scene, device)
                scene.cycles.seed = 2 if backend == "CPU_seed2" else 1
                rgb, seconds = render(scene, output / backend / f"{name}.exr",
                                      output / backend / f"{name}.png")
                images[backend] = rgb
                entry["renders"][backend] = {"seconds": seconds, "mean": float(rgb.mean()),
                                             "finite": bool(np.isfinite(rgb).all())}
                print(f"[jcycles-demo]   {backend:9s} {seconds:7.2f} s  mean {rgb.mean():.5f}",
                      flush=True)
            except Exception as ex:
                failures += 1
                entry["renders"][backend] = {"error": f"{ex}\n{traceback.format_exc()}"}
                print(f"[jcycles-demo]   {backend:9s} FAILED: {ex}", flush=True)
        scene.cycles.seed = 1
        if "CPU" in images and "CPU_seed2" in images:
            for backend in backends:
                if backend == "CPU" or backend not in images:
                    continue
                check = compare(images[backend], images["CPU"], images["CPU_seed2"], args.denoise)
                entry["renders"][backend]["comparison"] = check
                if not check["passed"]:
                    failures += 1
                print(f"[jcycles-demo]   {backend:9s} vs CPU: {'PASS' if check['passed'] else 'FAIL'} "
                      f"rmse {check['relative_rmse']:.4f} (limit {check['relative_rmse_limit']:.4f}) "
                      f"mean {check['mean_difference']:+.4f} (limit {check['mean_difference_limit']:.4f})",
                      flush=True)

    report["failures"] = failures
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_html(output / "report.html", report, backends + ["CPU_seed2"])
    print(f"[jcycles-demo] {failures} failure(s). Report: {output / 'report.html'}", flush=True)
    return 1 if failures else 0


sys.exit(main())
