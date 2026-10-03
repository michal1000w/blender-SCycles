# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Vertex connection and merging scenes for Cycles: demo scenes and validation scenes.

Vertex merging is a render setting (Render > Light Paths > Vertex Connection and Merging). It
must converge to the image of the regular path tracer in every configuration: with path tracing
and with bidirectional path tracing, with and without path guiding, on the CPU and on Metal.
The validation scenes have a path traced reference that converges in reasonable time; the demo
scenes show the transport that only merging samples well (caustics seen in a mirror or through
glass, caustics of point lights).

  blender -b --factory-startup --python tests/python/cycles_vcm_scenes.py -- \
      --output DIRECTORY [--device CPU|METAL] [--integrator pt|bdpt] [--vcm] [--guiding] \
      [--scenes a,b,c|validation|demo|all] [--samples N] [--resolution N] [--radius R] \
      [--light-paths N] [--time-limit SECONDS] [--save-blend] [--no-render] [--png]

Scenes without `--vcm` do not touch the setting, so they also render in builds without it.
Compare with tests/python/cycles_vcm_compare.py.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="CPU", choices=("METAL", "CPU"))
    parser.add_argument("--integrator", default="pt", choices=("pt", "bdpt"))
    parser.add_argument("--vcm", action="store_true")
    parser.add_argument("--guiding", action="store_true")
    parser.add_argument("--scenes", default="validation")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--bounces", type=int, default=8)
    parser.add_argument("--radius", type=float, default=0.0, help="Merge radius, 0 is automatic")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--merge-max", type=int, default=16)
    parser.add_argument("--light-paths", type=int, default=65536)
    parser.add_argument("--update-samples", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=0.0)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--png", action="store_true", help="Also save a tone mapped PNG")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------------
# Scene setup


def aim(obj, target):
    direction = Vector(target) - Vector(obj.location)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def new_scene(args, camera_location=(0.0, -3.9, 1.0), look_at=(0.0, 0.0, 1.0), lens=35.0,
              world_strength=0.0, world_color=(1.0, 1.0, 1.0)):
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
    cycles.seed = args.seed
    cycles.time_limit = args.time_limit
    # The same limit for every kind of bounce and for light subpaths, so that all strategies
    # sample the same set of paths.
    cycles.max_bounces = args.bounces
    cycles.diffuse_bounces = args.bounces
    cycles.glossy_bounces = args.bounces
    cycles.transmission_bounces = args.bounces
    cycles.transparent_max_bounces = args.bounces
    cycles.volume_bounces = args.bounces
    cycles.caustics_reflective = True
    cycles.caustics_refractive = True
    cycles.blur_glossy = 0.0
    cycles.sample_clamp_direct = 0.0
    cycles.sample_clamp_indirect = 0.0
    cycles.min_light_bounces = 64
    cycles.min_transparent_bounces = 64
    cycles.use_guiding = args.guiding
    if args.integrator == "bdpt":
        cycles.use_bidirectional_path_tracing = True
    if args.integrator == "bdpt" or args.vcm:
        cycles.bdpt_light_paths = args.light_paths
        cycles.bdpt_max_bounces = args.bounces
        cycles.bdpt_update_samples = args.update_samples
    if args.vcm:
        if not hasattr(cycles, "use_vertex_merging"):
            raise SystemExit("This build has no vertex merging")
        cycles.use_vertex_merging = True
        cycles.vcm_radius = args.radius
        cycles.vcm_radius_alpha = args.alpha
        cycles.vcm_merge_max = args.merge_max

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (*world_color, 1.0)
    background.inputs["Strength"].default_value = world_strength
    scene.world = world

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.data.lens = lens
    aim(camera, look_at)
    scene.camera = camera
    return scene


def new_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    return mat, nodes, mat.node_tree.links, output


def diffuse(name, color):
    mat, nodes, links, output = new_material(name)
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (*color, 1.0)
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat


def glossy(name, color, roughness):
    mat, nodes, links, output = new_material(name)
    bsdf = nodes.new("ShaderNodeBsdfGlossy")
    bsdf.inputs["Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat


def glass(name, ior=1.5, roughness=0.0, color=(1.0, 1.0, 1.0)):
    mat, nodes, links, output = new_material(name)
    bsdf = nodes.new("ShaderNodeBsdfGlass")
    bsdf.inputs["Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat


def principled(name, color, roughness=0.5, metallic=0.0, transmission=0.0, coat=0.0, ior=1.5):
    mat, nodes, links, output = new_material(name)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Transmission Weight"].default_value = transmission
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["IOR"].default_value = ior
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat


def emission(name, color, strength):
    mat, nodes, links, output = new_material(name)
    node = nodes.new("ShaderNodeEmission")
    node.inputs["Color"].default_value = (*color, 1.0)
    node.inputs["Strength"].default_value = strength
    links.new(node.outputs[0], output.inputs["Surface"])
    return mat


def checker(name, color_a, color_b, scale=6.0):
    mat, nodes, links, output = new_material(name)
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    tex = nodes.new("ShaderNodeTexChecker")
    tex.inputs["Color1"].default_value = (*color_a, 1.0)
    tex.inputs["Color2"].default_value = (*color_b, 1.0)
    tex.inputs["Scale"].default_value = scale
    links.new(tex.outputs["Color"], bsdf.inputs["Color"])
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat


def add_plane(name, location, rotation, size, material):
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = (size[0], size[1], 1.0)
    obj.data.materials.append(material)
    return obj


def add_sphere(name, location, radius, material, smooth=True, segments=48):
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments, ring_count=segments // 2, radius=radius, location=location)
    obj = bpy.context.object
    obj.name = name
    if smooth:
        bpy.ops.object.shade_smooth()
    obj.data.materials.append(material)
    return obj


def add_cube(name, location, size, material, rotation=(0.0, 0.0, 0.0)):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=rotation)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    obj.data.materials.append(material)
    return obj


def add_area_light(name, location, size, power, target=None, color=(1.0, 1.0, 1.0), shape="SQUARE"):
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.name = name
    light.data.shape = shape
    light.data.size = size
    light.data.energy = power
    light.data.color = color
    if target is not None:
        aim(light, target)
    return light


def add_box(white, left, right, light_size=0.5, light_power=60.0, open_front=True):
    """Box of 2 x 2 x 2 around (0, 0, 1) with a light under the ceiling."""
    half_pi = math.pi / 2
    add_plane("Floor", (0, 0, 0), (0, 0, 0), (2, 2), white)
    add_plane("Ceiling", (0, 0, 2), (math.pi, 0, 0), (2, 2), white)
    add_plane("Back", (0, 1, 1), (half_pi, 0, 0), (2, 2), white)
    add_plane("Left", (-1, 0, 1), (0, half_pi, 0), (2, 2), left)
    add_plane("Right", (1, 0, 1), (0, -half_pi, 0), (2, 2), right)
    if not open_front:
        add_plane("Front", (0, -1, 1), (-half_pi, 0, 0), (2, 2), white)
    if light_power:
        add_area_light("Light", (0, 0, 1.98), light_size, light_power, target=(0, 0, 0))


def box_materials():
    return (diffuse("White", (0.73, 0.73, 0.73)), diffuse("Red", (0.65, 0.06, 0.05)),
            diffuse("Green", (0.13, 0.45, 0.10)))


# ------------------------------------------------------------------------------------------------
# Validation scenes: the path traced reference converges.


def scene_diffuse_box(args):
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(white, red, green)
    add_sphere("Sphere", (-0.4, 0.3, 0.35), 0.35, diffuse("Blue", (0.2, 0.3, 0.7)))
    add_cube("Block", (0.45, 0.1, 0.3), (0.55, 0.55, 0.6), white, rotation=(0, 0, 0.4))
    return scene


def scene_glossy_box(args):
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(glossy("Rough", (0.7, 0.7, 0.7), 0.25), red, green, light_size=0.7)
    add_sphere("Metal", (-0.4, 0.3, 0.35), 0.35, glossy("Metal", (0.9, 0.7, 0.4), 0.15))
    add_sphere("Matte", (0.45, -0.1, 0.3), 0.3, white)
    return scene


def scene_emissive_mesh(args):
    """Mesh emitters of different sizes instead of lamps, one of them two-sided."""
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(white, red, green, light_power=0.0)
    add_plane("Panel", (0, 0, 1.97), (math.pi, 0, 0), (0.8, 0.8), emission("Warm", (1, 0.8, 0.6), 12))
    add_sphere("Bulb", (-0.55, 0.2, 0.5), 0.12, emission("Cool", (0.5, 0.7, 1.0), 25), segments=16)
    add_plane("Card", (0.5, 0.3, 0.6), (0.3, 1.2, 0.5), (0.4, 0.4), emission("Green", (0.4, 1, 0.4), 6))
    add_cube("Block", (0.3, -0.3, 0.2), (0.4, 0.4, 0.4), white, rotation=(0, 0, 0.6))
    return scene


def scene_caustic_glass(args):
    """Glass sphere under an area light: the caustic on the floor and its light on the walls."""
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(white, red, green, light_size=0.5, light_power=60.0)
    add_sphere("Glass", (0.0, 0.1, 0.45), 0.45, glass("Glass"))
    return scene


def scene_rough_glass(args):
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(white, red, green, light_size=0.6)
    add_sphere("Frosted", (-0.35, 0.2, 0.4), 0.4, glass("Frosted", roughness=0.3))
    add_cube("Slab", (0.45, 0.0, 0.5), (0.5, 0.08, 1.0), glass("Slab", roughness=0.15),
             rotation=(0, 0, 0.5))
    return scene


def scene_mirror_caustic(args):
    """Caustic of a glass sphere seen in a mirror: specular - diffuse - specular transport."""
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(white, red, glossy("Mirror", (0.95, 0.95, 0.95), 0.0), light_size=0.5)
    add_sphere("Glass", (-0.2, 0.1, 0.4), 0.4, glass("Glass"))
    return scene


def scene_ring(args):
    """Cardioid caustic inside a metal ring."""
    scene = new_scene(args, camera_location=(0.0, -2.6, 2.4), look_at=(0.0, 0.0, 0.0), lens=40.0)
    floor = diffuse("Floor", (0.7, 0.7, 0.7))
    add_plane("Floor", (0, 0, 0), (0, 0, 0), (6, 6), floor)
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=96, radius=0.8, depth=0.4, end_fill_type="NOTHING", location=(0, 0, 0.2))
    ring = bpy.context.object
    bpy.ops.object.shade_smooth()
    ring.data.materials.append(glossy("Gold", (1.0, 0.8, 0.4), 0.0))
    add_area_light("Light", (2.2, -0.6, 1.8), 0.25, 220.0, target=(0, 0, 0.1))
    return scene


def scene_sun_sky(args):
    """Infinite emitters: a sun with an angle and a uniform environment."""
    scene = new_scene(args, camera_location=(0.0, -4.5, 1.6), look_at=(0.0, 0.0, 0.5), lens=40.0,
                      world_strength=0.5, world_color=(0.6, 0.75, 1.0))
    add_plane("Floor", (0, 0, 0), (0, 0, 0), (8, 8), checker("Checker", (0.7, 0.7, 0.7), (0.3, 0.3, 0.3)))
    add_sphere("Glass", (-0.7, 0.0, 0.5), 0.5, glass("Glass"))
    add_sphere("Matte", (0.6, 0.3, 0.45), 0.45, diffuse("Orange", (0.8, 0.4, 0.1)))
    add_cube("Mirror", (1.4, 1.0, 0.6), (0.1, 1.5, 1.2), glossy("Mirror", (0.9, 0.9, 0.9), 0.0),
             rotation=(0, 0, 0.5))
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))
    sun = bpy.context.object
    sun.data.energy = 3.0
    sun.data.angle = math.radians(8.0)
    sun.rotation_euler = (math.radians(40), math.radians(15), math.radians(25))
    return scene


def scene_delta_lights(args):
    """Point and spot lights without radius on diffuse and rough surfaces, no caustics."""
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(white, red, green, light_power=0.0)
    bpy.ops.object.light_add(type="POINT", location=(-0.4, -0.2, 1.6))
    point = bpy.context.object
    point.data.energy = 40.0
    point.data.shadow_soft_size = 0.0
    bpy.ops.object.light_add(type="SPOT", location=(0.6, -0.5, 1.8))
    spot = bpy.context.object
    spot.data.energy = 120.0
    spot.data.shadow_soft_size = 0.0
    spot.data.spot_size = math.radians(50)
    spot.data.spot_blend = 0.3
    spot.data.color = (1.0, 0.8, 0.6)
    aim(spot, (0.2, 0.3, 0.0))
    add_sphere("Rough", (-0.4, 0.3, 0.35), 0.35, glossy("Rough", (0.8, 0.8, 0.8), 0.3))
    add_cube("Block", (0.45, 0.1, 0.3), (0.55, 0.55, 0.6), white, rotation=(0, 0, 0.4))
    return scene


def scene_sphere_lights(args, soft_falloff=False):
    """Point and spot lights with a radius, which emit light paths from their surface."""
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(white, red, green, light_power=0.0)
    bpy.ops.object.light_add(type="POINT", location=(-0.4, -0.2, 1.5))
    point = bpy.context.object
    point.data.energy = 40.0
    point.data.shadow_soft_size = 0.12
    point.data.use_soft_falloff = soft_falloff
    bpy.ops.object.light_add(type="SPOT", location=(0.6, -0.5, 1.7))
    spot = bpy.context.object
    spot.data.energy = 120.0
    spot.data.shadow_soft_size = 0.08
    spot.data.use_soft_falloff = soft_falloff
    spot.data.spot_size = math.radians(60)
    aim(spot, (0.2, 0.3, 0.0))
    add_sphere("Glass", (-0.3, 0.3, 0.35), 0.35, glass("Glass"))
    add_cube("Block", (0.45, 0.1, 0.3), (0.55, 0.55, 0.6), white, rotation=(0, 0, 0.4))
    return scene


def scene_soft_lights(args):
    """Point and spot lights with soft falloff: disks that face the shading point, which only
    next event estimation and the camera path sample."""
    return scene_sphere_lights(args, soft_falloff=True)


def scene_principled(args):
    """Layered Principled materials, whose closures depend on the incoming direction."""
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(principled("Walls", (0.7, 0.7, 0.7), roughness=0.6), red, green, light_size=0.6)
    add_sphere("Plastic", (-0.5, 0.3, 0.3), 0.3, principled("Plastic", (0.8, 0.2, 0.1), 0.2, coat=1.0))
    add_sphere("Metal", (0.1, 0.5, 0.3), 0.3, principled("Metal", (0.9, 0.8, 0.5), 0.25, metallic=1.0))
    add_sphere("Glass", (0.55, -0.1, 0.3), 0.3, principled("Glass", (1, 1, 1), 0.0, transmission=1.0))
    return scene


def scene_ortho_dof(args):
    """Orthographic camera with depth of field: no sensor connection exists for it."""
    scene = scene_caustic_glass(args)
    camera = scene.camera.data
    camera.type = "ORTHO"
    camera.ortho_scale = 2.3
    camera.dof.use_dof = True
    camera.dof.focus_distance = 3.8
    camera.dof.aperture_fstop = 0.4
    return scene


def scene_fog(args):
    """Scattering medium in front of the surfaces: merging happens on surfaces only."""
    scene = scene_diffuse_box(args)
    mat, nodes, links, output = new_material("Fog")
    volume = nodes.new("ShaderNodeVolumeScatter")
    volume.inputs["Density"].default_value = 0.25
    links.new(volume.outputs[0], output.inputs["Volume"])
    add_cube("Fog", (0, 0, 1), (1.9, 1.9, 1.9), mat)
    return scene


def scene_fog_caustic(args):
    """Glass sphere inside the scattering medium: caustics on the floor and in the medium."""
    scene = scene_fog(args)
    bpy.data.objects["Sphere"].data.materials[0] = glass("Glass")
    return scene


def scene_motion(args):
    """Moving caster with motion blur."""
    scene = scene_caustic_glass(args)
    scene.render.use_motion_blur = True
    scene.render.motion_blur_shutter = 1.0
    sphere = bpy.data.objects["Glass"]
    sphere.location = (-0.3, 0.1, 0.45)
    sphere.keyframe_insert("location", frame=0)
    sphere.location = (0.3, 0.1, 0.45)
    sphere.keyframe_insert("location", frame=2)
    scene.frame_set(1)
    return scene


# ------------------------------------------------------------------------------------------------
# Demo scenes: the reference of the path tracer does not converge in practical time.


def scene_demo_pool(args):
    """Water surface over a tiled floor lit by a small light: the caustics on the floor are seen
    through the surface that casts them."""
    scene = new_scene(args, camera_location=(0.0, -3.4, 2.2), look_at=(0.0, 0.2, 0.0), lens=32.0,
                      world_strength=0.05)
    add_plane("Floor", (0, 0, 0), (0, 0, 0), (4, 4), checker("Tiles", (0.75, 0.8, 0.85), (0.2, 0.35, 0.5), 10))
    half_pi = math.pi / 2
    wall = diffuse("Wall", (0.7, 0.7, 0.7))
    add_plane("Back", (0, 2, 0.6), (half_pi, 0, 0), (4, 1.2), wall)
    add_plane("Left", (-2, 0, 0.6), (0, half_pi, 0), (1.2, 4), wall)
    add_plane("Right", (2, 0, 0.6), (0, -half_pi, 0), (1.2, 4), wall)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=96, y_subdivisions=96, size=4.0, location=(0, 0, 0.6))
    water = bpy.context.object
    water.name = "Water"
    for vertex in water.data.vertices:
        x, y = vertex.co.x, vertex.co.y
        vertex.co.z = 0.035 * (math.sin(5.0 * x + 1.3 * y) + math.sin(3.1 * y - 2.2 * x) +
                                0.6 * math.sin(7.3 * x * y * 0.3 + 1.0))
    bpy.ops.object.shade_smooth()
    water.data.materials.append(glass("Water", ior=1.33))
    add_area_light("Light", (0.8, -0.6, 3.0), 0.08, 150.0, target=(0, 0, 0))
    return scene


def scene_demo_point_caustic(args):
    """Glass and mirror objects under a point light without radius, in front of a mirror."""
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(white, red, glossy("Mirror", (0.95, 0.95, 0.95), 0.0), light_power=0.0)
    bpy.ops.object.light_add(type="POINT", location=(-0.3, -0.1, 1.85))
    point = bpy.context.object
    point.data.energy = 25.0
    point.data.shadow_soft_size = 0.0
    add_sphere("Glass", (-0.3, 0.2, 0.4), 0.4, glass("Glass"))
    add_sphere("Chrome", (0.5, -0.2, 0.25), 0.25, glossy("Chrome", (0.95, 0.95, 0.95), 0.0))
    return scene


def scene_demo_lamp(args):
    """Light bulb inside a glass shade: all light reaches the room through glass."""
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(white, red, green, light_power=0.0)
    add_sphere("Bulb", (0.0, 0.0, 1.4), 0.04, emission("Bulb", (1.0, 0.85, 0.7), 700), segments=16)
    add_sphere("Shade", (0.0, 0.0, 1.4), 0.22, glass("Shade"))
    inner = add_sphere("ShadeInner", (0.0, 0.0, 1.4), 0.2, glass("Shade"))
    bpy.context.view_layer.objects.active = inner
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.flip_normals()
    bpy.ops.object.mode_set(mode="OBJECT")
    add_sphere("Matte", (-0.45, 0.3, 0.3), 0.3, diffuse("Blue", (0.2, 0.3, 0.7)))
    add_cube("Block", (0.45, 0.1, 0.3), (0.5, 0.5, 0.6), white, rotation=(0, 0, 0.4))
    return scene


def scene_demo_fog_beam(args):
    """Spot light without radius through a glass sphere in fog: the focused beam in the medium
    and its caustic on the floor are sampled by light subpaths only."""
    scene = new_scene(args)
    white, red, green = box_materials()
    add_box(white, red, green, light_power=4.0)
    bpy.ops.object.light_add(type="SPOT", location=(-0.8, -0.6, 1.8))
    spot = bpy.context.object
    spot.data.energy = 300.0
    spot.data.shadow_soft_size = 0.0
    spot.data.spot_size = math.radians(22)
    spot.data.spot_blend = 0.2
    aim(spot, (0.1, 0.1, 0.9))
    add_sphere("Glass", (0.1, 0.1, 0.9), 0.3, glass("Glass"))
    mat, nodes, links, output = new_material("Fog")
    volume = nodes.new("ShaderNodeVolumeScatter")
    volume.inputs["Density"].default_value = 0.35
    links.new(volume.outputs[0], output.inputs["Volume"])
    add_cube("Fog", (0, 0, 1), (1.9, 1.9, 1.9), mat)
    return scene


VALIDATION = {
    "diffuse_box": scene_diffuse_box,
    "glossy_box": scene_glossy_box,
    "emissive_mesh": scene_emissive_mesh,
    "caustic_glass": scene_caustic_glass,
    "rough_glass": scene_rough_glass,
    "mirror_caustic": scene_mirror_caustic,
    "ring": scene_ring,
    "sun_sky": scene_sun_sky,
    "delta_lights": scene_delta_lights,
    "sphere_lights": scene_sphere_lights,
    "soft_lights": scene_soft_lights,
    "principled": scene_principled,
    "ortho_dof": scene_ortho_dof,
    "fog": scene_fog,
    "fog_caustic": scene_fog_caustic,
    "motion": scene_motion,
}
DEMO = {
    "demo_pool": scene_demo_pool,
    "demo_point_caustic": scene_demo_point_caustic,
    "demo_lamp": scene_demo_lamp,
    "demo_fog_beam": scene_demo_fog_beam,
}
SCENES = {**VALIDATION, **DEMO}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.scenes == "validation":
        names = list(VALIDATION)
    elif args.scenes == "demo":
        names = list(DEMO)
    elif args.scenes == "all":
        names = list(SCENES)
    else:
        names = args.scenes.split(",")

    results = {}
    for name in names:
        scene = SCENES[name](args)
        if args.device == "METAL":
            # After the scene: loading factory settings resets the preferences.
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = "METAL"
            preferences.get_devices()
            metal = [device for device in preferences.devices if device.type == "METAL"]
            if not metal:
                raise SystemExit("No Metal device")
            for device in preferences.devices:
                device.use = device.type == "METAL"
            scene.cycles.device = "GPU"
        stem = name + args.suffix
        if args.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=str(args.output / (stem + ".blend")))
        if args.no_render:
            continue
        scene.render.filepath = str(args.output / (stem + ".exr"))
        start = time.time()
        bpy.ops.render.render(write_still=True)
        seconds = time.time() - start
        results[stem] = {"seconds": seconds, "samples": args.samples}
        print("VCM_SCENE %-24s %7.2f s" % (stem, seconds), flush=True)
        if args.png:
            image = bpy.data.images.load(scene.render.filepath)
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_depth = "8"
            scene.view_settings.view_transform = "Filmic" if "Filmic" in [
                item.identifier for item in
                type(scene.view_settings).bl_rna.properties["view_transform"].enum_items
            ] else "Standard"
            image.save_render(str(args.output / (stem + ".png")), scene=scene)
            bpy.data.images.remove(image)

    if results:
        path = args.output / ("timing%s.json" % args.suffix)
        previous = json.loads(path.read_text()) if path.exists() else {}
        previous.update(results)
        path.write_text(json.dumps(previous, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
