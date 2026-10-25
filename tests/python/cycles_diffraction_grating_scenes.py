# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Scenes for the Grating model of the Diffraction BSDF (1D/2D lattice, sinusoidal or rectangular
grooves), modeled on the Blender Artists thread "WIP: Cycles - Diffraction Grating Shader" and the
Al-Toto/blender `diffraction-bsdf` branch it comes from.

The same script runs in that branch (node without a Model setting) and in this fork, so renders
of the two builds can be compared pixel by pixel. Renders use fixed seeds and no denoising.

  blender -b --factory-startup --python tests/python/cycles_diffraction_grating_scenes.py -- \
      --output DIRECTORY [--device CPU|METAL] [--integrator pt|bdpt|photon|vcm] [--guiding] \
      [--osl] [--scenes a,b,c] [--resolution N] [--samples N] [--timing] [--save-blend]

Demo scenes (thread look): cd_rectangular, cd_sinusoidal, grid_2d, cd_hero, cloth, sign,
sphere_uv. Validation scenes: furnace_1d, furnace_2d, consistency (small, quick).
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bpy

# -------------------------------------------------------------------------------------------------
# Arguments


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="CPU", choices=("CPU", "METAL"))
    parser.add_argument("--integrator", default="pt", choices=("pt", "bdpt", "photon", "vcm"))
    parser.add_argument("--guiding", action="store_true")
    parser.add_argument("--osl", action="store_true", help="Open Shading Language (CPU only)")
    parser.add_argument("--scenes", default="demo",
                        help="Comma separated scene names, 'demo', 'validation' or 'all'")
    parser.add_argument("--resolution", type=int, default=0, help="0 = scene default")
    parser.add_argument("--samples", type=int, default=0, help="0 = scene default")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timing", action="store_true",
                        help="Render every scene twice and report the second render time")
    parser.add_argument("--save-blend", action="store_true",
                        help="Also save each scene as a .blend file next to the image")
    parser.add_argument("--format", default="EXR", choices=("EXR", "PNG"))
    parser.add_argument("--control-glossy", action="store_true",
                        help="Glossy BSDF instead of every grating, to measure the cost of the "
                        "rest of the scene")
    return parser.parse_args(argv)


# -------------------------------------------------------------------------------------------------
# Scene setup helpers


def new_scene(args, resolution, samples, max_bounces=8):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    res = args.resolution or resolution
    scene.render.resolution_x = res
    scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    if args.format == "EXR":
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
    else:
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    cycles = scene.cycles
    cycles.device = "GPU" if args.device == "METAL" else "CPU"
    cycles.samples = args.samples or samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.seed = args.seed
    cycles.max_bounces = max_bounces
    cycles.diffuse_bounces = max_bounces
    cycles.glossy_bounces = max_bounces
    cycles.transmission_bounces = max_bounces
    cycles.transparent_max_bounces = max_bounces
    cycles.caustics_reflective = True
    cycles.caustics_refractive = True
    cycles.sample_clamp_direct = 0.0
    cycles.sample_clamp_indirect = 0.0
    cycles.blur_glossy = 0.0
    set_integrator(args, cycles)
    if args.osl:
        cycles.shading_system = True
    return scene


def set_integrator(args, cycles):
    """Integrator settings of this fork. In builds without them (the reference branch), only plain
    path tracing exists and the settings are skipped."""
    def set_if(name, value):
        if hasattr(cycles, name):
            setattr(cycles, name, value)
        elif value not in (False, 0):
            raise RuntimeError(f"This build has no '{name}' setting")
    if args.integrator == "bdpt":
        set_if("use_bidirectional_path_tracing", True)
    elif args.integrator == "photon":
        set_if("use_photon_mapping", True)
    elif args.integrator == "vcm":
        set_if("use_vertex_merging", True)
    if args.guiding:
        cycles.use_guiding = True
        cycles.use_surface_guiding = True
        if hasattr(cycles, "use_deterministic_guiding"):
            cycles.use_deterministic_guiding = True


def enable_device(args):
    if args.device != "METAL":
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "METAL"


def gradient_world(scene, strength=1.0, left=(0.045, 0.032, 0.075), right=(0.16, 0.075, 0.065),
                   zenith=(0.42, 0.38, 0.44), nadir=(0.011, 0.009, 0.018)):
    """Dusk-like world of the thread's renders: below the horizon a purple to brown gradient from
    -X to +X (the backdrop seen around objects viewed from above), above it a brighter lilac sky
    that shiny surfaces reflect."""
    world = bpy.data.worlds.new("Gradient")
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    background = nodes["Background"]
    coord = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coord.outputs["Generated"], separate.inputs[0])
    # Horizontal gradient, x in [-1, 1] mapped to [0, 1], with a little of z mixed in.
    horizontal = nodes.new("ShaderNodeMath")
    horizontal.operation = "MULTIPLY_ADD"
    horizontal.inputs[1].default_value = 0.5
    horizontal.inputs[2].default_value = 0.5
    links.new(separate.outputs["X"], horizontal.inputs[0])
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.1
    ramp.color_ramp.elements[0].color = (*left, 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (*right, 1.0)
    links.new(horizontal.outputs[0], ramp.inputs["Fac"])
    # Darker towards the ground, as the bottom of the thread's renders.
    ground = nodes.new("ShaderNodeMapRange")
    ground.inputs["From Min"].default_value = 0.0
    ground.inputs["From Max"].default_value = -0.8
    links.new(separate.outputs["Z"], ground.inputs["Value"])
    below = nodes.new("ShaderNodeMix")
    below.data_type = "RGBA"
    below.inputs["B"].default_value = (*nadir, 1.0)
    links.new(ground.outputs["Result"], below.inputs["Factor"])
    links.new(ramp.outputs["Color"], below.inputs["A"])
    # Sky: fades in above the horizon.
    sky = nodes.new("ShaderNodeMapRange")
    sky.inputs["From Min"].default_value = -0.02
    sky.inputs["From Max"].default_value = 0.25
    # Dimmer sky between horizon and zenith.
    links.new(separate.outputs["Z"], sky.inputs["Value"])
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.inputs["B"].default_value = (*zenith, 1.0)
    links.new(sky.outputs["Result"], mix.inputs["Factor"])
    links.new(below.outputs["Result"], mix.inputs["A"])
    # Brighter towards the zenith: diffraction orders of the disc look higher into the sky than its
    # mirror reflection and turn this gradient into the colored arc of the thread's renders.
    top = nodes.new("ShaderNodeMapRange")
    top.inputs["From Min"].default_value = 0.84
    top.inputs["From Max"].default_value = 0.92
    links.new(separate.outputs["Z"], top.inputs["Value"])
    zenith_mix = nodes.new("ShaderNodeMix")
    zenith_mix.data_type = "RGBA"
    zenith_mix.inputs["B"].default_value = (2.6 * zenith[0], 2.6 * zenith[1], 2.9 * zenith[2], 1.0)
    links.new(top.outputs["Result"], zenith_mix.inputs["Factor"])
    links.new(mix.outputs["Result"], zenith_mix.inputs["A"])
    links.new(zenith_mix.outputs["Result"], background.inputs["Color"])
    background.inputs["Strength"].default_value = strength
    scene.world = world
    return world


def uniform_world(scene, color=(1.0, 1.0, 1.0), strength=1.0):
    world = bpy.data.worlds.new("Uniform")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (*color, 1.0)
    background.inputs["Strength"].default_value = strength
    scene.world = world
    return world


def look_at(obj, target):
    from mathutils import Vector
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(scene, location, target, lens=50.0):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = lens
    cam_data.clip_start = 0.001
    cam = bpy.data.objects.new("Camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = location
    look_at(cam, target)
    scene.camera = cam
    return cam


def add_light(scene, kind, location, energy, size=0.0, target=None, color=(1.0, 1.0, 1.0)):
    data = bpy.data.lights.new(kind.title(), kind)
    data.energy = energy
    data.color = color
    if kind in ("POINT", "SPOT"):
        data.shadow_soft_size = size
    elif kind == "AREA":
        data.size = size
    obj = bpy.data.objects.new(data.name, data)
    scene.collection.objects.link(obj)
    obj.location = location
    if target is not None:
        look_at(obj, target)
    return obj


def material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    output = nodes["Material Output"]
    return mat, nodes, links, output


CONTROL_GLOSSY = False


def diffraction_node(nodes, grating="1D", profile="RECTANGULAR", roughness=0.05, pitch=1.6,
                     depth=110.0, duty=0.45, rotation=(0.0, 0.25), color=(1.0, 1.0, 1.0)):
    """Diffraction BSDF with the Grating model. Works with the reference branch's node, which has
    no Model setting, and with this fork's node."""
    if CONTROL_GLOSSY:
        node = nodes.new("ShaderNodeBsdfAnisotropic")
        node.inputs["Color"].default_value = (*color, 1.0)
        node.inputs["Roughness"].default_value = roughness
        return node
    node = nodes.new("ShaderNodeBsdfDiffraction")
    if hasattr(node, "quality"):
        node.quality = "GRATING"
    node.grating_type = "GRATING_" + grating
    node.profile = profile
    pitch = pitch if isinstance(pitch, tuple) else (pitch, pitch, pitch)
    depth = depth if isinstance(depth, tuple) else (depth, depth, depth)
    duty = duty if isinstance(duty, tuple) else (duty, duty, duty)
    values = {"Color": (*color, 1.0), "Roughness": roughness, "Pitch": pitch,
              "Groove Depth": depth, "Duty Cycle": duty,
              "Rotation": (rotation[0], rotation[1], 0.0)}
    # Unavailable sockets (Duty Cycle of the sinusoidal profile) cannot be looked up by name.
    for socket in node.inputs:
        if socket.name in values:
            socket.default_value = values[socket.name]
    return node


def radial_tangent(nodes, links, node, axis="Z"):
    if node.bl_idname != "ShaderNodeBsdfDiffraction":
        return None
    tangent = nodes.new("ShaderNodeTangent")
    tangent.direction_type = "RADIAL"
    tangent.axis = axis
    links.new(tangent.outputs["Tangent"], node.inputs["Tangent"])
    return tangent


def add_mesh_object(scene, name, mesh, mat=None, location=(0, 0, 0), rotation=(0, 0, 0)):
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    obj.location = location
    obj.rotation_euler = rotation
    if mat is not None:
        mesh.materials.append(mat)
    return obj


def annulus_mesh(name, r_inner, r_outer, segments=256, rings=1, uv=True):
    """Flat ring in the XY plane facing +Z. UVs are planar so the UV tangent is defined."""
    import bmesh
    bm = bmesh.new()
    rings_v = []
    for j in range(rings + 1):
        r = r_inner + (r_outer - r_inner) * j / rings
        ring = []
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            ring.append(bm.verts.new((r * math.cos(a), r * math.sin(a), 0.0)))
        rings_v.append(ring)
    for j in range(rings):
        for i in range(segments):
            i1 = (i + 1) % segments
            bm.faces.new((rings_v[j][i], rings_v[j][i1], rings_v[j + 1][i1], rings_v[j + 1][i]))
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    if uv:
        layer = mesh.uv_layers.new(name="UVMap")
        for loop in mesh.loops:
            co = mesh.vertices[loop.vertex_index].co
            layer.data[loop.index].uv = (0.5 + 0.5 * co.x / r_outer, 0.5 + 0.5 * co.y / r_outer)
    for poly in mesh.polygons:
        poly.use_smooth = False
    return mesh


def grid_mesh(name, size_x, size_y, nx=1, ny=1):
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=nx + 1, y_subdivisions=ny + 1, size=1.0)
    obj = bpy.context.object
    mesh = obj.data
    mesh.name = name
    for v in mesh.vertices:
        v.co.x *= size_x
        v.co.y *= size_y
    bpy.data.objects.remove(obj)
    return mesh


def diffuse_material(name, color, roughness=0.5):
    mat, nodes, links, output = material(name)
    node = nodes.new("ShaderNodeBsdfDiffuse")
    node.inputs["Color"].default_value = (*color, 1.0)
    links.new(node.outputs[0], output.inputs["Surface"])
    return mat


# -------------------------------------------------------------------------------------------------
# Demo scenes, modeled on the thread's renders


def cd_material(profile, roughness=0.05):
    """The thread's "Diffraction CD" material: 1D grating, 1.6 um track pitch, 110 nm pits at 45%
    duty, grooves along the circles (radial tangent around Z, rotated a quarter turn so that the
    grating vector points away from the center)."""
    mat, nodes, links, output = material("Diffraction CD " + profile.title())
    node = diffraction_node(nodes, "1D", profile, roughness, 1.6, 110.0, 0.45, (0.25, 0.0))
    radial_tangent(nodes, links, node)
    links.new(node.outputs[0], output.inputs["Surface"])
    return mat


def cd_disc(scene, profile, location=(0, 0, 0), rotation=(0, 0, 0)):
    """A 12 cm disc: data area from 23 to 60 mm, a dark clamping area and a 7.5 mm hole."""
    data = add_mesh_object(scene, "CD", annulus_mesh("CD", 0.023, 0.06, rings=8),
                           cd_material(profile), location, rotation)
    hub = add_mesh_object(scene, "Hub", annulus_mesh("Hub", 0.0075, 0.023, rings=2),
                          diffuse_material("Hub", (0.06, 0.065, 0.075)), location, rotation)
    return data, hub


def scene_cd(args, profile):
    """Disc seen from above at about 36 degrees, a small light behind it on the mirrored side of
    the camera, purple to brown background. First orders form two rainbow streaks."""
    scene = new_scene(args, 512, 256)
    gradient_world(scene, 1.0)
    cd_disc(scene, profile)
    # Camera and light about 20 cm away on mirrored sides: first orders of green light leave a
    # point at x when x * (1 / d_camera + 1 / d_light) = wavelength / pitch, which puts the
    # streaks at 35 to 70 percent of the radius as in the thread.
    elevation = math.radians(36.0)
    distance = 0.20
    add_camera(scene, (0.0, -distance * math.cos(elevation), distance * math.sin(elevation)),
               (0.0, 0.0, -0.004), lens=46.0)
    light_distance = 0.21
    add_light(scene, "POINT", (0.0, light_distance * math.cos(elevation),
                               light_distance * math.sin(elevation)), 3.0, size=0.003)
    return scene


def scene_grid_2d(args):
    """The thread's "Diffraction Screen": a black plane coated with a crossed sinusoidal grating
    (4 um pitch, 55 nm), mixed with dark diffuse by Fresnel 1.3. A small light is reflected in its
    center, the orders form a cross with diagonal side orders."""
    scene = new_scene(args, 512, 256)
    gradient_world(scene, 1.0)
    mat, nodes, links, output = material("Diffraction Screen")
    fresnel = nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.3
    diffuse = nodes.new("ShaderNodeBsdfDiffuse")
    diffuse.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    diffuse.inputs["Roughness"].default_value = 0.0
    grating = diffraction_node(nodes, "2D", "SINUSOIDAL", 0.05, 4.0, 55.0, 0.45, (0.0, 0.25))
    mix = nodes.new("ShaderNodeMixShader")
    links.new(fresnel.outputs[0], mix.inputs[0])
    links.new(diffuse.outputs[0], mix.inputs[1])
    links.new(grating.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    add_mesh_object(scene, "Screen", grid_mesh("Screen", 2.4, 2.4), mat)
    elevation = math.radians(38.0)
    add_camera(scene, (0.0, -2.8 * math.cos(elevation), 2.8 * math.sin(elevation)),
               (0.0, 0.0, 0.0), lens=33.0)
    add_light(scene, "POINT", (0.0, 2.8 * math.cos(elevation), 2.8 * math.sin(elevation)),
              600.0, size=0.05)
    return scene


def checker_cross_image(size=1024, cells=16):
    """Checker board with small colored crosses in its cells, like the backdrop of the thread's
    hero render."""
    import random
    rng = random.Random(3)
    image = bpy.data.images.new("CheckerCross", size, size, alpha=False, float_buffer=False)
    palette = [(0.9, 0.2, 0.2), (0.2, 0.85, 0.3), (0.25, 0.4, 0.95), (0.95, 0.8, 0.2),
               (0.8, 0.3, 0.9), (0.2, 0.8, 0.85)]
    cell = size // cells
    pixels = [0.0] * (size * size * 4)
    colors = [[palette[rng.randrange(len(palette))] for _ in range(cells)] for _ in range(cells)]
    arm, thick = cell // 7, max(1, cell // 28)
    for y in range(size):
        cy, oy = divmod(y, cell)
        for x in range(size):
            cx, ox = divmod(x, cell)
            v = 0.42 if (cx + cy) % 2 == 0 else 0.07
            c = (v, v, v)
            dx, dy = ox - cell // 2, oy - cell // 2
            if (abs(dx) <= arm and abs(dy) <= thick) or (abs(dy) <= arm and abs(dx) <= thick):
                c = colors[cy][cx]
            i = (y * size + x) * 4
            pixels[i:i + 4] = (c[0], c[1], c[2], 1.0)
    image.pixels = pixels
    image.pack()
    return image


def scene_cd_hero(args):
    """A disc standing on a curved checker backdrop, lit by a big soft box above, the camera low
    and close (the thread's 2048 px "DiffractionRender")."""
    scene = new_scene(args, 768, 512)
    uniform_world(scene, (0.6, 0.62, 0.66), 0.25)
    # Curved backdrop: floor bending up into a wall behind the disc.
    import bmesh
    bm = bmesh.new()
    nu, nv = 2, 48
    width, radius = 1.2, 0.18
    verts = []
    for j in range(nv + 1):
        t = j / nv
        # Floor from y=-0.6 to 0.1, quarter circle, wall up to z=0.8.
        s = t * (0.7 + 0.5 * math.pi * radius + 0.6)
        if s < 0.7:
            y, z = -0.6 + s, 0.0
        elif s < 0.7 + 0.5 * math.pi * radius:
            a = (s - 0.7) / radius
            y, z = 0.1 + radius * math.sin(a), radius * (1.0 - math.cos(a))
        else:
            y, z = 0.1 + radius, radius + (s - 0.7 - 0.5 * math.pi * radius)
        row = [bm.verts.new((width * (i / nu - 0.5), y, z)) for i in range(nu + 1)]
        verts.append(row)
    for j in range(nv):
        for i in range(nu):
            bm.faces.new((verts[j][i], verts[j][i + 1], verts[j + 1][i + 1], verts[j + 1][i]))
    mesh = bpy.data.meshes.new("Backdrop")
    bm.to_mesh(mesh)
    bm.free()
    for poly in mesh.polygons:
        poly.use_smooth = True
    uv = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (co.x * 4.0, (co.y + co.z) * 4.0)
    mat, nodes, links, output = material("Checker")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = checker_cross_image()
    tex.interpolation = "Closest"
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    links.new(tex.outputs["Color"], bsdf.inputs["Color"])
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    add_mesh_object(scene, "Backdrop", mesh, mat)
    # Disc standing on its edge, leaning back a little, turned towards the camera.
    rotation = (math.radians(80.0), 0.0, math.radians(8.0))
    location = (0.0, 0.0, 0.06)
    data, hub = cd_disc(scene, "RECTANGULAR", location, rotation)
    # Clear polycarbonate rim and a thin glass hub ring.
    glass, gnodes, glinks, goutput = material("Polycarbonate")
    g = gnodes.new("ShaderNodeBsdfGlass")
    g.inputs["IOR"].default_value = 1.58
    g.inputs["Roughness"].default_value = 0.02
    glinks.new(g.outputs[0], goutput.inputs["Surface"])
    add_mesh_object(scene, "Rim", annulus_mesh("Rim", 0.06, 0.0605, rings=1), glass, location,
                    rotation)
    add_camera(scene, (0.035, -0.24, 0.085), (0.0, 0.0, 0.058), lens=50.0)
    # Big soft box above and behind the camera, reflected in the upper half of the disc, as the
    # bright window in the thread's render. Emissive mesh, invisible to the camera.
    box_mat, bnodes, blinks, boutput = material("Soft Box")
    emission = bnodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 4.0
    blinks.new(emission.outputs[0], boutput.inputs["Surface"])
    box = add_mesh_object(scene, "Soft Box", grid_mesh("Soft Box", 0.9, 0.35), box_mat,
                          (0.0, -0.55, 0.45), (math.radians(-60.0), 0.0, 0.0))
    box.visible_camera = False
    # Soft sphere lights for the backdrop rather than area lights, see scene_consistency().
    add_light(scene, "POINT", (0.0, -0.05, 0.6), 30.0, size=0.25)
    add_light(scene, "POINT", (0.25, 0.35, 0.35), 15.0, size=0.12, color=(1.0, 0.85, 0.7))
    return scene


def scene_cloth(args):
    """Waving cloth woven from crossed grooves whose direction alternates with a checker texture
    (the thread's "cloth with varying rotation driven by a Checker Texture")."""
    scene = new_scene(args, 512, 256)
    gradient_world(scene, 1.0)
    mesh = grid_mesh("Cloth", 2.0, 2.0, 96, 96)
    for v in mesh.vertices:
        x, y = v.co.x, v.co.y
        v.co.z = 0.08 * math.sin(4.0 * x + 1.0) * math.cos(3.0 * y) + 0.05 * math.sin(7.0 * y)
    for poly in mesh.polygons:
        poly.use_smooth = True
    uv = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (0.5 + 0.5 * co.x, 0.5 + 0.5 * co.y)
    mat, nodes, links, output = material("Diffraction Cloth")
    grating = diffraction_node(nodes, "2D", "RECTANGULAR", 0.12, (1.2, 1.6, 1.0), 150.0, 0.5,
                               (0.0, 0.25), color=(0.85, 0.82, 0.9))
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 12.0
    checker.inputs["Color1"].default_value = (1.0, 1.0, 1.0, 1.0)
    checker.inputs["Color2"].default_value = (0.0, 0.0, 0.0, 1.0)
    rot = nodes.new("ShaderNodeVectorMath")
    rot.operation = "MULTIPLY_ADD"
    # Rotation (x, y) = (0, 0.25) on one color, (0.125, 0.375) on the other.
    rot.inputs[1].default_value = (0.125, 0.125, 0.0)
    rot.inputs[2].default_value = (0.0, 0.25, 0.0)
    links.new(checker.outputs["Fac"], rot.inputs[0])
    links.new(rot.outputs["Vector"], grating.inputs["Rotation"])
    base = nodes.new("ShaderNodeBsdfDiffuse")
    base.inputs["Color"].default_value = (0.03, 0.03, 0.04, 1.0)
    mix = nodes.new("ShaderNodeMixShader")
    mix.inputs[0].default_value = 0.8
    links.new(base.outputs[0], mix.inputs[1])
    links.new(grating.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], output.inputs["Surface"])
    add_mesh_object(scene, "Cloth", mesh, mat)
    elevation = math.radians(40.0)
    add_camera(scene, (0.0, -3.0 * math.cos(elevation), 3.0 * math.sin(elevation)),
               (0.0, 0.0, 0.0), lens=40.0)
    add_light(scene, "POINT", (-0.6, 2.6 * math.cos(elevation), 2.6 * math.sin(elevation)),
              300.0, size=0.03)
    add_light(scene, "POINT", (1.2, 1.0, 1.6), 120.0, size=0.03, color=(1.0, 0.9, 0.8))
    return scene


def scene_sign(args):
    """Road sign with a holographic 2D grating sheet (the thread's traffic sign test, without its
    retroreflection), lit by a headlight next to the camera at night."""
    scene = new_scene(args, 512, 256)
    gradient_world(scene, 0.15)
    # Octagonal plate facing -Y.
    import bmesh
    bm = bmesh.new()
    center = bm.verts.new((0.0, 0.0, 0.0))
    ring = [bm.verts.new((0.4 * math.cos(math.pi / 8 + i * math.pi / 4), 0.0,
                          0.4 * math.sin(math.pi / 8 + i * math.pi / 4))) for i in range(8)]
    for i in range(8):
        bm.faces.new((center, ring[(i + 1) % 8], ring[i]))
    mesh = bpy.data.meshes.new("Sign")
    bm.to_mesh(mesh)
    bm.free()
    uv = mesh.uv_layers.new(name="UVMap")
    for loop in mesh.loops:
        co = mesh.vertices[loop.vertex_index].co
        uv.data[loop.index].uv = (0.5 + co.x, 0.5 + co.z)
    mat, nodes, links, output = material("Sign Sheeting")
    # Sub-micron crossed grating: near the Littrow angle its first orders go back towards a light
    # next to the camera, like the glass beads of retroreflective sheeting. A bumpy normal (the
    # beads) breaks the returns into sparkles of every color.
    grating = diffraction_node(nodes, "2D", "RECTANGULAR", 0.12, (0.72, 0.72, 1.0), 160.0, 0.5,
                               (0.0, 0.25))
    voronoi = nodes.new("ShaderNodeTexVoronoi")
    voronoi.feature = "DISTANCE_TO_EDGE"
    voronoi.inputs["Scale"].default_value = 90.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.6
    links.new(voronoi.outputs["Distance"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], grating.inputs["Normal"])
    paint = nodes.new("ShaderNodeBsdfDiffuse")
    paint.inputs["Color"].default_value = (0.45, 0.025, 0.02, 1.0)
    # White border and a white bar like a stop sign's lettering.
    coord = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coord.outputs["UV"], sep.inputs[0])
    bar = nodes.new("ShaderNodeMath")
    bar.operation = "COMPARE"
    bar.inputs[1].default_value = 0.5
    bar.inputs[2].default_value = 0.06
    links.new(sep.outputs["Y"], bar.inputs[0])
    white = nodes.new("ShaderNodeBsdfDiffuse")
    white.inputs["Color"].default_value = (0.7, 0.7, 0.7, 1.0)
    base = nodes.new("ShaderNodeMixShader")
    links.new(bar.outputs[0], base.inputs[0])
    links.new(paint.outputs[0], base.inputs[1])
    links.new(white.outputs[0], base.inputs[2])
    add = nodes.new("ShaderNodeAddShader")
    links.new(base.outputs[0], add.inputs[0])
    links.new(grating.outputs[0], add.inputs[1])
    links.new(add.outputs[0], output.inputs["Surface"])
    # Lights about 19 degrees to the right of the sign normal at -90 degrees: turned so that they
    # are 22.5 degrees off the normal, the Littrow angle of green light.
    add_mesh_object(scene, "Sign", mesh, mat, (0.0, 0.0, 1.4), (0.0, 0.0, math.radians(-3.5)))
    post = bpy.data.meshes.new("Post")
    add_mesh_object(scene, "Ground", grid_mesh("Ground", 6.0, 6.0),
                    diffuse_material("Asphalt", (0.04, 0.04, 0.045)))
    del post
    add_camera(scene, (1.2, -3.5, 1.3), (0.0, 0.0, 1.35), lens=55.0)
    add_light(scene, "SPOT", (1.0, -3.5, 1.1), 400.0, size=0.05, target=(0.0, 0.0, 1.4))
    add_light(scene, "SPOT", (1.4, -3.5, 1.1), 400.0, size=0.05, target=(0.0, 0.0, 1.4))
    return scene


def scene_sphere_uv(args):
    """Sphere with the default (UV map) tangent and a fine 1D grating: anisotropic rainbow
    highlights that follow the meridians."""
    scene = new_scene(args, 512, 256)
    gradient_world(scene, 1.0)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=0.5,
                                         location=(0.0, 0.0, 0.5))
    sphere = bpy.context.object
    bpy.ops.object.shade_smooth()
    mat, nodes, links, output = material("Grooved Sphere")
    grating = diffraction_node(nodes, "1D", "SINUSOIDAL", 0.08, 1.0, 180.0, 0.5, (0.0, 0.0))
    links.new(grating.outputs[0], output.inputs["Surface"])
    sphere.data.materials.append(mat)
    add_mesh_object(scene, "Floor", grid_mesh("Floor", 6.0, 6.0),
                    diffuse_material("Floor", (0.2, 0.2, 0.22)))
    add_camera(scene, (0.0, -2.6, 1.2), (0.0, 0.0, 0.45), lens=50.0)
    add_light(scene, "POINT", (-1.0, -1.0, 2.2), 150.0, size=0.1)
    add_light(scene, "POINT", (1.5, 0.8, 1.2), 80.0, size=0.02)
    return scene


# -------------------------------------------------------------------------------------------------
# Validation scenes


def scene_furnace(args, grating):
    """White furnace: a sphere with a white diffraction material in a uniform white world, seen
    by the camera directly. A passive BSDF can never be brighter than the world (1.0)."""
    scene = new_scene(args, 128, 256, max_bounces=64)
    uniform_world(scene, (1.0, 1.0, 1.0), 1.0)
    scene.view_settings.view_transform = "Standard"
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.0)
    sphere = bpy.context.object
    bpy.ops.object.shade_smooth()
    mat, nodes, links, output = material("Furnace")
    if grating == "1D":
        node = diffraction_node(nodes, "1D", "RECTANGULAR", 0.3, 0.8, 200.0, 0.5, (0.0, 0.0))
    else:
        node = diffraction_node(nodes, "2D", "SINUSOIDAL", 0.3, 1.2, 150.0, 0.5, (0.0, 0.25))
    links.new(node.outputs[0], output.inputs["Surface"])
    sphere.data.materials.append(mat)
    add_camera(scene, (0.0, -4.0, 0.0), (0.0, 0.0, 0.0), lens=50.0)
    return scene


def scene_consistency(args):
    """Small, fast scene with direct light and indirect diffraction caustics on a diffuse floor:
    a grating plate reflects a small light onto the floor. Integrators must agree."""
    scene = new_scene(args, 128, 512)
    uniform_world(scene, (0.05, 0.05, 0.06), 1.0)
    scene.view_settings.view_transform = "Standard"
    add_mesh_object(scene, "Floor", grid_mesh("Floor", 4.0, 4.0),
                    diffuse_material("Floor", (0.7, 0.7, 0.7)))
    mat, nodes, links, output = material("Plate")
    node = diffraction_node(nodes, "1D", "RECTANGULAR", 0.25, 1.0, 160.0, 0.5, (0.0, 0.0))
    links.new(node.outputs[0], output.inputs["Surface"])
    add_mesh_object(scene, "Plate", grid_mesh("Plate", 0.8, 0.8), mat, (0.0, 0.4, 0.5),
                    (math.radians(-60.0), 0.0, 0.0))
    add_camera(scene, (0.0, -3.0, 2.0), (0.0, 0.2, 0.3), lens=35.0)
    # A soft sphere light. (An area light with this scene's 1 mm clip start once hit a BDPT bug,
    # fixed since: see doc/cycles_bdpt_light_fixes.md.)
    add_light(scene, "POINT", (0.0, -1.0, 2.0), 200.0, size=0.15)
    return scene


DEMO = {
    "cd_rectangular": lambda a: scene_cd(a, "RECTANGULAR"),
    "cd_sinusoidal": lambda a: scene_cd(a, "SINUSOIDAL"),
    "grid_2d": scene_grid_2d,
    "cd_hero": scene_cd_hero,
    "cloth": scene_cloth,
    "sign": scene_sign,
    "sphere_uv": scene_sphere_uv,
}
VALIDATION = {
    "furnace_1d": lambda a: scene_furnace(a, "1D"),
    "furnace_2d": lambda a: scene_furnace(a, "2D"),
    "consistency": scene_consistency,
}


def main():
    global CONTROL_GLOSSY
    args = parse_args()
    CONTROL_GLOSSY = args.control_glossy
    args.output.mkdir(parents=True, exist_ok=True)
    scenes = dict(DEMO, **VALIDATION)
    if args.scenes == "demo":
        names = list(DEMO)
    elif args.scenes == "validation":
        names = list(VALIDATION)
    elif args.scenes == "all":
        names = list(scenes)
    else:
        names = args.scenes.split(",")
    report = {}
    ext = "exr" if args.format == "EXR" else "png"
    for name in names:
        scene = scenes[name](args)
        enable_device(args)
        scene.render.filepath = str(args.output / f"{name}.{ext}")
        if args.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=str(args.output / f"{name}.blend"))
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        entry = {"first_render_seconds": time.perf_counter() - start}
        if args.timing:
            start = time.perf_counter()
            bpy.ops.render.render(write_still=False)
            entry["render_seconds"] = time.perf_counter() - start
        report[name] = entry
        print("SCENE", name, json.dumps(entry), flush=True)
    (args.output / "timings.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
