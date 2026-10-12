# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Demo scenes for the volume index of refraction of Cycles: heat haze, a mirage, a lens.

* `hot_plate`: turbulent hot air above a glowing plate in front of a patterned wall. No smoke, no
  flame: only the refraction shows the air. Volume IOR node in Temperature mode with a
  procedural temperature. Animated: the turbulence rises with the frame.
* `road_mirage`: an inferior mirage, the "water" on a hot road. A thin layer of hot air above
  the asphalt, colder with the height, reflects the sky and the far objects at grazing angles.
* `fire_vdb`: a fire from voxel grids (`temperature` in kelvin, written by this script). The
  Principled Volume renders the flame and, with its Heat Refraction input, the shimmer above it.
* `hot_air_vdb`: the same grid above a hot stone, with nothing but the Volume IOR node reading
  the grid by name.
* `luneburg_lens`: a glass-like ball whose index falls from the square root of two at the
  center to one at the surface. No surface reflects or refracts: the volume does all of it.

  blender -b --factory-startup --python tests/python/cycles_volume_ior_demos.py -- \
      --output DIRECTORY [--scenes a,b] [--device CPU|METAL] [--integrator pt|bdpt] \
      [--samples N] [--resolution WIDTH] [--strength-scale X] [--no-refraction] [--render]
      [--frame N]

Saves `<scene>.blend` (and the voxel grids) into the directory, and with `--render` a PNG of
each next to it.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--device", default="METAL", choices=("METAL", "CPU"))
    parser.add_argument("--integrator", default="pt", choices=("pt", "bdpt"))
    parser.add_argument("--guiding", action="store_true")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--resolution", type=int, default=960)
    parser.add_argument("--strength-scale", type=float, default=1.0,
                        help="Scale of the exaggeration of the refraction, 0 switches it off")
    parser.add_argument("--no-refraction", action="store_true",
                        help="Switch the refraction off in the render settings")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-dir", type=Path, default=None)
    parser.add_argument("--frame", type=int, default=1)
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------------
# Helpers


def new_scene(args, aspect=16.0 / 9.0):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if args.device == "METAL":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == "METAL"
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = int(round(args.resolution / aspect))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.frame_set(args.frame)
    cycles = scene.cycles
    cycles.device = "GPU" if args.device == "METAL" else "CPU"
    cycles.samples = args.samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.max_bounces = 8
    cycles.transparent_max_bounces = 16
    cycles.use_bidirectional_path_tracing = args.integrator == "bdpt"
    cycles.use_guiding = args.guiding
    cycles.use_volume_ior = not args.no_refraction
    return scene


def add_sky(scene, strength=1.0, sun_elevation=35.0, sun_rotation=140.0):
    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    sky = nodes.new("ShaderNodeTexSky")
    for name, value in (("sun_elevation", math.radians(sun_elevation)),
                        ("sun_rotation", math.radians(sun_rotation))):
        if hasattr(sky, name):
            setattr(sky, name, value)
    links.new(sky.outputs["Color"], nodes["Background"].inputs["Color"])
    # The sky is in physical units: bring a sunny day to the range of the lamps of the scenes.
    nodes["Background"].inputs["Strength"].default_value = 0.04 * strength
    scene.world = world


def add_camera(scene, location, look_at, lens=50.0):
    data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", data)
    scene.collection.objects.link(camera)
    camera.location = location
    direction = Vector(look_at) - Vector(location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    data.lens = lens
    data.clip_start = 0.05
    data.clip_end = 20000.0
    scene.camera = camera
    return camera


def new_material(name):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    return material, nodes, material.node_tree.links, nodes["Material Output"]


def principled(name, color, roughness=0.6, metallic=0.0, emission=None, emission_strength=0.0):
    material, nodes, links, output = new_material(name)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission_strength
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def checker_material(name, scale, color1, color2, roughness=0.7):
    material, nodes, links, output = new_material(name)
    coord = nodes.new("ShaderNodeTexCoord")
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = scale
    checker.inputs["Color1"].default_value = (*color1, 1.0)
    checker.inputs["Color2"].default_value = (*color2, 1.0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = roughness
    links.new(coord.outputs["Object"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def add_mesh(op, material, smooth=False, **kwargs):
    op(**kwargs)
    obj = bpy.context.object
    obj.data.materials.append(material)
    if smooth:
        bpy.ops.object.shade_smooth()
    return obj


def add_box(material, location, size, name=None):
    obj = add_mesh(bpy.ops.mesh.primitive_cube_add, material, size=1.0, location=location)
    obj.scale = size
    if name:
        obj.name = name
    return obj


def math_node(nodes, links, operation, a, b=None, c=None, clamp=False):
    """Math node with sockets or constants as inputs. Returns its output socket."""
    node = nodes.new("ShaderNodeMath")
    node.operation = operation
    node.use_clamp = clamp
    for index, value in enumerate((a, b, c)):
        if value is None:
            continue
        if isinstance(value, (int, float)):
            node.inputs[index].default_value = value
        else:
            links.new(value, node.inputs[index])
    return node.outputs[0]


def frame_driver(socket_or_property, expression, path="default_value"):
    """Drives a value by the frame number, `frame` in the expression."""
    driver = socket_or_property.driver_add(path).driver
    driver.type = "SCRIPTED"
    driver.expression = expression


# ------------------------------------------------------------------------------------------------
# Hot plate


def plume_temperature(nodes, links, axis_xy, base_z, height, radius, rise, peak, scale,
                      detail=3.0):
    """Temperature rise of a turbulent plume above (axis_xy, base_z): socket in kelvin.

    The column widens and cools with the height, its turbulence rises with the frame and fades
    towards the sides, so that the air at the faces of the domain is as cold as outside."""
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geometry.outputs["Position"], separate.inputs[0])
    dx = math_node(nodes, links, "SUBTRACT", separate.outputs["X"], axis_xy[0])
    dy = math_node(nodes, links, "SUBTRACT", separate.outputs["Y"], axis_xy[1])
    r2 = math_node(nodes, links, "ADD", math_node(nodes, links, "MULTIPLY", dx, dx),
                   math_node(nodes, links, "MULTIPLY", dy, dy))
    r = math_node(nodes, links, "SQRT", r2)
    z = math_node(nodes, links, "SUBTRACT", separate.outputs["Z"], base_z)
    # Height as 0..1, and the radius of the column there.
    z01 = math_node(nodes, links, "DIVIDE", z, height, clamp=True)
    width = math_node(nodes, links, "MULTIPLY_ADD", z01, radius * 1.2, radius)
    radial = nodes.new("ShaderNodeMapRange")
    radial.interpolation_type = "SMOOTHERSTEP"
    radial.inputs["From Min"].default_value = 0.0
    radial.inputs["From Max"].default_value = 1.0
    radial.inputs["To Min"].default_value = 1.0
    radial.inputs["To Max"].default_value = 0.0
    links.new(math_node(nodes, links, "DIVIDE", r, width), radial.inputs["Value"])
    # Cools with the height, and nothing at the top and the bottom of the domain.
    cooling = math_node(nodes, links, "POWER", math_node(nodes, links, "SUBTRACT", 1.0, z01), 1.5)
    bottom = nodes.new("ShaderNodeMapRange")
    bottom.interpolation_type = "SMOOTHSTEP"
    bottom.inputs["From Min"].default_value = 0.0
    bottom.inputs["From Max"].default_value = 0.06
    links.new(z01, bottom.inputs["Value"])

    # Turbulence that rises: the lookup moves down with the time, a little faster in the middle.
    noise = nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = scale
    noise.inputs["Detail"].default_value = detail
    noise.inputs["Roughness"].default_value = 0.55
    noise.inputs["Distortion"].default_value = 0.6
    offset = nodes.new("ShaderNodeCombineXYZ")
    lift = nodes.new("ShaderNodeValue")
    lift.label = "Rise"
    frame_driver(lift.outputs[0], "-frame * %f" % (rise / 24.0))
    links.new(lift.outputs[0], offset.inputs["Z"])
    position = nodes.new("ShaderNodeVectorMath")
    position.operation = "ADD"
    links.new(geometry.outputs["Position"], position.inputs[0])
    links.new(offset.outputs[0], position.inputs[1])
    links.new(position.outputs[0], noise.inputs["Vector"])
    frame_driver(noise.inputs["W"], "frame * %f" % (0.25 / 24.0))
    # Noise is around a half: sharpen it into wisps.
    wisps = nodes.new("ShaderNodeMapRange")
    wisps.interpolation_type = "SMOOTHSTEP"
    wisps.inputs["From Min"].default_value = 0.35
    wisps.inputs["From Max"].default_value = 0.75
    links.new(noise.outputs["Fac"], wisps.inputs["Value"])

    shape = math_node(nodes, links, "MULTIPLY", radial.outputs["Result"], cooling)
    shape = math_node(nodes, links, "MULTIPLY", shape, bottom.outputs["Result"])
    turbulent = math_node(nodes, links, "MULTIPLY", shape,
                          math_node(nodes, links, "MULTIPLY_ADD", wisps.outputs["Result"], 0.85,
                                    0.15))
    return math_node(nodes, links, "MULTIPLY", turbulent, peak)


def scene_hot_plate(args):
    scene = new_scene(args)
    add_sky(scene, strength=0.35, sun_elevation=50.0, sun_rotation=200.0)
    add_camera(scene, (0.0, -4.2, 1.25), (0.0, 0.0, 1.3), lens=42.0)

    floor = checker_material("Floor", 6.0, (0.55, 0.52, 0.48), (0.42, 0.40, 0.37), 0.5)
    add_mesh(bpy.ops.mesh.primitive_plane_add, floor, size=14.0)
    wall = checker_material("Wall", 22.0, (0.85, 0.82, 0.75), (0.12, 0.22, 0.36))
    add_mesh(bpy.ops.mesh.primitive_plane_add, wall, size=14.0, location=(0.0, 2.6, 7.0),
             rotation=(math.radians(90.0), 0.0, 0.0))
    # Things with edges behind the air.
    for index, x in enumerate((-1.5, -0.75, 0.0, 0.75, 1.5)):
        color = ((0.8, 0.25, 0.15), (0.9, 0.7, 0.2), (0.2, 0.55, 0.3), (0.2, 0.4, 0.75),
                 (0.6, 0.3, 0.6))[index]
        add_mesh(bpy.ops.mesh.primitive_cylinder_add, principled("Pole %d" % index, color, 0.4),
                 radius=0.035, depth=2.6, location=(x, 1.9, 1.3))

    # The stove and its plate.
    add_box(principled("Stove", (0.08, 0.08, 0.09), 0.35, 1.0), (0.0, 0.0, 0.35),
            (1.3, 0.75, 0.7))
    add_box(principled("Plate", (0.02, 0.02, 0.02), 0.5, emission=(1.0, 0.28, 0.06),
                       emission_strength=6.0), (0.0, 0.0, 0.715), (1.1, 0.6, 0.03))

    light_data = bpy.data.lights.new("Key", "AREA")
    light_data.energy = 900.0
    light_data.size = 2.0
    light = bpy.data.objects.new("Key", light_data)
    light.location = (-2.5, -3.0, 4.0)
    light.rotation_euler = (Vector((0.0, 0.5, 1.0)) - Vector(light.location)).to_track_quat(
        "-Z", "Y").to_euler()
    scene.collection.objects.link(light)

    # Hot air: nothing to see but its refraction.
    material, nodes, links, output = new_material("Hot Air")
    ior = nodes.new("ShaderNodeVolumeIOR")
    ior.mode = "TEMPERATURE"
    ior.step_size = 0.03
    # The air right above a plate of 700 K is about 250 K above the room. The refraction it
    # causes is real at a strength of 1 and hard to see: this is 40 times that.
    ior.inputs["Strength"].default_value = 40.0 * args.strength_scale
    rise = plume_temperature(nodes, links, (0.0, 0.0), 0.73, 2.1, 0.42, 0.55, 250.0, 3.2)
    kelvin = math_node(nodes, links, "ADD", rise, 293.15)
    links.new(kelvin, ior.inputs["Temperature"])
    links.new(ior.outputs[0], output.inputs["Volume"])
    add_box(material, (0.0, 0.0, 1.8), (2.4, 2.0, 2.14), name="Hot Air")
    return scene


# ------------------------------------------------------------------------------------------------
# Road mirage


def scene_road_mirage(args):
    scene = new_scene(args)
    add_sky(scene, strength=1.0, sun_elevation=28.0, sun_rotation=150.0)
    add_camera(scene, (0.6, -2.0, 1.15), (0.0, 300.0, 1.0), lens=420.0)

    add_mesh(bpy.ops.mesh.primitive_plane_add, principled("Sand", (0.55, 0.42, 0.28), 0.9),
             size=20000.0, location=(0.0, 5000.0, -0.02))
    road = add_box(principled("Asphalt", (0.035, 0.035, 0.04), 0.85), (0.0, 600.0, 0.0),
                   (8.0, 1200.0, 0.02))
    road.name = "Road"
    for i in range(60):
        add_box(principled("Line", (0.8, 0.8, 0.75), 0.6) if i == 0 else
                bpy.data.materials["Line"], (0.0, 6.0 + 12.0 * i, 0.012), (0.16, 4.0, 0.004))
    pole = principled("Pole", (0.25, 0.2, 0.15), 0.7)
    for i in range(24):
        add_mesh(bpy.ops.mesh.primitive_cylinder_add, pole, radius=0.12, depth=7.0,
                 location=(-6.5, 30.0 + 40.0 * i, 3.5))
    # A truck and a car far down the road, and hills behind.
    add_box(principled("Truck", (0.75, 0.1, 0.08), 0.35), (-1.8, 330.0, 1.9), (2.5, 9.0, 3.4))
    add_box(principled("Car", (0.1, 0.25, 0.7), 0.3), (1.8, 240.0, 0.75), (1.8, 4.2, 1.4))
    hills = principled("Hills", (0.38, 0.3, 0.24), 0.9)
    for x, y, radius, height in ((-900.0, 6000.0, 1200.0, 70.0), (700.0, 6500.0, 1500.0, 95.0),
                                 (-100.0, 7500.0, 2000.0, 50.0)):
        add_mesh(bpy.ops.mesh.primitive_cone_add, hills, radius1=radius, depth=height,
                 vertices=64, location=(x, y, 0.5 * height - 2.0))

    # Hot air above the asphalt: 35 K hotter at the surface, falling off within centimeters.
    material, nodes, links, output = new_material("Hot Air")
    ior = nodes.new("ShaderNodeVolumeIOR")
    ior.mode = "TEMPERATURE"
    # Rays graze the layer: long steps along the road are short ones across it.
    ior.step_size = 0.5
    # The physical strength of 1 shows the mirage beyond 200 m from this eye height.
    ior.inputs["Strength"].default_value = 3.0 * args.strength_scale
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    links.new(geometry.outputs["Position"], separate.inputs[0])
    height = math_node(nodes, links, "SUBTRACT", separate.outputs["Z"], 0.01)
    falloff = math_node(nodes, links, "EXPONENT", math_node(nodes, links, "MULTIPLY", height,
                                                         -1.0 / 0.04))
    # Slow, weak waves along the road keep the mirage from being a perfect mirror.
    noise = nodes.new("ShaderNodeTexNoise")
    noise.noise_dimensions = "4D"
    noise.inputs["Scale"].default_value = 0.08
    noise.inputs["Detail"].default_value = 1.0
    frame_driver(noise.inputs["W"], "frame * %f" % (0.5 / 24.0))
    links.new(geometry.outputs["Position"], noise.inputs["Vector"])
    wave = math_node(nodes, links, "MULTIPLY_ADD", noise.outputs["Fac"], 0.5, 0.75)
    rise = math_node(nodes, links, "MULTIPLY", math_node(nodes, links, "MULTIPLY", falloff, wave),
                     35.0)
    links.new(math_node(nodes, links, "ADD", rise, 293.15), ior.inputs["Temperature"])
    links.new(ior.outputs[0], output.inputs["Volume"])
    add_box(material, (0.0, 600.0, 0.26), (8.0, 1190.0, 0.5), name="Hot Air")
    return scene


# ------------------------------------------------------------------------------------------------
# Voxel grids


def write_fire_grids(path, fire):
    """Writes the grids `temperature` (kelvin) and `density` of a plume of hot gas.

    A buoyant plume: a hot core that widens and cools with the height, bent and broken up by a
    few sine waves. Not a simulation, but smooth and plausible."""
    import openvdb

    voxel = 0.0125
    shape = (96, 96, 200)
    x = (np.arange(shape[0]) + 0.5 - 0.5 * shape[0]) * voxel
    y = (np.arange(shape[1]) + 0.5 - 0.5 * shape[1]) * voxel
    z = (np.arange(shape[2]) + 0.5) * voxel
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    rng = np.random.default_rng(7)

    # Sway of the axis and wrinkles of the column.
    sway_x = 0.05 * Z * np.sin(4.1 * Z + 0.6) + 0.02 * np.sin(11.0 * Z + 2.0)
    sway_y = 0.05 * Z * np.cos(3.3 * Z + 1.9) + 0.02 * np.sin(9.0 * Z + 0.3)
    wrinkle = np.zeros(shape, dtype=np.float64)
    for _ in range(14):
        k = rng.normal(0.0, 1.0, 3) * np.array((26.0, 26.0, 14.0))
        wrinkle += np.sin(k[0] * X + k[1] * Y + k[2] * Z + rng.uniform(0.0, 6.28))
    wrinkle /= math.sqrt(14.0)

    radius = np.hypot(X - sway_x, Y - sway_y) * (1.0 + 0.18 * wrinkle)
    width = 0.1 + 0.1 * Z
    column = np.exp(-0.5 * (radius / width) ** 2)
    # The gas spreads over a wider column as it rises: its excess temperature falls accordingly.
    excess = (0.1 / width) ** 2.2 * column * (1.0 + 0.25 * wrinkle)
    # Nothing leaves the grid hotter than the room.
    edge = np.clip(1.0 - np.maximum(np.abs(X), np.abs(Y)) / (0.5 * shape[0] * voxel - 0.05), 0.0,
                   1.0)
    top = np.clip((z[-1] - Z) / 0.3, 0.0, 1.0)
    excess *= np.clip(edge * 6.0, 0.0, 1.0) * top

    if fire:
        # The flame: a core of about the same temperature, not a thin hot line.
        excess = np.clip(excess * 2.5, 0.0, 1.0)
    peak = 1500.0 if fire else 380.0
    temperature = 293.15 + peak * np.clip(excess, 0.0, 1.2)
    grids = []
    grid = openvdb.FloatGrid(293.15)
    grid.name = "temperature"
    grid.copyFromArray(temperature.astype(np.float32), tolerance=0.5)
    grid.transform = openvdb.createLinearTransform(voxel)
    grid.transform.postTranslate((x[0], y[0], z[0]))
    grids.append(grid)
    if fire:
        # A little soot above the flame.
        soot = np.clip(excess - 0.05, 0.0, 0.5) * np.clip((Z - 0.25) / 0.3, 0.0, 1.0)
        grid = openvdb.FloatGrid(0.0)
        grid.name = "density"
        grid.copyFromArray(soot.astype(np.float32), tolerance=1e-3)
        grid.transform = openvdb.createLinearTransform(voxel)
        grid.transform.postTranslate((x[0], y[0], z[0]))
        grids.append(grid)
    openvdb.write(str(path), grids=grids)


def add_volume_object(scene, path, name, location):
    volume = bpy.data.volumes.new(name)
    volume.filepath = "//" + path.name
    obj = bpy.data.objects.new(name, volume)
    obj.location = location
    scene.collection.objects.link(obj)
    return obj


def campsite(args, scene):
    add_sky(scene, strength=0.12, sun_elevation=4.0, sun_rotation=300.0)
    add_camera(scene, (0.0, -4.0, 0.7), (0.0, 0.0, 0.85), lens=35.0)
    add_mesh(bpy.ops.mesh.primitive_plane_add,
             checker_material("Ground", 30.0, (0.3, 0.26, 0.2), (0.22, 0.2, 0.16), 0.9),
             size=30.0)
    # A fence and trees behind the fire, to see through the air above it.
    plank = principled("Plank", (0.5, 0.36, 0.2), 0.7)
    for i in range(21):
        add_box(plank, (-2.5 + 0.25 * i, 2.0, 0.9), (0.17, 0.04, 1.8))
    add_box(plank, (0.0, 1.96, 1.3), (5.4, 0.04, 0.1))
    add_box(plank, (0.0, 1.96, 0.5), (5.4, 0.04, 0.1))
    leaves = principled("Leaves", (0.08, 0.25, 0.08), 0.8)
    trunk = principled("Trunk", (0.2, 0.13, 0.08), 0.8)
    for x, y, h in ((-2.2, 4.0, 3.2), (-0.6, 4.6, 3.8), (1.0, 4.2, 3.4), (2.4, 4.8, 4.0)):
        add_mesh(bpy.ops.mesh.primitive_cylinder_add, trunk, radius=0.09, depth=h,
                 location=(x, y, 0.5 * h))
        add_mesh(bpy.ops.mesh.primitive_cone_add, leaves, radius1=0.9, depth=2.4,
                 location=(x, y, h))
    light_data = bpy.data.lights.new("Moon", "AREA")
    light_data.energy = 400.0
    light_data.size = 3.0
    light_data.color = (0.7, 0.8, 1.0)
    light = bpy.data.objects.new("Moon", light_data)
    light.location = (2.0, -2.0, 5.0)
    light.rotation_euler = (Vector((0.0, 1.0, 0.8)) - Vector(light.location)).to_track_quat(
        "-Z", "Y").to_euler()
    scene.collection.objects.link(light)


def scene_fire_vdb(args):
    scene = new_scene(args)
    campsite(args, scene)
    path = args.output / "fire.vdb"
    write_fire_grids(path, fire=True)
    wood = principled("Wood", (0.12, 0.07, 0.04), 0.9, emission=(1.0, 0.2, 0.03),
                      emission_strength=1.5)
    for angle in range(0, 180, 45):
        add_mesh(bpy.ops.mesh.primitive_cylinder_add, wood, radius=0.045, depth=0.6,
                 location=(0.0, 0.0, 0.06),
                 rotation=(math.radians(90.0), 0.0, math.radians(angle)))

    fire = add_volume_object(scene, path, "Fire", (0.0, 0.0, 0.05))
    material, nodes, links, output = new_material("Fire")
    volume = nodes.new("ShaderNodeVolumePrincipled")
    # The grid is in kelvin already.
    volume.inputs["Temperature"].default_value = 1.0
    volume.inputs["Blackbody Intensity"].default_value = 1.0
    # A flame is brighter than a black body of its soot at this density.
    volume.inputs["Blackbody Tint"].default_value = (25.0, 25.0, 25.0, 1.0)
    volume.inputs["Density"].default_value = 1.5
    volume.inputs["Color"].default_value = (0.05, 0.05, 0.05, 1.0)
    # Hot air is real at 1, and exaggerated 15 times here.
    volume.inputs["Heat Refraction"].default_value = 15.0 * args.strength_scale
    links.new(volume.outputs[0], output.inputs["Volume"])
    fire.data.materials.append(material)
    return scene


def scene_hot_air_vdb(args):
    scene = new_scene(args)
    campsite(args, scene)
    path = args.output / "hot_air.vdb"
    write_fire_grids(path, fire=False)
    add_mesh(bpy.ops.mesh.primitive_ico_sphere_add,
             principled("Stone", (0.1, 0.09, 0.08), 0.8, emission=(1.0, 0.25, 0.05),
                        emission_strength=0.6), smooth=True, subdivisions=3, radius=0.22,
             location=(0.0, 0.0, 0.05))
    air = add_volume_object(scene, path, "Hot Air", (0.0, 0.0, 0.05))
    material, nodes, links, output = new_material("Hot Air")
    ior = nodes.new("ShaderNodeVolumeIOR")
    ior.mode = "TEMPERATURE"
    # The grid, in kelvin, scales this input.
    ior.inputs["Temperature"].default_value = 1.0
    ior.inputs["Attribute"].default_value = "temperature"
    ior.inputs["Strength"].default_value = 40.0 * args.strength_scale
    links.new(ior.outputs[0], output.inputs["Volume"])
    air.data.materials.append(material)
    return scene


# ------------------------------------------------------------------------------------------------
# Luneburg lens


def scene_luneburg_lens(args):
    scene = new_scene(args)
    add_sky(scene, strength=0.8, sun_elevation=40.0, sun_rotation=120.0)
    add_camera(scene, (0.0, -6.0, 1.6), (0.0, 0.0, 1.05), lens=55.0)
    add_mesh(bpy.ops.mesh.primitive_plane_add,
             checker_material("Floor", 24.0, (0.8, 0.8, 0.78), (0.15, 0.17, 0.2), 0.4),
             size=40.0)
    for index, (x, y) in enumerate(((-2.2, 3.0), (-0.9, 4.0), (0.6, 3.4), (2.0, 4.4))):
        color = ((0.8, 0.2, 0.15), (0.9, 0.65, 0.15), (0.15, 0.5, 0.25), (0.2, 0.35, 0.8))[index]
        add_mesh(bpy.ops.mesh.primitive_cylinder_add, principled("Column %d" % index, color, 0.3),
                 radius=0.25, depth=2.4, location=(x, y, 1.2))

    material, nodes, links, output = new_material("Lens")
    ior = nodes.new("ShaderNodeVolumeIOR")
    ior.mode = "IOR"
    ior.step_size = 0.02
    ior.inputs["Strength"].default_value = args.strength_scale
    coord = nodes.new("ShaderNodeTexCoord")
    length = nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    links.new(coord.outputs["Object"], length.inputs[0])
    r2 = math_node(nodes, links, "MULTIPLY", length.outputs["Value"], length.outputs["Value"])
    index = math_node(nodes, links, "SQRT", math_node(nodes, links, "SUBTRACT", 2.0, r2,
                                                      clamp=False))
    links.new(index, ior.inputs["IOR"])
    links.new(ior.outputs[0], output.inputs["Volume"])
    add_mesh(bpy.ops.mesh.primitive_ico_sphere_add, material, subdivisions=6, radius=1.0,
             location=(0.0, 0.0, 1.05))
    return scene


SCENES = {
    "hot_plate": scene_hot_plate,
    "road_mirage": scene_road_mirage,
    "fire_vdb": scene_fire_vdb,
    "hot_air_vdb": scene_hot_air_vdb,
    "luneburg_lens": scene_luneburg_lens,
}


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    render_dir = args.render_dir or args.output
    render_dir.mkdir(parents=True, exist_ok=True)
    names = list(SCENES) if args.scenes == "all" else args.scenes.split(",")
    for name in names:
        scene = SCENES[name](args)
        scene.render.filepath = "//" + name + ".png"
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output / (name + ".blend")))
        if args.render:
            import time
            scene.render.filepath = str(render_dir / (name + ".png"))
            start = time.time()
            bpy.ops.render.render(write_still=True)
            print("VOLUME_IOR_DEMO %-14s rendered in %6.1f s" % (name, time.time() - start))


if __name__ == "__main__":
    main()
