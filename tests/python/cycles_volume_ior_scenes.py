# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Volume index of refraction scenes for Cycles: validation against closed forms, and pairs.

Scenes with a closed form render and then check themselves:

* `gradient`: an orthographic camera looks through a slab whose index grows linearly with the
  height at a backdrop that emits its own height. Rays in such a medium are catenaries, so the
  height every pixel must show is known. `gradient_absorb` adds absorption to the slab, which
  tests the path through media that are integrated one straight segment at a time, and
  `gradient_vdb` takes the same field from a voxel grid.
* `luneburg`: a Luneburg lens, a sphere with `n = sqrt(2 - r^2)`, focuses parallel rays onto the
  point of its surface opposite to them. A small emitter there must light up the whole lens.
* `caustic`: sunlight from a mirror through the same lens onto a wall half a radius behind it.
  The lens is hidden from the camera, which looks at the wall along the axis: what it sees was
  bent on the way from the light. The irradiance of the wall at the distance `r` from the axis
  is `R^2 d^2 / (d^2 + r^2)^2` of that of the mirror next to the lens. Path tracing cannot find
  these paths, so this is a test of light subpaths and photons only.
* `furnace`, `furnace_scatter`: a white diffuse sphere, in strongly refracting air with and
  without white fog, under a uniform white sky. Every pixel must be the sky.

The others are rendered for comparison with cycles_volume_ior_compare.py:

* `<name>_off` and `<name>_zero`: the same scene without the node and with a strength of zero
  that is linked, so that the feature runs. They must be the same image.
* `segments_fog` against `segments_fog_off`: fog with a refraction too weak to see, which splits
  every ray into segments. It must match the fog alone up to noise.
* `room`, `room_fog`: a lit interior with a column of hot air at 300 times its physical
  refraction, to compare devices. Integrators differ here by design: connections to lights are
  straight, so the focusing of light by the air (the shadowgraph on the walls) is rendered only
  as far as paths that are sampled through the air carry it, which bidirectional methods have
  more of.
* `room_weak`, `room_weak_fog`: the same at 30 times, where that focusing is below the noise:
  all integrators must agree.

  blender -b --factory-startup --python tests/python/cycles_volume_ior_scenes.py -- \
      --output DIRECTORY [--device CPU|METAL] [--integrator pt|bdpt|photon|vcm] [--guiding] \
      [--osl] [--scenes a,b,c] [--samples N] [--resolution N] [--step-rate X] [--save-blend]

Prints a line `VOLUME_IOR <scene> ... PASSED|FAILED` for every scene that checks itself and exits
non-zero if one failed.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

# Gradient of the index of refraction of the `gradient` scenes, per meter.
GRADIENT = 0.05
SLAB_LENGTH = 2.0
BACKDROP_DISTANCE = 4.0
ABSORPTION = 0.3


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="CPU", choices=("METAL", "CPU"))
    parser.add_argument("--integrator", default="pt", choices=("pt", "bdpt", "photon", "vcm"))
    parser.add_argument("--guiding", action="store_true")
    parser.add_argument("--osl", action="store_true", help="Open Shading Language (CPU only)")
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--step-rate", type=float, default=1.0)
    parser.add_argument("--ray-marching", action="store_true", help="Biased volume rendering")
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------------
# Scene setup


def new_scene(args, world_strength=0.0):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    enable_device(args)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.view_settings.view_transform = "Standard"
    cycles = scene.cycles
    cycles.device = "GPU" if args.device == "METAL" else "CPU"
    cycles.samples = args.samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.seed = args.seed
    cycles.max_bounces = 32
    cycles.diffuse_bounces = 32
    cycles.glossy_bounces = 32
    cycles.transmission_bounces = 32
    cycles.transparent_max_bounces = 32
    cycles.volume_bounces = 32
    cycles.sample_clamp_direct = 0.0
    cycles.sample_clamp_indirect = 0.0
    cycles.blur_glossy = 0.0
    cycles.volume_biased = args.ray_marching
    cycles.volume_ior_step_rate = args.step_rate
    if args.integrator == "bdpt":
        cycles.use_bidirectional_path_tracing = True
        cycles.bdpt_light_paths = 65536
    elif args.integrator == "photon":
        cycles.use_photon_mapping = True
        cycles.photon_count = 65536
    elif args.integrator == "vcm":
        cycles.use_vertex_merging = True
    cycles.use_guiding = args.guiding
    cycles.shading_system = args.osl and args.device == "CPU"

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    background.inputs["Strength"].default_value = world_strength
    scene.world = world
    return scene


def enable_device(args):
    if args.device != "METAL":
        return
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "METAL"


def add_camera(scene, location, look_at, ortho_scale=None, lens=50.0):
    data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", data)
    scene.collection.objects.link(camera)
    camera.location = location
    direction = Vector(look_at) - Vector(location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    if ortho_scale is not None:
        data.type = "ORTHO"
        data.ortho_scale = ortho_scale
    else:
        data.lens = lens
    data.clip_start = 0.01
    data.clip_end = 1000.0
    scene.camera = camera
    return camera


def add_object(scene, name, mesh_op, material, **kwargs):
    mesh_op(**kwargs)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def new_material(name):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    for node in list(nodes):
        if node.type != "OUTPUT_MATERIAL":
            nodes.remove(node)
    return material, nodes, material.node_tree.links, nodes["Material Output"]


def diffuse_material(name, color):
    material, nodes, links, output = new_material(name)
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (*color, 1.0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def emission_material(name, color=(1.0, 1.0, 1.0), strength=1.0):
    material, nodes, links, output = new_material(name)
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = strength
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def height_material(name, scale, offset):
    """Emits `scale * z + offset` of the world position."""
    material, nodes, links, output = new_material(name)
    geometry = nodes.new("ShaderNodeNewGeometry")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    value = nodes.new("ShaderNodeMath")
    value.operation = "MULTIPLY_ADD"
    value.inputs[1].default_value = scale
    value.inputs[2].default_value = offset
    emission = nodes.new("ShaderNodeEmission")
    links.new(geometry.outputs["Position"], separate.inputs[0])
    links.new(separate.outputs["Z"], value.inputs[0])
    links.new(value.outputs[0], emission.inputs["Strength"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def add_volume_ior(nodes, mode="IOR", strength=1.0, step_size=0.0):
    node = nodes.new("ShaderNodeVolumeIOR")
    node.mode = mode
    node.step_size = step_size
    node.inputs["Strength"].default_value = strength
    return node


def unfoldable_zero(nodes, links):
    """Zero that the shader compiler cannot tell: the node stays, and so does the feature."""
    noise = nodes.new("ShaderNodeTexNoise")
    minimum = nodes.new("ShaderNodeMath")
    minimum.operation = "MINIMUM"
    minimum.inputs[1].default_value = 0.0
    links.new(noise.outputs["Fac"], minimum.inputs[0])
    return minimum.outputs[0]


def combine(nodes, links, output, shaders):
    """Adds the shaders up into the volume output."""
    shaders = [shader for shader in shaders if shader is not None]
    result = shaders[0]
    for shader in shaders[1:]:
        add = nodes.new("ShaderNodeAddShader")
        links.new(result, add.inputs[0])
        links.new(shader, add.inputs[1])
        result = add.outputs[0]
    links.new(result, output.inputs["Volume"])


def linear_ior_material(name, gradient, absorption=0.0, attribute=""):
    """Index of refraction `1 + gradient * z`, from the position or from a voxel grid of z."""
    material, nodes, links, output = new_material(name)
    ior = add_volume_ior(nodes)
    if attribute:
        # The attribute scales the refractivity `IOR - 1`.
        ior.inputs["IOR"].default_value = 1.0 + gradient
        ior.inputs["Attribute"].default_value = attribute
    else:
        geometry = nodes.new("ShaderNodeNewGeometry")
        separate = nodes.new("ShaderNodeSeparateXYZ")
        value = nodes.new("ShaderNodeMath")
        value.operation = "MULTIPLY_ADD"
        value.inputs[1].default_value = gradient
        value.inputs[2].default_value = 1.0
        links.new(geometry.outputs["Position"], separate.inputs[0])
        links.new(separate.outputs["Z"], value.inputs[0])
        links.new(value.outputs[0], ior.inputs["IOR"])
    absorb = None
    if absorption:
        node = nodes.new("ShaderNodeVolumeAbsorption")
        node.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        node.inputs["Density"].default_value = absorption
        absorb = node.outputs[0]
    combine(nodes, links, output, [ior.outputs[0], absorb])
    return material


def noise_ior_material(name, amplitude, scale=3.0, strength=1.0, link_strength=False,
                       scatter=0.0, with_ior=True):
    """Index of refraction `1 + amplitude * (noise - 0.5)`, optionally in white fog."""
    material, nodes, links, output = new_material(name)
    shaders = []
    if with_ior:
        ior = add_volume_ior(nodes, strength=strength)
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = scale
        noise.inputs["Detail"].default_value = 1.0
        geometry = nodes.new("ShaderNodeNewGeometry")
        value = nodes.new("ShaderNodeMath")
        value.operation = "MULTIPLY_ADD"
        value.inputs[1].default_value = amplitude
        value.inputs[2].default_value = 1.0 - 0.5 * amplitude
        links.new(geometry.outputs["Position"], noise.inputs["Vector"])
        links.new(noise.outputs["Fac"], value.inputs[0])
        links.new(value.outputs[0], ior.inputs["IOR"])
        if link_strength:
            links.new(unfoldable_zero(nodes, links), ior.inputs["Strength"])
        shaders.append(ior.outputs[0])
    if scatter:
        node = nodes.new("ShaderNodeVolumeScatter")
        node.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
        node.inputs["Density"].default_value = scatter
        shaders.append(node.outputs[0])
    combine(nodes, links, output, shaders)
    return material


def add_box(scene, name, material, location, size):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = size
    obj.data.materials.append(material)
    return obj


# ------------------------------------------------------------------------------------------------
# Scenes


def scene_gradient(args, absorption=0.0, vdb=False):
    scene = new_scene(args)
    add_camera(scene, (-6.0, 0.0, 0.0), (0.0, 0.0, 0.0), ortho_scale=2.0)
    if vdb:
        make_height_volume(args, scene)
        material = linear_ior_material("Slab", GRADIENT, absorption, attribute="height")
        bpy.data.objects["Slab"].data.materials.append(material)
    else:
        material = linear_ior_material("Slab", GRADIENT, absorption)
        add_box(scene, "Slab", material, (0.0, 0.0, 0.0), (SLAB_LENGTH, 4.0, 4.0))
    # Emits (z + 5) / 10.
    backdrop = height_material("Backdrop", 0.1, 0.5)
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(1.0 + BACKDROP_DISTANCE, 0.0, 0.0),
                                     rotation=(0.0, math.radians(90.0), 0.0))
    bpy.context.object.data.materials.append(backdrop)
    return scene


def make_height_volume(args, scene):
    """Volume object with a grid `height` of the world z over the slab."""
    import openvdb

    voxel = 1.0 / 16.0
    shape = (int(SLAB_LENGTH / voxel), int(4.0 / voxel), int(4.0 / voxel))
    origin = np.array((-0.5 * SLAB_LENGTH, -2.0, -2.0))
    z = origin[2] + (np.arange(shape[2]) + 0.5) * voxel
    values = np.broadcast_to(z[None, None, :], shape).astype(np.float32).copy()
    grid = openvdb.FloatGrid()
    grid.name = "height"
    # Every voxel is active: a height of zero is not empty space.
    grid.copyFromArray(values, tolerance=-1.0)
    grid.transform = openvdb.createLinearTransform(voxel)
    grid.transform.postTranslate(tuple(origin + 0.5 * voxel))
    path = str(args.output / "height.vdb")
    openvdb.write(path, grids=[grid])
    volume = bpy.data.volumes.new("Slab")
    volume.filepath = path
    obj = bpy.data.objects.new("Slab", volume)
    scene.collection.objects.link(obj)
    return obj


def gradient_expected(resolution, absorption=0.0, interface=False):
    """Value of the backdrop every row of the image must show.

    A ray that enters horizontally at the height z0, where the index is n0, follows
    `z = z0 + n0 / g * (cosh(g * x / n0) - 1)`. With `interface` the index falls to 1 at the end
    of the slab, as that of a voxel grid does within its last voxel, and the ray is refracted
    there by the law of Snell."""
    z0 = -1.0 + (np.arange(resolution) + 0.5) * 2.0 / resolution
    n0 = 1.0 + GRADIENT * z0
    a = GRADIENT * SLAB_LENGTH / n0
    z_exit = z0 + n0 / GRADIENT * (np.cosh(a) - 1.0)
    slope = np.sinh(a)
    if interface:
        sine = (1.0 + GRADIENT * z_exit) * np.tanh(a)
        slope = sine / np.sqrt(1.0 - sine * sine)
    z_hit = z_exit + BACKDROP_DISTANCE * slope
    value = 0.1 * z_hit + 0.5
    if absorption:
        value = value * np.exp(-absorption * n0 / GRADIENT * np.sinh(a))
    return z0, value


def check_gradient(args, name, pixels, absorption=0.0, tolerance=0.004, interface=False):
    z0, expected = gradient_expected(pixels.shape[0], absorption, interface)
    # Rows are the height, every column sees the same.
    margin = pixels.shape[1] // 8
    rows = pixels[:, margin:-margin, 0].mean(axis=1)
    straight = 0.1 * z0 + 0.5
    if absorption:
        straight = straight * math.exp(-absorption * SLAB_LENGTH)
    # In units of the backdrop height.
    error = np.abs(rows - expected).max() * 10.0
    shift = np.abs(straight - expected).max() * 10.0
    noise = pixels[:, margin:-margin, 0].std(axis=1).max() * 10.0
    # The rows are averages of pixels with this noise: allow four standard deviations of them.
    allowed = tolerance + 4.0 * noise / math.sqrt(pixels.shape[1] - 2 * margin)
    passed = error < allowed
    return passed, "max error %.5f m of a shift of %.3f m (allowed %.5f, pixel noise %.5f)" % (
        error, shift, allowed, noise)


def luneburg_material(center=(0.0, 0.0, 0.0)):
    """Index of refraction `sqrt(2 - r^2)` around the center."""
    material, nodes, links, output = new_material("Lens")
    # A lens as strong as glass: finer steps than the automatic ones, which are made for air.
    ior = add_volume_ior(nodes, step_size=0.03)
    geometry = nodes.new("ShaderNodeNewGeometry")
    offset = nodes.new("ShaderNodeVectorMath")
    offset.operation = "SUBTRACT"
    offset.inputs[1].default_value = center
    length = nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    square = nodes.new("ShaderNodeMath")
    square.operation = "MULTIPLY"
    subtract = nodes.new("ShaderNodeMath")
    subtract.operation = "SUBTRACT"
    subtract.inputs[0].default_value = 2.0
    subtract.use_clamp = False
    root = nodes.new("ShaderNodeMath")
    root.operation = "SQRT"
    links.new(geometry.outputs["Position"], offset.inputs[0])
    links.new(offset.outputs[0], length.inputs[0])
    links.new(length.outputs["Value"], square.inputs[0])
    links.new(length.outputs["Value"], square.inputs[1])
    links.new(square.outputs[0], subtract.inputs[1])
    links.new(subtract.outputs[0], root.inputs[0])
    links.new(root.outputs[0], ior.inputs["IOR"])
    links.new(ior.outputs[0], output.inputs["Volume"])
    return material


def scene_luneburg(args):
    scene = new_scene(args)
    add_camera(scene, (0.0, -5.0, 0.0), (0.0, 0.0, 0.0), ortho_scale=2.4)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6, radius=1.0)
    bpy.context.object.data.materials.append(luneburg_material())
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=0.03, location=(0.0, 1.0, 0.0))
    bpy.context.object.data.materials.append(emission_material("Focus"))
    return scene


def check_luneburg(args, name, pixels):
    resolution = pixels.shape[0]
    coords = (np.arange(resolution) + 0.5) * 2.4 / resolution - 1.2
    x, z = np.meshgrid(coords, coords)
    radius = np.sqrt(x * x + z * z)
    inside = pixels[radius < 0.97, 0]
    outside = pixels[radius > 1.05, 0]
    lit = float(inside.mean())
    leak = float(outside.mean())
    passed = lit > 0.99 and leak < 1e-4
    return passed, "%.2f%% of the lens shows the emitter at its focus, %.4f outside" % (
        100.0 * lit, leak)


CAUSTIC_DISTANCE = 0.5
CAUSTIC_SUN = 3.0
# Dark, to keep what the wall reflects back into the lens and receives again out of the test.
CAUSTIC_ALBEDO = 0.1


def scene_caustic(args):
    scene = new_scene(args)
    # Inside of the lens, which camera rays do not see.
    add_camera(scene, (0.0, 3.0, 0.0), (0.0, 4.0, 0.0), ortho_scale=3.2)
    scene.camera.data.clip_start = 0.001

    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6, radius=1.0, location=(0.0, 3.0, 0.0))
    lens = bpy.context.object
    lens.data.materials.append(luneburg_material((0.0, 3.0, 0.0)))
    lens.visible_camera = False

    # Wall behind the lens, facing it.
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 4.0 + CAUSTIC_DISTANCE, 0.0),
                                     rotation=(math.radians(90.0), 0.0, 0.0))
    bpy.context.object.data.materials.append(
        diffuse_material("Wall", (CAUSTIC_ALBEDO, CAUSTIC_ALBEDO, CAUSTIC_ALBEDO)))

    # The sun shines along +X onto a mirror that turns it along +Y, the axis of the lens.
    material, nodes, links, output = new_material("Mirror")
    bsdf = nodes.new("ShaderNodeBsdfGlossy")
    bsdf.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.0
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=5.0, rotation=(math.radians(90.0), 0.0,
                                                         math.radians(45.0)))
    bpy.context.object.data.materials.append(material)
    light_data = bpy.data.lights.new("Sun", "SUN")
    light_data.energy = CAUSTIC_SUN
    light_data.angle = 0.0
    light = bpy.data.objects.new("Sun", light_data)
    light.location = (-10.0, 0.0, 0.0)
    light.rotation_euler = Vector((1.0, 0.0, 0.0)).to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(light)
    # Keeps the sun from the lens and the wall.
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(-1.8, 4.1, 0.0),
                                     rotation=(0.0, math.radians(90.0), 0.0))
    bpy.context.object.scale = (8.0, 5.0, 1.0)
    bpy.context.object.data.materials.append(diffuse_material("Screen", (0.0, 0.0, 0.0)))
    return scene


def check_caustic(args, name, pixels):
    resolution = pixels.shape[0]
    coords = (np.arange(resolution) + 0.5) * 3.2 / resolution - 1.6
    x, z = np.meshgrid(coords, coords)
    radius = np.sqrt(x * x + z * z)
    d = CAUSTIC_DISTANCE
    radiance = CAUSTIC_SUN * CAUSTIC_ALBEDO / math.pi
    image = pixels[:, :, 0]
    if args.integrator == "pt":
        # Only the light of the mirror that passes the lens by is found, by next event estimation
        # along the straight line through the lens too: not what this scene is about.
        return True, "not rendered by path tracing (mean %.4f inside)" % image[radius < 0.9].mean()
    # Rings behind the lens, where the wall has nothing but what the lens spreads over it.
    worst = 0.0
    report = []
    for low, high in ((0.25, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 0.9)):
        ring = (radius >= low) & (radius < high)
        expected = radiance * d * d / (d * d + radius[ring] ** 2) ** 2
        expected = expected.mean()
        ratio = image[ring].mean() / expected
        worst = max(worst, abs(ratio - 1.0))
        report.append("%.3f" % ratio)
    # Next to the lens the light of the mirror arrives as well.
    ring = (radius >= 1.15) & (radius < 1.5)
    expected = (radiance * (1.0 + d * d / (d * d + radius[ring] ** 2) ** 2)).mean()
    outside = image[ring].mean() / expected
    # Merging blurs the profile, which is steep in the middle.
    tolerance = 0.05 if args.integrator == "bdpt" else 0.12
    passed = worst < tolerance and abs(outside - 1.0) < tolerance
    return passed, "rendered / expected in four rings %s, next to the lens %.3f (tolerance %.2f)" % (
        " ".join(report), outside, tolerance)


def scene_furnace(args, scatter=0.0, amplitude=0.4):
    scene = new_scene(args, world_strength=1.0)
    add_camera(scene, (0.0, -6.0, 0.0), (0.0, 0.0, 0.0), lens=45.0)
    material = noise_ior_material("Air", amplitude, scatter=scatter)
    add_box(scene, "Air", material, (0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.8, segments=48, ring_count=24)
    bpy.ops.object.shade_smooth()
    bpy.context.object.data.materials.append(diffuse_material("White", (1.0, 1.0, 1.0)))
    return scene


def check_furnace(args, name, pixels, tolerance=0.01):
    size = 8
    height = pixels.shape[0] // size * size
    width = pixels.shape[1] // size * size
    blocks = pixels[:height, :width, :3].reshape(height // size, size, width // size, size, 3)
    blocks = blocks.mean(axis=(1, 3))
    mean = float(pixels[:, :, :3].mean())
    worst = float(np.abs(blocks - 1.0).max())
    passed = abs(mean - 1.0) < tolerance and worst < 8.0 * tolerance
    return passed, "mean %.5f, worst block %.4f off (tolerance %.3f)" % (mean, worst, tolerance)


def scene_room(args, kind="on", fog=0.0, strength=300.0):
    """Lit interior with a column of hot air. `kind`: on, off (no node), zero (no strength)."""
    scene = new_scene(args, world_strength=0.05)
    add_camera(scene, (0.0, -7.5, 1.6), (0.0, 0.0, 1.3), lens=40.0)

    # Floor and walls with a checker, so that the distortion has something to move.
    material, nodes, links, output = new_material("Checker")
    checker = nodes.new("ShaderNodeTexChecker")
    checker.inputs["Scale"].default_value = 14.0
    checker.inputs["Color1"].default_value = (0.8, 0.8, 0.8, 1.0)
    checker.inputs["Color2"].default_value = (0.15, 0.2, 0.3, 1.0)
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    links.new(checker.outputs["Color"], bsdf.inputs["Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=8.0)
    bpy.context.object.data.materials.append(material)
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 4.0, 4.0),
                                     rotation=(math.radians(90.0), 0.0, 0.0))
    bpy.context.object.data.materials.append(material)
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(-4.0, 0.0, 4.0),
                                     rotation=(0.0, math.radians(90.0), 0.0))
    bpy.context.object.data.materials.append(diffuse_material("Red", (0.7, 0.2, 0.15)))
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(4.0, 0.0, 4.0),
                                     rotation=(0.0, math.radians(-90.0), 0.0))
    bpy.context.object.data.materials.append(diffuse_material("Green", (0.2, 0.6, 0.25)))

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5, location=(1.6, 1.2, 0.5))
    bpy.ops.object.shade_smooth()
    glossy, nodes, links, output = new_material("Glossy")
    bsdf = nodes.new("ShaderNodeBsdfGlossy")
    bsdf.inputs["Roughness"].default_value = 0.15
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    bpy.context.object.data.materials.append(glossy)
    bpy.ops.mesh.primitive_cube_add(size=0.9, location=(-1.7, 1.5, 0.45))
    bpy.context.object.data.materials.append(diffuse_material("Box", (0.8, 0.7, 0.3)))

    light_data = bpy.data.lights.new("Light", "AREA")
    light_data.energy = 600.0
    light_data.size = 1.5
    light = bpy.data.objects.new("Light", light_data)
    light.location = (0.5, -1.0, 4.5)
    scene.collection.objects.link(light)

    # Hot air above a plate: temperature falls off from the axis of the column, broken up by
    # noise. The falloff makes the edge of the box as cold as the room.
    material, nodes, links, output = new_material("Hot Air")
    shaders = []
    if kind != "off":
        ior = add_volume_ior(nodes, mode="TEMPERATURE", strength=strength)
        geometry = nodes.new("ShaderNodeNewGeometry")
        separate = nodes.new("ShaderNodeSeparateXYZ")
        combine_xy = nodes.new("ShaderNodeCombineXYZ")
        length = nodes.new("ShaderNodeVectorMath")
        length.operation = "LENGTH"
        falloff = nodes.new("ShaderNodeMapRange")
        falloff.interpolation_type = "SMOOTHSTEP"
        falloff.inputs["From Min"].default_value = 0.2
        falloff.inputs["From Max"].default_value = 0.95
        falloff.inputs["To Min"].default_value = 1.0
        falloff.inputs["To Max"].default_value = 0.0
        noise = nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = 2.5
        noise.inputs["Detail"].default_value = 2.0
        heat = nodes.new("ShaderNodeMath")
        heat.operation = "MULTIPLY"
        kelvin = nodes.new("ShaderNodeMath")
        kelvin.operation = "MULTIPLY_ADD"
        kelvin.inputs[1].default_value = 500.0
        kelvin.inputs[2].default_value = 293.15
        links.new(geometry.outputs["Position"], separate.inputs[0])
        links.new(separate.outputs["X"], combine_xy.inputs["X"])
        links.new(separate.outputs["Y"], combine_xy.inputs["Y"])
        links.new(combine_xy.outputs[0], length.inputs[0])
        links.new(length.outputs["Value"], falloff.inputs["Value"])
        links.new(geometry.outputs["Position"], noise.inputs["Vector"])
        links.new(falloff.outputs["Result"], heat.inputs[0])
        links.new(noise.outputs["Fac"], heat.inputs[1])
        links.new(heat.outputs[0], kelvin.inputs[0])
        links.new(kelvin.outputs[0], ior.inputs["Temperature"])
        if kind == "zero":
            links.new(unfoldable_zero(nodes, links), ior.inputs["Strength"])
        shaders.append(ior.outputs[0])
    if fog:
        node = nodes.new("ShaderNodeVolumeScatter")
        node.inputs["Color"].default_value = (0.9, 0.9, 0.9, 1.0)
        node.inputs["Density"].default_value = fog
        shaders.append(node.outputs[0])
    if shaders:
        combine(nodes, links, output, shaders)
        bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=3.0, location=(0.0, 0.0, 1.55))
        bpy.context.object.data.materials.append(material)
    return scene


def scene_stress(args, kind):
    """Refraction together with other features: these only have to render finite images."""
    scene = scene_room(args, fog=0.1 if kind in ("fog_glass", "overlap") else 0.0)
    cycles = scene.cycles
    air = bpy.data.objects["Cylinder"]
    if kind == "camera_inside":
        scene.camera.location = (0.0, -0.5, 1.4)
    elif kind == "overlap":
        # A second refracting volume and a fog that both overlap the first.
        add_box(scene, "Air 2", noise_ior_material("Air 2", 0.1), (0.5, -0.3, 1.2),
                (1.6, 1.6, 1.6))
        add_box(scene, "Fog", noise_ior_material("Fog", 0.0, scatter=0.3, with_ior=False),
                (-0.6, 0.2, 1.0), (1.5, 1.5, 1.5))
    elif kind == "world":
        # The world as the refracting volume: denser air near the ground.
        nodes = scene.world.node_tree.nodes
        links = scene.world.node_tree.links
        ior = add_volume_ior(nodes, strength=1.0, step_size=0.2)
        geometry = nodes.new("ShaderNodeNewGeometry")
        separate = nodes.new("ShaderNodeSeparateXYZ")
        value = nodes.new("ShaderNodeMath")
        value.operation = "MULTIPLY_ADD"
        value.inputs[1].default_value = -0.02
        value.inputs[2].default_value = 1.1
        links.new(geometry.outputs["Position"], separate.inputs[0])
        links.new(separate.outputs["Z"], value.inputs[0])
        links.new(value.outputs[0], ior.inputs["IOR"])
        links.new(ior.outputs[0], nodes["World Output"].inputs["Volume"])
    elif kind == "fog_glass":
        # Glass in the hot air, and a glass shell around a part of it.
        material, nodes, links, output = new_material("Glass")
        bsdf = nodes.new("ShaderNodeBsdfGlass")
        links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.35, location=(0.2, -0.2, 1.3))
        bpy.context.object.data.materials.append(material)
        bpy.ops.mesh.primitive_cube_add(size=1.2, location=(0.9, 0.3, 0.9))
        bpy.context.object.data.materials.append(material)
    elif kind == "motion":
        scene.render.use_motion_blur = True
        air.keyframe_insert("location", frame=1)
        air.location.x += 0.8
        air.keyframe_insert("location", frame=2)
        scene.frame_set(1)
    elif kind == "passes":
        # Fast GI bounces, a shadow catcher, a transparent film, denoising data and clamping.
        cycles.use_fast_gi = True
        cycles.fast_gi_method = "REPLACE"
        cycles.ao_bounces_render = 2
        scene.render.film_transparent = True
        bpy.data.objects["Plane"].is_shadow_catcher = True
        cycles.sample_clamp_indirect = 5.0
        scene.view_layers[0].cycles.denoising_store_passes = True
        scene.view_layers[0].use_pass_z = True
        scene.view_layers[0].use_pass_normal = True
    elif kind == "max_steps":
        cycles.volume_ior_max_steps = 3
    elif kind == "switched_off":
        cycles.use_volume_ior = False
    return scene


SCENES = {
    "gradient": (lambda args: scene_gradient(args),
                 lambda args, name, pixels: check_gradient(args, name, pixels)),
    "gradient_absorb": (lambda args: scene_gradient(args, absorption=ABSORPTION),
                        lambda args, name, pixels: check_gradient(
                            args, name, pixels, absorption=ABSORPTION)),
    "gradient_vdb": (lambda args: scene_gradient(args, vdb=True),
                     lambda args, name, pixels: check_gradient(
                         args, name, pixels, tolerance=0.003, interface=True)),
    "luneburg": (scene_luneburg, check_luneburg),
    "caustic": (scene_caustic, check_caustic),
    "furnace": (lambda args: scene_furnace(args), check_furnace),
    "furnace_scatter": (lambda args: scene_furnace(args, scatter=0.4),
                        lambda args, name, pixels: check_furnace(
                            args, name, pixels, tolerance=0.02)),
    "room": (lambda args: scene_room(args), None),
    "room_off": (lambda args: scene_room(args, kind="off"), None),
    "room_zero": (lambda args: scene_room(args, kind="zero"), None),
    "room_fog": (lambda args: scene_room(args, fog=0.15), None),
    "room_fog_off": (lambda args: scene_room(args, kind="off", fog=0.15), None),
    "room_fog_zero": (lambda args: scene_room(args, kind="zero", fog=0.15), None),
    "room_weak": (lambda args: scene_room(args, strength=30.0), None),
    "room_weak_fog": (lambda args: scene_room(args, fog=0.15, strength=30.0), None),
    # Hot air at its physical strength: too weak to see at this resolution.
    "segments_fog": (lambda args: scene_room(args, fog=0.15, strength=1.0), None),
}

for _kind in ("camera_inside", "overlap", "world", "fog_glass", "motion", "passes", "max_steps",
              "switched_off"):
    SCENES["stress_" + _kind] = (lambda args, kind=_kind: scene_stress(args, kind), None)

GROUPS = {
    "stress": [name for name in SCENES if name.startswith("stress_")],
    "analytic": ["gradient", "gradient_absorb", "gradient_vdb", "luneburg", "caustic", "furnace",
                 "furnace_scatter"],
    "pairs": ["room", "room_off", "room_zero", "room_fog", "room_fog_off", "room_fog_zero",
              "segments_fog", "room_weak", "room_weak_fog"],
    "references": ["room", "room_off", "room_fog", "room_fog_off", "room_weak", "room_weak_fog"],
}


def load_pixels(path):
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    pixels = pixels.reshape(image.size[1], image.size[0], image.channels)
    bpy.data.images.remove(image)
    return pixels.astype(np.float64)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = []
    for name in (list(SCENES) if args.scenes == "all" else args.scenes.split(",")):
        names += GROUPS.get(name, [name])

    failed = 0
    for name in names:
        build, check = SCENES[name]
        scene = build(args)
        path = args.output / (name + ".exr")
        scene.render.filepath = str(path)
        if args.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=str(args.output / (name + ".blend")))
        if args.no_render:
            continue
        start = time.time()
        bpy.ops.render.render(write_still=True)
        seconds = time.time() - start
        pixels = load_pixels(path)
        finite = bool(np.isfinite(pixels).all())
        if check is None:
            print("VOLUME_IOR %-18s rendered in %6.1f s, mean %.5f%s" % (
                name, seconds, pixels[:, :, :3].mean(), "" if finite else " NOT FINITE FAILED"))
            failed += not finite
            continue
        passed, message = check(args, name, pixels)
        passed = passed and finite
        failed += not passed
        print("VOLUME_IOR %-18s %s in %.1f s: %s" % (
            name, "PASSED" if passed else "FAILED", seconds, message))

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
