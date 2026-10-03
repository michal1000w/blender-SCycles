# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Nested dielectric scenes for Cycles: demo scenes and validation pairs.

Every validation scene exists twice. `<name>_nested` models overlapping media with the material
setting Nested Priority. `<name>_reference` renders the same optics without that feature, from
geometry that was cut by hand and single interfaces whose shader has the relative index of
refraction. Both must converge to the same image. Reference scenes do not use the setting, so
they also render in builds without nested dielectrics.

Most pairs exist in two more kinds, which must match the same reference. `<name>_auto` has no
priorities at all and relies on Automatic Nested Dielectrics of the render settings: closed
refractive objects which overlap are media, the one which encloses the smaller volume first.
`<name>_equal` gives every medium the same priority with the automatic setting off, where the
same order decides. Reference scenes switch the automatic setting off, as they fake the media
with relative indices of refraction on overlapping objects.

Demo scenes (`drink_*`, `rod_*`, `spheres_*`) show the look; `drink_gap` and `drink_overlap` are
the two usual ways to model a drink without priorities.

  blender -b --factory-startup --python tests/python/cycles_nested_dielectric_scenes.py -- \
      --output DIRECTORY [--device CPU|METAL] [--integrator pt|bdpt|photon] [--guiding] \
      [--scenes a,b,c] [--samples N] [--resolution N] [--save-blend] [--no-render]

Compare with tests/python/cycles_nested_dielectric_compare.py.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import bmesh
import bpy

IOR_GLASS = 1.5
IOR_WATER = 1.33
IOR_ICE = 1.31

# How the scene being built resolves its media: "nested" by the priorities of the builder,
# "reference" not at all, "auto" without priorities by Automatic Nested Dielectrics, "equal" with
# the same priority for all media.
KIND = "nested"


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="CPU", choices=("METAL", "CPU"))
    parser.add_argument("--integrator", default="pt", choices=("pt", "bdpt", "photon"))
    parser.add_argument("--guiding", action="store_true")
    parser.add_argument("--osl", action="store_true", help="Open Shading Language (CPU only)")
    parser.add_argument("--scenes", default="all")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--bounces", type=int, default=24)
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--png", action="store_true", help="Also save a tone mapped PNG")
    return parser.parse_args(argv)


# ------------------------------------------------------------------------------------------------
# Scene setup


def new_scene(args, camera_location=(0.0, -6.0, 2.2), look_at=(0.0, 0.0, 0.9), lens=50.0):
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
    cycles.max_bounces = args.bounces
    cycles.transmission_bounces = args.bounces
    cycles.glossy_bounces = args.bounces
    cycles.transparent_max_bounces = args.bounces
    cycles.volume_bounces = 4
    cycles.caustics_reflective = True
    cycles.caustics_refractive = True
    cycles.blur_glossy = 0.0
    cycles.sample_clamp_direct = 0.0
    cycles.sample_clamp_indirect = 0.0
    if args.integrator == "bdpt":
        cycles.use_bidirectional_path_tracing = True
        cycles.bdpt_light_paths = 65536
    elif args.integrator == "photon":
        cycles.use_photon_mapping = True
        cycles.photon_count = 65536
    cycles.use_guiding = args.guiding
    cycles.shading_system = args.osl and args.device == "CPU"
    if hasattr(cycles, "use_auto_nested_dielectrics"):
        cycles.use_auto_nested_dielectrics = KIND == "auto"
    elif KIND == "auto":
        raise SystemExit("This build has no automatic nested dielectrics")

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    background = nodes["Background"]
    # A horizon gradient gives refraction something to bend.
    coord = nodes.new("ShaderNodeTexCoord")
    separate = nodes.new("ShaderNodeSeparateXYZ")
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.45
    ramp.color_ramp.elements[0].color = (0.35, 0.30, 0.25, 1)
    ramp.color_ramp.elements[1].position = 0.75
    ramp.color_ramp.elements[1].color = (0.55, 0.70, 1.0, 1)
    mapping = nodes.new("ShaderNodeMath")
    mapping.operation = "MULTIPLY_ADD"
    mapping.inputs[1].default_value = 0.5
    mapping.inputs[2].default_value = 0.5
    links.new(coord.outputs["Generated"], separate.inputs[0])
    links.new(separate.outputs["Z"], mapping.inputs[0])
    links.new(mapping.outputs[0], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], background.inputs["Color"])
    background.inputs["Strength"].default_value = 0.6
    scene.world = world

    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.data.lens = lens
    aim(camera, look_at)
    scene.camera = camera
    return scene


def aim(obj, target):
    direction = (target[0] - obj.location[0], target[1] - obj.location[1],
                 target[2] - obj.location[2])
    from mathutils import Vector
    obj.rotation_euler = Vector(direction).to_track_quat("-Z", "Y").to_euler()


def new_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    return mat, nodes, mat.node_tree.links, output


def set_priority(mat, priority):
    if KIND == "auto":
        return
    if priority and KIND == "equal":
        priority = 1
    if priority:
        if not hasattr(mat.cycles, "nested_priority"):
            raise SystemExit("This build has no nested dielectrics (Material > Nested Priority)")
        mat.cycles.nested_priority = priority


def dielectric(name, ior, priority=0, roughness=0.0, color=(1, 1, 1, 1), absorption=None,
               density=1.0, scatter=None, principled=False, dispersion=0.0):
    """Glass material, optionally with an absorbing or scattering interior."""
    mat, nodes, links, output = new_material(name)
    if principled:
        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Transmission Weight"].default_value = 1.0
        if dispersion and "Dispersion" in bsdf.inputs:
            bsdf.inputs["Dispersion"].default_value = dispersion
    else:
        bsdf = nodes.new("ShaderNodeBsdfGlass")
        bsdf.inputs["Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["IOR"].default_value = ior
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    if absorption is not None:
        volume = nodes.new("ShaderNodeVolumeAbsorption")
        volume.inputs["Color"].default_value = absorption
        volume.inputs["Density"].default_value = density
        links.new(volume.outputs[0], output.inputs["Volume"])
    elif scatter is not None:
        volume = nodes.new("ShaderNodeVolumeScatter")
        volume.inputs["Color"].default_value = scatter
        volume.inputs["Density"].default_value = density
        links.new(volume.outputs[0], output.inputs["Volume"])
    set_priority(mat, priority)
    return mat


def diffuse(name, color=(0.8, 0.8, 0.8, 1), roughness=0.6):
    mat, nodes, links, output = new_material(name)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat


def checker(name, scale=6.0, a=(0.75, 0.75, 0.75, 1), b=(0.12, 0.12, 0.14, 1)):
    mat, nodes, links, output = new_material(name)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.7
    tex = nodes.new("ShaderNodeTexChecker")
    tex.inputs["Scale"].default_value = scale
    tex.inputs["Color1"].default_value = a
    tex.inputs["Color2"].default_value = b
    links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat


def emission(name, color, strength):
    mat, nodes, links, output = new_material(name)
    node = nodes.new("ShaderNodeEmission")
    node.inputs["Color"].default_value = color
    node.inputs["Strength"].default_value = strength
    links.new(node.outputs[0], output.inputs["Surface"])
    return mat


def add_object(name, mesh, mat=None):
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    if mat is not None:
        mesh.materials.append(mat)
    return obj


def add_sphere(name, mat, location=(0.0, 0.0, 1.0), radius=1.0, flip=False, segments=96):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segments, v_segments=segments // 2, radius=radius)
    for face in bm.faces:
        face.smooth = True
        if flip:
            face.normal_flip()
    bm.to_mesh(mesh)
    bm.free()
    obj = add_object(name, mesh, mat)
    obj.location = location
    return obj


def add_box(name, mat, minimum, maximum):
    """Closed box with outward normals."""
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    verts = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return add_object(name, mesh, mat)


def add_revolved(name, mat, profile, segments=96, smooth=True):
    """Closed solid of revolution about Z from a (radius, height) profile running from the axis
    back to the axis, counter-clockwise in the (radius, height) plane for outward normals."""
    verts = []
    rings = []
    for radius, height in profile:
        if radius == 0.0:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, height))
        else:
            ring = []
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                ring.append(len(verts))
                verts.append((radius * math.cos(angle), radius * math.sin(angle), height))
            rings.append(ring)
    faces = []
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(segments):
            j = (i + 1) % segments
            if len(a) == 1 and len(b) == 1:
                continue
            if len(a) == 1:
                faces.append((a[0], b[j], b[i]))
            elif len(b) == 1:
                faces.append((a[i], a[j], b[0]))
            else:
                faces.append((a[i], a[j], b[j], b[i]))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for face in bm.faces:
        face.smooth = smooth
    bm.to_mesh(mesh)
    bm.free()
    return add_object(name, mesh, mat)


def add_floor(mat=None, size=14.0, height=0.0):
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, height))
    floor = bpy.context.object
    floor.data.materials.append(mat or checker("Floor"))
    return floor


def add_backdrop():
    """Colored stripes behind the subject, to read refraction and total internal reflection."""
    mat, nodes, links, output = new_material("Backdrop")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = 0.8
    coord = nodes.new("ShaderNodeTexCoord")
    wave = nodes.new("ShaderNodeTexChecker")
    wave.inputs["Scale"].default_value = 10.0
    wave.inputs["Color1"].default_value = (0.8, 0.25, 0.1, 1)
    wave.inputs["Color2"].default_value = (0.1, 0.35, 0.8, 1)
    links.new(coord.outputs["UV"], wave.inputs["Vector"])
    links.new(wave.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 4.0, 3.0),
                                     rotation=(math.radians(90), 0.0, 0.0))
    bpy.context.object.data.materials.append(mat)


def add_area_light(location=(2.5, -2.5, 5.0), energy=900.0, size=1.5, target=(0.0, 0.0, 0.8)):
    bpy.ops.object.light_add(type="AREA", location=location)
    light = bpy.context.object
    light.data.energy = energy
    light.data.size = size
    aim(light, target)
    return light


def stage(args, **camera):
    scene = new_scene(args, **camera)
    add_floor()
    add_backdrop()
    add_area_light()
    return scene


# ------------------------------------------------------------------------------------------------
# Validation pairs


def scene_shell(args, nested):
    """Water sphere inside a glass sphere: a true interface between two media."""
    stage(args)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0), radius=1.0)
    if nested:
        add_sphere("Water", dielectric("Water", IOR_WATER, 2), radius=0.7)
    else:
        add_sphere("Water", dielectric("WaterInGlass", IOR_WATER / IOR_GLASS), radius=0.7)


def scene_rough_shell(args, nested):
    """As `shell`, with a rough interface between the media and a rough outer surface."""
    stage(args)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0, roughness=0.15),
               radius=1.0)
    if nested:
        add_sphere("Water", dielectric("Water", IOR_WATER, 2, roughness=0.3), radius=0.7)
    else:
        add_sphere("Water", dielectric("WaterInGlass", IOR_WATER / IOR_GLASS, roughness=0.3),
                   radius=0.7)


def scene_rough_inner(args, nested):
    """As `shell`, with only the interface between the media rough."""
    stage(args)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0), radius=1.0)
    if nested:
        add_sphere("Water", dielectric("Water", IOR_WATER, 2, roughness=0.3), radius=0.7)
    else:
        add_sphere("Water", dielectric("WaterInGlass", IOR_WATER / IOR_GLASS, roughness=0.3),
                   radius=0.7)


def scene_rough_outer(args, nested):
    """As `shell`, with only the outer surface rough."""
    stage(args)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0, roughness=0.15),
               radius=1.0)
    if nested:
        add_sphere("Water", dielectric("Water", IOR_WATER, 2), radius=0.7)
    else:
        add_sphere("Water", dielectric("WaterInGlass", IOR_WATER / IOR_GLASS), radius=0.7)


def scene_mnee_shell(args, nested):
    """As `shell`, with the caustics of both media on the floor solved by manifold next event
    estimation (shadow caustics)."""
    scene_shell(args, nested)
    for obj in bpy.data.objects:
        if obj.type == "LIGHT":
            obj.data.cycles.is_caustics_light = True
        elif obj.name in ("Glass", "Water"):
            obj.cycles.is_caustics_caster = True
        elif obj.type == "MESH":
            obj.cycles.is_caustics_receiver = True


def scene_feature_only(args, nested):
    """The reference of `rough_shell`, with an unrelated glass sphere out of view which has a
    nested priority when `nested`: the feature is enabled in the kernel without changing any
    visible material."""
    scene_rough_shell(args, False)
    if not nested:
        add_sphere("Unrelated", dielectric("Unrelated", IOR_GLASS), location=(0.0, 4.6, 0.3),
                   radius=0.2, segments=16)
    else:
        # Behind the backdrop, inside the bounds of the scene.
        mat = dielectric("Unrelated", IOR_GLASS)
        mat.cycles.nested_priority = 1
        add_sphere("Unrelated", mat, location=(0.0, 4.6, 0.3), radius=0.2, segments=16)


def scene_principled_shell(args, nested):
    """As `shell`, with Principled BSDF transmission instead of the Glass BSDF."""
    stage(args)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0, principled=True,
                                   color=(0.9, 1.0, 0.9, 1)), radius=1.0)
    if nested:
        add_sphere("Water", dielectric("Water", IOR_WATER, 2, principled=True,
                                       color=(1.0, 0.85, 0.8, 1)), radius=0.7)
    else:
        add_sphere("Water", dielectric("WaterInGlass", IOR_WATER / IOR_GLASS, principled=True,
                                       color=(1.0, 0.85, 0.8, 1)), radius=0.7)


def scene_overlap(args, nested, small=False):
    """A glass block and a water block which overlap: the faces of the water inside the glass
    are false intersections, the face of the glass inside the water an interface of both.
    With `small` the glass block encloses less volume than the water, so that it also is the
    one which exists in the overlap without priorities."""
    stage(args, camera_location=(3.5, -5.5, 2.6), look_at=(0.0, 0.0, 0.7))
    if small:
        glass_min, glass_max = (-0.8, -0.45, 0.02), (0.1, 0.45, 1.22)
        water_min, water_max = (-0.2, -0.4, 0.32), (1.6, 0.4, 1.12)
    else:
        glass_min, glass_max = (-1.2, -0.7, 0.02), (0.1, 0.7, 1.42)
        water_min, water_max = (-0.2, -0.4, 0.32), (1.2, 0.4, 1.12)
    if nested:
        add_box("Glass", dielectric("Glass", IOR_GLASS, 2), glass_min, glass_max)
        add_box("Water", dielectric("Water", IOR_WATER, 1), water_min, water_max)
        return

    # Glass block whose +X face is split into the contact patch and the frame around it.
    x0, y0, z0 = glass_min
    x1, y1, z1 = glass_max
    ys = [y0, water_min[1], water_max[1], y1]
    zs = [z0, water_min[2], water_max[2], z1]
    verts = [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)]
    grid = {}
    for j, z in enumerate(zs):
        for i, y in enumerate(ys):
            grid[i, j] = len(verts)
            verts.append((x1, y, z))
    mesh = bpy.data.meshes.new("Glass")
    bm = bmesh.new()
    v = [bm.verts.new(co) for co in verts]
    bm.verts.ensure_lookup_table()

    def face(indices, material):
        f = bm.faces.new([v[i] for i in indices])
        f.material_index = material

    face((0, 3, 2, 1), 0)  # -X
    face([0, 1] + [grid[i, 0] for i in (3, 2, 1, 0)], 0)  # bottom
    face([3] + [grid[i, 3] for i in (0, 1, 2, 3)] + [2], 0)  # top
    face([0] + [grid[0, j] for j in (0, 1, 2, 3)] + [3], 0)  # -Y
    face([1, 2] + [grid[3, j] for j in (3, 2, 1, 0)], 0)  # +Y
    for j in range(3):
        for i in range(3):
            face((grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]),
                 1 if (i == 1 and j == 1) else 0)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()
    glass = add_object("Glass", mesh)
    mesh.materials.append(dielectric("Glass", IOR_GLASS))
    # Seen from the glass this face is the exit: eta = 1 / ior = water / glass.
    mesh.materials.append(dielectric("GlassToWater", IOR_GLASS / IOR_WATER))

    # Water block without the part inside the glass, open towards the glass.
    wx0, wy0, wz0 = x1, water_min[1], water_min[2]
    wx1, wy1, wz1 = water_max
    verts = [(wx0, wy0, wz0), (wx1, wy0, wz0), (wx1, wy1, wz0), (wx0, wy1, wz0),
             (wx0, wy0, wz1), (wx1, wy0, wz1), (wx1, wy1, wz1), (wx0, wy1, wz1)]
    faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6)]
    mesh = bpy.data.meshes.new("Water")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    add_object("Water", mesh, dielectric("Water", IOR_WATER))
    return glass


def scene_overlap_small(args, nested):
    return scene_overlap(args, nested, small=True)


def scene_shell_swapped(args, nested):
    """As `shell`, with the inner object created first: the order of the media must not depend
    on the order of the objects."""
    stage(args)
    if nested:
        add_sphere("Water", dielectric("Water", IOR_WATER, 2), radius=0.7)
    else:
        add_sphere("Water", dielectric("WaterInGlass", IOR_WATER / IOR_GLASS), radius=0.7)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0), radius=1.0)


def scene_unwelded(args, nested):
    """As `shell`, from meshes whose faces do not share vertices: closed by position only."""
    scene_shell(args, nested)
    for name in ("Glass", "Water"):
        mesh = bpy.data.objects[name].data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bmesh.ops.split_edges(bm, edges=bm.edges)
        bm.to_mesh(mesh)
        bm.free()


def scene_joined(args, nested):
    """As `shell`, with both spheres in one object: each material is a closed surface, so each
    is a medium, and the one a path entered last exists."""
    scene_shell(args, nested)
    if nested:
        glass = bpy.data.objects["Glass"]
        water = bpy.data.objects["Water"]
        bpy.ops.object.select_all(action="DESELECT")
        bpy.context.view_layer.objects.active = glass
        glass.select_set(True)
        water.select_set(True)
        bpy.ops.object.join()


def scene_two_materials(args, nested):
    """As `shell`, with two materials on the faces of the glass sphere: neither is a closed
    surface, so the sphere stays one medium which a path enters by one and leaves by the
    other."""
    scene_shell(args, nested)
    glass = bpy.data.objects["Glass"]
    other = bpy.data.materials["Glass"].copy()
    other.name = "GlassTop"
    glass.data.materials.append(other)
    for polygon in glass.data.polygons:
        polygon.material_index = 1 if polygon.center.z > 0.1 else 0


def scene_instance(args, nested):
    """Two objects of one mesh and material, one scaled down inside the other: the inner one is
    an index matched medium, so only the outer sphere is visible."""
    stage(args)
    glass = add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0), radius=1.0)
    if nested:
        inner = bpy.data.objects.new("Inner", glass.data)
        bpy.context.collection.objects.link(inner)
        inner.location = glass.location
        inner.scale = (0.6, 0.6, 0.6)


def scene_open(args, nested):
    """Meshes which cannot be media automatically, in front of a glass sphere with a water core:
    an open sheet of glass and a sphere with inward normals. A path that enters them never
    leaves, so they must stay plain interfaces with air and the media behind them correct."""
    stage(args)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0), radius=1.0)
    if nested:
        add_sphere("Water", dielectric("Water", IOR_WATER, 2), radius=0.7)
    else:
        add_sphere("Water", dielectric("WaterInGlass", IOR_WATER / IOR_GLASS), radius=0.7)
    # Reaches into the bounds of the sphere, without a priority in any kind.
    sheet_mat = dielectric("Sheet", 1.45)
    sheet_mat.cycles.nested_priority = 0
    bpy.ops.mesh.primitive_plane_add(size=1.6, location=(0.0, -0.9, 1.0),
                                     rotation=(math.radians(70), 0.0, 0.0))
    bpy.context.object.data.materials.append(sheet_mat)
    add_sphere("Inverted", sheet_mat, location=(-1.3, -0.9, 0.45), radius=0.4, flip=True)


def scene_mixed(args, nested):
    """A water sphere without a priority inside a glass sphere with one: the priority wins over
    the automatic order, so the water does not exist."""
    stage(args)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0), radius=1.0)
    if nested:
        water = dielectric("Water", IOR_WATER, 0)
        add_sphere("Water", water, radius=0.7)
        bpy.data.materials["Glass"].cycles.nested_priority = 5
        bpy.context.scene.cycles.use_auto_nested_dielectrics = True


def scene_separate(args, nested):
    """Refractive objects which do not touch, though the bounds of two of them overlap: the
    automatic setting must not change the image."""
    scene = stage(args)
    add_sphere("A", dielectric("A", IOR_GLASS), location=(-0.75, 0.0, 0.8), radius=0.6)
    add_sphere("B", dielectric("B", IOR_WATER), location=(0.25, 0.0, 1.5), radius=0.6)
    add_sphere("C", dielectric("C", IOR_ICE, roughness=0.2), location=(1.3, 0.3, 0.5),
               radius=0.45)
    return scene


def scene_matched(args, nested):
    """A sphere with the index of refraction of the glass around it does not exist."""
    stage(args)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0), radius=1.0)
    if nested:
        add_sphere("Core", dielectric("Core", IOR_GLASS, 2, roughness=0.4), radius=0.6)


def scene_matched_shadow(args, nested):
    """Light reaches a diffuse sphere inside a medium through an index matched interface, also
    for shadow rays: an emitter and the sphere share a water tank with a matched glass wall
    between them."""
    scene = new_scene(args, camera_location=(0.0, -9.0, 2.5), look_at=(0.0, 0.0, 1.0))
    add_floor()
    # Camera and all objects are inside the water, so shadow rays never cross a real interface.
    add_sphere("Tank", dielectric("Water", IOR_WATER, 1 if nested else 0), location=(0, 0, 1),
               radius=12.0, segments=64)
    add_sphere("Diffuse", diffuse("Diffuse", (0.8, 0.5, 0.3, 1)), location=(-1.2, 0.0, 1.0),
               radius=0.6)
    emitter = add_sphere("Emitter", emission("Emitter", (1, 1, 1, 1), 30.0),
                         location=(1.8, 0.0, 1.4), radius=0.3)
    emitter.visible_camera = True
    if nested:
        # A wall of the same index between emitter and sphere, overlapping nothing else.
        add_box("Wall", dielectric("Wall", IOR_WATER, 2), (0.2, -2.0, 0.05), (0.6, 2.0, 2.5))
    scene.cycles.max_bounces = 6


def scene_bubble(args, nested):
    """Air bubble in an absorbing liquid: only the medium of highest priority has its volume."""
    stage(args)
    absorption = (0.9, 0.45, 0.1, 1)
    if nested:
        add_sphere("Liquid", dielectric("Liquid", IOR_WATER, 1, absorption=absorption,
                                        density=1.2), radius=1.0)
        add_sphere("Bubble", dielectric("Bubble", 1.0, 2), location=(0.25, 0.0, 1.1),
                   radius=0.45)
    else:
        mat = dielectric("Liquid", IOR_WATER, absorption=absorption, density=1.2)
        liquid = add_sphere("Liquid", mat, radius=1.0)
        cavity = add_sphere("Cavity", mat, location=(0.25, 0.0, 1.1), radius=0.45, flip=True)
        bpy.context.view_layer.objects.active = liquid
        liquid.select_set(True)
        cavity.select_set(True)
        bpy.ops.object.join()


def scene_camera_inside(args, nested):
    """Camera inside the water, looking at a glass sphere in it and out through the surface."""
    scene = new_scene(args, camera_location=(0.0, -2.2, 1.0), look_at=(0.0, 0.0, 1.0), lens=28.0)
    add_floor(height=-3.5)
    add_backdrop()
    add_area_light(location=(2.0, -1.0, 7.0), energy=2500.0, target=(0.0, 0.0, 1.0))
    add_sphere("Water", dielectric("Water", IOR_WATER, 1 if nested else 0), radius=3.0)
    if nested:
        add_sphere("Glass", dielectric("Glass", IOR_GLASS, 2), location=(0.3, 0.2, 1.0),
                   radius=0.6)
    else:
        add_sphere("Glass", dielectric("GlassInWater", IOR_GLASS / IOR_WATER),
                   location=(0.3, 0.2, 1.0), radius=0.6)
    return scene


def scene_light_inside(args, nested):
    """A small emitter inside the water core of a glass sphere: light subpaths of the
    bidirectional integrators start inside two media."""
    scene = stage(args)
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0), radius=1.0)
    if nested:
        add_sphere("Water", dielectric("Water", IOR_WATER, 2), radius=0.7)
    else:
        add_sphere("Water", dielectric("WaterInGlass", IOR_WATER / IOR_GLASS), radius=0.7)
    add_sphere("Emitter", emission("Emitter", (1.0, 0.9, 0.7, 1), 400.0),
               location=(0.0, 0.0, 1.0), radius=0.08, segments=32)
    return scene


def scene_onion(args, nested):
    """Five concentric media: a deep list of media, every interface between two of them."""
    stage(args)
    iors = (1.5, 1.33, 1.6, 1.2, 1.45)
    radii = (1.0, 0.85, 0.7, 0.5, 0.3)
    for index, (ior, radius) in enumerate(zip(iors, radii)):
        if nested:
            mat = dielectric(f"Medium{index}", ior, index + 1)
        else:
            mat = dielectric(f"Medium{index}", ior / (iors[index - 1] if index else 1.0))
        add_sphere(f"Medium{index}", mat, radius=radius)


def scene_camera_inside_two(args, nested):
    """Camera inside water which is inside glass: the media of a path that starts inside two
    of them, with the index of refraction of their shaders."""
    scene = new_scene(args, camera_location=(0.0, -2.2, 1.0), look_at=(0.0, 0.0, 1.0), lens=28.0)
    add_floor(height=-4.5)
    add_backdrop()
    add_area_light(location=(2.0, -1.0, 8.0), energy=3500.0, target=(0.0, 0.0, 1.0))
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0), radius=3.6)
    if nested:
        add_sphere("Water", dielectric("Water", IOR_WATER, 2), radius=3.0)
        add_sphere("Ball", dielectric("Ball", 1.7, 3), location=(0.3, 0.2, 1.0), radius=0.6)
    else:
        add_sphere("Water", dielectric("WaterInGlass", IOR_WATER / IOR_GLASS), radius=3.0)
        add_sphere("Ball", dielectric("BallInWater", 1.7 / IOR_WATER), location=(0.3, 0.2, 1.0),
                   radius=0.6)
    return scene


def scene_ice_volume(args, nested):
    """A glass ball inside an absorbing liquid inside glass: the volume of the liquid ends at
    the ball, which a path enters through a true interface inside two media."""
    stage(args)
    absorption = (0.2, 0.6, 0.9, 1)
    add_sphere("Glass", dielectric("Glass", IOR_GLASS, 1 if nested else 0), radius=1.0)
    if nested:
        add_sphere("Liquid", dielectric("Liquid", IOR_WATER, 2, absorption=absorption,
                                        density=1.5), radius=0.85)
        add_sphere("Ball", dielectric("Ball", IOR_GLASS, 3), location=(0.15, 0.0, 1.1), radius=0.4)
    else:
        mat = dielectric("Liquid", IOR_WATER / IOR_GLASS, absorption=absorption, density=1.5)
        liquid = add_sphere("Liquid", mat, radius=0.85)
        # The cavity keeps the material of the liquid, so that its volume ends there. Seen from
        # the liquid its relative index is the inverse, glass over water: a ball of glass.
        cavity = add_sphere("Cavity", mat, location=(0.15, 0.0, 1.1), radius=0.4, flip=True)
        bpy.context.view_layer.objects.active = liquid
        liquid.select_set(True)
        cavity.select_set(True)
        bpy.ops.object.join()


def scene_stress(args, variant):
    """Everything at once, to check that it renders: the drink with a cloudy scattering ice
    cube, a subsurface scattering fruit and a diffuse object in the liquid, on a shadow catcher,
    with a volume in the world. `equal` gives all media the same priority, `open` adds an open
    mesh with a priority, `overflow` stacks more media than the list holds. The `legacy` variants
    are the same scenes without any priority, to compare render times."""
    legacy = variant.startswith("legacy")
    scene = scene_drink(args, "overlap" if legacy else "nested")
    floor = bpy.data.objects["Plane"]
    floor.is_shadow_catcher = True
    # Cloudy ice which sticks out of the drink.
    cloudy = dielectric("CloudyIce", IOR_ICE, 0 if legacy else 2, scatter=(0.9, 0.95, 1.0, 1),
                        density=6.0)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(-0.1, -0.25, 1.6), rotation=(0.2, 0.4, 0))
    bpy.context.object.data.materials.append(cloudy)
    # Subsurface scattering fruit across the surface of the drink.
    mat, nodes, links, output = new_material("Fruit")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.8, 0.1, 0.1, 1)
    bsdf.inputs["Subsurface Weight"].default_value = 1.0
    bsdf.inputs["Subsurface Radius"].default_value = (0.2, 0.05, 0.05)
    links.new(bsdf.outputs[0], output.inputs["Surface"])
    add_sphere("Fruit", mat, location=(0.3, 0.2, 1.5), radius=0.16, segments=32)
    world = scene.world.node_tree
    volume = world.nodes.new("ShaderNodeVolumeScatter")
    volume.inputs["Density"].default_value = 0.01
    world.links.new(volume.outputs[0], world.nodes["World Output"].inputs["Volume"])
    if variant == "equal":
        for mat in bpy.data.materials:
            if mat.cycles.nested_priority:
                mat.cycles.nested_priority = 1
    elif variant == "open":
        bpy.ops.mesh.primitive_plane_add(size=1.0, location=(0.0, 0.0, 1.0))
        bpy.context.object.data.materials.append(dielectric("Sheet", 1.4, 5))
    elif variant in ("overflow", "legacy_overflow"):
        for index in range(40):
            add_sphere(f"Shell{index}", dielectric(f"Shell{index}", 1.3 + 0.01 * index,
                                                   0 if legacy else 10 + index),
                       location=(0.0, 0.0, 0.8), radius=0.3 - 0.005 * index, segments=16)
    return scene


def scene_dispersion(args, nested):
    """Dispersive glass around a water core: the relative index follows the wavelength."""
    stage(args)
    add_sphere("Glass", dielectric("Glass", 1.6, 1 if nested else 0, principled=True,
                                   dispersion=0.6), radius=1.0)
    add_sphere("Water", dielectric("Water", IOR_WATER, 2 if nested else 0, principled=True),
               radius=0.7)


# ------------------------------------------------------------------------------------------------
# Demo scenes


def tumbler_profile(radius=0.8, height=2.2, wall=0.07, bottom=0.25):
    return [(0.0, 0.0), (radius, 0.0), (radius, height), (radius - wall, height),
            (radius - wall, bottom), (0.0, bottom)]


def scene_drink(args, mode):
    """Tumbler with a drink, ice and a straw. Modes: `nested` overlaps the media and resolves
    them by priority, `gap` leaves air between glass and liquid, `overlap` overlaps them without
    priorities."""
    scene = stage(args, camera_location=(0.0, -6.5, 2.6), look_at=(0.0, 0.0, 1.05), lens=60.0)
    nested = mode == "nested"
    radius, height, wall, bottom = 0.8, 2.2, 0.07, 0.25
    level = 1.55
    glass = add_revolved("Tumbler", dielectric("Glass", IOR_GLASS, 3 if nested else 0),
                         tumbler_profile(radius, height, wall, bottom))
    glass.location.z = 0.002

    inner = radius - wall
    grow = {"nested": 0.03, "overlap": 0.03, "gap": -0.02}[mode]
    liquid_mat = dielectric("Drink", IOR_WATER, 1 if nested else 0,
                            absorption=(0.95, 0.55, 0.15, 1), density=0.9)
    add_revolved("Drink", liquid_mat,
                 [(0.0, bottom - grow), (inner + grow, bottom - grow), (inner + grow, level),
                  (0.0, level)])

    ice_mat = dielectric("Ice", IOR_ICE, 2 if nested else 0, roughness=0.02)
    for index, (location, rotation) in enumerate((
            ((-0.22, 0.05, 1.45), (0.3, 0.2, 0.4)),
            ((0.26, -0.12, 1.38), (0.1, 0.5, 1.1)),
            ((0.05, 0.25, 0.95), (0.7, 0.1, 0.2)))):
        bpy.ops.mesh.primitive_cube_add(size=0.42, location=location, rotation=rotation)
        ice = bpy.context.object
        ice.name = f"Ice{index}"
        bevel = ice.modifiers.new("Bevel", "BEVEL")
        bevel.width = 0.04
        bevel.segments = 3
        bpy.ops.object.shade_smooth()
        ice.data.materials.append(ice_mat)

    bubble_mat = dielectric("Bubble", 1.0, 4 if nested else 0)
    if nested:
        for index, location in enumerate(((-0.4, -0.3, 0.6), (0.35, 0.3, 0.75),
                                          (0.1, -0.45, 1.0), (-0.3, 0.35, 1.15))):
            add_sphere(f"Bubble{index}", bubble_mat, location=location,
                       radius=0.05 + 0.015 * index, segments=32)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.045, depth=3.0, location=(0.42, 0.1, 1.55),
                                        rotation=(0.0, math.radians(14), 0.0))
    bpy.context.object.name = "Straw"
    bpy.context.object.data.materials.append(diffuse("Straw", (0.85, 0.08, 0.1, 1), 0.4))
    bpy.ops.object.shade_smooth()
    return scene


def scene_rod(args, matched):
    """A glass rod in a beaker of oil. With the index of the oil matched to the glass the
    immersed part of the rod disappears."""
    scene = stage(args, camera_location=(0.0, -6.5, 2.4), look_at=(0.0, 0.0, 1.0), lens=60.0)
    radius, height, wall, bottom = 0.85, 2.0, 0.06, 0.12
    add_revolved("Beaker", dielectric("Glass", IOR_GLASS, 2),
                 tumbler_profile(radius, height, wall, bottom)).location.z = 0.002
    inner = radius - wall + 0.03
    add_revolved("Oil", dielectric("Oil", 1.474 if matched else IOR_WATER, 1,
                                   absorption=(1.0, 0.95, 0.75, 1), density=0.15),
                 [(0.0, bottom - 0.03), (inner, bottom - 0.03), (inner, 1.3), (0.0, 1.3)])
    bpy.ops.mesh.primitive_cylinder_add(radius=0.12, depth=3.2, vertices=64,
                                        location=(0.2, 0.0, 1.7),
                                        rotation=(0.0, math.radians(12), 0.0))
    rod = bpy.context.object
    rod.name = "Rod"
    bpy.ops.object.shade_smooth()
    rod.data.materials.append(dielectric("Pyrex", 1.474, 3))
    return scene


def scene_spheres(args, order):
    """Three overlapping spheres. `order` lists their priorities; the highest keeps its shape."""
    scene = stage(args, camera_location=(0.0, -6.5, 2.4), look_at=(0.0, 0.0, 1.0))
    colors = ((1.0, 0.75, 0.75, 1), (0.75, 1.0, 0.75, 1), (0.75, 0.8, 1.0, 1))
    iors = (1.5, 1.33, 1.2)
    for index in range(3):
        angle = 2.0 * math.pi * index / 3.0 + math.pi / 2.0
        add_sphere(f"Sphere{index}",
                   dielectric(f"Medium{index}", iors[index], order[index], color=colors[index]),
                   location=(0.55 * math.cos(angle), 0.0, 1.05 + 0.55 * math.sin(angle)),
                   radius=0.8)
    return scene


def arc(radius, center_height, start_degrees, end_degrees, steps=24):
    """Points of a circle about (0, center_height) in the (radius, height) plane."""
    points = []
    for step in range(steps + 1):
        angle = math.radians(start_degrees + (end_degrees - start_degrees) * step / steps)
        distance = radius * math.cos(angle)
        points.append((distance if distance > 1e-6 * radius else 0.0,
                       center_height + radius * math.sin(angle)))
    return points


def scene_showcase(args, mode="nested"):
    """Three still lifes whose media have no priority at all: the tumbler of `drink`, a fish
    bowl with water, marbles and bubbles, and a paperweight with a liquid core and bubbles."""
    scene = scene_drink(args, mode)
    for obj in bpy.data.objects:
        if obj.name.split(".")[0].rstrip("0123456789") in ("Tumbler", "Drink", "Ice", "Bubble",
                                                            "Straw"):
            obj.location.x -= 2.0
    camera = scene.camera
    camera.location = (0.0, -9.0, 3.0)
    camera.data.lens = 48.0
    aim(camera, (0.0, 0.0, 0.95))
    scene.render.resolution_x = args.resolution * 2

    # Fish bowl: a spherical shell which is open at the top, with thickness.
    center, outer, inner, rim, level = 0.9, 0.9, 0.84, 1.5, 1.2
    top_outer = math.degrees(math.asin((rim - center) / outer))
    top_inner = math.degrees(math.asin((rim - center) / inner))
    add_revolved("Bowl", dielectric("BowlGlass", IOR_GLASS),
                 arc(outer, center, -90, top_outer) + arc(inner, center, top_inner, -90))
    grow = 0.02
    top_water = math.degrees(math.asin((level - center) / (inner + grow)))
    add_revolved("BowlWater", dielectric("BowlWater", IOR_WATER, absorption=(0.75, 0.9, 1.0, 1),
                                         density=0.25),
                 arc(inner + grow, center, -90, top_water) + [(0.0, level)])
    for index, (location, color) in enumerate((((0.22, -0.1, 0.27), (1.0, 0.45, 0.3, 1)),
                                               ((-0.2, 0.12, 0.25), (0.35, 0.8, 0.45, 1)),
                                               ((0.0, -0.3, 0.2), (0.4, 0.55, 1.0, 1)))):
        add_sphere(f"Marble{index}", dielectric(f"Marble{index}", 1.55, color=color),
                   location=location, radius=0.13, segments=48)
    air = dielectric("Air", 1.0)
    for index, location in enumerate(((0.35, -0.2, 0.7), (-0.3, -0.35, 0.9), (0.1, 0.3, 1.0),
                                      (-0.45, 0.1, 0.55), (0.42, 0.15, 1.05))):
        add_sphere(f"BowlBubble{index}", air, location=location, radius=0.035 + 0.008 * index,
                   segments=24)
    # An ice cube floating across the surface of the water.
    bpy.ops.mesh.primitive_cube_add(size=0.36, location=(-0.1, -0.15, 1.17),
                                    rotation=(0.3, 0.5, 0.2))
    bpy.context.object.name = "BowlIce"
    bpy.context.object.data.materials.append(dielectric("BowlIce", IOR_ICE, roughness=0.03))

    # Paperweight: a liquid core and bubbles inside solid glass.
    add_sphere("Paperweight", dielectric("Crystal", 1.6), location=(1.95, 0.0, 0.62), radius=0.6)
    add_sphere("Core", dielectric("Core", IOR_WATER, absorption=(0.9, 0.2, 0.35, 1), density=4.0),
               location=(1.95, 0.0, 0.62), radius=0.3)
    for index, location in enumerate(((2.2, -0.25, 0.85), (1.7, -0.3, 0.5), (2.05, -0.35, 0.35),
                                      (1.75, -0.15, 0.95))):
        add_sphere(f"CrystalBubble{index}", air, location=location, radius=0.03 + 0.01 * index,
                   segments=24)
    return scene


def build(builder, kind, *arguments):
    """Scene of one kind, see KIND."""
    def function(args):
        global KIND
        KIND = kind
        try:
            return builder(args, *arguments)
        finally:
            KIND = "nested"
    return function


def pair(builder, kinds=("nested", "reference", "auto", "equal")):
    return {kind: build(builder, kind, kind != "reference") for kind in kinds}


# Kinds which a pair does not have: the larger block wins `overlap` and the unrelated medium of
# `feature_only` needs a priority, `mixed` is about one priority among automatic media, and
# `separate` has no media.
SKIP_KINDS = {("overlap", "auto"), ("overlap", "equal"),
              ("feature_only", "auto"), ("feature_only", "equal"),
              ("mixed", "auto"), ("mixed", "equal"),
              ("separate", "nested"), ("separate", "equal")}

SCENES = {}
for _name, _builder in (("shell", scene_shell),
                        ("rough_shell", scene_rough_shell),
                        ("rough_inner", scene_rough_inner),
                        ("rough_outer", scene_rough_outer),
                        ("mnee_shell", scene_mnee_shell),
                        ("feature_only", scene_feature_only),
                        ("principled_shell", scene_principled_shell),
                        ("overlap", scene_overlap),
                        ("matched", scene_matched),
                        ("matched_shadow", scene_matched_shadow),
                        ("bubble", scene_bubble),
                        ("camera_inside", scene_camera_inside),
                        ("light_inside", scene_light_inside),
                        ("overlap_small", scene_overlap_small),
                        ("shell_swapped", scene_shell_swapped),
                        ("unwelded", scene_unwelded),
                        ("instance", scene_instance),
                        ("joined", scene_joined),
                        ("two_materials", scene_two_materials),
                        ("open", scene_open),
                        ("mixed", scene_mixed),
                        ("separate", scene_separate)):
    for _kind, _function in pair(_builder).items():
        if (_name, _kind) in SKIP_KINDS:
            continue
        SCENES[f"{_name}_{_kind}"] = _function
for _name, _builder in (("onion", scene_onion),
                        ("camera_inside_two", scene_camera_inside_two),
                        ("ice_volume", scene_ice_volume)):
    for _kind, _function in pair(_builder).items():
        SCENES[f"{_name}_{_kind}"] = _function
SCENES.update({
    "stress_mixed": lambda args: scene_stress(args, "mixed"),
    "stress_equal": lambda args: scene_stress(args, "equal"),
    "stress_open": lambda args: scene_stress(args, "open"),
    "stress_overflow": lambda args: scene_stress(args, "overflow"),
    "legacy_stress": build(scene_stress, "reference", "legacy"),
    "legacy_stress_overflow": build(scene_stress, "reference", "legacy_overflow"),
    "stress_auto": build(scene_stress, "auto", "mixed"),
    "stress_open_auto": build(scene_stress, "auto", "open"),
    "stress_overflow_auto": build(scene_stress, "auto", "overflow"),
    "dispersion_nested": lambda args: scene_dispersion(args, True),
    "dispersion_auto": build(scene_dispersion, "auto", True),
    "showcase_auto": build(scene_showcase, "auto"),
    "showcase_off": build(scene_showcase, "reference", "overlap"),
    "drink_auto": build(scene_drink, "auto", "nested"),
    "drink_equal": build(scene_drink, "equal", "nested"),
    "rod_matched_auto": build(scene_rod, "auto", True),
    "rod_water_auto": build(scene_rod, "auto", False),
    "spheres_auto": build(scene_spheres, "auto", (1, 1, 1)),
    "drink_nested": lambda args: scene_drink(args, "nested"),
    "drink_gap": build(scene_drink, "reference", "gap"),
    "drink_overlap": build(scene_drink, "reference", "overlap"),
    "rod_matched": lambda args: scene_rod(args, True),
    "rod_water": lambda args: scene_rod(args, False),
    "spheres_123": lambda args: scene_spheres(args, (1, 2, 3)),
    "spheres_321": lambda args: scene_spheres(args, (3, 2, 1)),
    "spheres_equal": lambda args: scene_spheres(args, (1, 1, 1)),
})

EXTRA_PAIRS = ("onion", "camera_inside_two", "ice_volume")
VALIDATION_PAIRS = ("shell", "rough_shell", "rough_inner", "rough_outer", "mnee_shell",
                    "feature_only",
                    "principled_shell", "overlap", "matched", "matched_shadow", "bubble",
                    "camera_inside", "light_inside", "overlap_small", "shell_swapped", "unwelded",
                    "instance", "joined", "two_materials", "open", "mixed", "separate")


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
    kinds = ("nested", "reference", "auto", "equal")
    if args.scenes == "all":
        names = list(SCENES)
    elif args.scenes == "validation":
        names = [f"{name}_{kind}" for name in VALIDATION_PAIRS for kind in kinds]
    elif args.scenes == "extra":
        names = [f"{name}_{kind}" for name in EXTRA_PAIRS for kind in kinds]
    elif args.scenes == "stress":
        names = [name for name in SCENES if name.startswith("stress_")]
        names += ["dispersion_nested", "dispersion_auto"]
    elif args.scenes == "references":
        names = [f"{name}_reference" for name in VALIDATION_PAIRS]
    elif args.scenes == "demo":
        names = [name for name in SCENES if name.split("_")[0] in ("drink", "rod", "spheres", "showcase")]
    else:
        names = args.scenes.split(",")
    report = {}
    for name in names:
        if name not in SCENES:
            continue
        SCENES[name](args)
        enable_device(args)
        scene = bpy.context.scene
        if args.save_blend:
            bpy.ops.wm.save_as_mainfile(filepath=str(args.output / f"{name}.blend"),
                                        check_existing=False)
        if args.no_render:
            continue
        scene.render.filepath = str(args.output / f"{name}.exr")
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        seconds = time.perf_counter() - start
        report[name] = {"render_seconds": seconds}
        if args.png:
            image = bpy.data.images.load(str(args.output / f"{name}.exr"))
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_depth = "8"
            image.save_render(str(args.output / f"{name}.png"), scene=scene)
            bpy.data.images.remove(image)
        print(f"SCENE {name} {json.dumps(report[name])}", flush=True)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


main()
