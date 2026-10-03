# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: GPL-2.0-or-later
"""Toggle the nested priority of materials between renders of one persistent session.

Renders the glass sphere with a water core as nested dielectrics, switches the materials to the
equivalent scene without priorities (a relative index of refraction on the inner sphere),
switches back, and compares every render with the first of its kind. Checks that changing the
setting updates shaders, object flags, kernel features and the integrator state.

  blender -b --factory-startup --python tests/python/cycles_nested_dielectric_update.py -- \
      [--device CPU|METAL] [--integrator pt|bdpt]
"""

import argparse
import math
import sys

import bmesh
import bpy
import numpy as np


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="CPU", choices=("METAL", "CPU"))
    parser.add_argument("--integrator", default="pt", choices=("pt", "bdpt"))
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--resolution", type=int, default=96)
    return parser.parse_args(argv)


def glass(name, ior):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfGlass")
    bsdf.inputs["Roughness"].default_value = 0.0
    bsdf.inputs["IOR"].default_value = ior
    mat.node_tree.links.new(bsdf.outputs[0], output.inputs["Surface"])
    return mat


def sphere(name, mat, radius):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=radius)
    for face in bm.faces:
        face.smooth = True
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(mat)
    obj.location = (0.0, 0.0, 1.0)
    return obj


def render(scene):
    bpy.ops.render.render()
    result = bpy.data.images["Render Result"]
    path = bpy.app.tempdir + "nested_update.exr"
    result.save_render(path, scene=scene)
    image = bpy.data.images.load(path, check_existing=False)
    pixels = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(pixels)
    bpy.data.images.remove(image)
    return pixels.astype(np.float64)


def main():
    args = parse_args()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"
    scene.render.use_persistent_data = True
    cycles = scene.cycles
    cycles.device = "GPU" if args.device == "METAL" else "CPU"
    cycles.samples = args.samples
    cycles.use_adaptive_sampling = False
    cycles.use_denoising = False
    cycles.max_bounces = cycles.transmission_bounces = cycles.glossy_bounces = 16
    cycles.use_bidirectional_path_tracing = args.integrator == "bdpt"
    if args.device == "METAL":
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "METAL"
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == "METAL"

    world = bpy.data.worlds.new("World")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.4, 0.5, 0.7, 1)
    scene.world = world
    bpy.ops.object.camera_add(location=(0.0, -6.0, 2.2), rotation=(math.radians(78), 0.0, 0.0))
    scene.camera = bpy.context.object
    bpy.ops.mesh.primitive_plane_add(size=14.0)
    bpy.ops.object.light_add(type="AREA", location=(2.5, -2.5, 5.0))
    bpy.context.object.data.energy = 900.0

    outer = glass("Glass", 1.5)
    inner = glass("Water", 1.33)
    sphere("Glass", outer, 1.0)
    sphere("Water", inner, 0.7)
    ior = inner.node_tree.nodes["Glass BSDF"].inputs["IOR"]

    def set_nested(enabled):
        outer.cycles.nested_priority = 1 if enabled else 0
        inner.cycles.nested_priority = 2 if enabled else 0
        ior.default_value = 1.33 if enabled else 1.33 / 1.5

    failed = False
    reference = {}
    for step, nested in enumerate((True, False, True, False, True)):
        set_nested(nested)
        pixels = render(scene)
        if nested not in reference:
            reference[nested] = pixels
        difference = float(np.abs(pixels - reference[nested]).max())
        across = float(np.abs(pixels - reference[True]).mean() / reference[True].mean())
        ok = np.isfinite(pixels).all() and difference < 1e-3 and across < 2e-3
        failed |= not ok
        print("NESTED_UPDATE step %d nested=%s mean=%.6f max difference to first of its kind=%.2e "
              "relative difference to nested=%.2e %s" % (
                  step, nested, pixels.mean(), difference, across, "ok" if ok else "FAILED"),
              flush=True)
    if failed:
        sys.exit(1)


main()
