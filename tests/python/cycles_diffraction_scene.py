#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Reproducible Cycles grating integration scenes, with raw linear render data.

CD/DVD fixtures use radial grating axes and 120 mm disc geometry. By default
they isolate the reflective grating; --cover adds a 1.2 mm clear substrate.
These are validation fixtures, not claims of measured disc appearance.
"""

import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys
import time

import bpy
import numpy as np
from mathutils import Vector


def aim(obj, target):
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', 'Y').to_euler()


def annulus(name, material, bottom=0.0, top=None, inner_radius=0.0075, outer_radius=0.06):
    count = 256
    vertices = []
    levels = [bottom] if top is None else [bottom, top]
    for z in levels:
        for radius in [inner_radius, outer_radius]:
            vertices.extend((radius * math.cos(2 * math.pi * j / count),
                             radius * math.sin(2 * math.pi * j / count), z)
                            for j in range(count))
    faces = []
    for j in range(count):
        k = (j + 1) % count
        face = (j, count + j, count + k, k)
        faces.append(face if top is None else tuple(reversed(face)))
        if top is not None:
            faces.append(tuple(index + 2 * count for index in face))
            faces.append((j, k, 2 * count + k, 2 * count + j))
            faces.append((count + k, count + j, 3 * count + j, 3 * count + k))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj


def build(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    cycles = scene.cycles
    cycles.device = args.device
    cycles.shading_system = args.osl
    cycles.samples = args.samples
    cycles.seed = args.seed
    cycles.use_denoising = False
    cycles.use_adaptive_sampling = False
    cycles.use_guiding = args.guiding
    cycles.guiding_training_samples = 64
    cycles.guiding_gpu_memory_mb = 64
    cycles.use_bidirectional_path_tracing = args.bdpt
    cycles.bdpt_light_paths = 16384
    cycles.bdpt_update_samples = 4
    cycles.use_photon_mapping = False
    cycles.max_bounces = 12
    cycles.transmission_bounces = 12
    cycles.glossy_bounces = 12
    cycles.sample_clamp_direct = 0
    cycles.sample_clamp_indirect = 0
    cycles.blur_glossy = 0
    devices = []
    if args.device == 'GPU':
        if args.osl:
            raise ValueError('Metal OSL is not supported by this build; use --device CPU for OSL')
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == 'METAL'
            if device.use:
                devices.append(device.name)
        if not devices:
            raise RuntimeError('No Metal GPU available')
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'AgX'
    scene.world = bpy.data.worlds.new('Dark studio')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.015

    material = bpy.data.materials.new('Spectral binary grating')
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    glossy = nodes.new('ShaderNodeBsdfAnisotropic')
    glossy.distribution = 'GGX'
    glossy.inputs['Color'].default_value = (0.85, 0.85, 0.85, 1)
    glossy.inputs['Roughness'].default_value = args.roughness
    if 'Diffraction Weight' in glossy.inputs:
        glossy.inputs['Diffraction Weight'].default_value = args.weight
        glossy.inputs['Diffraction Pitch'].default_value = args.pitch
        glossy.inputs['Diffraction Depth'].default_value = args.depth
        glossy.inputs['Diffraction Duty Cycle'].default_value = args.duty
        glossy.inputs['Diffraction Medium IOR'].default_value = args.medium_ior
    elif args.weight != 0:
        raise RuntimeError('This build does not have diffraction support')
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(glossy.outputs['BSDF'], output.inputs['Surface'])
    if args.scene == 'plane':
        bpy.ops.mesh.primitive_plane_add(size=0.12)
        bpy.context.object.data.materials.append(material)
        bpy.context.object.cycles.is_caustics_receiver = args.mnee
        glossy.inputs['Tangent'].default_value = (1, 0, 0)
    else:
        annulus('Reflective grating', material).cycles.is_caustics_receiver = args.mnee
        coordinates = nodes.new('ShaderNodeTexCoord')
        transform = nodes.new('ShaderNodeVectorTransform')
        transform.vector_type = 'VECTOR'
        transform.convert_from = 'OBJECT'
        transform.convert_to = 'WORLD'
        links.new(coordinates.outputs['Object'], transform.inputs['Vector'])
        links.new(transform.outputs['Vector'], glossy.inputs['Tangent'])
        if args.cover:
            cover = bpy.data.materials.new('Clear substrate')
            cover.use_nodes = True
            cn = cover.node_tree.nodes
            cn.clear()
            glass = cn.new('ShaderNodeBsdfGlass')
            glass.inputs['Color'].default_value = (1, 1, 1, 1)
            glass.inputs['IOR'].default_value = 1.58
            glass.inputs['Roughness'].default_value = 0
            out = cn.new('ShaderNodeOutputMaterial')
            cover.node_tree.links.new(glass.outputs['BSDF'], out.inputs['Surface'])
            # Enclose the reflector in glass. A bottom face above z=0 would
            # create an air gap and contradict the grating's incident index.
            annulus('1.2 mm clear substrate', cover, bottom=-0.000001,
                    top=0.0012).cycles.is_caustics_caster = args.mnee

    bpy.ops.object.camera_add(location=(0, -0.16, 0.24))
    scene.camera = bpy.context.object
    scene.camera.data.type = 'ORTHO'
    scene.camera.data.ortho_scale = 0.15
    aim(scene.camera, (0, 0, 0))
    light = bpy.data.lights.new('White area source', 'AREA')
    light.energy = 3.0
    light.shape = 'DISK'
    light.size = 0.008
    light.cycles.is_caustics_light = args.mnee
    obj = bpy.data.objects.new('White area source', light)
    bpy.context.collection.objects.link(obj)
    obj.location = (-0.08, -0.06, 0.18)
    aim(obj, (0, 0, 0))
    if args.world_scale != 1.0:
        for obj in scene.objects:
            obj.location *= args.world_scale
            if obj.type == 'MESH':
                for vertex in obj.data.vertices:
                    vertex.co *= args.world_scale
            elif obj.type == 'CAMERA':
                obj.data.ortho_scale *= args.world_scale
            elif obj.type == 'LIGHT':
                obj.data.size *= args.world_scale
                obj.data.energy *= args.world_scale ** 2
    return scene, devices


def main():
    script_sha256 = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['plane', 'cd', 'dvd'], default='cd')
    parser.add_argument('--pitch', type=float)
    parser.add_argument('--depth', type=float, default=150.0)
    parser.add_argument('--duty', type=float, default=0.5)
    parser.add_argument('--medium-ior', type=float)
    parser.add_argument('--world-scale', type=float, default=1.0,
                        help='Scale macroscopic scene geometry with constant source radiance')
    parser.add_argument('--weight', type=float, default=1.0)
    parser.add_argument('--roughness', type=float, default=0.15)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--osl', action='store_true')
    parser.add_argument('--bdpt', action='store_true')
    parser.add_argument('--guiding', action='store_true')
    parser.add_argument('--cover', action='store_true')
    parser.add_argument('--mnee', action='store_true', help='Enable shadow caustics for the cover')
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--save-scene', action='store_true')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.repeat < 1:
        parser.error('--repeat must be positive')
    if args.world_scale <= 0:
        parser.error('--world-scale must be positive')
    if args.pitch is None:
        args.pitch = 740.0 if args.scene == 'dvd' else 1600.0
    if args.medium_ior is None:
        args.medium_ior = 1.58 if args.cover else 1.0
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene, devices = build(args)
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.filepath = str(args.output.with_suffix('.exr'))
    if args.save_scene:
        bpy.ops.wm.save_as_mainfile(filepath=str(args.output.with_suffix('.blend')))
    warmup_seconds = None
    if args.warmup:
        start = time.perf_counter()
        bpy.ops.render.render()
        warmup_seconds = time.perf_counter() - start
    durations = []
    for _ in range(args.repeat):
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        durations.append(time.perf_counter() - start)
    seconds = statistics.median(durations)
    import OpenImageIO as oiio
    image = oiio.ImageInput.open(scene.render.filepath)
    if image is None:
        raise RuntimeError('No render output')
    rgb = np.asarray(image.read_image(format=oiio.FLOAT))[..., :3]
    image.close()
    if not np.isfinite(rgb).all():
        raise RuntimeError('Nonfinite render values')
    np.save(args.output.with_suffix('.npy'), rgb)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '8'
    bpy.data.images['Render Result'].save_render(str(args.output.with_suffix('.png')), scene=scene)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from cycles_metal_guiding_benchmark import kernel_source_digest
    binary = pathlib.Path(bpy.app.binary_path)
    with binary.open('rb') as stream:
        binary_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    report = {key: str(value) if isinstance(value, pathlib.Path) else value
              for key, value in vars(args).items()}
    report.update(seconds=seconds, render_durations_seconds=durations,
                  warmup_seconds=warmup_seconds, devices=devices,
                  mean=float(rgb.mean()), peak=float(rgb.max()),
                  build_hash=bpy.app.build_hash.decode(), binary_sha256=binary_hash,
                  kernel_sha256=kernel_source_digest(binary),
                  script_sha256=script_sha256)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print('DIFFRACTION_RENDER ' + json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
