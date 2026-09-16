#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Metal render regression for a dielectric surface enclosing a scattering medium.

Run with Blender --background --factory-startup --python-exit-code 1 --python
cycles_bdpt_internal_volume.py -- --output /path/to/results. The reference uses
ordinary path tracing; denoising, guiding, and adaptive sampling are disabled.
"""

import argparse
import json
import pathlib
import sys
import time

import bpy
import numpy as np
from mathutils import Vector


def scene_setup(density=0.3, absorption=0.0, roughness=0.0, enclosing=False, heterogeneous=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'METAL'
    preferences.get_devices()
    metal = [device for device in preferences.devices if device.type == 'METAL']
    if not metal:
        raise RuntimeError('This regression requires a real Metal GPU')
    for device in preferences.devices:
        device.use = device.type == 'METAL'
    print('METAL_DEVICES', [device.name for device in metal], flush=True)
    scene.cycles.device = 'GPU'
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_guiding = False
    scene.cycles.use_volume_guiding = False
    # Match the sensor splat measure and disable glossy mollification in the reference.
    scene.cycles.pixel_filter_type = 'BOX'
    scene.cycles.blur_glossy = 0.0
    scene.cycles.sample_clamp_direct = 0.0
    scene.cycles.sample_clamp_indirect = 0.0
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 12
    scene.cycles.diffuse_bounces = 12
    scene.cycles.volume_bounces = 1
    scene.cycles.bdpt_max_bounces = 12
    scene.cycles.bdpt_update_samples = 1
    scene.render.resolution_x = scene.render.resolution_y = 48
    scene.render.resolution_percentage = 100
    world = bpy.data.worlds.new('Black world')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
    scene.world = world

    bpy.ops.mesh.primitive_cube_add(scale=(1, 1, 0.5))
    glass_object = bpy.context.object
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    material = bpy.data.materials.new('Surface plus interior volume')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    glass = nodes.new('ShaderNodeBsdfGlass')
    glass.inputs['IOR'].default_value = 1.45
    glass.inputs['Roughness'].default_value = roughness
    material.node_tree.links.new(glass.outputs[0], output.inputs['Surface'])
    scatter = nodes.new('ShaderNodeVolumeScatter')
    scatter.inputs['Density'].default_value = density
    if heterogeneous:
        texture = nodes.new('ShaderNodeTexNoise')
        texture.inputs['Scale'].default_value = 2.0
        texture.inputs['Detail'].default_value = 0.0
        scale = nodes.new('ShaderNodeMath')
        scale.operation = 'MULTIPLY'
        scale.inputs[1].default_value = 2.0 * density
        material.node_tree.links.new(texture.outputs['Fac'], scale.inputs[0])
        material.node_tree.links.new(scale.outputs[0], scatter.inputs['Density'])
    if absorption:
        absorb = nodes.new('ShaderNodeVolumeAbsorption')
        absorb.inputs['Color'].default_value = (0, 0, 0, 1)
        absorb.inputs['Density'].default_value = absorption
        add = nodes.new('ShaderNodeAddShader')
        material.node_tree.links.new(scatter.outputs[0], add.inputs[0])
        material.node_tree.links.new(absorb.outputs[0], add.inputs[1])
        material.node_tree.links.new(add.outputs[0], output.inputs['Volume'])
    else:
        material.node_tree.links.new(scatter.outputs[0], output.inputs['Volume'])
    glass_object.data.materials.append(material)

    if enclosing:
        bpy.ops.mesh.primitive_cube_add(scale=(5, 5, 5))
        medium = bpy.data.materials.new('Overlapping outer medium')
        medium.use_nodes = True
        medium.node_tree.nodes.clear()
        outer_output = medium.node_tree.nodes.new('ShaderNodeOutputMaterial')
        outer_absorb = medium.node_tree.nodes.new('ShaderNodeVolumeAbsorption')
        outer_absorb.inputs['Color'].default_value = (0, 0, 0, 1)
        outer_absorb.inputs['Density'].default_value = 0.1
        medium.node_tree.links.new(outer_absorb.outputs[0], outer_output.inputs['Volume'])
        bpy.context.object.data.materials.append(medium)

    light = bpy.data.lights.new('Finite area emitter', 'AREA')
    light.energy = 100
    light.shape = 'DISK'
    light.size = 1
    emitter = bpy.data.objects.new('Finite area emitter', light)
    scene.collection.objects.link(emitter)
    emitter.location = (0, 0, 3)
    camera = bpy.data.cameras.new('Camera')
    sensor = bpy.data.objects.new('Camera', camera)
    scene.collection.objects.link(sensor)
    sensor.location = (0, -4, 2)
    sensor.rotation_euler = (-sensor.location).to_track_quat('-Z', 'Y').to_euler()
    camera.lens = 50
    scene.camera = sensor
    return scene


def render(scene, folder, name, bdpt, samples, paths=65536, volume_bounces=1):
    scene.cycles.use_bidirectional_path_tracing = bdpt
    scene.cycles.samples = samples
    scene.cycles.bdpt_light_paths = paths
    scene.cycles.volume_bounces = volume_bounces
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.filepath = str(folder / (name + '.exr'))
    started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    pixels = np.array(image.pixels[:], dtype=np.float64).reshape(48, 48, 4)[..., :3]
    bpy.data.images.remove(image)
    if not np.isfinite(pixels).all():
        raise AssertionError(name + ': non-finite radiance')
    metrics = dict(mean=float(pixels.mean()), center=float(pixels[16:32, 16:32].mean()),
                   peak=float(pixels.max()), seconds=time.perf_counter() - started)
    print('INTERNAL_VOLUME', name, json.dumps(metrics), flush=True)
    scene.render.image_settings.file_format = 'PNG'
    bpy.data.images['Render Result'].save_render(str(folder / (name + '.png')), scene=scene)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--reference-samples', type=int, default=65536)
    parser.add_argument('--samples', type=int, default=4096)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    report = {}
    errors = []
    for name, options in [('glass_volume', {}),
                          ('absorbing_glass', {'absorption': 0.6}),
                          ('overlapping_volume', {'enclosing': True}),
                          ('heterogeneous_volume', {'heterogeneous': True})]:
        scene = scene_setup(**options)
        reference = render(scene, args.output, name + '_pt', False, args.reference_samples)
        actual = render(scene, args.output, name + '_bdpt', True, args.samples, paths=8192)
        error = abs(actual['mean'] - reference['mean']) / max(reference['mean'], 1e-12)
        center_error = abs(actual['center'] - reference['center']) / max(reference['center'], 1e-12)
        report[name] = dict(reference=reference, bdpt=actual, relative_error=error,
                            center_relative_error=center_error)
        if center_error >= 0.05:
            errors.append(f'{name}: center region differs from PT by {center_error:.2%}')
        if error >= 0.05:
            errors.append(f'{name}: mean differs from independent PT by {error:.2%}')
        if name == 'glass_volume':
            low_budget = render(scene, args.output, name + '_low_budget', True, 4 * args.samples, paths=2048)
            budget_error = abs(low_budget['mean'] - actual['mean']) / actual['mean']
            report[name]['low_budget'] = low_budget
            if budget_error >= 0.05:
                errors.append(f'Light-path budget changed energy by {budget_error:.2%}')
    if report['absorbing_glass']['bdpt']['mean'] >= 0.7 * report['glass_volume']['bdpt']['mean']:
        errors.append('Interior absorption did not attenuate the caustic')
    scene = scene_setup()
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, -2, 1))
    blocker = bpy.context.object
    blocker.rotation_euler = scene.camera.rotation_euler
    black = bpy.data.materials.new('Opaque black camera blocker')
    black.use_nodes = True
    black.node_tree.nodes.clear()
    output = black.node_tree.nodes.new('ShaderNodeOutputMaterial')
    diffuse = black.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    diffuse.inputs['Color'].default_value = (0, 0, 0, 1)
    black.node_tree.links.new(diffuse.outputs[0], output.inputs['Surface'])
    blocker.data.materials.append(black)
    blocked = render(scene, args.output, 'opaque_blocker', True, 16, paths=8192)
    if blocked['peak'] > 1e-8:
        errors.append('Volumetric sensor connection leaked through an opaque blocker')
    report['opaque_blocker'] = blocked
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if errors:
        raise AssertionError('; '.join(errors))
    print('INTERNAL_VOLUME_PASS', flush=True)


if __name__ == '__main__':
    main()
