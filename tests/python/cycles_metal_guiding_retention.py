#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Check retained training observations across many mixed-transparent visits.

Run in Blender. Arguments follow --. The default one-SPP comparison isolates
history retention before learned proposals affect later paths. Multi-SPP runs
also exercise learning; equal-budget repetitions distinguish repeatability from
budget dependence. Both retain the same strict image comparison.
"""

import argparse
import json
import os
import pathlib
import runpy
import sys

import bpy
import numpy as np
import OpenImageIO as oiio
from mathutils import Vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--integrator', choices=('pt', 'bdpt', 'both'), default='both')
    parser.add_argument('--budgets', type=int, nargs=2, default=(16, 128))
    parser.add_argument('--dump-cache', action='store_true')
    parser.add_argument('--dump-guiding', action='store_true',
                        help='Save learned-field snapshots separately for each render')
    parser.add_argument('--samples', type=int, default=1,
                        help='Use more than one sample to exercise training publications')
    parser.add_argument('--max-bounces', type=int, default=2)
    parser.add_argument('--sheets', type=int, default=24,
                        help='Zero uses the opaque room to exercise smooth RIS guiding')
    parser.add_argument('--resize-between', type=int, default=0,
                        help='Render an intermediate square resolution in the same session')
    parser.add_argument('--crop-between', action='store_true',
                        help='Render an intermediate central crop to change the light-cache budget')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.samples < 1 or args.max_bounces < 1 or args.sheets < 0:
        parser.error('samples and max-bounces must be positive; sheets must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    scene_script = pathlib.Path(__file__).with_name('cycles_metal_guiding_scene.py')
    sys.argv = [str(scene_script), '--', '--build-only', '--guiding', '--resolution', '128',
                '--samples', str(args.samples), '--output', str(args.output / 'scene')]
    runpy.run_path(str(scene_script), run_name='__main__')
    scene = bpy.context.scene
    scene.render.use_persistent_data = True
    scene.cycles.max_bounces = args.max_bounces
    scene.cycles.transparent_max_bounces = 64
    scene.cycles.guiding_training_samples = 0
    material = bpy.data.materials.new('Mixed transparent retention control')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    mix = nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = .05
    transparent = nodes.new('ShaderNodeBsdfTransparent')
    diffuse = nodes.new('ShaderNodeBsdfDiffuse')
    material.node_tree.links.new(transparent.outputs[0], mix.inputs[1])
    material.node_tree.links.new(diffuse.outputs[0], mix.inputs[2])
    material.node_tree.links.new(mix.outputs[0], output.inputs['Surface'])
    camera = scene.camera
    direction = camera.rotation_euler.to_quaternion() @ Vector((0, 0, -1))
    for i in range(args.sheets):
        bpy.ops.mesh.primitive_plane_add(size=20,
            location=camera.location + direction * (.15 + .025 * i),
            rotation=camera.rotation_euler)
        bpy.context.object.name = 'Mixed transparent sheet %02d' % i
        bpy.context.object.data.materials.append(material)
    results = {}
    modes = (False, True) if args.integrator == 'both' else (args.integrator == 'bdpt',)
    for bidirectional in modes:
        integrator = 'bdpt' if bidirectional else 'pt'
        scene.cycles.use_bidirectional_path_tracing = bidirectional
        images = []
        caches = []
        for repetition, budget in enumerate(args.budgets):
            name = '%s_%dmb' % (integrator, budget)
            if args.budgets[0] == args.budgets[1]:
                name += '_run%d' % repetition
            scene.cycles.guiding_gpu_history_memory_mb = budget
            scene.render.filepath = str(args.output / (name + '.exr'))
            if args.dump_cache:
                os.environ['CYCLES_BDPT_CACHE_DUMP'] = str(args.output / name)
            if args.dump_guiding:
                os.environ['CYCLES_METAL_GUIDING_DUMP'] = str(args.output / name)
            print('GUIDING_RETENTION_BEGIN ' + name, flush=True)
            bpy.ops.render.render(write_still=True)
            image = oiio.ImageInput.open(scene.render.filepath)
            assert image is not None, name
            channels = list(image.spec().channelnames)
            rgb_indices = [channels.index(channel) for channel in ('R', 'G', 'B')]
            pixels = np.asarray(image.read_image(format=oiio.FLOAT))[..., rgb_indices]
            image.close()
            assert np.isfinite(pixels).all() and float(pixels.max()) > 0, name
            images.append(pixels)
            results[name] = {'mean': float(pixels.mean()), 'peak': float(pixels.max()),
                             'samples': args.samples, 'history_mb': budget,
                             'sheets': args.sheets, 'max_bounces': args.max_bounces}
            if args.dump_cache and bidirectional:
                cache = (args.output / (name + '_0.vertices')).read_bytes()
                assert cache, name + ': empty light cache'
                caches.append(cache)
                results[name]['cache_bytes'] = len(cache)
            if repetition == 0 and (args.resize_between > 0 or args.crop_between):
                intermediate = integrator + '_intermediate'
                width, height = scene.render.resolution_x, scene.render.resolution_y
                if args.resize_between > 0:
                    scene.render.resolution_x = scene.render.resolution_y = args.resize_between
                if args.crop_between:
                    scene.render.use_border = scene.render.use_crop_to_border = True
                    scene.render.border_min_x = scene.render.border_min_y = .25
                    scene.render.border_max_x = scene.render.border_max_y = .75
                scene.render.filepath = str(args.output / (intermediate + '.exr'))
                if args.dump_cache:
                    os.environ['CYCLES_BDPT_CACHE_DUMP'] = str(args.output / intermediate)
                if args.dump_guiding:
                    os.environ['CYCLES_METAL_GUIDING_DUMP'] = str(args.output / intermediate)
                print('GUIDING_RETENTION_BEGIN ' + intermediate, flush=True)
                bpy.ops.render.render(write_still=True)
                scene.render.resolution_x, scene.render.resolution_y = width, height
                scene.render.use_border = scene.render.use_crop_to_border = False
        results[integrator + '_maximum_budget_difference'] = float(np.max(np.abs(images[0] - images[1])))
        results[integrator + '_comparison_mse'] = float(np.mean((images[0] - images[1]) ** 2))
        results[integrator + '_equal'] = bool(np.allclose(images[0], images[1], rtol=1e-6, atol=1e-6))
        if caches:
            results[integrator + '_cache_equal'] = caches[0] == caches[1]
        (args.output / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
        if caches:
            assert results[integrator + '_cache_equal'], integrator + ': light cache changed'
        assert results[integrator + '_equal'], integrator + ': history comparison changed image'
    (args.output / 'report.json').write_text(json.dumps(results, indent=2) + '\n')
    print('GUIDING_RETENTION_PASSED', flush=True)


if __name__ == '__main__':
    main()
