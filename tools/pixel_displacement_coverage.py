# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""Measure actual geometry holes separately from displacement shading.

Load a scene, then pass --output STEM [--reference REFERENCE.exr] after --.
The named object's surface is replaced by white emission in memory, preserving its
original displacement graph. A reference alpha mask is eroded to exclude silhouette
and filtering differences. Neither the input .blend nor its textures are saved.
"""

import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def pixels(image):
    data = np.empty(len(image.pixels), dtype=np.float32)
    image.pixels.foreach_get(data)
    return data.reshape(image.size[1], image.size[0], 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--object', default='SSand')
    parser.add_argument('--percentage', type=int, default=25)
    parser.add_argument('--mode', default='unlimited')
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--margin', type=int, default=12)
    parser.add_argument('--max-misses', type=int, default=0)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if not 1 <= args.percentage <= 100 or args.margin < 0 or args.max_misses < 0:
        parser.error('invalid resolution, margin, or miss threshold')
    scene = bpy.context.scene
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'METAL'
    prefs.metalrt = 'ON'
    prefs.get_devices()
    if not any(device.type == 'METAL' for device in prefs.devices):
        raise RuntimeError('This regression requires a Metal GPU')
    for device in prefs.devices:
        device.use = device.type == 'METAL'
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.use_pixel_displacement = True
    scene.cycles.use_pixel_displacement_resolution_clamp = args.mode != 'unlimited'
    scene.cycles.pixel_displacement_resolution = 16384 if args.mode == 'unlimited' else int(args.mode)
    scene.render.resolution_percentage = args.percentage
    scene.cycles.samples = 1
    scene.cycles.seed = 7
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.film_transparent = True
    scene.camera.data.dof.use_dof = False
    target = bpy.data.objects[args.object]
    for obj in scene.objects:
        if obj.type == 'MESH' and obj != target:
            obj.hide_render = True
    for material in target.data.materials:
        nodes = material.node_tree
        emission = nodes.nodes.new('ShaderNodeEmission')
        emission.inputs['Color'].default_value = (1, 1, 1, 1)
        emission.inputs['Strength'].default_value = 1
        output = next(node for node in nodes.nodes
                      if node.type == 'OUTPUT_MATERIAL' and node.is_active_output)
        nodes.links.new(emission.outputs[0], output.inputs['Surface'])
    scene.update_tag()
    bpy.context.view_layer.update()
    bpy.ops.render.render()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    exr = args.output.with_suffix('.exr')
    bpy.data.images['Render Result'].save_render(str(exr), scene=scene)
    scene.render.image_settings.file_format = 'PNG'
    bpy.data.images['Render Result'].save_render(str(args.output.with_suffix('.png')), scene=scene)
    # Reload saved pixels because Render Result may not expose its combined buffer.
    image = bpy.data.images.load(str(exr), check_existing=False)
    data = pixels(image)
    report = {'scene': bpy.data.filepath, 'mode': args.mode,
              'size': list(image.size), 'finite': bool(np.isfinite(data).all())}
    bpy.data.images.remove(image)
    if args.reference:
        reference = bpy.data.images.load(str(args.reference), check_existing=False)
        ref = pixels(reference)
        bpy.data.images.remove(reference)
        if ref.shape != data.shape or not np.isfinite(ref).all():
            raise ValueError('Reference must have matching dimensions and finite pixels')
        mask = ref[:, :, 3] > 0.99
        for _ in range(args.margin):
            mask &= (np.roll(mask, 1, axis=0) & np.roll(mask, -1, axis=0) &
                     np.roll(mask, 1, axis=1) & np.roll(mask, -1, axis=1))
            mask[[0, -1], :] = False
            mask[:, [0, -1]] = False
        report.update(reference=str(args.reference), margin=args.margin,
                      interior_pixels=int(mask.sum()),
                      interior_misses=int(((data[:, :, 3] < 0.5) & mask).sum()))
        if not report['interior_pixels']:
            raise ValueError('Reference has no interior pixels after erosion')
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print('COVERAGE_RESULT', json.dumps(report), flush=True)
    if not report['finite'] or report.get('interior_misses', 0) > args.max_misses:
        raise RuntimeError('Displacement coverage regression failed')


if __name__ == '__main__':
    main()
