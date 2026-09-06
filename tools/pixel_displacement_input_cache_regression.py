# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""JAR displacement cache regression fixture; never saves the loaded .blend.

Run baseline and candidate sequentially with the user-provided JAR-test-Cycles.blend:
  DISP_REGRESSION_OUTPUT=/tmp/jar-cache-reference Blender --background \
    --factory-startup JAR-test-Cycles.blend --python-exit-code 1 \
    --python tools/pixel_displacement_input_cache_regression.py

Requires SSand and Material.002/Mapping in the JAR scene. Saves seven 32-bit EXRs
and completion/timing JSON. Compare images between builds, and verify that edits
visibly change the reference output. Timing is diagnostic, not a throughput test.
"""

import json
import math
import os
from pathlib import Path
import time

import bpy


def main():
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
    scene.cycles.use_pixel_displacement_resolution_clamp = False
    scene.cycles.pixel_displacement_resolution = 16384
    scene.cycles.samples = 2
    scene.cycles.seed = 0
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.resolution_percentage = 5
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    output = Path(os.environ['DISP_REGRESSION_OUTPUT'])
    output.mkdir(parents=True, exist_ok=True)
    results = []

    def render(label):
        start = time.perf_counter()
        bpy.ops.render.render()
        bpy.data.images['Render Result'].save_render(str(output / (label + '.exr')), scene=scene)
        results.append({'case': label, 'seconds': time.perf_counter() - start})
        print('EDIT_REGRESSION', json.dumps(results[-1]), flush=True)

    mesh = bpy.data.objects['SSand'].data
    nodes = bpy.data.materials['Material.002'].node_tree
    if mesh.uv_layers.active is None:
        raise RuntimeError('SSand must have an active UV layer')

    render('initial')
    for index, vertex_data in enumerate(mesh.vertices):
        vertex_data.co.z += 0.05 * math.sin(index * 1.37)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    mesh.update()
    render('normals')

    for item in mesh.uv_layers.active.data:
        item.uv = (item.uv.x * 0.8 + 0.07, item.uv.y * 1.1 - 0.03)
    mesh.update()
    render('uv_edit')

    second = mesh.uv_layers.new(name='DisplacementCacheRegression')
    for item in second.data:
        item.uv = (item.uv.y * 0.7 + 0.13, item.uv.x * 0.9 + 0.17)
    mesh.update()
    uv_node = nodes.nodes.new('ShaderNodeUVMap')
    uv_node.uv_map = second.name
    nodes.links.new(uv_node.outputs['UV'], nodes.nodes['Mapping'].inputs['Vector'])
    render('uv_switch')

    nodes.links.new(nodes.nodes['Texture Coordinate'].outputs['Generated'],
                    nodes.nodes['Mapping'].inputs['Vector'])
    render('generated')

    uv_node.uv_map = 'DisplacementCacheMissingUV'
    nodes.links.new(uv_node.outputs['UV'], nodes.nodes['Mapping'].inputs['Vector'])
    render('missing_uv')
    render('missing_uv_repeated')
    (output / 'results.json').write_text(json.dumps({'completed': True, 'renders': results}, indent=2))


if __name__ == '__main__':
    main()
