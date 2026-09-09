#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Pure-diffuse BDPT transport must be invariant to the two caustic switches."""
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
    parser.add_argument('--transparent', action='store_true',
                        help='Also exercise null transmission through a camera-facing sheet')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    script = pathlib.Path(__file__).with_name('cycles_metal_guiding_scene.py')
    sys.argv = [str(script), '--', '--build-only', '--resolution', '64', '--samples', '1',
                '--output', str(args.output / 'scene')]
    runpy.run_path(str(script), run_name='__main__')
    scene = bpy.context.scene
    scene.cycles.use_bidirectional_path_tracing = True
    scene.cycles.bdpt_light_paths = 16384
    scene.cycles.bdpt_update_samples = 1
    scene.cycles.bdpt_max_bounces = 8
    scene.cycles.use_guiding = False
    scene.render.use_persistent_data = True
    for material in bpy.data.materials:
        material.use_nodes = True
        material.node_tree.nodes.clear()
        output = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
        diffuse = material.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
        diffuse.inputs['Color'].default_value = (.7, .7, .7, 1)
        material.node_tree.links.new(diffuse.outputs[0], output.inputs['Surface'])

    if args.transparent:
        material = bpy.data.materials.new('Null transmission control')
        material.use_nodes = True
        material.node_tree.nodes.clear()
        output = material.node_tree.nodes.new('ShaderNodeOutputMaterial')
        transparent = material.node_tree.nodes.new('ShaderNodeBsdfTransparent')
        material.node_tree.links.new(transparent.outputs[0], output.inputs['Surface'])
        camera = scene.camera
        direction = camera.rotation_euler.to_quaternion() @ Vector((0, 0, -1))
        bpy.ops.mesh.primitive_plane_add(size=20, location=camera.location + .15 * direction,
                                        rotation=camera.rotation_euler)
        bpy.context.object.data.materials.append(material)

    reference = None
    reference_cache = None
    report = {}
    for reflective, refractive in [(True, True), (False, True), (True, False), (False, False)]:
        name = f'reflect_{int(reflective)}_refract_{int(refractive)}'
        scene.cycles.caustics_reflective = reflective
        scene.cycles.caustics_refractive = refractive
        scene.render.filepath = str(args.output / (name + '.exr'))
        os.environ['CYCLES_BDPT_CACHE_DUMP'] = str(args.output / name)
        print('CAUSTIC_CONTROL_BEGIN ' + name, flush=True)
        bpy.ops.render.render(write_still=True)
        source = oiio.ImageInput.open(scene.render.filepath)
        assert source is not None, name
        rgb = np.asarray(source.read_image(format=oiio.FLOAT))[..., :3]
        source.close()
        assert np.isfinite(rgb).all() and rgb.max() > 0, name
        cache = (args.output / (name + '_0.vertices')).read_bytes()
        assert cache, name
        if reference is None:
            reference, reference_cache = rgb, cache
        report[name] = {'image_equal': bool(np.allclose(reference, rgb, rtol=1e-6, atol=1e-6)),
                         'maximum_image_difference': float(np.max(np.abs(reference - rgb))),
                         'cache_equal': cache == reference_cache, 'cache_bytes': len(cache)}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    assert all(x['image_equal'] and x['cache_equal'] for x in report.values()), report
    print('BDPT_CAUSTIC_CONTROLS_PASSED', flush=True)


if __name__ == '__main__':
    main()
