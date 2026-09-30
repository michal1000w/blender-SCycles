# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Polarizer films whose surface does not face object +Z.

Three closed IOR 1 Glass slabs face +-X (thin in X), viewed along -X in a
constant unpolarized world, as in a typical "three polarizer" setup. The
transmission axis must lie in the film plane: 90/45/0 transmits 1/8, crossed
0/90 transmits 0, a single filter 1/2, uniformly across the slab. Rotating the
object about its film normal must act like the angle socket.

  Blender -b --factory-startup --python this.py -- [--device CPU|METAL]
"""
import argparse
import math
import sys

import bpy
import numpy as np

WORLD = 0.25
RES = 64


def glass(name, angle):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    node = mat.node_tree.nodes.new('ShaderNodeBsdfGlass')
    node.inputs['Roughness'].default_value = 0.0
    node.inputs['IOR'].default_value = 1.0
    node.inputs['Polarizer'].default_value = True
    node.inputs['Polarizer Angle'].default_value = angle
    out = mat.node_tree.nodes['Material Output']
    mat.node_tree.links.new(node.outputs[0], out.inputs['Surface'])
    return node


def setup(device):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.use_polarization = True
    scene.cycles.samples = 4
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.transparent_max_bounces = scene.cycles.transmission_bounces = 16
    scene.cycles.max_bounces = 16
    scene.render.resolution_x = scene.render.resolution_y = RES
    scene.view_settings.view_transform = 'Standard'
    if device == 'METAL':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        for d in prefs.devices:
            d.use = d.type == 'METAL'
        scene.cycles.device = 'GPU'
    world = bpy.data.worlds.new('World')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = WORLD
    scene.world = world
    nodes = []
    for i, angle in enumerate((math.pi / 2, math.pi / 4, 0.0)):
        bpy.ops.mesh.primitive_cube_add(size=2, location=(1.8 - 0.9 * i, 0, 0))
        obj = bpy.context.object
        obj.scale = (0.09, 1, 1)
        nodes.append(glass('film%d' % i, angle))
        obj.data.materials.append(bpy.data.materials['film%d' % i])
    bpy.ops.object.camera_add(location=(8, 0, 0), rotation=(math.pi / 2, 0, math.pi / 2))
    scene.camera = bpy.context.object
    scene.camera.data.type = 'ORTHO'
    scene.camera.data.ortho_scale = 1.5
    return scene, nodes


def render(scene):
    bpy.ops.render.render()
    result = bpy.data.images['Render Result']
    path = bpy.app.tempdir + 'side_film.exr'
    scene.render.image_settings.file_format = 'OPEN_EXR'
    result.save_render(path)
    img = bpy.data.images.load(path)
    px = np.array(img.pixels[:]).reshape(RES, RES, 4)[..., :3].mean(axis=2)
    bpy.data.images.remove(img)
    return px


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('CPU', 'METAL'), default='CPU')
    args = parser.parse_args(argv)
    scene, nodes = setup(args.device)
    objects = [o for o in scene.objects if o.type == 'MESH']
    cases = [
        ('three_90_45_0', lambda: None, 1 / 8),
        ('crossed_90_0', lambda: setattr(nodes[1].inputs['Polarizer'], 'default_value', False), 0.0),
        ('single_0', lambda: setattr(nodes[0].inputs['Polarizer'], 'default_value', False), 0.5),
        ('object_rotated_60', lambda: (setattr(nodes[0].inputs['Polarizer'], 'default_value', True),
                                       setattr(nodes[0].inputs['Polarizer Angle'], 'default_value', 0.0),
                                       setattr(objects[0], 'rotation_euler', (math.radians(60), 0, 0))),
         0.5 * math.cos(math.radians(60)) ** 2),
    ]
    failed = False
    for name, edit, expected in cases:
        edit()
        px = render(scene)
        roi = px[RES // 4:3 * RES // 4, RES // 4:3 * RES // 4]
        mean, spread = float(roi.mean()), float(roi.max() - roi.min())
        ok = abs(mean - expected * WORLD) < 1e-3 and spread < 2e-3
        failed |= not ok
        print('%s %-18s mean=%.5f expected=%.5f spread=%.2e %s' %
              (args.device, name, mean, expected * WORLD, spread, 'PASS' if ok else 'FAIL'))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
