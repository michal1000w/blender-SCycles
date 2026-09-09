#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Material regression variants of the shared guiding room.

Use --scene rough_glass, coated, transmission, or subsurface and the shared room
arguments. Only the sphere and tall block change material. Metrics include their
projected bounding rectangles so whole-image averages cannot conceal local errors.
"""

import importlib.util
import math
import pathlib
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def material(variant):
    result = bpy.data.materials.new('Guiding regression ' + variant)
    result.use_nodes = True
    nodes = result.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    if variant == 'rough_glass':
        shader = nodes.new('ShaderNodeBsdfGlass')
        shader.inputs['Color'].default_value = (.98, .98, .98, 1)
        shader.inputs['Roughness'].default_value = .18
        shader.inputs['IOR'].default_value = 1.45
    else:
        shader = nodes.new('ShaderNodeBsdfPrincipled')
        shader.inputs['Base Color'].default_value = (.12, .25, .65, 1)
        shader.inputs['Roughness'].default_value = .3
        if variant == 'coated':
            shader.inputs['Metallic'].default_value = .2
            shader.inputs['Coat Weight'].default_value = 1
            shader.inputs['Coat Roughness'].default_value = .12
        elif variant == 'transmission':
            shader.inputs['Base Color'].default_value = (.8, .97, 1, 1)
            shader.inputs['Transmission Weight'].default_value = 1
            shader.inputs['Roughness'].default_value = .2
            shader.inputs['IOR'].default_value = 1.45
            shader.inputs['Coat Weight'].default_value = .3
        else:
            shader.inputs['Base Color'].default_value = (.85, .12, .05, 1)
            shader.inputs['Subsurface Weight'].default_value = 1
            shader.inputs['Subsurface Radius'].default_value = (.3, .15, .05)
            shader.inputs['Subsurface Scale'].default_value = .2
    result.node_tree.links.new(shader.outputs[0], output.inputs['Surface'])
    return result


def main():
    base_path = pathlib.Path(__file__).with_name('cycles_metal_guiding_scene.py')
    spec = importlib.util.spec_from_file_location('cycles_guiding_room', base_path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    args = sys.argv[sys.argv.index('--') + 1:]
    if '--scene' not in args:
        raise ValueError('A material scene variant is required')
    variant = args[args.index('--scene') + 1]
    if variant not in {'rough_glass', 'coated', 'transmission', 'subsurface'}:
        raise ValueError('Unknown material variant: ' + variant)
    args[args.index('--scene') + 1] = 'indirect'
    sys.argv = [str(base_path), '--', *args]
    original_box, original_build = generator.box, generator.build_scene

    def box(name, location, scale, mat):
        if name == 'Tall block':
            mat = material(variant)
            sphere = bpy.data.objects['Guiding test sphere']
            sphere.data.materials.clear()
            sphere.data.materials.append(mat)
        return original_box(name, location, scale, mat)

    def build_scene(settings):
        settings.scene = variant

        def regions(scene):
            width, height = scene.render.resolution_x, scene.render.resolution_y
            boxes = {}
            for name in ['Guiding test sphere', 'Tall block']:
                obj = scene.objects[name]
                corners = [world_to_camera_view(scene, scene.camera, obj.matrix_world @ Vector(point))
                           for point in obj.bound_box]
                boxes[name] = [max(0, math.floor(min(p.x for p in corners) * width)),
                               max(0, height - math.ceil(max(p.y for p in corners) * height)),
                               min(width, math.ceil(max(p.x for p in corners) * width)),
                               min(height, height - math.floor(min(p.y for p in corners) * height))]
            settings.object_regions = boxes

        # Factory settings clear handlers, so register after the shared factory has begun
        # constructing geometry, before it invokes either the warmup or measured render.
        # Registering after geometry construction via the last box avoids replacing bpy operators.
        def box_and_register(name, location, scale, mat):
            obj = box(name, location, scale, mat)
            if name == 'Tall block' and regions not in bpy.app.handlers.render_pre:
                bpy.app.handlers.render_pre.append(regions)
            return obj

        generator.box = box_and_register
        try:
            return original_build(settings)
        finally:
            if regions in bpy.app.handlers.render_pre:
                bpy.app.handlers.render_pre.remove(regions)

    generator.box = box
    generator.build_scene = build_scene
    generator.main()


if __name__ == '__main__':
    main()
