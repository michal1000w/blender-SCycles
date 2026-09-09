#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Flat-normal Lambertian control for isolating bidirectional estimator errors.

Uses the shared indirect-room arguments. Geometry and illumination are unchanged;
all surfaces use an exactly reciprocal diffuse BSDF and geometric shading normals.
"""

import importlib.util
import argparse
import json
import pathlib
import sys

import bpy


def main():
    controls = argparse.ArgumentParser(add_help=False)
    controls.add_argument('--control-depth', type=int)
    controls.add_argument('--control-near-clip', type=float)
    controls.add_argument('--control-fstop', type=float)
    controls.add_argument('--control-camera-motion', action='store_true')
    controls.add_argument('--control-null-surfaces', action='store_true')
    controls.add_argument('--control-transparent-depth', type=int)
    delimiter = sys.argv.index('--') + 1
    options, remaining = controls.parse_known_args(sys.argv[delimiter:])
    if options.control_depth is not None and options.control_depth < 0:
        controls.error('--control-depth must be nonnegative')
    if options.control_transparent_depth is not None and not 0 <= options.control_transparent_depth <= 1024:
        controls.error('--control-transparent-depth must be between 0 and 1024')
    sys.argv[delimiter:] = remaining
    path = pathlib.Path(__file__).with_name('cycles_metal_guiding_scene.py')
    spec = importlib.util.spec_from_file_location('cycles_guiding_lambert_room', path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    def material(name, color, roughness=.6, metallic=0):
        result = bpy.data.materials.new(name)
        result.use_nodes = True
        nodes = result.node_tree.nodes
        nodes.clear()
        output = nodes.new('ShaderNodeOutputMaterial')
        diffuse = nodes.new('ShaderNodeBsdfDiffuse')
        diffuse.inputs['Color'].default_value = (*color, 1)
        diffuse.inputs['Roughness'].default_value = 0
        result.node_tree.links.new(diffuse.outputs[0], output.inputs['Surface'])
        return result

    original_box = generator.box
    original_aim = generator.look_at

    def aim(obj, target):
        original_aim(obj, target)
        if obj.type == 'CAMERA' and options.control_transparent_depth is not None:
            bpy.context.scene.cycles.transparent_max_bounces = options.control_transparent_depth
        if obj.type == 'CAMERA' and options.control_near_clip is not None:
            obj.data.clip_start = options.control_near_clip
        if obj.type == 'CAMERA' and options.control_fstop is not None:
            obj.data.dof.use_dof = True
            obj.data.dof.aperture_fstop = options.control_fstop
            obj.data.dof.focus_distance = 8.8
        if obj.type == 'CAMERA' and options.control_camera_motion:
            scene = bpy.context.scene
            scene.render.use_motion_blur = True
            location, lens = obj.location.copy(), obj.data.lens
            for frame, offset, zoom in ((0, -.35, .9), (2, .35, 1.1)):
                scene.frame_set(frame)
                obj.location = location
                obj.location.x += offset
                obj.data.lens = lens * zoom
                original_aim(obj, target)
                obj.keyframe_insert('location')
                obj.keyframe_insert('rotation_euler')
                obj.data.keyframe_insert('lens')
            scene.frame_set(1)
        if obj.type == 'CAMERA' and options.control_null_surfaces:
            transparent = bpy.data.materials.new('Unit transparent control')
            transparent.use_nodes = True
            transparent.node_tree.nodes.clear()
            output = transparent.node_tree.nodes.new('ShaderNodeOutputMaterial')
            closure = transparent.node_tree.nodes.new('ShaderNodeBsdfTransparent')
            closure.inputs['Color'].default_value = (1, 1, 1, 1)
            transparent.node_tree.links.new(closure.outputs[0], output.inputs['Surface'])
            for position, rotation in (((0, -2, 2), (1.5707963267948966, 0, 0)),
                                        ((0, 1, 2.3), (0, 0, 0))):
                bpy.ops.mesh.primitive_plane_add(size=20, location=position, rotation=rotation)
                bpy.context.object.name = 'Null transport control'
                bpy.context.object.data.materials.append(transparent)

    def box(name, location, scale, mat):
        if name == 'Tall block':
            for polygon in bpy.data.objects['Guiding test sphere'].data.polygons:
                polygon.use_smooth = False
            if options.control_depth is not None:
                # Cycles' user depth counts continuations after the first visible surface.
                # Light generation counts scattering vertices, including that first surface.
                bpy.context.scene.cycles.max_bounces = options.control_depth
                bpy.context.scene.cycles.bdpt_max_bounces = options.control_depth + 1
        return original_box(name, location, scale, mat)

    generator.material, generator.box = material, box
    generator.look_at = aim
    generator.main()
    if (any(value is not None for value in (options.control_depth, options.control_near_clip,
                                            options.control_fstop, options.control_transparent_depth))
            or options.control_camera_motion or options.control_null_surfaces):
        output = pathlib.Path(remaining[remaining.index('--output') + 1]).with_suffix('.json')
        if output.exists():
            metadata = json.loads(output.read_text())
            metadata['control_depth'] = options.control_depth
            metadata['control_near_clip'] = options.control_near_clip
            metadata['control_fstop'] = options.control_fstop
            metadata['control_camera_motion'] = options.control_camera_motion
            metadata['control_null_surfaces'] = options.control_null_surfaces
            metadata['transparent_max_bounces'] = bpy.context.scene.cycles.transparent_max_bounces
            output.write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
