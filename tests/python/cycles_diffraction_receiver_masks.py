# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Camera-ray object masks for measuring the saved indirect diffraction scene."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Vector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resolution', type=int, default=320)
    parser.add_argument('--border', type=int, default=3)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.resolution < 16 or not 1 <= args.border < args.resolution // 4:
        parser.error('Invalid resolution or exclusion border')
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))
    scene = bpy.context.scene
    camera = scene.camera
    assert camera.data.type == 'PERSP' and not camera.data.dof.use_dof
    width = args.resolution
    height = round(width * scene.render.resolution_y / scene.render.resolution_x)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    projection = camera.calc_matrix_camera(depsgraph, x=width, y=height,
                                          scale_x=scene.render.pixel_aspect_x,
                                          scale_y=scene.render.pixel_aspect_y).inverted()
    origin = camera.matrix_world.translation
    rotation = camera.matrix_world.to_3x3()
    names = ['Floor', 'Back / indirect receiver', 'Left / indirect receiver',
             'Right / indirect receiver', 'Ceiling / indirect receiver',
             'Grating tile', 'Occluder', 'Sphere']
    identifiers = {name: index + 1 for index, name in enumerate(names)}
    labels = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            point = projection @ Vector((2 * (x + 0.5) / width - 1,
                                         2 * (y + 0.5) / height - 1, -1, 1))
            direction = (rotation @ (point.xyz / point.w)).normalized()
            hit, location, normal, index, obj, matrix = scene.ray_cast(depsgraph, origin, direction)
            if hit:
                labels[y, x] = identifiers.get(obj.name, 0)
    # Exclude silhouettes from region statistics. This is a center-ray geometric
    # mask, not an exact reconstruction of the stochastic pixel filter.
    masks = {}
    for name, identifier in identifiers.items():
        selected = labels == identifier
        interior = np.zeros_like(selected)
        border = args.border
        interior[border:-border, border:-border] = np.logical_and.reduce([
            selected[dy:height - 2 * border + dy, dx:width - 2 * border + dx]
            for dy in range(2 * border + 1) for dx in range(2 * border + 1)])
        masks[name] = interior
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / 'receiver_masks.npz'
    np.savez_compressed(path, **masks)
    report = dict(input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
                  mask_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  resolution=[width, height], array_origin='bottom_left',
                  exclusion_border_pixels=args.border,
                  scope='Eroded center-ray object masks; no depth of field or motion blur.',
                  pixels={name: int(mask.sum()) for name, mask in masks.items()})
    (args.output / 'receiver_masks.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
