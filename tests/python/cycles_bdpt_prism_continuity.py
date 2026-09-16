#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the supplied prismNew.blend against an independent geometric beam trace.

Blender --background --factory-startup --python-exit-code 1 --python
cycles_bdpt_prism_continuity.py -- --scene /path/prismNew.blend --output /path/results
Use --image existing.exr to check a previous render without rendering again.
The external fixture is read only. No denoising or guiding is used.
"""

import argparse
import json
import pathlib
import sys

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def beam_segments(scene, prism, ior):
    """Trace the spotlight axis with Snell refraction and total internal reflection."""
    light = scene.objects['Spot']
    origin = light.matrix_world.translation.copy()
    direction = light.matrix_world.to_quaternion() @ Vector((0, 0, -1))
    inverse = prism.matrix_world.inverted()
    normal_matrix = inverse.to_3x3().transposed()
    inside = False
    segments = []
    for _ in range(12):
        hit, point, normal, _ = prism.ray_cast(inverse @ origin, inverse.to_3x3() @ direction)
        if not hit:
            break
        point = prism.matrix_world @ point
        normal = (normal_matrix @ normal).normalized()
        if inside:
            segments.append((origin.copy(), point.copy()))
        if direction.dot(normal) > 0:
            normal.negate()
        eta = ior if inside else 1 / ior
        cosine = -direction.dot(normal)
        discriminant = 1 - eta * eta * (1 - cosine * cosine)
        if discriminant < 0:
            direction += 2 * cosine * normal
        else:
            direction = (eta * direction + (eta * cosine - discriminant ** 0.5) * normal).normalized()
            inside = not inside
            if not inside:
                break
        origin = point + direction * 1e-5
    if len(segments) < 3:
        raise AssertionError('Fixture must exercise multiple internal reflections')
    return segments


def top_interface(point, camera, top, ior):
    """Independent scalar Snell solve through the fixture's horizontal top face."""
    delta = camera - point
    distance = Vector((delta.x, delta.y)).length
    depth = top - point.z
    height = camera.z - top
    if depth <= 0 or height <= 0:
        raise AssertionError('Expected a volume point below the top and camera above it')
    lo, hi = 0.0, distance
    for _ in range(60):
        internal = (lo + hi) * 0.5
        external = distance - internal
        residual = external / (external * external + height * height) ** 0.5
        residual -= ior * internal / (internal * internal + depth * depth) ** 0.5
        if residual > 0:
            lo = internal
        else:
            hi = internal
    fraction = (lo + hi) * 0.5 / max(distance, 1e-20)
    return Vector((point.x + fraction * delta.x, point.y + fraction * delta.y, top))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True, type=pathlib.Path)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--image', type=pathlib.Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(args.scene))
    scene = bpy.context.scene
    scene.render.resolution_percentage = 50
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_guiding = False
    scene.cycles.use_volume_guiding = False
    scene.cycles.use_bidirectional_path_tracing = True
    scene.cycles.samples = 32
    scene.cycles.bdpt_light_paths = 655360
    scene.cycles.bdpt_update_samples = 1
    if args.image is None:
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = 'METAL'
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == 'METAL'
        devices = [d.name for d in preferences.devices if d.use]
        if not devices:
            raise RuntimeError('Metal GPU required')
        print('METAL_DEVICES', devices, flush=True)
        scene.cycles.device = 'GPU'
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.filepath = str(args.output / 'prism.exr')
        bpy.ops.render.render(write_still=True)
        args.image = pathlib.Path(scene.render.filepath)
        scene.render.image_settings.file_format = 'PNG'
        bpy.data.images['Render Result'].save_render(str(args.output / 'prism.png'), scene=scene)
    image = bpy.data.images.load(str(args.image), check_existing=False)
    width, height = image.size
    pixels = np.array(image.pixels[:], dtype=np.float64).reshape(height, width, 4)[..., :3]
    if not np.isfinite(pixels).all():
        raise AssertionError('Non-finite radiance')
    prism = scene.objects['Circle'].evaluated_get(bpy.context.evaluated_depsgraph_get())
    glass = next(n for n in prism.active_material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED')
    ior = glass.inputs['IOR'].default_value
    top = max((prism.matrix_world @ v.co).z for v in prism.data.vertices)
    camera = scene.camera.matrix_world.translation
    traces = []
    # Exclude the short exit segment and corner neighborhoods: dispersive branches
    # legitimately separate there. All three long internal segments must remain lit.
    for start, end in beam_segments(scene, prism, ior)[:3]:
        samples = []
        for fraction in np.linspace(0.1, 0.9, 25):
            point = start.lerp(end, float(fraction))
            surface = top_interface(point, camera, top, ior)
            raster = world_to_camera_view(scene, scene.camera, surface)
            x, y = int(raster.x * width), int(raster.y * height)
            radius = max(2, round(width / 180))
            patch = pixels[max(0, y-radius):y+radius+1, max(0, x-radius):x+radius+1]
            if not patch.size:
                raise AssertionError('Expected beam outside the image')
            samples.append(float(patch.mean(axis=2).max()))
        traces.append(samples)
    reference = float(np.median(traces[0]))
    ratios = [min(segment) / max(reference, 1e-20) for segment in traces]
    report = dict(image=str(args.image), first_segment_median=reference,
                  segment_minimum_ratios=ratios, segment_samples=traces)
    (args.output / 'continuity.json').write_text(json.dumps(report, indent=2) + '\n')
    # A hard missing segment falls to the dark background; attenuation and dispersion
    # along these short segments remain well above this conservative spatial threshold.
    if reference <= 0.1 or min(ratios) < 0.1:
        raise AssertionError(f'Premature internal beam cutoff: segment ratios {ratios}')
    print('PRISM_CONTINUITY_PASS', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
