#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Host support-boundary checks for streamed closed-convex Glass (Blender, CPU).

  blender --background --factory-startup --python THIS -- PRISM_PHASE_0_BLEND REPORT_JSON

Each case edits the saved prism fixture and renders a few CPU samples. Invalid
configurations must be rejected before the kernel with their specific message;
valid variants must render finite, nonzero detector radiance.
"""
import json
import sys
from pathlib import Path

import bmesh
import bpy
import numpy as np


def load(blend):
    bpy.ops.wm.open_mainfile(filepath=blend)
    scene = bpy.context.scene
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 4
    scene.render.resolution_x = scene.render.resolution_y = 8
    return scene, bpy.data.objects['Glass prism'], bpy.data.objects['Unit Lambertian detector']


def edit_mesh(obj, function):
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    function(mesh)
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()


def render(scene, path):
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = 'OPEN_EXR'
    try:
        bpy.ops.render.render(write_still=True)
    except RuntimeError as error:
        return 'error', str(error)
    image = bpy.data.images.load(str(path), check_existing=False)
    pixels = np.asarray(image.pixels[:], dtype=np.float64)
    bpy.data.images.remove(image)
    return 'rendered', {'finite': bool(np.isfinite(pixels).all()), 'mean': float(pixels.mean())}


def main():
    blend, report_path = sys.argv[sys.argv.index('--') + 1:][:2]
    out = Path(report_path).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    cases = []

    def case(name, modify, expect_error=None):
        scene, prism, detector = load(blend)
        modify(scene, prism, detector)
        status, detail = render(scene, out / f'{name}.exr')
        if expect_error is None:
            passed = status == 'rendered' and detail['finite'] and detail['mean'] > 0
        else:
            passed = status == 'error' and expect_error in detail
        cases.append({'case': name, 'expected_error': expect_error, 'status': status,
                      'detail': detail, 'passed': passed})
        print(name, 'PASS' if passed else 'FAIL', status, detail if status == 'error' else '')

    case('valid_reference', lambda s, p, d: None)
    case('open_mesh', lambda s, p, d: edit_mesh(p, lambda m: m.faces.remove(list(m.faces)[0])),
         'is open')
    case('inward_normals',
         lambda s, p, d: edit_mesh(p, lambda m: bmesh.ops.reverse_faces(m, faces=list(m.faces))),
         'point inward')

    def concave(s, p, d):
        # Move one apex vertex inside: the prism is no longer convex.
        def f(m):
            m.verts.ensure_lookup_table()
            top = max(m.verts, key=lambda v: v.co.z)
            top.co.z *= 0.2
        edit_mesh(p, f)
    case('concave_mesh', concave, 'convex volume')

    def two_parts(s, p, d):
        def f(m):
            geom = bmesh.ops.duplicate(m, geom=list(m.verts) + list(m.edges) + list(m.faces))['geom']
            bmesh.ops.translate(m, verts=[g for g in geom if isinstance(g, bmesh.types.BMVert)],
                                vec=(0.0, 0.05, 0.0))
        edit_mesh(p, f)
    case('disconnected_parts', two_parts, 'one connected closed volume')

    def source_inside(s, p, d):
        bpy.data.objects[[o.name for o in bpy.data.objects if o.type == 'LIGHT'][0]].location = (0, 0, 0.004)
    case('source_inside_glass', source_inside, 'strictly outside every declared Glass volume')

    def overlap(s, p, d):
        d.location = (0.004, 0.0, 0.004)  # Inside the prism bounding box.
    case('detector_overlaps_glass', overlap, 'bounds overlap declared object')

    case('scalar_polarization', lambda s, p, d: setattr(s.cycles, 'coherent_polarization_mode', 'SCALAR'),
         'Vector Dipole Ensemble')

    def smooth(s, p, d):
        for poly in p.data.polygons:
            poly.use_smooth = True
    case('smooth_shading', smooth, 'smooth shading normals')

    case('max_one_event_exterior_only',
         lambda s, p, d: setattr(s.cycles, 'coherent_max_interface_events', 1))

    def second_glass(s, p, d):
        copy = p.copy()
        copy.data = p.data.copy()
        s.collection.objects.link(copy)
        copy.location = (0.0, -0.2, 0.0)
    case('second_separate_glass', second_glass)

    report = {'scope': 'Streamed closed convex Glass host support boundary, CPU',
              'fixture': blend, 'passed': all(c['passed'] for c in cases), 'cases': cases}
    Path(report_path).write_text(json.dumps(report, indent=1) + '\n')
    print('PASS' if report['passed'] else 'FAIL', len(cases), 'cases')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
