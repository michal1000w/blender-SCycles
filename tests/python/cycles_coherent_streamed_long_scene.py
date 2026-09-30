#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Author streamed fixtures with three and four interface events (Blender only).

  blender --background --factory-startup --python THIS -- OUTPUT_DIR

Layouts (variants phase_0, phase_pi, distinct):
  prism_TIR (3 events): entry, total internal reflection on the prism base, exit;
    plus the direct path.
  slab_long (4 events): closed slab with TT, side-wall T-R-T and T-R-R-T.
  concave_L (4 events): concave L-shaped Glass extrusion: TT, exit into the notch
    followed by exterior reflection or re-entry, internal reflections.
  mirror_corner (3 events): three mutually perpendicular finite mirrors (R, RR, RRR).
Budgets admit every route of at most K events (glossy = transmission = K,
max bounces K + 1), so native BDPT has no unowned path within the budget.
References are computed separately by cycles_coherent_streamed_glass_plan.py.
"""
import math
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_coherent_mirror_acceptance import set_diffuse_material
from cycles_coherent_polarizer_scene import glass, pane
from cycles_coherent_streamed_glass_scene import (add_camera, add_sources, base_scene,
                                                  cube_mesh, mesh_object, prism_mesh, record)


def l_mesh():
    outline = [(0, 0), (0.012, 0), (0.012, 0.004), (0.004, 0.004), (0.004, 0.012), (0, 0.012)]
    verts = [(x, -0.01, z) for x, z in outline] + [(x, 0.01, z) for x, z in outline]
    faces = []
    for a, b, c in [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 5)]:
        faces += [(a, c, b), (a + 6, b + 6, c + 6)]
    for i in range(6):
        j = (i + 1) % 6
        faces += [(i, j, j + 6), (i, j + 6, i + 6)]
    v = np.array(verts)
    volume = sum(np.dot(v[a], np.cross(v[b], v[c])) for a, b, c in faces)
    if volume < 0:
        faces = [(a, c, b) for a, b, c in faces]
    return verts, faces


def detector_and_camera(scene, receiver, source, tilt):
    towards = np.asarray(source, float) - np.asarray(receiver, float)
    towards = towards / np.linalg.norm(towards) + np.asarray(tilt, float)
    normal = towards / np.linalg.norm(towards)
    u = np.cross(normal, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    detector = pane('Unit Lambertian detector', receiver, u, v, 0.002,
                    set_diffuse_material('Unit detector', (1, 1, 1)), 'DETECTOR')
    return detector, add_camera(scene, receiver, normal, 0.0003)


def budgets(scene, events):
    scene.cycles.coherent_max_interface_events = events
    scene.cycles.max_bounces = events + 1
    scene.cycles.bdpt_max_bounces = events + 1
    scene.cycles.glossy_bounces = events
    scene.cycles.transmission_bounces = events


def variants(folder, build):
    for name in ('phase_0', 'phase_pi', 'distinct'):
        scene, mirror_material = base_scene()
        objects, source, receiver, tilt, events = build(scene, mirror_material)
        budgets(scene, events)
        detector, camera = detector_and_camera(scene, receiver, source, tilt)
        sources = add_sources([tuple(source), tuple(np.asarray(source) + [0.0004, -0.0003, 0.0002])],
                              100000.0, 10.0)
        if name == 'phase_pi':
            sources[1].data.cycles.coherence_phase = math.pi
        if name == 'distinct':
            sources[1].data.cycles.coherence_group = 2
        record(folder, name, scene, objects, detector, sources, camera, events)


def main():
    output = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    output.mkdir(parents=True, exist_ok=False)

    def prism(scene, _):
        material, _ = glass('Ideal n1.5 prism', 1.5)
        verts, faces = prism_mesh(0.008, 0.0139, 0.02)
        obj = mesh_object('Glass prism', verts, faces, material, 'GLASS')
        return [obj], (-0.02, 0.0005, 0.016), (0.02, 0.0, 0.016), (0.0, 0.0, -0.2), 3

    def slab(scene, _):
        material, _ = glass('Ideal n1.5 slab', 1.5)
        verts, faces = cube_mesh(1.0)
        verts = [(x * 0.1, y * 0.1, z * 0.01 + 0.015) for x, y, z in verts]
        obj = mesh_object('Glass slab', verts, faces, material, 'GLASS')
        return [obj], (0.0, 0.0, 0.0), (0.01, 0.0, 0.03), (0.0, 0.0, 0.0), 4

    def concave(scene, _):
        material, _ = glass('Ideal n1.5 L', 1.5)
        verts, faces = l_mesh()
        obj = mesh_object('Glass L concave', verts, faces, material, 'GLASS')
        return [obj], (-0.02, 0.0005, 0.008), (0.03, 0.0, 0.008), (0.0, 0.0, 0.0), 4

    def corner(scene, mirror_material):
        h = 0.02
        mirrors = [
            mesh_object('Mirror x0', [(0, 0, 0), (0, h, 0), (0, h, h), (0, 0, h)], [(0, 1, 2), (0, 2, 3)],
                        mirror_material, 'MIRROR'),
            mesh_object('Mirror y0', [(0, 0, 0), (h, 0, 0), (h, 0, h), (0, 0, h)], [(0, 1, 2), (0, 2, 3)],
                        mirror_material, 'MIRROR'),
            mesh_object('Mirror z0', [(0, 0, 0), (h, 0, 0), (h, h, 0), (0, h, 0)], [(0, 1, 2), (0, 2, 3)],
                        mirror_material, 'MIRROR'),
        ]
        return mirrors, (0.03, 0.026, 0.035), (0.033, 0.031, 0.03), (0.0, 0.0, 0.0), 3

    def medium(kind):
        def build(scene, _):
            material, _ = glass('Ideal n1.5 prism with medium', 1.5)
            tree = material.node_tree
            node = tree.nodes.new('ShaderNodeVolumeAbsorption' if kind == 'absorbing' else 'ShaderNodeVolumeScatter')
            node.inputs['Color'].default_value = (0.35, 0.7, 0.9, 1.0)
            node.inputs['Density'].default_value = 60.0
            tree.links.new(node.outputs[0], [n for n in tree.nodes if n.type == 'OUTPUT_MATERIAL'][0].inputs['Volume'])
            verts, faces = prism_mesh(0.008, 0.0139, 0.02)
            obj = mesh_object('Glass prism', verts, faces, material, 'GLASS')
            normal_tilt = (0.0, 0.0, 0.25)
            return [obj], (-0.03, 0.0, 0.004), (0.03, 0.0, -0.01), normal_tilt, 2
        return build

    for layout, build in [('prism_absorbing', medium('absorbing')), ('prism_scattering', medium('scattering')),
                          ('prism_TIR', prism), ('slab_long', slab), ('concave_L', concave),
                          ('mirror_corner', corner)]:
        folder = output / layout
        folder.mkdir()
        variants(folder, build)
    print('authored', output)


if __name__ == '__main__':
    main()
