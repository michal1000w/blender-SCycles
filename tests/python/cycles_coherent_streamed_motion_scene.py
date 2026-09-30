#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Author a streamed Glass motion-blur fixture (Blender only, no render calls).

  blender --background --factory-startup --python THIS -- OUTPUT_DIR

prism_motion: the v46 prism TT layout with the Glass prism translating linearly
through the shutter. The saved scene's exact shutter-open and shutter-close
object positions are recorded; the oracle time-averages over the shutter.
"""
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_coherent_mirror_acceptance import set_diffuse_material
from cycles_coherent_polarizer_scene import glass, pane
from cycles_coherent_streamed_glass_scene import (add_camera, add_sources, base_scene, mesh_object,
                                                  prism_mesh, record)


def main():
    output = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
    folder = output / 'prism_motion'
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('phase_0', 'phase_pi'):
        scene, _ = base_scene()
        scene.render.use_motion_blur = True
        scene.render.motion_blur_shutter = 0.5
        scene.render.motion_blur_position = 'CENTER'
        material, _ = glass('Ideal n1.5 moving prism', 1.5)
        verts, faces = prism_mesh(0.008, 0.0139, 0.02)
        prism = mesh_object('Glass prism', verts, faces, material, 'GLASS')
        # Linear translation along +x and -z: 4 mm per frame over frames 0..2.
        for frame, x in ((0, -0.004), (2, 0.004)):
            prism.location = (x, 0.0, -0.25 * x)
            prism.keyframe_insert('location', frame=frame)
        for curve in prism.animation_data.action.fcurves if hasattr(prism.animation_data.action, 'fcurves') else []:
            for key in curve.keyframe_points:
                key.interpolation = 'LINEAR'
        try:
            for layer in prism.animation_data.action.layers:
                for strip in layer.strips:
                    for bag in strip.channelbags:
                        for curve in bag.fcurves:
                            for key in curve.keyframe_points:
                                key.interpolation = 'LINEAR'
        except AttributeError:
            pass
        scene.frame_set(1)
        receiver = np.array([0.03, 0.0, -0.01])
        towards = np.array([-0.03, 0.0, 0.004]) - receiver
        towards = towards / np.linalg.norm(towards) + np.array([0.0, 0.0, 0.25])
        normal = towards / np.linalg.norm(towards)
        u = np.cross(normal, [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        detector = pane('Unit Lambertian detector', receiver, u, v, 0.002,
                        set_diffuse_material('Unit detector', (1, 1, 1)), 'DETECTOR')
        sources = add_sources([(-0.03, 0.0, 0.004), (-0.03, 0.0006, 0.0046)], 100000.0, 10.0)
        if name == 'phase_pi':
            sources[1].data.cycles.coherence_phase = math.pi
        camera = add_camera(scene, receiver, normal, 0.0003)
        geometry = record(folder, name, scene, [prism], detector, sources, camera, 2)
        # Exact object positions at shutter open/close (centered shutter 0.5).
        center = np.array(prism.matrix_world.translation)
        offsets = {}
        for label, sub in (('offset_open', 0.75), ('offset_close', 1.25)):
            scene.frame_set(int(math.floor(sub)), subframe=sub - math.floor(sub))
            offsets[label] = (np.array(prism.matrix_world.translation) - center).tolist()
        scene.frame_set(1)
        bpy.ops.wm.save_as_mainfile(filepath=geometry['blend'])
        geometry['motion'] = {prism.name: offsets}
        import hashlib
        geometry['blend_sha256'] = hashlib.sha256(Path(geometry['blend']).read_bytes()).hexdigest()
        (folder / (name + '_geometry.json')).write_text(json.dumps(geometry, indent=1) + '\n')
        print(name, offsets)


if __name__ == '__main__':
    main()
