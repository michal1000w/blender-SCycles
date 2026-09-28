# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Render the saved full-domain lossless furnace across four transport modes.

Requires the relief_dielectric/both/all fixture created by
cycles_physical_diffraction_scene.py. Unit radiance is the analytic reference.
Persistent cache reuse must be verified separately from the --debug-cycles log.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=1024)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.samples <= 0:
        parser.error('Samples must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    nodes = [n for m in bpy.data.materials if m.node_tree for n in m.node_tree.nodes
             if n.bl_idname == 'ShaderNodeBsdfDiffraction']
    if len(nodes) != 1 or nodes[0].quality != 'REALISTIC':
        raise RuntimeError('Expected one Realistic grating in the saved furnace')
    node = nodes[0]
    if (node.ridge_extinction != 0 or node.substrate_extinction != 0 or
            node.depth != 150 or node.pitch != 740):
        raise RuntimeError('Unexpected furnace material')
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'METAL'
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == 'METAL' and 'Apple M5' in device.name
    devices = [d.name for d in preferences.devices if d.use]
    if len(devices) != 1:
        raise RuntimeError('Expected exactly one Apple M5 Metal device')
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = args.samples
    scene.cycles.seed = 11
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.time_limit = 0
    scene.render.use_persistent_data = True
    scene.view_settings.view_transform = 'Standard'
    scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1
    report = dict(devices=devices, samples=args.samples, adaptive=False, denoising=False,
                  analytic_reference=1.0, mean_error_gate=0.005, results=[],
                  scope='Analytic furnace regression; fixed N16 material reference. '
                        'Not modal convergence, appearance, or repeated performance benchmark.')
    for label, bdpt, guiding in [('pt', False, False), ('bdpt', True, False),
                                 ('guided', False, True), ('bdpt_guided', True, True)]:
        scene.cycles.use_bidirectional_path_tracing = bdpt
        scene.cycles.use_guiding = guiding
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        path = output / (label + '.exr')
        scene.render.filepath = str(path)
        bpy.ops.wm.save_as_mainfile(filepath=str(output / (label + '.blend')))
        start = time.monotonic()
        bpy.ops.render.render(write_still=True)
        seconds = time.monotonic() - start
        image = bpy.data.images.load(str(path), check_existing=False)
        pixels = np.array(image.pixels[:], dtype=np.float64).reshape(-1, image.channels)[:, :3]
        finite = bool(np.all(np.isfinite(pixels)))
        mean = pixels.mean(axis=0)
        error = float(np.max(np.abs(mean - 1))) if finite else None
        row = dict(transport=label, bdpt=bool(scene.cycles.use_bidirectional_path_tracing),
                   guiding=bool(scene.cycles.use_guiding), finite=finite, mean_rgb=mean.tolist(),
                   max_mean_error=error, passed=finite and error < 0.005,
                   seconds=seconds, exr_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        report['results'].append(row)
        (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '8'
        image.save_render(str(output / (label + '.png')), scene=scene)
    if not all(row['passed'] for row in report['results']):
        raise RuntimeError('Furnace regression failed; see per-method report')


if __name__ == '__main__':
    main()
