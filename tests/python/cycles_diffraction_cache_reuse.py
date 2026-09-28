# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Render a saved Realistic fixture across color edits with persistent data.

Run with --debug-cycles and retain its log to verify Built/Reused cache events.
The image check alone cannot prove cache reuse or absence of initialization.
"""
import argparse
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device-name', default='Apple M5')
    parser.add_argument('--samples', type=int, default=64)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.samples <= 0:
        parser.error('Samples must be positive')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scene = bpy.context.scene
    nodes = [node for material in bpy.data.materials if material.node_tree
             for node in material.node_tree.nodes
             if node.bl_idname == 'ShaderNodeBsdfDiffraction']
    if len(nodes) != 1 or nodes[0].quality != 'REALISTIC':
        raise RuntimeError('Expected one Realistic diffraction material')
    node = nodes[0]
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'METAL'
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == 'METAL' and args.device_name in device.name
    devices = [device.name for device in preferences.devices if device.use]
    if len(devices) != 1:
        raise RuntimeError('Expected exactly one selected Metal device')
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = args.samples
    scene.cycles.seed = 11
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    images = []
    original = tuple(node.inputs['Color'].default_value)
    try:
        for label, scale in [('initial', 1.0), ('edited', 0.5), ('restored', 1.0)]:
            node.inputs['Color'].default_value = tuple(v * scale for v in original[:3]) + (original[3],)
            scene.render.filepath = str(output / (label + '.exr'))
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(scene.render.filepath, check_existing=False)
            pixels = np.array(image.pixels[:], dtype=np.float64).reshape(-1, image.channels)[:, :3]
            if not np.all(np.isfinite(pixels)):
                raise RuntimeError('Nonfinite rendered pixels')
            images.append(pixels)
    finally:
        node.inputs['Color'].default_value = original
    denominator = max(float(np.abs(images[0]).sum()), 1e-30)
    restored_error = float(np.abs(images[2] - images[0]).sum()) / denominator
    edited_change = float(np.abs(images[1] - images[0]).sum()) / denominator
    report = dict(devices=devices, samples=args.samples, adaptive=False, denoising=False,
                  restored_relative_l1=restored_error, edited_relative_l1=edited_change,
                  cache_reuse_requires_log_verification=True,
                  scope='Persistent-data application regression, not a performance benchmark')
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if restored_error > 1e-5 or edited_change < 0.1:
        raise RuntimeError('Color update/restoration regression failed; see report')


if __name__ == '__main__':
    main()
