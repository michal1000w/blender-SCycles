# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Non-diffraction control for the physical grating room and BDPT diagnostics."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import bpy
import numpy as np


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def render_provenance(input_path):
    root = Path(os.environ['CYCLES_KERNEL_PATH']).resolve()
    if not (root / 'kernel').is_dir():
        raise RuntimeError('CYCLES_KERNEL_PATH must identify the tested source tree')
    fingerprint = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in {'.h', '.cpp', '.metal', '.cl', '.cu', '.osl'}:
            fingerprint.update(str(path.relative_to(root)).encode() + b'\0')
            fingerprint.update(bytes.fromhex(file_hash(path)))
            count += 1
    return dict(input_sha256=file_hash(input_path), binary_sha256=file_hash(bpy.app.binary_path),
                script_sha256=file_hash(__file__), kernel_root=str(root),
                kernel_sha256=fingerprint.hexdigest(), kernel_files=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['diffuse', 'mirror', 'mixed', 'translucent'],
                        default='diffuse')
    parser.add_argument('--save-only', action='store_true')
    parser.add_argument('--emitter', choices=['sphere', 'compatibility'], default='sphere')
    parser.add_argument('--persistent', action='store_true')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--diffuse-bounces', type=int, choices=range(0, 1025))
    parser.add_argument('--glossy-bounces', type=int, choices=range(0, 1025))
    parser.add_argument('--transmission-bounces', type=int, choices=range(0, 1025))
    parser.add_argument('--volume-bounces', type=int, choices=range(0, 1025))
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--resolution', type=int, default=160)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.samples <= 0 or args.resolution <= 0:
        parser.error('samples and resolution must be positive')
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = None if args.save_only else render_provenance(args.input)
    if provenance is not None:
        (args.output / 'provenance.json').write_text(
            json.dumps(dict(status='running', before=provenance), indent=2) + '\n')
    bpy.ops.wm.open_mainfile(filepath=str(args.input.resolve()))
    scene = bpy.context.scene
    scene.objects['Restricted white source'].data.use_soft_falloff = args.emitter == 'compatibility'
    material = bpy.data.objects['Grating tile'].data.materials[0]
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    shader_type = {'diffuse': 'ShaderNodeBsdfDiffuse', 'mirror': 'ShaderNodeBsdfAnisotropic',
                   'mixed': 'ShaderNodeBsdfDiffuse', 'translucent': 'ShaderNodeBsdfTranslucent'}
    shader = nodes.new(shader_type[args.kind])
    shader.inputs['Color'].default_value = (0.75, 0.75, 0.75, 1)
    if 'Roughness' in shader.inputs:
        shader.inputs['Roughness'].default_value = 0
    if args.kind == 'mixed':
        glossy = nodes.new('ShaderNodeBsdfAnisotropic')
        glossy.inputs['Color'].default_value = (0.75, 0.75, 0.75, 1)
        glossy.inputs['Roughness'].default_value = 0.2
        mix = nodes.new('ShaderNodeMixShader')
        mix.inputs[0].default_value = 0.5
        links.new(shader.outputs[0], mix.inputs[1])
        links.new(glossy.outputs[0], mix.inputs[2])
        shader = mix
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(shader.outputs[0], output.inputs['Surface'])
    scene['validation_status'] = 'Non-diffraction control: ' + args.kind
    material.name = 'Transport control - ' + args.kind
    if 'test_parameters' in material:
        del material['test_parameters']
    material['test_parameters'] = json.dumps(dict(
        closure=args.kind, color=[0.75, 0.75, 0.75], roughness=0,
        diffraction_enabled=False,
        mixed_glossy_roughness=0.2 if args.kind == 'mixed' else None,
        mix_factor=0.5 if args.kind == 'mixed' else None))
    notes = bpy.data.texts.get('READ ME - transport control') or bpy.data.texts.new(
        'READ ME - transport control')
    notes.clear()
    notes.write(
        'Non-diffraction transport control\n'
        'The grating tile is replaced by a built-in ' + args.kind + ' closure.\n'
        'The neutral room, occluder and sphere expose indirect transport.\n'
        'Compare PT and BDPT with matched seeds, then repeat independent seeds.\n'
        'The sphere emitter exercises forward light paths; the compatibility\n'
        'emitter tests the camera-path fallback. No diffraction or coherence\n'
        'is tested by this scene. Raw EXR values are the measurement output.\n')
    if args.kind == 'mixed':
        notes.write('The tile mixes equal diffuse and rough glossy closures (roughness 0.2).\n'
                    'It tests separate diffuse/glossy bounce budgets within one node graph.\n')
    if args.kind == 'translucent':
        notes.write('The tile uses diffuse transmission. This must consume transmission\n'
                    'bounces even though its light-pass classification is diffuse.\n')
        tile = bpy.data.objects['Grating tile']
        tile.location = (0, 0, 0.22)
        tile.rotation_euler = (math.radians(60), 0, 0)
        source = scene.objects['Restricted white source']
        source.rotation_euler = (tile.location - source.location).to_track_quat('-Z', 'Y').to_euler()
        notes.write('The tile is raised to 0.22 m and tilted 60 degrees so transmitted\n'
                    'illumination can reach visible floor and rear receivers, rather than\n'
                    'being hidden immediately beneath a flush floor tile.\n')
    for name in ('diffuse_bounces', 'glossy_bounces', 'transmission_bounces', 'volume_bounces'):
        value = getattr(args, name)
        if value is not None:
            setattr(scene.cycles, name, value)
    scene.cycles.samples = args.samples
    scene.cycles.seed = args.seed
    scene.cycles.use_guiding = False
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.render.use_persistent_data = args.persistent
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = round(args.resolution * 136 / 160)
    scene.render.resolution_percentage = 100
    preferences = bpy.context.preferences.addons['cycles'].preferences
    if not args.save_only:
        preferences.compute_device_type = 'METAL'
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == 'METAL'
        assert any(device.use and device.type == 'METAL' for device in preferences.devices)
    scene.cycles.device = 'GPU'
    reports = []
    for mode in ['pt', 'bdpt']:
        scene.cycles.use_bidirectional_path_tracing = mode == 'bdpt'
        prefix = args.output / (args.kind + '_' + mode)
        if prefix.with_suffix('.json').exists() or prefix.with_suffix('.blend').exists():
            raise FileExistsError('Use a fresh output directory')
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.filepath = str(prefix.with_suffix('.exr'))
        bpy.ops.wm.save_as_mainfile(filepath=str(prefix.with_suffix('.blend')))
        if args.save_only:
            reports.append(dict(kind=args.kind, transport=mode, status='saved_not_rendered',
                                blend_sha256=hashlib.sha256(
                                    prefix.with_suffix('.blend').read_bytes()).hexdigest()))
            continue
        bpy.ops.render.render(write_still=True)
        image = bpy.data.images.load(str(prefix.with_suffix('.exr')), check_existing=False)
        pixels = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(pixels)
        rgb = pixels.reshape(image.size[1], image.size[0], image.channels)[:, :, :3].copy()
        assert np.isfinite(rgb).all()
        np.save(prefix.with_suffix('.npy'), rgb)
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '8'
        bpy.data.images['Render Result'].save_render(str(prefix.with_suffix('.png')), scene=scene)
        report = dict(kind=args.kind, emitter=args.emitter, diffraction_enabled=False, transport=mode,
                      persistent_data=args.persistent, seed=args.seed, samples=args.samples,
                      bounce_limits={key: getattr(scene.cycles, key) for key in
                                     ['max_bounces', 'diffuse_bounces', 'glossy_bounces',
                                      'transmission_bounces', 'volume_bounces']},
                      resolution=[scene.render.resolution_x, scene.render.resolution_y],
                      raw_mean_rgb=rgb.mean(axis=(0, 1), dtype=np.float64).tolist(),
                      status='control_render_pending_analysis',
                      sha256={ext: hashlib.sha256(prefix.with_suffix('.' + ext).read_bytes()).hexdigest()
                              for ext in ['blend', 'exr', 'npy', 'png']})
        prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
    if provenance is not None:
        after = render_provenance(args.input)
        if after != provenance:
            raise RuntimeError('Input, binary, script or kernel source changed during rendering')
        (args.output / 'provenance.json').write_text(
            json.dumps(dict(status='complete', before=provenance, after=after), indent=2) + '\n')
    (args.output / 'manifest.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
