# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Render a saved physical grating scene with matched transport variants/seeds.

Keeps one Blender scene alive to permit cache reuse. Reports observations, not
an automatic convergence verdict; uncertainty must be estimated across seeds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import bpy
import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def kernel_fingerprint():
    root = Path(os.environ['CYCLES_KERNEL_PATH']).resolve()
    if not (root / 'kernel').is_dir():
        raise RuntimeError('CYCLES_KERNEL_PATH must identify the tested Cycles source tree')
    result = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix in {'.h', '.cpp', '.metal', '.cl', '.cu', '.osl'}:
            result.update(str(path.relative_to(root)).encode() + b'\0')
            result.update(bytes.fromhex(digest(path)))
            count += 1
    return dict(sha256=result.hexdigest(), files=count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--transports', nargs='+', choices=['pt', 'bdpt', 'guided', 'bdpt_guided'],
                        default=['pt', 'bdpt_guided'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 29, 47, 83])
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--resolution', type=int, default=320)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if (len(set(args.seeds)) != len(args.seeds) or len(args.seeds) < 2 or
            any(seed < 0 or seed > 2147483647 for seed in args.seeds) or
            len(set(args.transports)) != len(args.transports) or
            args.samples < 1 or args.resolution < 16):
        parser.error('Require distinct seeds/transports, at least two seeds, and positive dimensions')
    args.input = args.input.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / 'manifest.json'
    if manifest_path.exists():
        raise FileExistsError('Use a new output directory for each batch')
    bpy.ops.wm.open_mainfile(filepath=str(args.input))
    scene = bpy.context.scene
    assert scene.render.engine == 'CYCLES'
    assigned = {slot.material for obj in scene.objects for slot in obj.material_slots
                if slot.material is not None and slot.material.use_nodes}
    assert any(node.bl_idname == 'ShaderNodeBsdfDiffraction'
               for material in assigned for node in material.node_tree.nodes)
    assert not any('Diffraction Pitch' in node.inputs
                   for material in assigned for node in material.node_tree.nodes)
    scene.render.use_persistent_data = True
    scene.render.resolution_y = round(args.resolution * scene.render.resolution_y /
                                      scene.render.resolution_x)
    scene.render.resolution_x = args.resolution
    scene.render.resolution_percentage = 100
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.sample_clamp_direct = 0
    scene.cycles.sample_clamp_indirect = 0
    scene.cycles.blur_glossy = 0
    scene.cycles.use_photon_mapping = False
    if not args.validate_only:
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = 'METAL'
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == 'METAL'
        if not any(device.use and device.type == 'METAL' for device in preferences.devices):
            raise RuntimeError('No Metal device available')
        scene.cycles.device = 'GPU'
    binary = Path(bpy.app.binary_path)
    binary_hash = digest(binary)
    kernel_hash = None if args.validate_only else kernel_fingerprint()
    manifest = dict(status='validating' if args.validate_only else 'running',
                    input=str(args.input), input_sha256=digest(args.input),
                    binary_sha256=binary_hash, script_sha256=digest(Path(__file__)),
                    kernel_source_fingerprint=kernel_hash,
                    emitters=[dict(name=light.name, type=light.type, energy=light.energy,
                                   use_soft_falloff=getattr(light, 'use_soft_falloff', None),
                                   radius=getattr(light, 'shadow_soft_size', None))
                              for light in bpy.data.lights],
                    bdpt_light_paths=scene.cycles.bdpt_light_paths,
                    bdpt_update_samples=scene.cycles.bdpt_update_samples,
                    caustics_reflective=scene.cycles.caustics_reflective,
                    caustics_refractive=scene.cycles.caustics_refractive,
                    samples=args.samples, seeds=args.seeds, transports=args.transports,
                    persistent_data=True, isolated_benchmark=False, runs=[])

    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')

    write_manifest()
    try:
        for seed in args.seeds:
            for transport in args.transports:
                scene.cycles.seed = seed
                scene.cycles.use_bidirectional_path_tracing = transport in ['bdpt', 'bdpt_guided']
                scene.cycles.use_guiding = transport in ['guided', 'bdpt_guided']
                assert scene.cycles.use_bidirectional_path_tracing == ('bdpt' in transport)
                assert scene.cycles.use_guiding == ('guided' in transport)
                prefix = args.output / f'{transport}_seed{seed}'
                report = dict(transport=transport, seed=seed, samples=args.samples,
                              resolution=[scene.render.resolution_x, scene.render.resolution_y])
                if args.validate_only:
                    report['status'] = 'settings_validated_not_rendered'
                else:
                    scene.render.image_settings.file_format = 'OPEN_EXR'
                    scene.render.image_settings.color_depth = '32'
                    scene.render.filepath = str(prefix.with_suffix('.exr'))
                    bpy.ops.wm.save_as_mainfile(filepath=str(prefix.with_suffix('.blend')))
                    start = time.perf_counter()
                    bpy.ops.render.render(write_still=True)
                    report['wall_seconds_including_updates'] = time.perf_counter() - start
                    image = bpy.data.images.load(str(prefix.with_suffix('.exr')), check_existing=False)
                    width, height = image.size
                    assert (width, height) == tuple(report['resolution']) and image.channels >= 3
                    pixels = np.empty(width * height * image.channels, dtype=np.float32)
                    image.pixels.foreach_get(pixels)
                    rgb = pixels.reshape(height, width, image.channels)[:, :, :3].copy()
                    bpy.data.images.remove(image)
                    if not np.isfinite(rgb).all():
                        raise RuntimeError('Nonfinite raw render pixels')
                    np.save(prefix.with_suffix('.npy'), rgb)
                    report['raw_mean_rgb'] = rgb.mean(axis=(0, 1), dtype=np.float64).tolist()
                    report['array_origin'] = 'bottom_left'
                    scene.render.image_settings.file_format = 'PNG'
                    scene.render.image_settings.color_depth = '8'
                    bpy.data.images['Render Result'].save_render(str(prefix.with_suffix('.png')), scene=scene)
                    report['sha256'] = {ext: digest(prefix.with_suffix('.' + ext))
                                        for ext in ['blend', 'exr', 'npy', 'png']}
                    report['status'] = 'rendered_pending_convergence_analysis'
                prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
                manifest['runs'].append(report)
                write_manifest()
        if digest(binary) != binary_hash:
            raise RuntimeError('Blender binary changed during the batch')
        if not args.validate_only and kernel_fingerprint() != kernel_hash:
            raise RuntimeError('Cycles sources changed during the batch')
        manifest['status'] = 'settings_validated' if args.validate_only else 'rendered_pending_analysis'
        write_manifest()
    except Exception as error:
        manifest['status'] = 'failed'
        manifest['error'] = str(error)
        write_manifest()
        raise


if __name__ == '__main__':
    main()
