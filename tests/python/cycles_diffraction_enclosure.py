# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Absolute geometric-series reference for transport in a closed emitting enclosure."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--kind', choices=['diffuse', 'mirror'], default='diffuse')
    parser.add_argument('--bounces', type=int, choices=range(0, 65), default=4)
    parser.add_argument('--total-bounces', type=int, choices=range(0, 65))
    parser.add_argument('--samples', type=int, default=512)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--resolution', type=int, default=128)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.samples < 1 or args.resolution < 16:
        parser.error('Require positive samples and resolution at least 16')
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.world = bpy.data.worlds.new('Black exterior')
    scene.world.use_nodes = True
    scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0
    bpy.ops.mesh.primitive_cube_add(size=2)
    wall = bpy.context.object
    wall.name = 'Closed uniform emitting enclosure'
    material = bpy.data.materials.new('Emission 0.25 plus reflectance 0.5')
    material.use_nodes = True
    material.cycles.emission_sampling = 'BACK'
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.clear()
    bsdf = nodes.new('ShaderNodeBsdfDiffuse' if args.kind == 'diffuse' else 'ShaderNodeBsdfAnisotropic')
    bsdf.inputs['Color'].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs['Roughness'].default_value = 0
    emission = nodes.new('ShaderNodeEmission')
    emission.inputs['Color'].default_value = (1, 1, 1, 1)
    emission.inputs['Strength'].default_value = 0.25
    add = nodes.new('ShaderNodeAddShader')
    links.new(bsdf.outputs[0], add.inputs[0])
    links.new(emission.outputs[0], add.inputs[1])
    output = nodes.new('ShaderNodeOutputMaterial')
    links.new(add.outputs[0], output.inputs['Surface'])
    wall.data.materials.append(material)
    bpy.ops.object.camera_add(location=(0.13, -0.07, 0.11))
    scene.camera = bpy.context.object
    scene.camera.rotation_euler = (0.13, 0.21, 0.07)
    scene.camera.data.lens = 40
    scene.camera.data.clip_start = 0.001
    scene.render.resolution_x = scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.cycles.samples = args.samples
    scene.cycles.seed = args.seed
    total_bounces = args.bounces if args.total_bounces is None else args.total_bounces
    scene.cycles.max_bounces = total_bounces
    scene.cycles.diffuse_bounces = args.bounces
    scene.cycles.glossy_bounces = args.bounces
    scene.cycles.transmission_bounces = args.bounces
    scene.cycles.use_denoising = False
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.sample_clamp_direct = scene.cycles.sample_clamp_indirect = 0
    scene.cycles.blur_glossy = 0
    scene.cycles.use_photon_mapping = False
    scene.cycles.use_guiding = False
    # Every ray hits another uniform wall. L_N = E + rho L_(N-1), L_0 = E.
    # Cycles allows user limit + 1 real scatters before emission-only termination.
    scatter_count = min(args.bounces, total_bounces) + 1
    expected = 0.25 * sum(0.5 ** k for k in range(scatter_count + 1))
    text = bpy.data.texts.new('READ ME - absolute reference')
    text.write('Uniform closed enclosure: emitted radiance E=0.25, reflectance rho=0.5.\n'
               'Every reflected ray encounters another wall with the same response.\n'
               'After N scatters, L=E*(1+rho+...+rho^N); infinite-depth L=0.5.\n'
               f'This scene uses N={scatter_count}; expected linear RGB={expected}.\n'
               'The uniform image is intentional: this measures absolute transport, not appearance.\n'
               'No diffraction is enabled; this isolates the shared integrator.\n')
    scene['analytic_radiance'] = expected
    reports = []
    if args.render:
        helper_path = Path(__file__).with_name('cycles_diffraction_indirect_control.py')
        spec = importlib.util.spec_from_file_location('control_provenance', helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == 'METAL'
        assert any(d.use and d.type == 'METAL' for d in prefs.devices)
    scene.cycles.device = 'GPU'
    for mode in ['pt', 'bdpt']:
        scene.cycles.use_bidirectional_path_tracing = mode == 'bdpt'
        stem = out / (args.kind + '_' + mode)
        scene.render.image_settings.file_format = 'OPEN_EXR'
        scene.render.image_settings.color_depth = '32'
        scene.render.filepath = str(stem.with_suffix('.exr'))
        bpy.ops.wm.save_as_mainfile(filepath=str(stem.with_suffix('.blend')))
        report = dict(kind=args.kind, transport=mode, expected_rgb=[expected]*3,
                      infinite_depth_rgb=[0.5]*3, user_bounces=args.bounces,
                      total_bounces=total_bounces,
                      allowed_scattering_events=scatter_count, seed=args.seed,
                      samples=args.samples, status='saved_not_rendered')
        if args.render:
            before = helper.render_provenance(stem.with_suffix('.blend'))
            before['fixture_script_sha256'] = helper.file_hash(__file__)
            bpy.ops.render.render(write_still=True)
            image = bpy.data.images.load(str(stem.with_suffix('.exr')), check_existing=False)
            pixels = np.empty(len(image.pixels), dtype=np.float32)
            image.pixels.foreach_get(pixels)
            rgb = pixels.reshape(image.size[1], image.size[0], image.channels)[:, :, :3].copy()
            assert np.isfinite(rgb).all()
            np.save(stem.with_suffix('.npy'), rgb)
            mean = rgb.mean((0, 1), dtype=np.float64)
            report.update(status='rendered_pending_analysis', mean_rgb=mean.tolist(),
                          mean_error_rgb=(mean-expected).tolist())
            after = helper.render_provenance(stem.with_suffix('.blend'))
            after['fixture_script_sha256'] = helper.file_hash(__file__)
            if before != after:
                raise RuntimeError('Fixture provenance changed during rendering')
            report['provenance'] = dict(before=before, after=after)
        report['sha256'] = {p.suffix[1:]: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in out.glob(stem.name+'.*')}
        stem.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
        reports.append(report)
    (out/'manifest.json').write_text(json.dumps(reports, indent=2)+'\n')


if __name__ == '__main__':
    main()
