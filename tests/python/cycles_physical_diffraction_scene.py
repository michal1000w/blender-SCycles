# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Save and optionally render actual Blender Diffraction BSDF control scenes.

Use a Blender build containing ShaderNodeBsdfDiffraction. These are planar
measurement fixtures; they do not replace disc presentation scenes.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import bpy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=['flat_metal', 'relief_metal', 'relief_dielectric'],
                        default='flat_metal')
    parser.add_argument('--illumination', choices=['both', 'reflection', 'transmission'], default='both')
    parser.add_argument('--angular-region', choices=['all', 'central', 'middle', 'outer'],
                        default='all')
    parser.add_argument('--device', choices=['CPU', 'METAL'], default='CPU')
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--transport', choices=['pt', 'bdpt', 'guided', 'bdpt_guided'], default='pt')
    parser.add_argument('--quality', choices=['FAST', 'REALISTIC'], default='REALISTIC')
    parser.add_argument('--osl', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--mix-mirror', action='store_true',
                        help='Mix the physical grating equally with a smooth unit mirror')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    if args.samples <= 0:
        parser.error('Samples must be positive')
    if args.osl and args.device != 'CPU':
        parser.error('This OSL fixture requires the CPU device')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = args.samples
    scene.cycles.seed = 11
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.shading_system = args.osl
    scene.cycles.time_limit = 0
    scene.cycles.use_layer_samples = 'IGNORE'
    scene.cycles.use_bidirectional_path_tracing = args.transport in ['bdpt', 'bdpt_guided']
    scene.cycles.use_guiding = args.transport in ['guided', 'bdpt_guided']
    scene.cycles.sample_clamp_direct = 0
    scene.cycles.sample_clamp_indirect = 0
    scene.cycles.max_bounces = 8
    scene.render.resolution_x = 128
    scene.render.resolution_y = 128
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.film_transparent = False
    if args.device == 'METAL':
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = 'METAL'
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == 'METAL'
        if not any(d.use and d.type == 'METAL' for d in preferences.devices):
            raise RuntimeError('No Metal device available')
        scene.cycles.device = 'GPU'
    else:
        scene.cycles.device = 'CPU'

    material = bpy.data.materials.new('Physical grating measurement')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    grating = nodes.new('ShaderNodeBsdfDiffraction')
    grating.quality = args.quality
    grating.name = 'Physical grating'
    grating.pitch = 740
    grating.depth = 0 if args.case == 'flat_metal' else 150
    grating.duty_cycle = 0.41
    if args.case == 'relief_dielectric':
        grating.ridge_ior, grating.ridge_extinction = 1.5, 0
        grating.substrate_ior, grating.substrate_extinction = 1, 0
    else:
        # Generic constant-index conductor, not a claim of measured aluminum.
        grating.ridge_ior, grating.ridge_extinction = 0.9, 6
        grating.substrate_ior, grating.substrate_extinction = 0.9, 6
    tangent = nodes.new('ShaderNodeCombineXYZ')
    tangent.inputs['X'].default_value = 1
    tangent.location = (-240, -100)
    material.node_tree.links.new(tangent.outputs['Vector'], grating.inputs['Tangent'])
    surface = nodes.new('ShaderNodeOutputMaterial')
    surface.location = (300, 0)
    material.node_tree.links.new(grating.outputs['BSDF'], surface.inputs['Surface'])
    if args.mix_mirror:
        mirror = nodes.new('ShaderNodeBsdfAnisotropic')
        mirror.name = 'Unit mirror'
        mirror.inputs['Color'].default_value = (1, 1, 1, 1)
        mirror.inputs['Roughness'].default_value = 0
        mix = nodes.new('ShaderNodeMixShader')
        mix.name = 'Equal grating mirror mixture'
        mix.inputs[0].default_value = .5
        material.node_tree.links.new(grating.outputs['BSDF'], mix.inputs[1])
        material.node_tree.links.new(mirror.outputs[0], mix.inputs[2])
        material.node_tree.links.new(mix.outputs[0], surface.inputs['Surface'])
    properties = ['pitch', 'depth', 'duty_cycle', 'incident_ior', 'ridge_ior',
                  'ridge_extinction', 'groove_ior', 'substrate_ior', 'substrate_extinction']
    expected = {name: getattr(grating, name) for name in properties}

    bpy.ops.mesh.primitive_plane_add(size=4)
    bpy.context.object.name = 'Planar grating, normal +Z, grooves along Y'
    bpy.context.object.data.materials.append(material)
    camera_data = bpy.data.cameras.new('Normal incidence camera')
    camera = bpy.data.objects.new('Normal incidence camera', camera_data)
    scene.collection.objects.link(camera)
    camera.location = (0, 0, 3)
    camera_data.type = 'ORTHO'
    camera_data.ortho_scale = 2
    scene.camera = camera
    world = bpy.data.worlds.new('Calibrated unit environment')
    world.use_nodes = True
    scene.world = world
    background = world.node_tree.nodes.get('Background')
    background.inputs['Color'].default_value = (1, 1, 1, 1)
    background.inputs['Strength'].default_value = 1
    if args.illumination != 'both' or args.angular_region != 'all':
        geometry = world.node_tree.nodes.new('ShaderNodeNewGeometry')
        separate = world.node_tree.nodes.new('ShaderNodeSeparateXYZ')
        world.node_tree.links.new(geometry.outputs['Incoming'], separate.inputs['Vector'])
    mask = None
    if args.illumination != 'both':
        hemisphere = world.node_tree.nodes.new('ShaderNodeMath')
        hemisphere.operation = 'LESS_THAN' if args.illumination == 'reflection' else 'GREATER_THAN'
        hemisphere.inputs[1].default_value = 0
        world.node_tree.links.new(separate.outputs['Z'], hemisphere.inputs[0])
        mask = hemisphere.outputs[0]
    if args.angular_region != 'all':
        absolute = world.node_tree.nodes.new('ShaderNodeMath')
        absolute.operation = 'ABSOLUTE'
        world.node_tree.links.new(separate.outputs['X'], absolute.inputs[0])
        bounds = [('LESS_THAN', 0.25)] if args.angular_region == 'central' else (
            [('GREATER_THAN', 0.9)] if args.angular_region == 'outer' else
            [('GREATER_THAN', 0.25), ('LESS_THAN', 0.9)])
        for operation, threshold in bounds:
            comparison = world.node_tree.nodes.new('ShaderNodeMath')
            comparison.operation = operation
            comparison.inputs[1].default_value = threshold
            world.node_tree.links.new(absolute.outputs[0], comparison.inputs[0])
            if mask is None:
                mask = comparison.outputs[0]
            else:
                product = world.node_tree.nodes.new('ShaderNodeMath')
                product.operation = 'MULTIPLY'
                world.node_tree.links.new(mask, product.inputs[0])
                world.node_tree.links.new(comparison.outputs[0], product.inputs[1])
                mask = product.outputs[0]
    if mask is not None:
        world.node_tree.links.new(mask, background.inputs['Strength'])

    suffix = '' if args.angular_region == 'all' else '_' + args.angular_region
    if args.mix_mirror:
        suffix += '_mirror_mix'
    prefix = output / (args.case + '_' + args.illumination + suffix)
    description = (
        'Physical diffraction measurement: normal incidence, tangent +X, pitch 740 nm.\n'
        'Environment radiance is one inside the selected angular region and zero outside.\n'
        'Measure the raw linear EXR mean; this deliberately uniform plane is a radiometric control.\n'
        'Angular regions use |Incoming.x|: central <0.25, middle 0.25..0.9, outer >0.9.\n'
        'For equal air media, order m has |direction.x| = |m| * wavelength / pitch.\n'
        'The outer boundary selects first-order wavelengths above 666 nm.\n'
        'This is not a diffraction screen or a cross-object coherence test.\n'
        f'Case: {args.case}; hemisphere: {args.illumination}; region: {args.angular_region}.\n')
    if args.mix_mirror:
        description += ('Equal physical-grating/unit-mirror mixture. The mirror direction overlaps '
                        'the grating zero reflection order; nonzero orders remain distinct.\n')
    bpy.data.texts.new('READ ME - measurement setup').write(description)
    material_name = material.name
    scene.render.filepath = str(prefix.with_suffix('.exr'))
    bpy.ops.wm.save_as_mainfile(filepath=str(prefix.with_suffix('.blend')))
    # Verify the real DNA/RNA serialization, including the tangent connection.
    bpy.ops.wm.open_mainfile(filepath=str(prefix.with_suffix('.blend')))
    restored = bpy.data.materials.get(material_name)
    if restored is None:
        raise RuntimeError('Material missing after loading saved scene')
    node = restored.node_tree.nodes['Physical grating']
    assert node.bl_idname == 'ShaderNodeBsdfDiffraction'
    assert node.quality == args.quality
    assert bpy.context.scene.cycles.shading_system == args.osl
    assert all(abs(getattr(node, name) - value) < 1e-6 for name, value in expected.items())
    assert node.inputs['Tangent'].is_linked
    if args.mix_mirror:
        assert restored.node_tree.nodes['Equal grating mirror mixture'].inputs[0].default_value == .5
        assert restored.node_tree.nodes['Unit mirror'].inputs['Roughness'].default_value == 0
    assert bpy.context.scene.cycles.use_bidirectional_path_tracing == (args.transport in ['bdpt', 'bdpt_guided'])
    assert bpy.context.scene.cycles.use_guiding == (args.transport in ['guided', 'bdpt_guided'])
    report = {'case': args.case, 'illumination': args.illumination, 'properties': expected,
              'quality': args.quality,
              'shading_system': 'OSL' if args.osl else 'SVM',
              'angular_region': args.angular_region,
              'mix_mirror': args.mix_mirror,
              'device_requested': args.device, 'samples': args.samples,
              'transport_requested': args.transport,
              'blender_version': bpy.app.version_string, 'serialization_verified': True,
              'status': 'saved_not_rendered', 'coherent_transport': False,
              'scope': 'Planar diffraction-node measurement fixture; requested quality is recorded separately. No modal-convergence or performance claim.',
              'blend_sha256': hashlib.sha256(prefix.with_suffix('.blend').read_bytes()).hexdigest()}
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.render:
        bpy.ops.render.render(write_still=True)
        if not prefix.with_suffix('.exr').exists():
            raise RuntimeError('Renderer did not write the expected EXR')
        report['status'] = 'rendered_unvalidated'
        report['exr_sha256'] = hashlib.sha256(prefix.with_suffix('.exr').read_bytes()).hexdigest()
        image = bpy.data.images.load(str(prefix.with_suffix('.exr')), check_existing=False)
        assert tuple(image.size) == (128, 128) and image.channels >= 3
        pixels = tuple(image.pixels[:])
        assert len(pixels) == 128 * 128 * image.channels
        if not all(math.isfinite(value) for value in pixels):
            raise RuntimeError('Nonfinite pixel in physical render')
        report['raw_mean_rgb'] = [statistics.fmean(pixels[c::image.channels]) for c in range(3)]
        analytic = None
        if args.case == 'flat_metal' and args.angular_region == 'all':
            n = complex(expected['substrate_ior'], expected['substrate_extinction'])
            analytic = 0 if args.illumination == 'transmission' else abs((1 - n) / (1 + n)) ** 2
        elif (args.case == 'relief_dielectric' and args.illumination == 'both' and
              args.angular_region == 'all'):
            analytic = 1
        if analytic is not None:
            if args.mix_mirror:
                analytic = .5 * analytic + (.5 if args.illumination != 'transmission' else 0)
            report['analytic_neutral_reference'] = analytic
            report['max_mean_error'] = max(abs(value - analytic) for value in report['raw_mean_rgb'])
            report['smoke_tolerance'] = 0.005
            report['status'] = 'analytic_smoke_pass' if report['max_mean_error'] < 0.005 else 'analytic_smoke_fail'
        prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
        if report['status'] == 'analytic_smoke_fail':
            raise RuntimeError('Physical Blender render failed analytic smoke comparison')


if __name__ == '__main__':
    main()
