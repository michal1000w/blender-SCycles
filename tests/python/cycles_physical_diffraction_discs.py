# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Physical smooth lamellar disc and indirect-light presentation fixtures."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cycles_diffraction_suite import aim, discs, indirect


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=['discs', 'covered', 'indirect'], default='discs')
    parser.add_argument('--samples', type=int, default=1024)
    parser.add_argument('--resolution', type=int, default=960)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    parser.add_argument('--transport', choices=['pt', 'bdpt', 'guided', 'bdpt_guided'], default='pt')
    parser.add_argument('--quality', choices=['FAST', 'REALISTIC'], default='REALISTIC')
    parser.add_argument('--optical-constants', type=Path,
                        help='Embed a conductor CSV with wavelength_nm,n,k columns.')
    parser.add_argument('--lighting', choices=['pilot', 'broad'], default='broad',
                        help='Broad uses two white strips to cover more diffraction angles.')
    parser.add_argument('--emitter', choices=['sphere', 'compatibility'], default='sphere',
                        help='Indirect fixture: physical sphere or camera-facing soft-light approximation.')
    parser.add_argument('--indirect-source-radius-mm', type=float, default=5.0,
                        help='Physical source radius; use a separate output directory for each size.')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    if args.samples < 1 or args.resolution < 16 or not 0 <= args.seed <= 2147483647:
        parser.error('Invalid samples, resolution or seed')
    if not 0 < args.indirect_source_radius_mm <= 100:
        parser.error('Indirect source radius must be in (0, 100] mm')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    prefix = args.output / ('physical_' + args.case + '_' + args.transport)
    if any(prefix.with_suffix(ext).exists() for ext in ('.blend', '.json', '.exr', '.png')):
        raise RuntimeError('Refusing to overwrite an existing fixture or render')
    scene, devices, parameters = (indirect(args) if args.case == 'indirect' else
                                 discs(args, covered=args.case=='covered'))
    scene.cycles.time_limit = 0
    scene.cycles.use_layer_samples = 'IGNORE'
    if args.case == 'covered':
        # Keep the whole disc and its label within the presentation frame.
        scene.camera.data.ortho_scale = 0.20
    if args.case == 'indirect':
        scene.objects['Restricted white source'].data.use_soft_falloff = args.emitter == 'compatibility'
        scene.objects['Restricted white source'].data.shadow_soft_size = args.indirect_source_radius_mm * 0.001
        source = scene.objects['Restricted white source']
        source.location.z = min(0.48, 0.5 - source.data.shadow_soft_size - 0.01)
        aim(source, (0, 0.03, 0))
    if args.case in ('discs', 'covered') and args.lighting == 'broad':
        first = scene.objects['White studio strip']
        second = first.copy()
        second.data = first.data.copy()
        second.name = 'White studio strip - opposite azimuth'
        scene.collection.objects.link(second)
        second.location = (-0.18, -0.20, 0.45)
        aim(second, (0, 0, 0))
    # Reuse geometry and lighting only. Replace every used provisional grating
    # material; preserve the explicit radial object-to-world tangent connection.
    materials = {slot.material for obj in scene.objects for slot in obj.material_slots
                 if slot.material is not None}
    profiles = []
    optical_text = None
    optical_contents = None
    if args.optical_constants:
        optical_contents = args.optical_constants.read_text()
        optical_text = bpy.data.texts.new('Conductor optical constants - ' + args.optical_constants.name)
        optical_text.write(optical_contents)
    for material in materials:
        if not material.use_nodes:
            continue
        nodes, links = material.node_tree.nodes, material.node_tree.links
        # Only replace the original reflective grating, never the dielectric cover.
        old = next((node for node in nodes
                    if node.bl_idname == 'ShaderNodeBsdfAnisotropic'
                    and 'Diffraction Pitch' in node.inputs), None)
        if old is None:
            continue
        if old.inputs['Tangent'].is_linked:
            tangent_source = old.inputs['Tangent'].links[0].from_socket
        else:
            tangent = nodes.new('ShaderNodeCombineXYZ')
            tangent.inputs['X'].default_value = 1
            tangent_source = tangent.outputs['Vector']
        destinations = [link.to_socket for link in old.outputs['BSDF'].links]
        node = nodes.new('ShaderNodeBsdfDiffraction')
        node.quality = args.quality
        node.name = 'Physical lamellar grating'
        node.pitch = old.inputs['Diffraction Pitch'].default_value
        node.incident_ior = old.inputs['Diffraction Medium IOR'].default_value
        # The enclosing dielectric fills the relief as well as the upper medium.
        node.groove_ior = node.incident_ior
        node.depth, node.duty_cycle = 150, 0.41
        node.ridge_ior, node.ridge_extinction = 0.9, 6
        node.substrate_ior, node.substrate_extinction = 0.9, 6
        node.optical_constants = optical_text
        node.inputs['Color'].default_value = (1, 1, 1, 1)
        links.new(tangent_source, node.inputs['Tangent'])
        for socket in destinations:
            links.new(node.outputs['BSDF'], socket)
        nodes.remove(old)
        profiles.append({'material': material.name, 'pitch_nm': node.pitch,
                         'depth_nm': node.depth, 'duty_cycle': node.duty_cycle,
                         'incident_ior': node.incident_ior,
                         'groove_ior': node.groove_ior,
                         'index': [node.substrate_ior, node.substrate_extinction],
                         'index_source': 'embedded_csv' if optical_text else 'constant',
                         'optical_constants_sha256': digest(args.optical_constants) if optical_text else None})
        material['test_parameters'] = json.dumps(profiles[-1], sort_keys=True)
    assert sorted(profile['pitch_nm'] for profile in profiles) == (
        [1600] if args.case in ('indirect', 'covered') else [740, 1600])
    if args.case == 'indirect':
        for parameter in parameters:
            parameter.pop('roughness', None)
            parameter['surface_model'] = 'smooth physical lamellar grating'
            parameter['emitter_model'] = args.emitter
            parameter['source_radius_mm'] = args.indirect_source_radius_mm
            parameter['source_position_m'] = list(scene.objects['Restricted white source'].location)
    # Remove the unused provisional template too, so inspection cannot confuse it
    # with either of the materials actually assigned to a reflector.
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    scene['validation_status'] = args.quality + ' diffraction model; presentation awaiting validation'
    disc_description = ('One 120 mm CD under a 1.2 mm clear cover, IOR 1.58.\n'
                        'The grating upper medium and filled grooves are 1.58; the cover uses explicit glass interfaces.\n'
                        'Pitch 1600 nm, radial tangent across concentric grooves.\n'
                        if args.case == 'covered' else
                        'Two 120 mm discs, with radial tangent across concentric grooves.\n'
                        'Pitches: DVD 740 nm, CD 1600 nm.\n')
    setup_notes = (disc_description +
                   f'White strip lighting preset: {args.lighting}.\n'
                   if args.case in ('discs', 'covered') else
                   'Smooth 1600 nm grating tile, tangent +X, in an achromatic 0.5 m room.\n'
                   'Restricted white spotlight illuminates the tile; walls receive indirect light.\n'
                   f'Emitter model: {args.emitter}; compatibility emitters use the camera-path fallback.\n'
                   f'Source radius: {args.indirect_source_radius_mm:g} mm.\n'
                   'Occluder and sphere test shadowing and multiple surface interactions.\n'
                   'Compare identical seeds/sample budgets across PT, BDPT and guiding variants.\n')
    material_notes = (f'Embedded wavelength-dependent conductor data: {args.optical_constants.name}.\n'
                      if optical_text else
                      'Generic conductor n=0.9+6i, constant with wavelength; not measured aluminum.\n')
    notes = (setup_notes + f'Quality: {args.quality}.\n' + 'Depth 150 nm; duty 0.41.\n' + material_notes +
             'Smooth lamellar surface, no recorded pits or coherent transport.\n'
             'Only the covered fixture includes a dielectric cover; its IOR is constant with wavelength.\n'
             'Use calibrated planar controls for absolute efficiency; this scene tests appearance and transport.\n')
    bpy.data.texts.new('READ ME - physical grating fixture').write(notes)
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    scene.render.filepath = str(prefix.with_suffix('.exr'))
    bpy.ops.wm.save_as_mainfile(filepath=str(prefix.with_suffix('.blend')))
    bpy.ops.wm.open_mainfile(filepath=str(prefix.with_suffix('.blend')))
    scene = bpy.context.scene
    assert scene.cycles.seed == args.seed and scene.cycles.samples == args.samples
    assert scene.cycles.use_bidirectional_path_tracing == (args.transport in ['bdpt', 'bdpt_guided'])
    assert scene.cycles.use_guiding == (args.transport in ['guided', 'bdpt_guided'])
    assert not scene.cycles.use_adaptive_sampling and not scene.cycles.use_denoising
    assert scene.cycles.time_limit == 0 and scene.cycles.use_layer_samples == 'IGNORE'
    if args.case == 'covered':
        cover = scene.objects['1.2 mm enclosing cover']
        glass = next(node for node in cover.active_material.node_tree.nodes
                     if node.bl_idname == 'ShaderNodeBsdfGlass')
        assert abs(glass.inputs['IOR'].default_value - 1.58) < 1e-6
        assert abs(cover.dimensions.z - 0.001201) < 1e-7
        assert all(abs(profile['incident_ior'] - 1.58) < 1e-6 and
                   abs(profile['groove_ior'] - 1.58) < 1e-6 for profile in profiles)
    if args.case == 'indirect':
        assert scene.objects['Restricted white source'].data.use_soft_falloff == (args.emitter == 'compatibility')
        source = scene.objects['Restricted white source']
        assert source.location.z + source.data.shadow_soft_size < 0.5
        assert source.location.z - source.data.shadow_soft_size > 0.13
        assert abs(scene.objects['Restricted white source'].data.shadow_soft_size -
                   args.indirect_source_radius_mm * 0.001) < 1e-7
    for profile in profiles:
        node = bpy.data.materials[profile['material']].node_tree.nodes['Physical lamellar grating']
        assert node.bl_idname == 'ShaderNodeBsdfDiffraction' and node.inputs['Tangent'].is_linked
        assert node.quality == args.quality
        if optical_contents is not None:
            assert node.optical_constants.as_string() == optical_contents
        assert abs(node.pitch - profile['pitch_nm']) < 1e-6
        assert abs(node.depth - profile['depth_nm']) < 1e-6
        assert abs(node.duty_cycle - profile['duty_cycle']) < 1e-6
        assert abs(node.incident_ior - profile['incident_ior']) < 1e-6
        assert abs(node.groove_ior - profile['groove_ior']) < 1e-6
    report = dict(status='saved_not_rendered', physical_cache_integrated=args.quality=='REALISTIC',
                  quality=args.quality,
                  coherent_transport_integrated=False, profiles=profiles, geometry=parameters,
                  device_requested=args.device, devices=devices, transport=args.transport,
                  case=args.case,
                  lighting=args.lighting if args.case in ('discs', 'covered') else 'restricted_spot',
                  samples=args.samples, seed=args.seed, serialization_verified=True,
                  adaptive_sampling=scene.cycles.use_adaptive_sampling,
                  denoising=scene.cycles.use_denoising,
                  blend_sha256=digest(prefix.with_suffix('.blend')),
                  binary_sha256=digest(Path(bpy.app.binary_path)),
                  script_sha256={name: digest(Path(__file__).with_name(name)) for name in
                                 ['cycles_physical_diffraction_discs.py',
                                  'cycles_diffraction_suite.py', 'cycles_diffraction_scene.py']})
    prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.render:
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        report['wall_seconds_including_preparation'] = time.perf_counter() - start
        image = bpy.data.images.load(str(prefix.with_suffix('.exr')), check_existing=False)
        pixels = tuple(image.pixels[:])
        assert tuple(image.size) == (scene.render.resolution_x, scene.render.resolution_y)
        if not pixels or not all(math.isfinite(value) for value in pixels):
            raise RuntimeError('Empty or nonfinite disc render')
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_depth = '8'
        bpy.data.images['Render Result'].save_render(str(prefix.with_suffix('.png')), scene=scene)
        report.update(status='rendered_pending_visual_and_numerical_review',
                      exr_sha256=digest(prefix.with_suffix('.exr')),
                      png_sha256=digest(prefix.with_suffix('.png')))
        prefix.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
