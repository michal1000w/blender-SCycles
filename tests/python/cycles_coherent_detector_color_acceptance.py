#!/usr/bin/env python3
"""Prepare passive colored Lambertian receiver tests; performs no render.

The independent mirror reference is multiplied by known RGB reflectance.
Spatial test uses a world checker with exactly black alternating squares. Pixels
near its boundaries are excluded, so no pixel-center texture approximation
enters the comparison. This models a coherent Lambertian receiver only.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np


def main():
    args = sys.argv[sys.argv.index('--') + 1:]
    output = Path(args[0]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name('cycles_coherent_mirror_acceptance.py')
    spec = importlib.util.spec_from_file_location('mirror_reference', source)
    mirror = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mirror)
    scene, detector, interface, masks, sources, camera = mirror.configure_scene(True)
    scene.cycles.use_coherent_specular_connections = True
    scene.cycles.samples = 128
    detector.cycles.coherent_interface = 'DETECTOR'
    interface.cycles.coherent_interface = 'MIRROR'
    scene.cycles.coherent_max_interface_events = 1
    bpy.context.view_layer.update()
    mirror.make_reference(sources, camera, output)
    white = np.load(output / 'virtual_source_reference.npz')
    color = tuple(float(np.float32(v)) for v in (0.25, 0.5, 0.75))
    cases = {}
    for principled in (False, True):
        for textured in (False, True):
            name = ('principled' if principled else 'diffuse') + ('_black_checker' if textured else '_color')
            case_dir = output / name
            case_dir.mkdir()
            material = bpy.data.materials.new(name)
            material.use_nodes = True
            nodes = material.node_tree.nodes
            nodes.clear()
            out = nodes.new('ShaderNodeOutputMaterial')
            bsdf = nodes.new('ShaderNodeBsdfPrincipled' if principled else 'ShaderNodeBsdfDiffuse')
            color_input = bsdf.inputs['Base Color' if principled else 'Color']
            color_input.default_value = (*color, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.5 if principled else 0.0
            if principled:
                for weight in ('Metallic', 'Transmission Weight', 'Specular IOR Level',
                               'Coat Weight', 'Sheen Weight', 'Subsurface Weight', 'Emission Strength'):
                    bsdf.inputs[weight].default_value = 0.0
                bsdf.inputs['Diffuse Roughness'].default_value = 0.0
                bsdf.inputs['Alpha'].default_value = 1.0
            material.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
            if textured:
                geometry = nodes.new('ShaderNodeNewGeometry')
                offset = nodes.new('ShaderNodeVectorMath')
                offset.operation = 'ADD'
                offset.inputs[1].default_value = (0.0, 0.0, 0.031)
                checker = nodes.new('ShaderNodeTexChecker')
                checker.inputs['Color1'].default_value = (*color, 1.0)
                checker.inputs['Color2'].default_value = (0.0, 0.0, 0.0, 1.0)
                checker.inputs['Scale'].default_value = 10.0
                links = material.node_tree.links
                links.new(geometry.outputs['Position'], offset.inputs[0])
                links.new(offset.outputs[0], checker.inputs['Vector'])
                links.new(checker.outputs['Color'], color_input)
            detector.data.materials.clear()
            detector.data.materials.append(material)
            factor = np.broadcast_to(np.array(color), (*white['y_m'].shape, 3)).copy()
            roi = white['roi_mask'].copy()
            if textured:
                # The +.031 world-z offset fixes the detector's z cell at20.
                # Odd parity selects Color1; even parity is exactly black.
                ix = np.floor(white['x_m'] * 10.0).astype(np.int64)
                iy = np.floor(white['y_m'] * 10.0).astype(np.int64)
                colored = ((ix + iy + 20) % 2) == 1
                factor[~colored] = 0.0
                # Fixed physical margin exceeds the pixel integration footprint.
                fx = np.mod(white['x_m'], 0.1)
                fy = np.mod(white['y_m'], 0.1)
                roi &= (np.minimum(fx, 0.1 - fx) > 0.003)
                roi &= (np.minimum(fy, 0.1 - fy) > 0.003)
            reference = {key: white[key][..., None] * factor for key in
                         ('phase_0_radiance', 'phase_pi_radiance', 'incoherent_radiance')}
            reference.update(roi_mask=roi, reflectance_rgb=factor, x_m=white['x_m'], y_m=white['y_m'])
            np.savez_compressed(case_dir / 'virtual_source_reference.npz', **reference)
            blends = {}
            for phase_name, phase in (('phase_0', 0.0), ('phase_pi', math.pi),
                                      ('incoherent_connector_control', 0.0)):
                sources[0].data.cycles.coherence_phase = 0.0
                sources[1].data.cycles.coherence_phase = phase
                for index, light in enumerate(sources):
                    light.data.cycles.coherence_group = index + 1 if phase_name == 'incoherent_connector_control' else 1
                    light.data.cycles.coherence_length_m = 1.0
                scene.render.filepath = f'//{name}_{phase_name}'
                path = case_dir / f'{phase_name}.blend'
                bpy.ops.wm.save_as_mainfile(filepath=str(path))
                blends[phase_name] = {'path': str(path), 'sha256': mirror.sha256(path)}
            (case_dir / 'manifest.json').write_text(json.dumps(
                {'specular_connections': True, 'scene_variants': blends, 'scope': name}, indent=2) + '\n')
            cases[name] = {'scenes': blends, 'reference': str(case_dir / 'virtual_source_reference.npz'),
                           'black_interior_pixels': int(np.count_nonzero(roi & np.all(factor == 0.0, axis=2))) if textured else 0}
    # Host-only negative: an effective film recreates Principled's specular
    # layer even when Specular IOR Level is zero. It must be rejected.
    bsdf.inputs['Thin Film Thickness'].default_value = 250.0
    film_path = output / 'principled_effective_film_invalid.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(film_path))
    bsdf.inputs['Thin Film Thickness'].default_value = 0.0
    for polygon in detector.data.polygons:
        polygon.use_smooth = True
    smooth_path = output / 'smooth_detector_invalid.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(smooth_path))
    for polygon in detector.data.polygons:
        polygon.use_smooth = False
    detector.shadow_terminator_shading_offset = .25
    terminator_path = output / 'terminator_detector_invalid.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(terminator_path))
    detector.shadow_terminator_shading_offset = 0.0
    manifest = {'scope': 'passive Lambertian RGB detector, one planar mirror; no arbitrary BSDF phase',
                'specular_connections': scene.cycles.use_coherent_specular_connections,
                'declared_detector': detector.cycles.coherent_interface,
                'declared_interface': interface.cycles.coherent_interface,
                'constant_reflectance_rgb': color, 'cases': cases,
                'negative_cases': {'effective_film': {'blend': str(film_path),
                    'expected_error': 'requires one passive Lambertian Diffuse or pure diffuse Principled shader'},
                    'smooth': {'blend': str(smooth_path), 'expected_error': 'require flat normals and zero shading terminator offset'},
                    'terminator': {'blend': str(terminator_path), 'expected_error': 'require flat normals and zero shading terminator offset'}},
                'settings': {'samples': scene.cycles.samples, 'adaptive': scene.cycles.use_adaptive_sampling,
                             'denoise': scene.cycles.use_denoising},
                'gates': {'absolute_rgb_rmse_max_radiance': 0.012,
                          'absolute_rgb_mean_error_max_radiance': 0.005,
                          'black_interior_absolute_max': 1e-6},
                'texture': 'world Checker(Position + z .031), Scale10: odd floor(x*10)+floor(y*10)+20 is RGB; even is black; .003m edge margin',
                'binary': bpy.app.binary_path, 'script_sha256': mirror.sha256(Path(__file__))}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'prepared': len(cases) * 3, 'rendered': False}))


def validate_negative(output):
    output = Path(output).resolve()
    manifest = json.loads((output / 'manifest.json').read_text())
    results = {}
    for name, case in manifest['negative_cases'].items():
        bpy.ops.wm.open_mainfile(filepath=case['blend'])
        bpy.context.scene.render.resolution_x = bpy.context.scene.render.resolution_y = 8
        bpy.context.scene.cycles.samples = 1
        preferences = bpy.context.preferences.addons['cycles'].preferences
        preferences.compute_device_type = 'METAL'
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == 'METAL'
        result = {'passed': False, 'scope': 'host rejection of unsupported detector'}
        try:
            bpy.ops.render.render()
            result['error'] = 'invalid detector unexpectedly rendered'
        except Exception as error:
            result['error'] = str(error)
            result['passed'] = case['expected_error'] in str(error)
        results[name] = result
    results['passed'] = all(case['passed'] for case in results.values())
    (output / 'detector_negative_validation.json').write_text(json.dumps(results, indent=2)+'\n')
    if not results['passed']:
        raise RuntimeError(str(results))

if __name__ == '__main__':
    args = sys.argv[sys.argv.index('--') + 1:]
    if args[0] == '--validate-negative':
        validate_negative(args[1])
    else:
        main()
