# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Blender worker: coherent Combined invariance and native surface-data parity."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
import numpy as np
import OpenImageIO as oiio


def read_passes(path):
    image = oiio.ImageInput.open(str(path))
    if image is None:
        raise RuntimeError(oiio.geterror())
    result = {}
    subimage = 0
    while image.seek_subimage(subimage, 0):
        spec = image.spec()
        pixels = np.asarray(image.read_image(format=oiio.FLOAT))
        for channel, name in enumerate(spec.channelnames):
            result[name] = pixels[..., channel]
        subimage += 1
    image.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blend', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--denoising-data', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:])
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'adaptive_sampling': False, 'samples': 32,
              'binary_sha256': hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
              'source_scene_sha256': hashlib.sha256(args.blend.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report_path = args.output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2))
    try:
        bpy.ops.wm.open_mainfile(filepath=str(args.blend.resolve()))
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'GPU'
        scene.cycles.samples = 32
        scene.cycles.seed = 19
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        scene.cycles.use_bidirectional_path_tracing = True
        scene.cycles.use_guiding = False
        scene.cycles.blur_glossy = 0.0
        scene.render.resolution_x = scene.render.resolution_y = 64
        scene.render.resolution_percentage = 100
        scene.render.image_settings.media_type = 'MULTI_LAYER_IMAGE'
        scene.render.image_settings.file_format = 'OPEN_EXR_MULTILAYER'
        scene.render.image_settings.color_depth = '32'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'METAL'
        prefs.get_devices()
        for device in prefs.devices:
            device.use = device.type == 'METAL'
        assert any(d.use for d in prefs.devices), 'No Metal device'
        layer = scene.view_layers[0]
        for prop in layer.bl_rna.properties:
            if (prop.identifier.startswith('use_pass_') and prop.identifier != 'use_pass_combined'
                    and prop.type == 'BOOLEAN' and not prop.is_readonly):
                setattr(layer, prop.identifier, False)
        layer.use_pass_combined = True
        layer.cycles.denoising_store_passes = False
        for aov in list(layer.aovs):
            layer.aovs.remove(aov)
        for obj in scene.objects:
            obj.pass_index = 7
        for material in bpy.data.materials:
            material.pass_index = 11
        outputs = {}
        report['renders'] = {}
        for case in ('combined_only', 'coherent_data', 'native_data'):
            scene.cycles.use_coherent_specular_connections = case != 'native_data'
            if case == 'coherent_data':
                layer.cycles.denoising_store_passes = args.denoising_data
                for prop in ('use_pass_z', 'use_pass_position', 'use_pass_normal', 'use_pass_uv',
                             'use_pass_object_index', 'use_pass_material_index',
                             'use_pass_diffuse_color', 'use_pass_glossy_color',
                             'use_pass_transmission_color', 'use_pass_mist',
                             'use_pass_cryptomatte_object', 'use_pass_cryptomatte_material'):
                    setattr(layer, prop, True)
                for name, kind in (('ProbeColor', 'COLOR'), ('ProbeValue', 'VALUE')):
                    aov = layer.aovs.add()
                    aov.name, aov.type = name, kind
                    for material in bpy.data.materials:
                        if not material.use_nodes:
                            continue
                        node = material.node_tree.nodes.new('ShaderNodeOutputAOV')
                        node.name = name
                        node.aov_name = name
                        node.inputs['Color'].default_value = (0.2, 0.4, 0.6, 1)
                        node.inputs['Value'].default_value = 0.375
            path = args.output / (case + '.exr')
            scene.render.filepath = str(path.resolve())
            bpy.ops.wm.save_as_mainfile(filepath=str((args.output / (case + '.blend')).resolve()))
            start = time.perf_counter()
            bpy.ops.render.render(write_still=True)
            report['renders'][case] = {'seconds_including_initialization': time.perf_counter() - start}
            outputs[case] = read_passes(path)
        baseline, coherent, native = (outputs[c] for c in ('combined_only', 'coherent_data', 'native_data'))
        assert coherent.keys() == native.keys(), 'Native/coherent channel mismatch'
        assert all(np.isfinite(v).all() for passes in outputs.values() for v in passes.values()), 'Non-finite pass'
        combined = [key for key in baseline if '.Combined.' in key]
        data = [key for key in coherent if '.Combined.' not in key]
        assert combined and data, 'Missing Combined or data channels'
        report['combined_max_delta'] = max(float(np.max(np.abs(baseline[k] - coherent[k]))) for k in combined)
        report['data_max_delta_by_channel'] = {
            k: float(np.max(np.abs(coherent[k] - native[k]))) for k in data}
        assert report['combined_max_delta'] <= 1e-6, 'Data passes changed coherent Combined'
        assert max(report['data_max_delta_by_channel'].values()) <= 1e-6, 'Data differs from native surface output'
        for suffix, expected in (('.ProbeValue.X', 0.375), ('.Object Index.X', 7), ('.Material Index.X', 11)):
            matched = [v for k, v in coherent.items() if k.endswith(suffix)]
            assert matched and any(np.any(np.isclose(v, expected)) for v in matched), suffix
        scene.cycles.use_coherent_specular_connections = True
        layer.cycles.use_pass_volume_direct = True
        try:
            bpy.ops.render.render()
        except RuntimeError as error:
            report['volume_pass_rejection'] = str(error)
            assert 'unsupported' in str(error).lower()
        else:
            raise AssertionError('Unsupported light pass was accepted')
        report['status'] = 'passed'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
