# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Convert a saved absorbing-disc fixture to the existing scalar GPU model.

No physical-equivalence claim. The normal-incidence
Fresnel reflectance sets a constant colour; angular metal Fresnel is approximated.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import bpy

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--transport', choices=('pt', 'bdpt', 'guided', 'bdpt_guided'), default='pt')
p.add_argument('--samples', type=int, default=1024)
p.add_argument('--render', action='store_true')
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
if a.samples < 1:
    p.error('Samples must be positive')
a.output.mkdir(parents=True, exist_ok=False)
bpy.context.preferences.filepaths.file_preview_type = 'NONE'
bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()))
converted = []
for material in bpy.data.materials:
    if not material.node_tree:
        continue
    tree = material.node_tree
    for old in list(tree.nodes):
        if old.bl_idname != 'ShaderNodeBsdfDiffraction':
            continue
        if old.substrate_extinction <= 0:
            raise ValueError('This comparison fixture supports absorbing reflectors only')
        n, k, ni = old.substrate_ior, old.substrate_extinction, old.incident_ior
        reflectance = ((n - ni)**2 + k*k) / ((n + ni)**2 + k*k)
        if old.inputs['Color'].is_linked:
            raise ValueError('Linked colour needs explicit spectral reflectance handling')
        new = tree.nodes.new('ShaderNodeBsdfAnisotropic')
        new.distribution = 'GGX'
        new.label = 'Fast scalar diffraction candidate'
        new.location = old.location
        new.inputs['Roughness'].default_value = 0
        new.inputs['Diffraction Weight'].default_value = 1
        for socket, value in [('Diffraction Pitch', old.pitch),
                              ('Diffraction Depth', old.depth),
                              ('Diffraction Duty Cycle', old.duty_cycle),
                              ('Diffraction Medium IOR', ni)]:
            new.inputs[socket].default_value = value
        colour = old.inputs['Color'].default_value
        new.inputs['Color'].default_value = (*[reflectance * colour[c] for c in range(3)], 1)
        for name in ('Normal', 'Tangent'):
            new.inputs[name].default_value = old.inputs[name].default_value
            for link in list(old.inputs[name].links):
                tree.links.new(link.from_socket, new.inputs[name])
        for link in list(old.outputs['BSDF'].links):
            tree.links.new(new.outputs['BSDF'], link.to_socket)
        converted.append(dict(material=material.name, pitch_nm=old.pitch,
                              depth_nm=old.depth, duty_cycle=old.duty_cycle,
                              constant_reflectance=reflectance))
        tree.nodes.remove(old)
if len(converted) != 2:
    raise ValueError('Expected exactly two disc materials')
scene = bpy.context.scene
scene['validation_status'] = 'Fast scalar approximation; awaiting visual and transport validation'
for note in list(bpy.data.texts):
    if note.name.startswith('READ ME - physical grating fixture'):
        bpy.data.texts.remove(note)
bpy.data.texts.new('READ ME - fast grating fixture').write(__doc__)
scene.cycles.samples = a.samples
scene.cycles.use_bidirectional_path_tracing = a.transport in ('bdpt', 'bdpt_guided')
scene.cycles.use_guiding = a.transport in ('guided', 'bdpt_guided')
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.time_limit = 0
scene.cycles.use_layer_samples = 'IGNORE'
output = a.output / 'fast_discs.blend'
scene.render.image_settings.file_format = 'OPEN_EXR'
scene.render.image_settings.color_depth = '32'
scene.render.filepath = str(output.with_suffix('.exr').resolve())
bpy.ops.wm.save_as_mainfile(filepath=str(output.resolve()))
bpy.ops.wm.open_mainfile(filepath=str(output.resolve()))
assert not any(n.bl_idname == 'ShaderNodeBsdfDiffraction'
               for m in bpy.data.materials if m.node_tree for n in m.node_tree.nodes)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
report = dict(
    status='saved_not_rendered', scope=__doc__, materials=converted,
    source_scene_sha256=digest(a.scene), blend_sha256=digest(output),
    script_sha256=digest(Path(__file__)), serialization_verified=True,
    coherent_transport=False, maxwell_cache_required=False,
    transport=a.transport, samples=a.samples,
    binary_sha256=digest(Path(bpy.app.binary_path)))
scene = bpy.context.scene
def settings():
    return dict(samples=scene.cycles.samples, adaptive=scene.cycles.use_adaptive_sampling,
                denoising=scene.cycles.use_denoising, time_limit=scene.cycles.time_limit,
                layer_samples=scene.cycles.use_layer_samples,
                bdpt=scene.cycles.use_bidirectional_path_tracing, guiding=scene.cycles.use_guiding)
expected = dict(samples=a.samples, adaptive=False, denoising=False, time_limit=0,
                layer_samples='IGNORE', bdpt=a.transport in ('bdpt', 'bdpt_guided'),
                guiding=a.transport in ('guided', 'bdpt_guided'))
assert settings() == expected
report['settings'] = settings()
def save_report():
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
save_report()
if a.render:
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = 'METAL'
    preferences.get_devices()
    for device in preferences.devices:
        device.use = device.type == 'METAL'
    report['devices'] = [d.name for d in preferences.devices if d.use]
    if not report['devices']:
        raise RuntimeError('No Metal device')
    scene.cycles.device = 'GPU'
    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    report['wall_seconds_including_preparation'] = time.perf_counter() - start
    assert settings() == expected
    image = bpy.data.images.load(str(output.with_suffix('.exr').resolve()), check_existing=False)
    pixels = tuple(image.pixels[:])  # Force Blender's lazy EXR loading before validation.
    if not pixels or not all(math.isfinite(v) for v in pixels):
        raise RuntimeError('Invalid rendered pixels')
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '8'
    preview = output.with_suffix('.preview.png')
    image.save_render(str(preview.resolve()), scene=scene)
    report.update(status='rendered_requires_review', exr_sha256=digest(output.with_suffix('.exr')),
                  preview_sha256=digest(preview), timing_scope='Includes preparation; isolation must be checked separately')
    save_report()
