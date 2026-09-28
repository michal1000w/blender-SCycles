# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired covered-grating diagnostic; preserve MNEE-on/off images without normalization."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--samples', type=int, default=512)
p.add_argument('--resolution', type=int, default=384)
p.add_argument('--device', choices=['CPU', 'METAL'], default='METAL')
p.add_argument('--transport', choices=['pt', 'bdpt', 'guided', 'bdpt_guided'], default='pt')
p.add_argument('--cover', choices=['glass', 'hidden', 'transparent'], default='glass')
p.add_argument('--disable-object-caustics', action='store_true')
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()))
scene = bpy.context.scene
cover = scene.objects['1.2 mm enclosing cover']
if a.cover == 'hidden':
    cover.hide_render = True
elif a.cover == 'transparent':
    tree = cover.active_material.node_tree
    out = next(node for node in tree.nodes if node.bl_idname == 'ShaderNodeOutputMaterial')
    transparent = tree.nodes.new('ShaderNodeBsdfTransparent')
    tree.links.new(transparent.outputs[0], out.inputs['Surface'])
if a.disable_object_caustics:
    for obj in scene.objects:
        obj.cycles.is_caustics_receiver = False
        obj.cycles.is_caustics_caster = False
scene.cycles.samples = a.samples
scene.cycles.seed = 11
scene.cycles.use_adaptive_sampling = False
scene.cycles.use_denoising = False
scene.cycles.time_limit = 0
scene.cycles.use_layer_samples = 'IGNORE'
scene.cycles.use_bidirectional_path_tracing = a.transport in ('bdpt', 'bdpt_guided')
scene.cycles.use_guiding = a.transport in ('guided', 'bdpt_guided')
scene.cycles.device = 'CPU'
if a.device == 'METAL':
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'METAL'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == 'METAL'
    assert any(device.use for device in prefs.devices)
    scene.cycles.device = 'GPU'
scene.render.resolution_y = round(a.resolution * scene.render.resolution_y / scene.render.resolution_x)
scene.render.resolution_x = a.resolution
scene.render.resolution_percentage = 100
scene.render.use_persistent_data = True
root = Path(__file__).resolve().parents[2]
report = dict(scope=__doc__, samples=a.samples, device=a.device,
              transport=a.transport,
              fresh_saved_scene_per_case=True,
              cover=a.cover, disable_object_caustics=a.disable_object_caustics,
              adaptive_sampling=False, denoising=False, seed=11,
              scene_sha256=digest(a.scene), binary_sha256=digest(Path(bpy.app.binary_path)),
              script_sha256=digest(Path(__file__)),
              intersection_source_sha256=digest(root/'intern/cycles/kernel/integrator/intersect_closest.h'),
              renders=[])
for enabled in [True, False]:
    name = 'mnee_on' if enabled else 'mnee_off'
    for light in bpy.data.lights:
        light.cycles.is_caustics_light = enabled
        light.update_tag()
    scene.render.filepath = str(a.output/(name+'.exr'))
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_depth = '32'
    bpy.ops.wm.save_as_mainfile(filepath=str(a.output/(name+'.blend')))
    # Custom light properties must not rely on persistent-session invalidation.
    # Reload the saved case so each treatment has a fresh render engine.
    bpy.ops.wm.open_mainfile(filepath=str(a.output/(name+'.blend')))
    scene = bpy.context.scene
    assert all(light.cycles.is_caustics_light == enabled for light in bpy.data.lights)
    assert scene.cycles.samples == a.samples and not scene.cycles.use_adaptive_sampling
    start = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    elapsed = time.perf_counter() - start
    image = bpy.data.images.load(scene.render.filepath, check_existing=False)
    pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(-1,4)[:,:3]
    assert np.isfinite(pixels).all()
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '8'
    image.save_render(str(a.output/(name+'.png')), scene=scene)
    report['renders'].append(dict(mnee=enabled, mean_rgb=pixels.mean(axis=0,dtype=np.float64).tolist(),
        render_seconds_including_preparation=elapsed,
        exr_sha256=digest(a.output/(name+'.exr')), png_sha256=digest(a.output/(name+'.png')),
        blend_sha256=digest(a.output/(name+'.blend'))))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
report['completed'] = True
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
