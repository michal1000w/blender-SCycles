# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""One bounded warm-render benchmark of an existing scene; no adaptive sampling."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import bpy
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--warmup', action='store_true', required=True,
               help='Also tells the Metal backend to finish specialization during warmup')
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
a.output = a.output.resolve()
a.output.mkdir(parents=True, exist_ok=False)
bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()))
s = bpy.context.scene
s.cycles.samples = 512
s.cycles.seed = 11
s.cycles.use_adaptive_sampling = False
s.cycles.use_denoising = False
s.cycles.time_limit = 0
s.cycles.use_layer_samples = 'IGNORE'
s.cycles.use_bidirectional_path_tracing = False
s.cycles.use_guiding = False
s.cycles.use_photon_mapping = False
s.render.resolution_y = round(512 * s.render.resolution_y / s.render.resolution_x)
s.render.resolution_x = 512
s.render.resolution_percentage = 100
s.render.use_persistent_data = True
s.render.use_compositing = False
for layer in s.view_layers:
    layer.cycles.use_denoising = False
    layer.cycles.denoising_store_passes = False
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'METAL'
prefs.get_devices()
for device in prefs.devices:
    device.use = device.type == 'METAL'
assert any(device.use for device in prefs.devices)
s.cycles.device = 'GPU'
assert any(n.bl_idname == 'ShaderNodeBsdfDiffraction' and n.quality == 'FAST'
           for m in bpy.data.materials if m.use_nodes for n in m.node_tree.nodes)
s.render.image_settings.file_format = 'OPEN_EXR'
s.render.image_settings.color_depth = '32'
s.render.filepath = str(a.output / 'render.exr')
bpy.ops.wm.save_as_mainfile(filepath=str(a.output / 'benchmark.blend'))
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
report = {'binary_sha256': digest(bpy.app.binary_path), 'scene_sha256': digest(a.scene),
          'script_sha256': digest(__file__), 'samples': 512, 'adaptive_sampling': False,
          'denoising': False, 'transport': 'pt', 'persistent_data': True,
          'resolution': [s.render.resolution_x, s.render.resolution_y],
          'devices': [d.name for d in prefs.devices if d.use], 'runs': [],
          'scope': 'One warmup and three unchanged-scene measured renders. Times include EXR writing; not GPU timestamps, an equal-error comparison, or a universal speed ranking.'}
for index in range(4):
    start = time.perf_counter()
    bpy.ops.render.render(write_still=index > 0)
    elapsed = time.perf_counter() - start
    report['runs'].append({'warmup': index == 0, 'seconds': elapsed})
    (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
image = bpy.data.images.load(s.render.filepath, check_existing=False)
pixels = np.asarray(image.pixels[:], dtype=np.float32)
assert pixels.size and np.isfinite(pixels).all()
np.save(a.output / 'pixels.npy', pixels, allow_pickle=False)
bpy.data.images.remove(image)
s.render.image_settings.file_format = 'PNG'
s.render.image_settings.color_depth = '8'
bpy.data.images['Render Result'].save_render(str(a.output / 'render.png'), scene=s)
report['all_finite'] = True
report['median_measured_seconds'] = statistics.median(r['seconds'] for r in report['runs'][1:])
(a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
