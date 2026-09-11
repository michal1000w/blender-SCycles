#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Blender Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Render a supplied blend without saving changes, retaining linear pixels and timings."""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import bpy
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene', type=pathlib.Path, required=True)
p.add_argument('--output', type=pathlib.Path, required=True)
p.add_argument('--samples', type=int)
p.add_argument('--percentage', type=int)
p.add_argument('--seed', type=int, default=11)
p.add_argument('--integrator', choices=['saved', 'pt', 'bdpt'], default='saved')
p.add_argument('--warmup', action='store_true')
p.add_argument('--raw', action='store_true', help='Disable denoising for variance comparisons')
p.add_argument('--fixed-samples', action='store_true', help='Disable adaptive stopping and time limits')
p.add_argument('--guiding', action=argparse.BooleanOptionalAction, default=True)
a = p.parse_args(sys.argv[sys.argv.index('--') + 1:])
a.output.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()), load_ui=False)
s = bpy.context.scene
s.render.engine = 'CYCLES'
c = s.cycles
c.device = 'GPU'
c.use_guiding = a.guiding
if a.integrator != 'saved':
    c.use_bidirectional_path_tracing = a.integrator == 'bdpt'
if a.samples is not None:
    c.samples = a.samples
if a.percentage is not None:
    s.render.resolution_percentage = a.percentage
c.seed = a.seed
if a.raw:
    c.use_denoising = False
if a.fixed_samples:
    c.use_adaptive_sampling = False
    c.time_limit = 0
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'METAL'
prefs.get_devices()
devices = [d.name for d in prefs.devices if d.type == 'METAL']
if not devices:
    raise RuntimeError('No Metal GPU')
for d in prefs.devices:
    d.use = d.type == 'METAL'
s.render.image_settings.file_format = 'OPEN_EXR'
s.render.image_settings.color_depth = '32'
s.render.image_settings.color_mode = 'RGB'
s.render.filepath = str(a.output.resolve().with_suffix('.exr'))
metadata = dict(scene=str(a.scene.resolve()), scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),
                devices=devices, resolution=[s.render.resolution_x, s.render.resolution_y,
                                             s.render.resolution_percentage],
                cycles={p.identifier: getattr(c, p.identifier) for p in c.bl_rna.properties
                        if p.type in {'INT', 'FLOAT', 'BOOLEAN', 'ENUM', 'STRING'}})
binary = pathlib.Path(bpy.app.binary_path)
metadata['binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
metadata['benchmark_script_sha256'] = hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
metadata['cycles_environment'] = {key: value for key, value in os.environ.items()
                                  if key.startswith('CYCLES_')}
source = binary.parents[1] / 'Resources' / '5.3/scripts/addons_core/cycles/source'
digest = hashlib.sha256()
for path in sorted(source.rglob('*')):
    if path.is_file():
        digest.update(str(path.relative_to(source)).encode() + b'\0')
        digest.update(path.read_bytes())
metadata['kernel_source_sha256'] = digest.hexdigest()
a.output.with_suffix('.settings.json').write_text(json.dumps(metadata, indent=2))
times = []
for i in range(2 if a.warmup else 1):
    start = time.perf_counter()
    bpy.ops.render.render(write_still=False)
    times.append(time.perf_counter() - start)
    print('GUIDING_FILE_TIMING', i, times[-1], flush=True)
bpy.data.images['Render Result'].save_render(s.render.filepath, scene=s)
import OpenImageIO as oiio
img = oiio.ImageInput.open(s.render.filepath)
if img is None:
    raise RuntimeError('Cannot read the saved linear EXR')
channels = list(img.spec().channelnames)
if not all(channel in channels for channel in ['R', 'G', 'B']):
    img.close()
    raise RuntimeError(f'Expected explicit RGB channels, found {channels}')
rgb = np.asarray(img.read_image(format=oiio.FLOAT))[..., [channels.index(c) for c in ['R', 'G', 'B']]]
img.close()
if not np.isfinite(rgb).all():
    raise RuntimeError('Nonfinite rendered pixels')
np.save(a.output.with_suffix('.npy'), rgb)
metadata.update(seconds=times[-1], all_render_seconds=times, mean=float(rgb.mean()),
                peak=float(rgb.max()), shape=list(rgb.shape), exr_channels=channels)
a.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
print('GUIDING_FILE_RESULT', json.dumps(metadata), flush=True)
