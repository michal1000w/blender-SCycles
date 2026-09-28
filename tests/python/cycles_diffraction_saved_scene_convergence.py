# SPDX-FileCopyrightText: 2026 Blender Authors
# SPDX-License-Identifier: Apache-2.0
"""Rerender a saved physical grating scene with independent random seeds.

Uncertainty is estimated between render means, not between pixels. Raw signed
linear EXR values are never clipped, brightness-normalized or denoised.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import bpy

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--scene',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--seeds',type=int,nargs='+',default=[29,47,83])
p.add_argument('--samples',type=int,default=4096)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:])
if len(a.seeds)<3 or len(set(a.seeds))!=len(a.seeds) or min(a.seeds)<0 or a.samples<=0:
    p.error('Use at least three distinct nonnegative seeds and positive samples')
a.output.mkdir(parents=True,exist_ok=False)
bpy.ops.wm.open_mainfile(filepath=str(a.scene.resolve()))
scene=bpy.context.scene
scene.render.use_persistent_data=True
scene.render.image_settings.file_format='OPEN_EXR'
scene.render.image_settings.color_depth='32'
scene.cycles.samples=a.samples
scene.cycles.use_adaptive_sampling=False
scene.cycles.use_denoising=False
scene.cycles.time_limit=0
scene.cycles.use_layer_samples='IGNORE'
preferences=bpy.context.preferences.addons['cycles'].preferences
preferences.compute_device_type='METAL'
preferences.get_devices()
for device in preferences.devices:device.use=device.type=='METAL'
if not any(d.use for d in preferences.devices):raise RuntimeError('No Metal device')
scene.cycles.device='GPU'
def sampling_settings():
    return dict(samples=scene.cycles.samples,
                adaptive_sampling=scene.cycles.use_adaptive_sampling,
                denoising=scene.cycles.use_denoising,
                time_limit=scene.cycles.time_limit,
                use_layer_samples=scene.cycles.use_layer_samples,
                resolution_percentage=scene.render.resolution_percentage)
settings=sampling_settings()
report=dict(input_scene_sha256=hashlib.sha256(a.scene.read_bytes()).hexdigest(),samples=a.samples,
            binary_sha256=hashlib.sha256(Path(bpy.app.binary_path).read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            sampling_settings=settings,status='running',
            device_names=[d.name for d in preferences.devices if d.use],
            use_bdpt=scene.cycles.use_bidirectional_path_tracing,use_guiding=scene.cycles.use_guiding,
            resolution=[scene.render.resolution_x,scene.render.resolution_y],runs=[],scope=__doc__)
for seed in a.seeds:
    scene.cycles.seed=seed
    prefix=a.output/f'seed_{seed}'
    scene.render.filepath=str(prefix.with_suffix('.exr'))
    bpy.ops.wm.save_as_mainfile(filepath=str(prefix.with_suffix('.blend')))
    if sampling_settings()!=settings:
        raise RuntimeError('Sampling settings changed before rendering')
    bpy.ops.render.render(write_still=True)
    if sampling_settings()!=settings:
        raise RuntimeError('Sampling settings changed during rendering')
    image=bpy.data.images.load(str(prefix.with_suffix('.exr')),check_existing=False)
    pixels=tuple(image.pixels[:])
    if not pixels or image.channels<3 or not all(math.isfinite(x) for x in pixels):
        raise RuntimeError('Invalid render pixels')
    mean=[statistics.fmean(pixels[c::image.channels]) for c in range(3)]
    bpy.data.images.remove(image)
    report['runs'].append(dict(seed=seed,mean_rgb=mean,sampling_settings=sampling_settings(),
      blend_sha256=hashlib.sha256(prefix.with_suffix('.blend').read_bytes()).hexdigest(),
      exr_sha256=hashlib.sha256(prefix.with_suffix('.exr').read_bytes()).hexdigest()))
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
report['mean_rgb']=[statistics.fmean(r['mean_rgb'][c] for r in report['runs']) for c in range(3)]
report['between_seed_standard_error_rgb']=[statistics.stdev(r['mean_rgb'][c] for r in report['runs'])/
                                         math.sqrt(len(report['runs'])) for c in range(3)]
report['status']='completed_requires_reference_comparison'
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
